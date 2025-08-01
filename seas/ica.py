import os
import re
import numpy as np
from datetime import datetime
from sklearn.decomposition import FastICA
from scipy import linalg
from timeit import default_timer as timer
from typing import Tuple

from seas.waveletAnalysis import waveletAnalysis
from seas.signalanalysis import butterworth, sort_noise, lag_n_autocorr
from seas.hdf5manager import hdf5manager
from seas.video import rotate, save, rescale, play, scale_video


def project(vector: np.ndarray,
            shape: Tuple[int, int, int],
            roimask: np.ndarray = None,
            n_components: int = None,
            svd_multiplier: float = 5,
            calc_residuals: bool = True,
            max_iter: int = 1000):
    '''
    Apply an ica decomposition to the first axis of the input vector.  
    If a roimask is provided, the flattened roimask will be used to crop the vector before decomposition.

    If n_components is not set, an adaptive svd threshold is used 
    (see approximate_svd_linearity_transition), 
    with the hyperparameter svd_mutliplier.  

    Residuals lost in the ICA projection are captured if calc_residuals == True.  
    This represents the signal lost by ICA compression.

    Arguments:
        vector: 
            The (x*y, t) vector to be spatially ICA projected.
        shape:
            The shape of the original movie (t,x,y).
        roimask:
            The roimask to crop the vectorized movie (x,y).
        n_components:
            Manually request a set number of ICA components.
        svd_multiplier:
            The hyperparameter for svd adaptive thresholding.
        calc_residuals:
            Whether to calculate spatial and temporal residuals of projection compression.
        max_iter:
            Maximum iterations assigned for FastICA

    Returns:
        components: A dictionary containing all the results, metadata, and information regarding the filter applied.

            mean: 
                the original video mean
            roimask: 
                the mask applied to the video before decomposing
            shape: 
                the original shape of the movie array
            eig_mix: 
                the ICA mixing matrix
            timecourses: 
                the ICA component time series
            eig_vec: 
                the eigenvectors
            n_components:
                the number of components in eig_vec (reduced to only have 25% of total components as noise)
            project_meta:
                The metadata for the ica projection
            expmeta:
                All metadata created for this class
            lag1: 
                the lag-1 autocorrelation
            noise_components: 
                a vector (n components long) to store binary representation of which components were detected as noise 
            cutoff: 
                the signal-noise cutoff value

        if the n_components was automatically set, the following additional keys are also returned in components

            svd_cutoff: 
                the number of components originally decomposed
            lag1_full: 
                the lag-1 autocorrelation of the full set of components decomposed before cropping to only 25% noise components
            svd_multiplier: 
                the svd multiplier value used to determine cutoff
    '''
    print('\nCalculating Eigenspace\n-----------------------')
    assert (vector.ndim == 2), (
        'vector was not a two-dimensional np array.'
        'If input is a movie, be sure to convert shape to (xy, t)')

    if roimask is not None:
        print('Using roimask to crop video')
        assert roimask.size == vector.shape[0], \
        'Vector was not the same size as the cropped mask'

        print('Original size:', vector.shape)
        maskind = np.where(roimask.flat == 1)
        vector = vector[maskind]
        print('Reduced size:', vector.shape)

    mean = np.mean(vector, 0).flatten()
    vector = vector - mean

    components = {}
    components['mean'] = mean
    components['roimask'] = roimask
    components['shape'] = shape

    if svd_multiplier is None:
        svd_multiplier = 5

    if vector.dtype == np.float16:
        vector = vector.astype('float32', copy=False)

    if n_components is None:
        print('Calculating ICA (with n_component SVD estimator)...')

        t0 = timer()
        try:
            u, ev, _ = linalg.svd(vector, full_matrices=False)
        except ValueError:
            # LAPACK error if matricies are too big
            u, ev, _ = linalg.svd(vector,
                                  full_matrices=False,
                                  lapack_driver='gesvd')

        components['svd_eigval'] = ev

        #Get starting point for decomposition based on svd mutliplier * the approximate
        # point of transition to linearity in tail of ev components.
        cross_1 = approximate_svd_linearity_transition(ev)
        n_components = cross_1 * svd_multiplier

        components['increased_cutoff'] = 0

        while True:
            print('\nCalculating ICA with', n_components, 'components...')

            w_init = u[:n_components, :n_components].astype('float64')
            ica = FastICA(n_components=n_components,
                          max_iter=max_iter,
                          random_state=1000,
                          w_init=w_init)

            eig_vec = ica.fit_transform(vector)
            eig_mix = ica.mixing_

            noise, cutoff = sort_noise(eig_mix.T)

            p_signal = (1 - noise.sum() / noise.size) * 100

            if noise.size == shape[0]:  # All components are being used.
                break
            elif p_signal < 75:
                print('ICA components were under 75% signal ({0}% signal).'\
                    .format(p_signal))
                break
            elif n_components >= shape[0]:
                print('ICA components were under 75% signal ({0}% signal).'\
                    .format(p_signal))
                print('However, number of components is maxed out.')
                print('Using this decomposition...')
                break
            else:
                print('ICA components were over 75% signal ({0}% signal).'\
                    .format(p_signal))
                print('Recalculating with more components...')
                n_components += n_components // 2
                components['increased_cutoff'] += 1

                if n_components > shape[0]:
                    print('\nComponents maxed out!')
                    print('\tAttempted:', n_components)
                    n_components = shape[0]
                    print('\tReduced to:', shape[0])

        components['lag1_full'] = lag_n_autocorr(eig_mix.T, 1)
        components['svd_multiplier'] = svd_multiplier

        print('Cropping excess noise components')
        components['svd_cutoff'] = n_components
        reduced_n_components = int((noise.size - noise.sum()) * 1.25)

        print('reduced_n_components:', reduced_n_components)

        if reduced_n_components < n_components:
            print('Cropping', n_components, 'to', reduced_n_components)

            ev_sort = np.argsort(eig_mix.std(axis=0))
            eig_vec = eig_vec[:, ev_sort][:, ::-1]
            eig_mix = eig_mix[:, ev_sort][:, ::-1]
            noise = noise[ev_sort][::-1]

            eig_vec = eig_vec[:, :reduced_n_components]
            eig_mix = eig_mix[:, :reduced_n_components]
            n_components = reduced_n_components
            noise = noise[:reduced_n_components]

            components['lag1_full'] = components['lag1_full'][ev_sort][::-1]
        else:
            print('Less than 75% signal.  Not cropping excess noise.')

        components['noise_components'] = noise
        components['cutoff'] = cutoff
        t = timer() - t0
        print('Independent Component Analysis took: {0} sec'.format(t))

    else:
        print('Calculating ICA (' + str(n_components) + ' components)...')

        t0 = timer()
        ica = FastICA(n_components=n_components, max_iter=max_iter, random_state=1000)

        try:
            eig_vec = ica.fit_transform(vector)  # Eigenbrains
        except ValueError:
            print('Calculation exceeded float32 maximum.')
            print('Trying again with float64 vector...')
            # Value error if any value exceeds float32 maximum.
            # Overcome this by converting to float64.
            eig_vec = ica.fit_transform(vector.astype('float64'))

        t = timer() - t0
        print('Independent Component Analysis took: {0} sec'.format(t))
        eig_mix = ica.mixing_

        # Sort components by their eig val influence (approximated by timecourse standard deviation).
        ev_sort = np.argsort(eig_mix.std(axis=0))
        eig_vec = eig_vec[:, ev_sort][:, ::-1]
        eig_mix = eig_mix[:, ev_sort][:, ::-1]

        # Track component orientation and ensure positive spatial patterns
        flipped = np.ones(n_components)
        for i in range(n_components):
            # Find the index of maximum absolute value
            max_idx = np.argmax(np.abs(eig_vec[:, i]))
            # If that maximum value is negative, flip the component
            if eig_vec[max_idx, i] < 0:
                eig_vec[:, i] *= -1
                eig_mix[:, i] *= -1
                flipped[i] = -1
                
        noise, cutoff = sort_noise(eig_mix.T)
        components['noise_components'] = noise
        components['cutoff'] = cutoff
        components['flipped'] = flipped

    print('components shape:', eig_vec.shape)

    components['eig_mix'] = eig_mix
    components['timecourses'] = eig_mix.T

    n_components = eig_vec.shape[1]
    components['eig_vec'] = eig_vec
    components['n_components'] = n_components
    components['lag1'] = lag_n_autocorr(components['timecourses'], 1)

    if calc_residuals:
        try:
            vector = vector.astype('float64')
            rebuilt = rebuild(components,
                              artifact_components='none',
                              vector=True).T

            rebuilt -= rebuilt.mean(axis=0)
            vector -= vector.mean(axis=0)

            residuals = np.abs(vector - rebuilt)

            residuals_temporal = residuals.mean(axis=0)

            if roimask is not None:
                residuals_spatial = np.zeros(roimask.shape)
                residuals_spatial.flat[maskind] = residuals.mean(axis=1)
            else:
                residuals_spatial = np.reshape(residuals.mean(axis=1),
                                               (shape[1], shape[2]))

            components['residuals_spatial'] = residuals_spatial
            components['residuals_temporal'] = residuals_temporal

        except Exception as e:
            print('Residual Calculation Failed!!')
            print('\t', e)

    # Save filter metadata information about how and when movie was filtered in dictionary.
    project_meta = {}
    project_meta['time_elapsed'] = t
    project_meta['date'] = \
        datetime.now().strftime('%Y%m%d')[2:]
    fmt = '%Y-%m-%dT%H:%M:%SZ'
    project_meta['tstmp'] = \
        datetime.now().strftime(fmt)
    project_meta['n_components'] = n_components
    components['project_meta'] = project_meta

    print('\n')
    return components


def rebuild(components: dict,
            artifact_components: np.ndarray = None,
            t_start: int = None,
            t_stop: int = None,
            apply_mean_filter: bool = True,
            mlow: float = 0.5,
            mhigh: float = 1.0,
            apply_component_filter: bool = False,
            chigh: float = 1.0,
            apply_masked_mean: bool = False,
            filter_method: str = 'butterworth_highpass',
            fps: float = 7.5,
            include_noise: bool = True):
    '''
    Rebuild original vector space based on a subset of principal 
    components of the data.  Eigenvectors to use are specified where 
    artifact_components == False.  Returns a matrix data_r, the reconstructed 
    vector projected back into its original dimensions.

    Arguments:
        components: 
            The components from ica_project.  artifact_components must be assigned to components before rebuilding, or passed in explicitly
        artifact_components:
            Overrides the artifact_components key in components, to rebuild all components except those specified
        t_start: 
            The frame to start rebuilding the movie at.  If none is provided, the rebuilt movie starts at the first frame
        t_stop: 
            The frame to stop rebuilding the movie at.  If none is provided, the rebuilt movie ends at the last frame
        apply_mean_filter:
            Whether to apply a filter to the mean signal.
        mlow:
            A float determining the highpass cutoff for the mean filter, if used.
        mhigh:
            A float determining the lowpass cutoff for the mean filter, if used.
        apply_component_filter:
            Whether to apply a butterworth_lowpass filter to IC timecourses before rebuild.
        chigh:
            A float determining the lowpass cutoff for the component filter, if used.
        apply_masked_mean:
            If True, only re-adds the mean signal to pixels where at least one IC is defined. To be used for thresholded ICs.
        filter_method:
            The filter method to apply to the mean. Choose from 'butterworth_bandpass', 'butterworth_lowpass', 'butterworth_highpass', or 'constant'. Behaviour for 'wavelet' as yet undefined.
        fps:
            A float determining the fps for the source video.
        include_noise:
            Whether to include noise components when rebuilding.  If noise_components should not be included in the rebuilt movie, set this to False

    Returns:
        data_r: The ICA filtered video.
    '''
    print('\nRebuilding Data from Selected ICs\n-----------------------')

    if type(components) is str:
        f = hdf5manager(components)
        components = f.load()

    assert type(components) is dict, 'Components were not in format expected'

    eig_vec = components['eig_vec']
    roimask = components['roimask']
    shape = components['shape']
    mean = components['mean']
    n_components = components['n_components']
    dtype = np.float32

    t, x, y = shape
    l = eig_vec[:, 0].size

    if mean.ndim > 1:  # why is there sometimes an extra dimension added?
        mean = mean.flatten()

    if artifact_components is None:
        artifact_components = components['artifact_components']
    elif artifact_components == 'none':
        print('including all components')
        artifact_components = np.zeros(n_components)
    elif ((not include_noise) and ('noise_components' in components.keys())):
        print('Not rebuilding noise components')
        artifact_components += components['noise_components']
        artifact_components[np.where(artifact_components > 1)] = 1

    reconstruct_indices = np.where(artifact_components == 0)[0]

    if reconstruct_indices.size == 0:
        print('No indices were selected for reconstruction.')
        print('Returning empty matrix...')
        data_r = np.zeros((t, x, y), dtype='uint8')
        data_r = data_r[t_start:t_stop]
        return data_r

    n_components = reconstruct_indices.size

    # Make sure vector extracted properly matches the roimask given.
    if roimask is None:
        assert eig_vec[:, 0].size == x * y, (
            "Eigenvector size isn't compatible with the shape of the output "
            'matrix')
    else:
        maskind = np.where(roimask.flat == 1)
        assert eig_vec[:,0].size == maskind[0].size, \
        "Eigenvector size is not compatible with the masked region's size"

    # Filter component timecourses
    if apply_component_filter:
        print('Filtering component timecourses using butterworth_lowpass at 0.5Hz...')
        eig_mix = components['eig_mix']
        timecourses = eig_mix.T
        lpf_timecourses = np.zeros_like(timecourses)
        for index in range(timecourses.shape[0]):
            lpf_timecourses[index] = butterworth(timecourses[index], high=chigh)
        eig_mix = lpf_timecourses.T

    if (t_start == None):
        t_start = 0

    if (t_stop == None):
        t_stop = eig_mix.shape[0]

    if (t_stop - t_start) is not shape[0]:
        shape = (t_stop - t_start, shape[1], shape[2])

    t = t_stop - t_start

    print('\nRebuilding ICA...')
    print('number of elements included:', n_components)
    print('eig_vec:', eig_vec.shape)
    print('eig_mix:', eig_mix.shape)

    print('\nReconstructing....')
    data_r = np.dot(eig_vec[:, reconstruct_indices],
                    eig_mix[t_start:t_stop, reconstruct_indices].T).T

    if apply_masked_mean:
        masks = components['masks']
        assert masks is not None, \
        "Masks have not been assigned to dictionary"
        # Apply mean to masks only, zeroing unmasked pixels
        if apply_mean_filter:
            combined_mask = np.any(masks[:, reconstruct_indices], axis=1)
            mean_to_add = np.zeros_like(data_r)
            mean_filtered = filter_mean(mean, filter_method, low_cutoff=mlow, high_cutoff=mhigh, fps=fps)
            mean_to_add[:, combined_mask] = mean_filtered[t_start:t_stop, None]
            data_r += mean_to_add

        else:
            print('Not filtering mean')
            combined_mask = np.any(masks[:, reconstruct_indices], axis=1)
            mean_to_add = np.zeros_like(data_r)
            mean_filtered = None
            mean_to_add[:, combined_mask] = mean[t_start:t_stop, None]
            data_r += mean_to_add
    else:
        # Run original readdition of mean
        if apply_mean_filter:
            mean_filtered = filter_mean(mean, filter_method, low_cutoff=mlow, high_cutoff=mhigh, fps=fps)
            data_r += mean_filtered[t_start:t_stop, None]

        else:
            print('Not filtering mean')
            mean_filtered = None
            data_r += mean[t_start:t_stop, None]

    print('Done!')

    if roimask is None:
        data_r = data_r.reshape(shape)
    else:
        reconstructed = np.zeros((x * y, t), dtype=dtype)
        reconstructed[maskind] = data_r.swapaxes(0, 1)
        reconstructed = reconstructed.swapaxes(0, 1)
        data_r = reconstructed.reshape(t, x, y)

    return data_r


def approximate_svd_linearity_transition(eig_val: np.ndarray):
    '''
    Approximates the transition between the svd signal distribution and 
    the noise floor.

    Calculates the integral of the eigenvalue 'influence' per component, 
    fits a 2 degree polynomial to the curve, and looks for the point at 
    which the integrated eigenvalues first overshoot the polynomial fit.
    This transition point (multiplied by a hyperparameter) is used to inform 
    the ICA n_components parameter.

    Arguments:
        eig_val: 
            The eigenvalues of the SVD decomposition.

    Returns:
        transition: 
            The estimate of the SVD noise floor cutoff.
    '''
    eig_val -= eig_val.min()
    eig_val = eig_val / eig_val.sum()
    eig_val_integrated = np.cumsum(eig_val)
    x = np.arange(eig_val.size)

    p = np.polyfit(x, eig_val_integrated, deg=2)
    y = np.polyval(p, x)

    transition = np.where(eig_val_integrated > y)[0][0]

    return transition


def filter_mean(mean: np.ndarray,
                filter_method: str = 'wavelet',
                fps: float = 7.5,
                low_cutoff: float = 0.5,
                high_cutoff: float = 1.0):
    '''
    Applies a high pass filtration to the ica mean signal.

    Arguments:
        mean: 
            The mean timecourse signal.
        filter_method:
            Which filtration method to apply.  
            Default is 'wavelet', but 'butterworth' is also accepted.
        low_cutoff:
            The frequency cutoff to apply the high pass filter at.

    Returns:
        mean_filtered: The filtered mean.
    '''
    print('Filter method:', filter_method)

    if filter_method == 'butterworth':
        print('Highpass filter signal timecourse: ' + str(low_cutoff) + 'Hz')
        variance = mean.var()
        mean_filtered = butterworth(mean, fps=fps, low=low_cutoff)
        percent_variance = np.round(mean.var() / variance * 100)
        print(str(percent_variance) + '% variance retained')

    elif filter_method == 'butterworth_lowpass':
        print('Lowpass filter signal timecourse: ' + str(low_cutoff) + 'Hz')
        variance = mean.var()
        mean_filtered = butterworth(mean, fps=fps, high=low_cutoff)
        percent_variance = np.round(mean.var() / variance * 100)
        print(str(percent_variance) + '% variance retained')

    elif filter_method == 'butterworth_bandpass':
        print('Bandpass filter signal timecourse: ' + str(low_cutoff) + 'Hz to ' + str(high_cutoff) + 'Hz')
        variance = mean.var()
        mean_filtered = butterworth(mean, fps=fps, low=low_cutoff, high=high_cutoff)
        percent_variance = np.round(mean.var() / variance * 100)
        print(str(percent_variance) + '% variance retained')

    elif filter_method == 'wavelet':
        print('Highpass filter signal timecourse: ' + str(low_cutoff) + 'Hz')
        wavelet = waveletAnalysis(mean.astype('float64'), fps=fps)
        mean_filtered = wavelet.noiseFilter(upperPeriod=1 / low_cutoff)

    elif filter_method == 'constant':
        mean_template = np.zeros_like(mean)
        meanest_mean = np.mean(mean)
        mean_filtered = mean_template + meanest_mean
        print('Mean set as constant: dfof = ' + str(meanest_mean))

    else:
        raise Exception("Filter method '" + str(filter_method)\
         + "' not supported!\n\t Supported methods: butterworth, butterworth_bandpass, wavelet")

    return mean_filtered


def rebuild_mean_roi_timecourse(components: np.ndarray,
                                mask: np.ndarray,
                                include_zero: bool = True,
                                filter: bool = True,
                                invert_artifact: bool = False,
                                include_noise: bool = True):
    '''
    Rebuild a mean timecourse under a specific region of interest (ROI), 
    or set of ROIs.

    Arguments:
        components: 
            The components result dictionary from ica.project
        mask:
            The (x,y) mask to apply to the video for rebuilding.  
            If the mask has multiple unique indices (n_components), 
            rather than just a single domain, they are all returned in an 
            array.

    Returns:
        timecourses:
            The set of rebuilt time courses (n_components,t).
    '''
    eig_vec = components['eig_vec']
    roimask = components['roimask']
    eig_mix = components['eig_mix']

    if filter and 'artifact_components' in components.keys():
        artifact_components = components['artifact_components'].copy()

        if not include_noise and 'noise_components' in components.keys():
            artifact_components += components['noise_components']
            artifact_components[np.where(artifact_components > 1)] = 1

        if invert_artifact:
            print('inverting to use artifact indices..')
            signal_indices = np.where(artifact_components == 1)[0]
        else:
            print('using signal components to rebuild.')
            signal_indices = np.where(artifact_components == 0)[0]
        eig_vec = eig_vec[:, signal_indices]
        eig_mix = eig_mix[:, signal_indices]

    if roimask is not None:
        maskind = np.where(roimask.flat == 1)[0]

    indices = np.unique(mask[~np.isnan(mask)]).astype('uint16')

    n_indices = indices.max() + 1
    timecourses = np.empty((n_indices, eig_mix.shape[0]))
    timecourses[:] = np.nan

    print('Rebuilding timecourses...')
    for i in indices:
        if (i == 0) and not include_zero:
            continue
        elif i % 50 == 0:
            print(i, '/', n_indices)

        if roimask is not None:
            domain_index = np.where(mask.flat[maskind] == i)[0]
        else:
            domain_index = np.where(mask.flat == i)[0]
        rebuilt = np.dot(eig_vec[domain_index, :], eig_mix.T)

        trace = rebuilt.mean(axis=0)
        timecourses[i] = trace
    print(n_indices, '/', n_indices)

    if not include_zero:
        timecourses = timecourses[1:]

    return timecourses


def rebuild_eigenbrain(eig_vec: np.ndarray,
                       index: int = None,
                       roimask: np.ndarray = None,
                       eigb_shape: Tuple[int, int] = None,
                       maskind: float = 1,
                       bulk: bool = False):
    '''
    Reshape components from (n_components, xy) shape into (n_components, x, y), 
    either through reassigning pixels where the roimask indicates, or by reshaping 
    it into the original dimensions.

    If one component is requested with index, just that components is returned.
    If the bulk flag is used instead, all are rebuilt and returned.

    Arguments:
        eig_vec: 
            The component eigenvectors (from components dictionary).
        index:
            Which index to rebuild.
        roimask:
            The roimask used to extract the xy coordinates (if applicable).
        eigb_shape:
            The xy shape of the original movie (if roimask was not used).
        bulk:
            Whether to rebuild all components, or just the one indicated by index.

    Returns:
        eigenbrain:
            The reshaped eigenvector (x,y)
        OR eigenbrains:
            The array of reshaped eigenvectors (n_components, x, y)
    '''
    assert (roimask is not None) or (eigb_shape is not None), (
        'Not enough information to rebuild eigenbrain')

    if bulk:
        assert eig_vec.ndim == 2, (
            'For bulk rebuild, give a 2d array of the eigenbrains')
        if roimask is not None:
            x, y = np.where(roimask == 1)

        if roimask is None:
            h, w = eigb_shape
            eigenbrains = eig_vec.reshape(h, w, eig_vec[1])
        else:
            eigenbrains = np.empty(
                (roimask.shape[0], roimask.shape[1], eig_vec.shape[1]))
            eigenbrains[:] = np.NAN
            eigenbrains[x, y, :] = eig_vec
        eigenbrains = np.swapaxes(eigenbrains, 0, 2)
        eigenbrains = np.swapaxes(eigenbrains, 1, 2)

        return eigenbrains

    else:
        assert index != None, ('Provide index to rebuild')
        if roimask is not None:
            maskind = np.where(roimask.flat == 1)

        if roimask is None:
            eigenbrain = eig_vec.T[index]
            eigenbrain = eigenbrain.reshape(eigb_shape)
        else:
            eigenbrain = np.empty(roimask.shape)
            eigenbrain[:] = np.NAN
            eigenbrain.flat[maskind] = eig_vec.T[index]

        return eigenbrain


def filter_comparison(components: dict,
                      downsample: int = 4,
                      savepath: str = None,
                      filtered_path: str = None,
                      include_noise: bool = True,
                      t_start: int = None,
                      t_stop: int = None,
                      apply_mean_filter: bool = True,
                      n_rotations: int = 0):
    '''
    Create a filter comparison movie, displaying the original movie, 
    artifacts removed, and the filtered movie side by side.


    Arguments:
        components: 
            The ICA components returned by ica.project.
        downsample:
            The factor to downsample by before writing the video.
        savepath:
            The path to save the video at (mp4).
        filtered_path:
            The hdf5 path to save the filtered movie to. 
        include_noise:
            Whether noise components should be included in the filtered video.
        t_start: 
            The frame to start rebuilding the movie at.  If none is provided, 
            the rebuilt movie starts at the first frame.
        t_stop: 
            The frame to stop rebuilding the movie at.  If none is provided, 
            the rebuilt movie ends at the last frame.
        filter_mean:
            Whether to filter the mean before readding.
        n_rotations:
            The number of CCW rotations to apply before saving the video.

    Returns:
        Nothing.
    '''
    print('\n-----------------------', '\nBuilding Filter Comparison Movies',
          '\n-----------------------')

    print('\nFiltered Movie\n-----------------------')
    filtered = rebuild(components,
                       include_noise=include_noise,
                       t_start=t_start,
                       t_stop=t_stop,
                       apply_mean_filter=apply_mean_filter)

    if filtered_path is not None:
        print('Saving filtered movie to:', filtered_path)
        f = hdf5manager(filtered_path)
        f.save({'filtered_movie': filtered})

    filtered = scale_video(filtered, downsample)
    filtered = rotate(filtered, n_rotations)

    print('\nArtifact Movie\n-----------------------')
    artifact_index = np.where(components['artifact_components'] == 1)[0]
    components['artifact_components'] = np.ones(
        components['artifact_components'].shape)
    components['artifact_components'][artifact_index] = 0
    if not include_noise:
        components['artifact_components'][np.where(
            components['noise_components'] == 1)] = 0
    artifact_movie = rebuild(components, t_start=t_start, t_stop=t_stop)
    print('rescaling video...')
    artifact_movie = scale_video(artifact_movie, downsample)
    artifact_movie = rotate(artifact_movie, n_rotations)

    print('\nOriginal Movie\n-----------------------')
    components['artifact_components'] = np.zeros(
        components['artifact_components'].shape)
    raw_movie = rebuild(components,
                        t_start=t_start,
                        t_stop=t_stop,
                        apply_mean_filter=apply_mean_filter)
    print('rescaling video...')
    raw_movie = scale_video(raw_movie, downsample)
    raw_movie = rotate(raw_movie, n_rotations)

    movies = np.concatenate((raw_movie, artifact_movie, filtered), axis=2)

    if 'roimask' in components.keys():
        roimask = components['roimask']
        overlay = (roimask == 0).astype('uint8')
        overlay = rotate(overlay, n_rotations)

        overlay = scale_video(overlay[None, :, :], downsample)[0]
        overlay = np.concatenate((overlay, overlay, overlay), axis=1)

    else:
        overlay = None

    print('overlay', overlay.shape)
    print('movies', movies.shape)

    save(movies,
         savepath,
         rescale_range=True,
         resize_factor=1 / 2,
         save_cbar=True,
         overlay=overlay)
