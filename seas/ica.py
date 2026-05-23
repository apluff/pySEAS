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

import cv2
from skimage.morphology import remove_small_objects
from skimage import draw, measure
from scipy import ndimage
import tifffile as tif

# Refactor additions
from dataclasses import dataclass, field, fields, asdict
from typing import Dict
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
import zarr
from picard import Picard

@dataclass
class PyseasRecord(MutableMapping):
    mean:  np.ndarray
    roimask: np.ndarray
    shape: Tuple[int, int, int]
    eig_mix: np.ndarray
    timecourses: np.ndarray
    eig_vec: np.ndarray
    n_components: int
    lag1: np.ndarray
    lag1_full: np.ndarray
    noise_components: np.ndarray
    cutoff: float
    svd_cutoff: int
    svd_multiplier: int
    increased_cutoff: int
    flipped: np.ndarray = None
    project_meta: dict = field(default_factory=dict)
    _extra: dict = field(default_factory=dict)

    def __getitem__(self, key):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return self._extra[key]
    
    def __setitem__(self, key, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            self._extra[key] = value
    
    def __delitem__(self, key):
        if key in self.__dataclass_fields__:
            raise KeyError("Cannot delete core dataclass field.")
        else:
            del self._extra[key]

    def __iter__(self):
        yield from (f.name for f in fields(self) if not f.name.startswith("_"))
        yield from self._extra

    def __len__(self):
        return (len([f for f in fields(self) if not f.name.startswith("_")])
                + len(self._extra))

    def __post_init__(self):
        if self.mean.ndim > 1:  # why is there sometimes an extra dimension added?
            self.mean = self.mean.flatten()

        # Make sure vector extracted properly matches the roimask given.
        _, x, y = self.shape
        if self.roimask is None:
            self.maskind = None
            assert self.eig_vec[:, 0].size == x * y, (
                "Eigenvector size isn't compatible with the shape of the output "
                'matrix')
        else:
            self.maskind = np.where(self.roimask.flat == 1)
            assert self.eig_vec[:,0].size == self.maskind[0].size, \
            "Eigenvector size is not compatible with the masked region's size"

    def save_creation_metadata(self, projection: str, n_components: int, time_elapsed: float):
        # Save filter metadata information about how and when movie was filtered in dictionary.
        project_meta = {}
        project_meta['time_elapsed'] = time_elapsed
        project_meta['date'] = \
            datetime.now().strftime('%Y%m%d')[2:]
        fmt = '%Y-%m-%dT%H:%M:%SZ'
        project_meta['tstmp'] = \
            datetime.now().strftime(fmt)
        project_meta['n_components'] = n_components
        project_meta['projection'] = projection
        self.project_meta = project_meta


class Projector(ABC):

    @abstractmethod
    def project(self, vector):
        pass


class _FastICA(Projector):

    def __init__(self, 
                 n_components = None, 
                 svd_multiplier = None, 
                 max_iter = 1000):
        self.n_components = n_components
        self.svd_multiplier = svd_multiplier
        self.max_iter = max_iter

    def project(self, vector: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int]:
        '''
        Replicates original FastICA processing in conjunction with the
        top-level function project() (original wrapper) per Weiser et al. 2023.
        '''
        # ========================= START ICA BLOCK ======================== #
        increased_cutoff = 0
        if self.n_components is None:
            u, n_components = estimate_n_components(vector, 
                                                    self.svd_multiplier)
            w_init = u[:n_components, :n_components].astype('float64')
        else:
            n_components = self.n_components
            w_init = None
            
        while True:
            print('\nCalculating ICA with', n_components, 'components...')

            ica = FastICA(n_components = n_components,
                        max_iter = self.max_iter,
                        random_state = 1000,
                        w_init = w_init)

            try:
                eig_vec = ica.fit_transform(vector)  # Eigenbrains
            except ValueError:
                print('Calculation exceeded float32 maximum.')
                print('Trying again with float64 vector...')
                # Value error if any value exceeds float32 maximum.
                # Overcome this by converting to float64.
                eig_vec = ica.fit_transform(vector.astype('float64'))
            print("n_iter:" , ica.n_iter_)
            
            eig_mix = ica.mixing_
            noise, cutoff = sort_noise(eig_mix.T)

            #Signal/noise check #1
            p_signal = (1 - noise.sum() / noise.size) * 100
            if self.n_components is not None: # No dynamic threshold required
                break
            elif noise.size == input.shape[0]:  # All components are being used.
                break
            elif p_signal < 75:
                print('ICA components were under 75% signal ({0}% signal).'\
                    .format(p_signal))
                break
            elif n_components >= input.shape[0]:
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
                increased_cutoff += 1

                if n_components > input.shape[0]:
                    print('\nComponents maxed out!')
                    print('\tAttempted:', n_components)
                    n_components = input.shape[0]
                    print('\tReduced to:', input.shape[0])

        if self.n_components is None:
            svd_cutoff = n_components
        else:
            svd_cutoff = None
        # ========================= FINISH ICA BLOCK ======================== #

        return n_components, eig_vec, eig_mix, noise, cutoff, svd_cutoff, increased_cutoff


class _InfoMaxICA(Projector):
    pass

class _JadeICA(Projector):
    pass

class _PicardICA(Projector):
    def __init__(self, 
                 n_components = None, 
                 svd_multiplier = None, 
                 max_iter = 1000):
        self.n_components = n_components
        self.svd_multiplier = svd_multiplier
        self.max_iter = max_iter

    def project(self, vector: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int]:
        '''
        Replicates original FastICA processing in conjunction with the
        top-level function project() (original wrapper) per Weiser et al. 2023.
        '''
        # ========================= START ICA BLOCK ======================== #
        increased_cutoff = 0
        if self.n_components is None:
            u, n_components = estimate_n_components(vector, 
                                                    self.svd_multiplier)
            w_init = u[:n_components, :n_components].astype('float64')
        else:
            n_components = self.n_components
            w_init = None
            
        while True:
            print('\nCalculating ICA with', n_components, 'components...')

            ica = Picard(n_components = n_components,
                         max_iter = 500, # Default for testing
                         random_state = 1000,
                         w_init = w_init)

            try:
                eig_vec = ica.fit_transform(vector)  # Eigenbrains
            except ValueError:
                print('Calculation exceeded float32 maximum.')
                print('Trying again with float64 vector...')
                # Value error if any value exceeds float32 maximum.
                # Overcome this by converting to float64.
                eig_vec = ica.fit_transform(vector.astype('float64'))
            print("n_iter:" , ica.n_iter_)
            
            eig_mix = ica.mixing_
            noise, cutoff = sort_noise(eig_mix.T)

            #Signal/noise check #1
            p_signal = (1 - noise.sum() / noise.size) * 100
            if self.n_components is not None: # No dynamic threshold required
                break
            elif noise.size == input.shape[0]:  # All components are being used.
                break
            elif p_signal < 75:
                print('ICA components were under 75% signal ({0}% signal).'\
                    .format(p_signal))
                break
            elif n_components >= input.shape[0]:
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
                increased_cutoff += 1

                if n_components > input.shape[0]:
                    print('\nComponents maxed out!')
                    print('\tAttempted:', n_components)
                    n_components = input.shape[0]
                    print('\tReduced to:', input.shape[0])

        if self.n_components is None:
            lag1_full = lag_n_autocorr(eig_mix.T, 1)
            svd_cutoff = n_components
        else: # For compatability with original pySEAS dicts
            lag1_full = None # Maybe change this to return all the time.
            svd_cutoff = None
        # ========================= FINISH ICA BLOCK ======================== #

        return n_components, eig_vec, eig_mix, lag1_full, noise, cutoff, svd_cutoff, increased_cutoff

class _AMICA(Projector):
    pass

class _PCA(Projector):

    def __init__(self):
        pass

    def project(vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            u, ev, v = linalg.svd(vector, full_matrices = False)
            print('PCA run with scipy.linalg.svd and gesdd lapack.')
        except ValueError:
            try:
                print('Initial PCA failed.')
                # LAPACK error if matricies are too big
                u, ev, _ = linalg.svd(vector,
                                      full_matrices = False,
                                      lapack_driver = 'gesvd')
                print('PCA run with scipy.linalg.svid and gesvd lapack.')
            except ValueError:
                print('Secondary PCA failed.')
                u, ev, _ = np.linalg.svd(vector,
                                         full_matrices = False)
                print('PCA run with numpy.linalg.svd.')
        
        return u, ev, v


@dataclass
class PyseasConfig:
    '''
    Attributes:
        n_components:
            Manual override for n_components into the ICA, defaults to None for automatic calulation.
        svd_multiplier:
            The hyperparameter for svd adaptive thresholding.
        calc_residuals:
            Whether to calculate spatial and temporal residuals of projection compression.
        max_iter:
            Maximum iterations assigned for FastICA
    '''
    n_components: int = None
    crop_excess_noise: bool = True
    svd_multiplier: float = 5
    calc_residuals: bool = False
    projector: str = 'FastICA'
    max_iter: int = None

    def __post_init__(self):
        valid_projectors = ['FastICA', 'InfoMax', 'JADE', 'PCA']
        
        assert self.projector in valid_projectors, \
            'Specified projector is not valid, must be "FastICA", "InfoMax",' \
            ' "JADE", or "PCA".'
        if self.n_components is None:
            assert self.svd_multiplier is not None, \
                'n_components is unset, so SVD multiplier must be specified.'
        else:
            'n_components has been specified, erasing svd_multiplier.'            
            self.svd_multiplier = None
        

@dataclass
class PyseasInput:
    '''
    Attributes:
        vector: 
            The (x*y, t) vector to be spatially ICA projected.
        shape:
            The shape of the original movie (t,x,y).
        roimask:
            The roimask to crop the vectorized movie (x,y).
        maskind:
            The indices of each vector frame corresponding to the mask.
    '''
    vector: np.ndarray
    shape: Tuple[int, int, int]
    roimask: np.ndarray = None
    maskind: np.ndarray = None

    def __post_init__(self):
        assert (self.vector.ndim == 2), (
        'vector was not a two-dimensional np array.'
        'If input is a movie, be sure to convert shape to (xy, t)')

        if self.vector.dtype == np.float16:
            self.vector = self.vector.astype('float32', copy=False)
            
        if self.roimask is not None:
            print('Roimask will be used to crop video.')
            assert self.roimask.size == self.vector.shape[0], \
            'Vector was not the same size as the cropped mask'

            print('Original vector size:', self.vector.shape)
            self.maskind = np.where(self.roimask.flat == 1)
            self.vector = self.vector[self.maskind]
            print('Original vector reduced to size:', self.vector.shape)

def estimate_n_components(vector: np.ndarray, 
                          svd_multiplier: float,
                          Estimator: Projector = _PCA) -> Tuple[np.ndarray, int]:
    estimator = Estimator
    print('Estimating n_components with SVD...')
    u, ev, _ = estimator.project(vector)
    # components['svd_eigval'] = ev # Not used anywhere, should I store this?

    # Get starting point for decomposition based on svd mutliplier * the approximate
    # point of transition to linearity in tail of ev components.
    cross_1 = approximate_svd_linearity_transition(ev)
    n_components = cross_1 * svd_multiplier
    
    return u, n_components
    
def calculate_residuals(input: PyseasInput, components: PyseasRecord) -> dict:
    vector = input.vector.astype('float64')
    rebuilt = rebuild(components,
                      artifact_components='none',
                      apply_mean_filter=False).T
    rebuilt -= rebuilt.mean(axis=0)
    vector -= vector.mean(axis=0)
    residuals = np.abs(vector - rebuilt)
    residuals_temporal = residuals.mean(axis=0)

    if input.roimask is not None:
        residuals_spatial = np.zeros(input.roimask.shape)
        residuals_spatial.flat[input.maskind] = residuals.mean(axis=1)
    else:
        residuals_spatial = np.reshape(residuals.mean(axis=1),
                                       (shape[1], shape[2]))
        
    output = {}
    output['residuals_spatial'] = residuals_spatial
    output['residuals_temporal'] = residuals_temporal

    return output

def flip_components(components: PyseasRecord) -> dict:
    # Track component orientation and ensure positive spatial patterns
    n_components = components.n_components
    eig_vec = np.zeros_like(components.eig_vec)
    eig_mix = np.zeros_like(components.eig_mix)
    flipped = np.ones(n_components)
    
    for i in range(n_components):
        # Find the index of maximum absolute value
        max_idx = np.argmax(np.abs(components.eig_vec[:, i]))
        # If that maximum value is negative, flip the component
        if components.eig_vec[max_idx, i] < 0:
            eig_vec[:, i] = -1 * components.eig_vec[:, i]
            eig_mix[:, i] = -1 * components.eig_mix[:, i]
            flipped[i] = -1

        output = {}
        output['eig_vec'] = eig_vec
        output['eig_mix'] = eig_mix
        output['flipped'] = flipped

        return output
    
def crop_excess_noise(components: PyseasRecord) -> dict:
    n_components = components.n_components
    eig_vec = components.eig_vec
    eig_mix = components.eig_mix
    noise = components.noise_components
    
    print('Cropping excess noise components')
    reduced_n_components = int((noise.size - noise.sum()) * 1.25)
    print('reduced_n_components:', reduced_n_components)
    if reduced_n_components < n_components:
        print('Cropping', n_components, 'to', reduced_n_components)

        eig_vec = eig_vec[:, :reduced_n_components]
        eig_mix = eig_mix[:, :reduced_n_components]
        n_components = reduced_n_components
        noise_components = noise[:reduced_n_components]
        
        # Recalculate lag1 for reduced components
        timecourses = eig_mix.T
        lag1 = lag_n_autocorr(timecourses, 1)

        output = {}
        output['eig_vec'] = eig_vec
        output['eig_mix'] = eig_mix
        output['n_components'] = n_components
        output['noise_components'] = noise_components
        output['lag1'] = lag1

        return output
    else:
        print('Less than 75% signal.  Not cropping excess noise.')

def sort_components(sort_by: str = 'timecourse_std', 
                    components: PyseasRecord = None) -> dict:
    '''
    Sorts components by some metric before cropping excess noise.
    '''
    assert components is not None, 'PyseasRecord must be provided for sort.'
    eig_mix = components.eig_mix
    eig_vec = components.eig_vec
    noise = components.noise_components
    lag1 = components.lag1
    lag1_full = components.lag1_full
    match sort_by:
        case 'timecourse_std': # Original pySEAS default.
            # Sort components by their eig val influence (approximated by timecourse standard deviation).
            # NOTE: This doesn't guarantee all removed components are noise.
            ev_sort = np.argsort(eig_mix.std(axis=0))
        case 'lag1': # Guarantees correct ordering for noise cropping
            ev_sort = np.argsort(eig_mix.std(axis=0))
    eig_vec = eig_vec[:, ev_sort][:, ::-1]
    eig_mix = eig_mix[:, ev_sort][:, ::-1]
    noise_components = noise[ev_sort][::-1]
    lag1 = [ev_sort][::-1]
    lag1_full = lag1_full[ev_sort][::-1]

    output = {}
    output['eig_vec'] = eig_vec
    output['eig_mix'] = eig_mix
    output['noise_components'] = noise_components
    output['lag1'] = lag1
    output['lag1_full'] = lag1_full

    return output

def project(Input: PyseasInput, Config: PyseasConfig) -> PyseasRecord:
    '''
    Apply an ica decomposition to the first axis of the input vector.  
    If a roimask was provided, the flattened roimask will be used to crop the 
    vector before decomposition.

    If n_components is not set, an adaptive svd threshold is used 
    (see approximate_svd_linearity_transition), with the hyperparameter 
    svd_mutliplier.  

    Residuals lost in the ICA projection are captured if calc_residuals == True.  
    This represents the signal lost by ICA compression.
    
    Arguments:
        input: 
            a PyseasInput object containing the video to be projected and 
            associated data.
        config:
            a PyseasConfig object containing config for the projection process.
        
    Returns:
        components: A PyseasRecord dictionary containing all the results, 
        metadata, and information regarding the filter applied.

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
                the number of components in eig_vec (reduced to only have 25% 
                of total components as noise)
            project_meta:
                The metadata for the ica projection
            expmeta:
                All metadata created for this class
            lag1: 
                the lag-1 autocorrelation
            noise_components: 
                a vector (n components long) to store binary representation of 
                which components were detected as noise 
            cutoff: 
                the signal-noise cutoff value

        if the n_components was automatically set, the following keys are also
        set in components (otherwise default to None)

            svd_cutoff: 
                the number of components originally decomposed
            lag1_full: 
                the lag-1 autocorrelation of the full set of components 
                decomposed before cropping to only 25% noise components
            svd_multiplier: 
                the svd multiplier value used to determine cutoff
    '''
    print('\nCalculating Eigenspace\n-----------------------')
    config = Config
    input = Input

    # ========================== Preprocessing ========================== #

    mean = np.mean(input.vector, 0).flatten()
    vector = input.vector - mean
    
    # ========================== Projection ========================== #

    # TODO: Add cases
    match config.projector:
        case 'FastICA':
            calculator = _FastICA(n_components = config.n_components, 
                                  max_iter = config.max_iter)
        case 'PicardICA':
            calculator = _PicardICA(n_components = config.n_components,
                                    max_iter = config.max_iter)
        case 'PCA':
            calculator = _PCA()

    t0 = timer()
    n_components, eig_vec, eig_mix, noise, cutoff, svd_cutoff, increased_cutoff \
        = calculator.project(vector)
    t = timer() - t0
    print('Independent Component Analysis took: {0} sec'.format(t))

    # HUHHHHH????
    print('components shape:', eig_vec.shape)
    assert n_components == eig_vec.shape[1], "HUUUHHHH????"

    timecourses = eig_mix.T
    lag1_full = lag_n_autocorr(timecourses, 1)
    lag1 = lag1_full

    # ========================== Saving ========================== #  

    components = PyseasRecord(mean = mean,
                              roimask = input.roimask,
                              shape = input.shape,
                              eig_mix = eig_mix,
                              timecourses = timecourses,
                              eig_vec = eig_vec,
                              n_components = n_components,
                              lag1 = lag1,
                              noise_components = noise,
                              cutoff = cutoff,
                              svd_cutoff = svd_cutoff,
                              svd_multiplier = config.svd_multiplier,
                              increased_cutoff = increased_cutoff,
                              lag1_full = lag1_full)
    components.save_creation_metadata(config.projector, n_components, t)

    # ========================== Postprocessing ========================== #

    # Sort components by timecourse standard deviation per pyseas default
    sorted_components = sort_components(sort_by = 'timecourse_std', 
                                        components = components)
    components.update(sorted_components)

    # Crop excess noise components from record
    if config.crop_excess_noise and config.n_components is None:
        cropped_components = crop_excess_noise(components)
        if cropped_components is not None:
            components.update(cropped_components)
    else:
        print('Noise retention enabled. Not cropping excess noise.')
    
    # Calculate residuals
    if config.calc_residuals:
        try:
            residuals = calculate_residuals(input, components)
            components.update(residuals)
        except Exception as e:
            print('Residual Calculation Failed!!')
            print('\t', e)

    # Flip inverted components
    flipped_components = flip_components(components)
    components.update(flipped_components)

    print('\n')
    return components

def derive_reconstruct_indices(components: dict, 
                               artifact_components: np.ndarray = None, 
                               include_noise: bool = False) -> np.ndarray:
    n_components = components['n_components']

    if artifact_components is None:
        artifact_components = components['artifact_components']
    elif artifact_components == 'none':
        print('including all components')
        artifact_components = np.zeros(n_components)
    
    if ((not include_noise) and ('noise_components' in components.keys())):
        print('Not rebuilding noise components')
        artifact_components += components['noise_components']
        artifact_components[np.where(artifact_components > 1)] = 1

    reconstruct_indices = np.where(artifact_components == 0)[0]

    return reconstruct_indices

def rebuild(components: dict | str,
            artifact_components: np.ndarray = None,
            t_start: int = None,
            t_stop: int = None,
            apply_mean_filter: bool = True,
            mlow: float = 0.5,
            mhigh: float = 1.0,
            apply_component_filter: bool = False,
            chigh: float = 1.0,
            apply_component_threshold: bool = False,
            cthresh: float = 2.0,
            apply_masked_mean: bool = False,
            binary_threshold: bool = False,
            filter_method: str = 'wavelet',
            fps: float = 7.5,
            include_noise: bool = True,
            splitting: str = None):
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
        apply_component_threshold:
            Whether to apply a z-score threshold on the component timeseries.
        cthresh:
            A float determining the z-score threshold for the component threshold, if used.
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
    def _rebuild_full_video(eig_vec, eig_mix, 
                        mean, reconstruct_indices, 
                        t_start, t_stop,
                        shape, roimask, maskind):
        print('\nReconstructing full video...')
        data_r = np.dot(eig_vec[:, reconstruct_indices],
                        eig_mix[t_start:t_stop, reconstruct_indices].T).T
        # Re-add mean timecourse
        data_r += mean[t_start:t_stop, None]
        print('Done!')
        # Reshaping
        data_r = reshape_rebuilt_video(data_r, shape, roimask, maskind)

        return data_r

    def _rebuild_component_videos(eig_vec, eig_mix, 
                                mean, reconstruct_indices, 
                                t_start, t_stop,
                                shape, roimask, maskind):
        print('\nReconstructing per component...')
        data_r = {}
        for idx in reconstruct_indices:
            i = idx + 1 # Convert to 1-base for component labelling
            print('\nRebuilding component: ', i)
            data_s = [eig_vec[:,i] * m for m in eig_mix[t_start:t_stop, i]]
            data_c = np.stack(data_s, axis = 0)
            #Re-add mean timecourse
            data_c += mean[t_start:t_stop, None]
            # Reshaping
            data_c = scale_dfof_to_8bit(data_c)
            data_c = reshape_rebuilt_video(data_c, shape, roimask, maskind)
            # Assign to compressed array (defaults to zstd)
            data_z = zarr.create_array(shape = data_c.shape, 
                                       chunks = (8, 8), 
                                       dtype = data_c.dtype)
            data_z[:] = data_c
            data_r[int(i)] = data_z
        
        return data_r

    def _rebuild_cluster_videos(eig_vec, eig_mix, 
                                mean, cluster_indices, 
                                t_start, t_stop,
                                shape, roimask, maskind):
            print('\nReconstructing per cluster...')
            data_r = {}
            for i in cluster_indices[0]:
                if len(cluster_indices[i]) == 1:
                    data_c = [eig_vec[:,i] * m for m in eig_mix[t_start:t_stop, i]]
                else:    
                    data_c = np.dot(eig_vec[:, cluster_indices[i]],
                                    eig_mix[t_start:t_stop, cluster_indices[i]].T).T
                data_c += mean[t_start:t_stop, None]
                data_c = reshape_rebuilt_video(data_c, shape, roimask, maskind)
                data_c = scale_dfof_to_8bit(data_c)
                data_r[int(i)] = data_c
            
            return data_r

    print('\nRebuilding Data from Selected ICs\n-----------------------')

    if type(components) is str:
        f = hdf5manager(components)
        components = f.load()

    #assert type(components) is dict, 'Components were not in format expected'

    # Localising variables
    eig_vec = components.eig_vec
    eig_mix = components.eig_mix
    shape = components.shape
    roimask = components.roimask
    maskind = components.maskind
    mean = components.mean
    t, x, y = shape

    # Determine reconstruction indices
    reconstruct_indices = derive_reconstruct_indices(components, 
                                                     artifact_components, 
                                                     include_noise)
    if reconstruct_indices.size == 0:
        print('No indices were selected for reconstruction.')
        print('Returning empty matrix...')
        data_r = np.zeros((t, x, y), dtype='uint8')
        data_r = data_r[t_start:t_stop]
        return data_r
    n_components = reconstruct_indices.size

    # Determine cluster indices (if available)
    if hasattr(components, 'clusters') and components.clusters is not None:
        cluster_indices = {}
        for i in np.unique(components.clusters):
            current_cluster_indices = np.where(components.clusters == i)
            cluster_indices[i] = np.intersect1d(current_cluster_indices, 
                                                reconstruct_indices)

    # Filter mean timecourse
    if apply_mean_filter:
        mean = filter_mean(mean, 
                           filter_method, 
                           low_cutoff = mlow, 
                           high_cutoff = mhigh, 
                           fps = fps)
    else:
        print('Not filtering mean timecourse.')

    # Filter component timecourses
    if apply_component_filter:
        lpf_eig_mix = filter_components(eig_mix, 
                                        fps = fps, 
                                        high_cutoff = chigh)
        eig_mix = lpf_eig_mix
    else:
        print('Not filtering component timecourses.')

    # Threshold component timecourses
    if apply_component_threshold:
        thresh_eig_mix = threshold_components(eig_mix, thresh_param = cthresh)
        eig_mix = thresh_eig_mix
    else:
        print('Not thresholding component timecourses.')

    # Determine start and stop bounds
    if (t_start == None):
        t_start = 0
    if (t_stop == None):
        t_stop = eig_mix.shape[0]
    if (t_stop - t_start) is not shape[0]:
        shape = (t_stop - t_start, shape[1], shape[2])
    t = t_stop - t_start

    #eig_mix = eig_mix[t_start:t_stop, :]

    # Reconstruction
    print('\nRebuilding ICA...')
    print('number of elements included:', n_components)
    print('eig_vec:', eig_vec.shape)
    print('eig_mix:', eig_mix.shape)
    match splitting:
        case None:
            data_r = _rebuild_full_video(eig_vec, eig_mix, mean,
                                         reconstruct_indices, t_start, 
                                         t_stop, shape, roimask, maskind)
        case 'components':
            data_r = _rebuild_component_videos(eig_vec, eig_mix, mean,
                                               reconstruct_indices, t_start, 
                                               t_stop, shape, roimask, maskind)
        case 'clusters':
            data_r = _rebuild_cluster_videos(eig_vec, eig_mix, mean,
                                             cluster_indices, t_start, 
                                             t_stop, shape, roimask, maskind)
        
    # spatiotemporal_event_masks = data_r[data_r > 0]

    # # More of my extra stuff, integration could be clearer.
    # if apply_masked_mean:

    #     # Apply mean to masks only, zeroing unmasked pixels
    #     spatiotemporal_event_masks = np.zeros_like(data_r)
    #     spatiotemporal_event_masks[data_r > 0] = 255
    #     spatiotemporal_event_masks = spatiotemporal_event_masks.astype(bool)
    #     masks = components['thresh_masks']
    #     assert masks is not None, \
    #     "Masks have not been assigned to dictionary"
    #     if apply_mean_filter:
    #         combined_mask = np.any(masks[:, reconstruct_indices], axis=1)
    #         mean_to_add = np.zeros_like(data_r)
    #         mean_filtered = filter_mean(mean, filter_method, low_cutoff=mlow, high_cutoff=mhigh, fps=fps)
    #         mean_to_add[:, combined_mask] = mean_filtered[t_start:t_stop, None]
    #         data_r += mean_to_add
    #         data_r[~spatiotemporal_event_masks] = 0

    #     else:
    #         print('Not filtering mean')
    #         combined_mask = np.any(masks[:, reconstruct_indices], axis=1)
    #         mean_to_add = np.zeros_like(data_r)
    #         mean_filtered = None
    #         mean_to_add[:, combined_mask] = mean[t_start:t_stop, None]
    #         data_r += mean_to_add
    #         data_r[~spatiotemporal_event_masks] = 0
    # else:
    #     # Run original readdition of mean
    #     if apply_mean_filter:
    #         mean_filtered = filter_mean(mean, filter_method, low_cutoff=mlow, high_cutoff=mhigh, fps=fps)
    #         data_r += mean_filtered[t_start:t_stop, None]

    #     else:
    #         print('Not filtering mean')
    #         mean_filtered = None
    #         data_r += mean[t_start:t_stop, None]

    # if binary_threshold:
    #     data_binary = np.zeros(data_r)
    #     data_binary[data_r > 0] = 255
    #     data_r = data_binary

    return data_r

def reshape_rebuilt_video(data_r: np.ndarray, 
                          shape: Tuple[int, int, int], 
                          roimask: np.ndarray = None, 
                          maskind: np.ndarray = None):
    if roimask is None:
        data_r = data_r.reshape(shape)
    else:
        t, x, y = shape
        reconstructed = np.zeros((x * y, t), dtype = np.float32)
        print(f'data_r shape is: {data_r.shape}')
        print(f'reconstructed shape is: {reconstructed.shape}')
        print(f'maskind is: {maskind}')
        reconstructed[maskind] = data_r.swapaxes(0, 1)
        reconstructed = reconstructed.swapaxes(0, 1)
        data_r = reconstructed.reshape(t, x, y)

    return data_r

def scale_dfof_to_8bit(data_r: np.ndarray) -> np.ndarray:
    assert data_r.dtype == np.float32, "Data is not in dF/F format."
    data_r[data_r < 0.0] = 0.0
    data_r = data_r*255
    data_r[data_r > 255.0] = 255.0
    data_r = data_r.astype(np.uint8)
    return data_r

def rebuild_split_components(components: dict,
            artifact_components: np.ndarray = None,
            t_start: int = None,
            t_stop: int = None,
            apply_mean_filter: bool = True,
            mlow: float = 0.5,
            mhigh: float = 1.0,
            apply_component_filter: bool = False,
            chigh: float = 1.0,
            apply_component_threshold: bool = False,
            cthresh: float = 2.0,
            apply_masked_mean: bool = False,
            binary_threshold: bool = False,
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
        apply_component_threshold:
            Whether to apply a z-score threshold on the component timeseries.
        cthresh:
            A float determining the z-score threshold for the component threshold, if used.
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
    eig_mix = components['eig_mix']
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
    
    if ((not include_noise) and ('noise_components' in components.keys())):
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
        lpf_eig_mix = filter_components(eig_mix, fps=fps, high_cutoff=chigh)
        eig_mix = lpf_eig_mix

    # Threshold component timecourses
    if apply_component_threshold:
        thresh_eig_mix = threshold_components(eig_mix, thresh_param=cthresh)
        eig_mix = thresh_eig_mix

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
    print(f'reconstruct_indices are: {reconstruct_indices}')
    for i in reconstruct_indices:
        print(f'i is: {i}')
        print(f'Selected eig_vec shape is: {eig_vec[:, i].shape}')
        print(f'Selected eig_mix shape is: {eig_mix[t_start:t_stop, i].T.shape}')
        #data_r = np.dot(eig_vec[:, i],
        #                eig_mix[t_start:t_stop, i].T).T
        data_rl = [eig_vec[:,i] * m for m in eig_mix[t_start:t_stop, i]]
        data_r = np.stack(data_rl, axis=0)
        # spatiotemporal_event_masks = data_r[data_r > 0]

        if apply_masked_mean:
            # Apply mean to masks only, zeroing unmasked pixels
            spatiotemporal_event_masks = np.zeros_like(data_r)
            spatiotemporal_event_masks[data_r > 0] = 255
            spatiotemporal_event_masks = spatiotemporal_event_masks.astype(bool)
            masks = components['thresh_masks']
            assert masks is not None, \
            "Masks have not been assigned to dictionary"
            if apply_mean_filter:
                combined_mask = masks[:, i]
                mean_to_add = np.zeros_like(data_r)
                mean_filtered = filter_mean(mean, filter_method, low_cutoff=mlow, high_cutoff=mhigh, fps=fps)
                #mean_to_add[:, combined_mask] = mean_filtered[t_start:t_stop, None]
                mean_to_add = 0.028687261
                data_r += mean_to_add
                data_r[~spatiotemporal_event_masks] = 0

            else:
                print('Not filtering mean')
                combined_mask = np.any(masks[:, i], axis=1)
                mean_to_add = np.zeros_like(data_r)
                mean_filtered = None
                mean_to_add[:, combined_mask] = mean[t_start:t_stop, None]
                data_r += mean_to_add
                data_r[~spatiotemporal_event_masks] = 0
        else:
            # Run original readdition of mean
            if apply_mean_filter:
                mean_filtered = filter_mean(mean, filter_method, low_cutoff=mlow, high_cutoff=mhigh, fps=fps)
                data_r += mean_filtered[t_start:t_stop, None]

            else:
                print('Not filtering mean')
                mean_filtered = None
                data_r += mean[t_start:t_stop, None]

        if binary_threshold:
            data_binary = np.zeros(data_r)
            data_binary[data_r > 0] = 255
            data_r = data_binary

        print('Done!')

        if roimask is None:
            data_r = data_r.reshape(shape)
        else:
            reconstructed = np.zeros((x * y, t), dtype=dtype)
            print(f'data_r shape is: {data_r.shape}')
            print(f'reconstructed shape is: {reconstructed.shape}')
            print(f'maskind shape: {maskind}')
            reconstructed[maskind] = data_r.swapaxes(0, 1)
            reconstructed = reconstructed.swapaxes(0, 1)
            data_r = reconstructed.reshape(t, x, y)
        comp_out = 'sub-071_rec-baseline_run-02_id-' + str(i) + '.tif'
        data_r[data_r < 0.0] = 0.0
        data_r = data_r*255
        data_r[data_r > 255.0] = 255.0
        data_r = data_r.astype(np.uint8)
        #data_r[data_r > 0] = 255
        tif.imwrite('/QRISdata/Q5451/temp/tests/pyseas_split_components/' + comp_out, data_r,compression='lzw', imagej=True)

def rebuild_split_clusters(components: dict,
            artifact_components: np.ndarray = None,
            t_start: int = None,
            t_stop: int = None,
            apply_mean_filter: bool = True,
            mlow: float = 0.5,
            mhigh: float = 1.0,
            apply_component_filter: bool = False,
            chigh: float = 1.0,
            apply_component_threshold: bool = False,
            cthresh: float = 2.0,
            apply_masked_mean: bool = False,
            binary_threshold: bool = False,
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
        apply_component_threshold:
            Whether to apply a z-score threshold on the component timeseries.
        cthresh:
            A float determining the z-score threshold for the component threshold, if used.
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
    eig_mix = components['eig_mix']
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
    
    if ((not include_noise) and ('noise_components' in components.keys())):
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
        lpf_eig_mix = filter_components(eig_mix, fps=fps, high_cutoff=chigh)
        eig_mix = lpf_eig_mix

    # Threshold component timecourses
    if apply_component_threshold:
        thresh_eig_mix = threshold_components(eig_mix, thresh_param=cthresh)
        eig_mix = thresh_eig_mix

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
    clusters = components['component_clusters']
    for i in range(1,np.max(clusters)):
        print(f'Reuilding cluster: {i}')
        cluster_indices = np.where(clusters == i)[0]
        signal_indices = np.where(artifact_components == 0)[0]
        reconstruct_indices = np.intersect1d(cluster_indices, signal_indices)
        data_r = np.dot(eig_vec[:, reconstruct_indices],
                        eig_mix[t_start:t_stop, reconstruct_indices].T).T
        # spatiotemporal_event_masks = data_r[data_r > 0]

        if apply_masked_mean:
            # Apply mean to masks only, zeroing unmasked pixels
            spatiotemporal_event_masks = np.zeros_like(data_r)
            spatiotemporal_event_masks[data_r > 0] = 255
            spatiotemporal_event_masks = spatiotemporal_event_masks.astype(bool)
            masks = components['thresh_masks']
            assert masks is not None, \
            "Masks have not been assigned to dictionary"
            if apply_mean_filter:
                combined_mask = np.any(masks[:, reconstruct_indices], axis=1)
                mean_to_add = np.zeros_like(data_r)
                mean_filtered = filter_mean(mean, filter_method, low_cutoff=mlow, high_cutoff=mhigh, fps=fps)
                mean_to_add[:, combined_mask] = mean_filtered[t_start:t_stop, None]
                data_r += mean_to_add
                data_r[~spatiotemporal_event_masks] = 0

            else:
                print('Not filtering mean')
                combined_mask = np.any(masks[:, reconstruct_indices], axis=1)
                mean_to_add = np.zeros_like(data_r)
                mean_filtered = None
                mean_to_add[:, combined_mask] = mean[t_start:t_stop, None]
                data_r += mean_to_add
                data_r[~spatiotemporal_event_masks] = 0
        else:
            # Run original readdition of mean
            if apply_mean_filter:
                mean_filtered = filter_mean(mean, filter_method, low_cutoff=mlow, high_cutoff=mhigh, fps=fps)
                data_r += mean_filtered[t_start:t_stop, None]

            else:
                print('Not filtering mean')
                mean_filtered = None
                data_r += mean[t_start:t_stop, None]

        if binary_threshold:
            data_binary = np.zeros(data_r)
            data_binary[data_r > 0] = 255
            data_r = data_binary

        print('Done!')

        if roimask is None:
            data_r = data_r.reshape(shape)
        else:
            reconstructed = np.zeros((x * y, t), dtype=dtype)
            print(f'data_r shape is: {data_r.shape}')
            print(f'reconstructed shape is: {reconstructed.shape}')
            print(f'maskind is: {maskind}')
            reconstructed[maskind] = data_r.swapaxes(0, 1)
            reconstructed = reconstructed.swapaxes(0, 1)
            data_r = reconstructed.reshape(t, x, y)
        comp_out = 'sub-116_rec-baseline_run-01_cluster-' + str(i) + '.tif'
        data_r[data_r < 0.0] = 0.0
        data_r = data_r*255
        data_r[data_r > 255.0] = 255.0
        data_r = data_r.astype(np.uint8)
        #data_r[data_r > 0] = 255
        print("Writing data to tiff...")
        tif.imwrite('C:/Users/aluff/Git_Clones/test_data/' + comp_out, data_r,compression='lzw', imagej=True)

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


def filter_components(eig_mix: np.ndarray,
                      fps: float = 7.5,
                      high_cutoff: float = 0.5):
    '''
    Applies a butterworth low pass filter to the IC timecourses.

    Arguments:
        eig_mix: 
            The mixing matrix containing IC timecourses.
        fps:
            Sampling rate of the video.
        high_cutoff:
            The frequency cutoff to apply the low pass filter at.

    Returns:
        lpf_eig_mix: The filtered IC timecourses reconstructed as the eig_mix matrix.
    '''
    
    print('Filtering component timecourses using butterworth_lowpass at '+ str(high_cutoff) +'Hz...')
    timecourses = eig_mix.T
    lpf_timecourses = np.zeros_like(timecourses)
    for index in range(timecourses.shape[0]):
        lpf_timecourses[index] = butterworth(timecourses[index], fps=fps, high=high_cutoff)
    lpf_eig_mix = lpf_timecourses.T

    return lpf_eig_mix

def threshold_components(eig_mix: np.ndarray,
                         thresh_param: float):
    '''
    Applies a z-score threshold to the IC timecourses.

    Arguments:
        eig_mix: 
            The mixing matrix containing IC timecourses.
        thresh_param:
            Z-score thresholding parameter (standard deviations).

    Returns:
        thresh_eig_mix: The thresholded IC timecourses reconstructed as the eig_mix matrix.
    '''

    print('Thresholding component timecourses using z-score: >' + str(thresh_param) +'s.d.')
    timecourses = eig_mix.T
    thresh_timecourses = np.zeros_like(timecourses)
    for index in range(timecourses.shape[0]):
        timecourse = timecourses[index]
        mean = np.mean(timecourse)
        std = np.std(timecourse)
        threshold = mean + thresh_param*std
        timecourse[np.abs(timecourse) < np.abs(threshold)] = 0
        thresh_timecourses[index] = timecourse
    thresh_eig_mix = thresh_timecourses.T

    return thresh_eig_mix

def threshold_by_domains(components: dict,
                   blur: int = 1,
                   min_mask_size: int = 64,
                   thresh_type: str = 'max',
                   thresh_param: float = None,
                   schematic: bool = False):
    '''
    Function based on modified get_domain_map(). Thresholds ICs using a variety of methods for selective rebuild.

    Arguments:
        components: 
            The dictionary of components returned from seas.ica.project.  ROIs are most interesting if artifacts has already been assigned through seas.gui.run_gui.
        blur: 
            An odd integer kernel Gaussian blur to run before segmenting.  ROIs look smoother with larger blurs, but you can lose some smaller domains.
        min_mask_size:
            An integer determining the minimum ROIs passed from each thresholded IC.
        thresh_type:
            A string used to determine IC threshold method. Choose from either 'max', 'z-score' or 'percentile'.
        thresh_param:
            A float used to determine the parameter for the given thresh_type. For 'z-score', this is the z-score threshold (eg; 2.0 for 2std). For 'percentile' this is the percentile used to threshold (eg; 95th percentile = 0.95).

    Returns:
        output: a dictionary containing the results of the operation, containing the following keys
            domain_blur:
                The Gaussian blur value used when generating the map
            eig_vec: 
                The thresholded eigenvectors (ICs).  
            thresh_masks: 
                The boolean masks used to threshold eig_vec.
    '''
    print('\nExtracting Domain ROIs\n-----------------------')
    output = {}
    output['domain_blur'] = blur

    eig_vec = components['eig_vec'].copy()

    shape = components['shape']
    shape = (shape[1], shape[2])

    if 'roimask' in components.keys() and components['roimask'] is not None:
        roimask = components['roimask']
        maskind = np.where(roimask.flat == 1)[0]
    else:
        roimask = None

    if 'artifact_components' in components.keys():
        artifact_components = components['artifact_components']

        print('Switching to signal indices only for domain detection')

        if 'noise_components' in components.keys():
            noise_components = components['noise_components']

            signal_indices = np.where((artifact_components +
                                       noise_components) == 0)[0]
        else:
            print('no noise components found')
            signal_indices = np.where(artifact_components == 0)[0]
        # eig_vec = eig_vec[:, signal_indices] # Don't change number of ICs, we're updating back to dict
    
    mask = np.zeros_like(eig_vec, dtype = bool)
    print(f'eig_vec shape is: {eig_vec.shape}')

    match thresh_type:
        case 'max':
            # Return indices across each eig_vec (loading vector for component) where loading is max
            threshold_ROIs_vector = np.argmax(np.abs(eig_vec), axis=1)
            # Then threshold by clearing eig_vec outside of max indices
            mask[np.arange(eig_vec.shape[0]), threshold_ROIs_vector] = True
        case 'z-score':
            mean_ROIs_vector = np.nanmean(eig_vec, axis=0)
            std_ROIs_vector = np.nanstd(eig_vec, axis=0)
            z_ROIs_vector = (eig_vec - mean_ROIs_vector)/std_ROIs_vector
            for i in np.arange(eig_vec.shape[1]):
                abs_z = np.abs(z_ROIs_vector[:, i])
                mask[:, i] = abs_z > thresh_param
                # event = abs_z[mask[i, :]]
                # Deprecated but produced an interesting result
                # if schematic and event.size != 0:
                #     schem_thresh = np.percentile(event, 75) 
                #     mask[i, :] = abs_z > schem_thresh
        case 'percentile':
            flipped = components['flipped']
            # Flip ICs where necessary using flipped from dict
            flipped_threshold_vec = np.multiply(flipped, eig_vec)
            # Calculate 95 percentile cutoff for each IC
            cutoff_vector = np.percentile(flipped, thresh_param, axis=0)
            # Mask for all values above cutoff
            for i in np.arange(eig_vec.shape[0]):
                mask[i, :] = flipped_threshold_vec[i] > cutoff_vector[i]
        case 'max_value':
            max_ROIs_vector = np.max(eig_vec, axis=0)
            print(f'max_ROIs_vector shape is: {max_ROIs_vector.shape}')
            for i in np.arange(eig_vec.shape[1]):
                mask[:, i] = eig_vec[:, i] >= max_ROIs_vector[i]
        case 'dynamic':
            # We calculate the bounds of the eig_vec distribution
            min = np.min(eig_vec, axis = 0)
            max = np.max(eig_vec, axis = 0)

            # And check the distribution is centred around zero
            assert np.all(max > 0), "eig_vec distribution is deviant, max is less than 0."
            assert np.all(min < 0), "eig_vec distribution is deviant, min is greater than 0."
            
            # Then we identify return short tail as threshold, adjusting for flipping by ICA
            short_tail = np.where(np.abs(min) > max, max, min)
            flipped = -1 * np.sign(short_tail)
            thresholds = -1 * short_tail
            
            flipped_vec = np.multiply(flipped, eig_vec)
            flipped_thresholds = np.multiply(flipped, thresholds)

            for i in np.arange(eig_vec.shape[1]):
                mask[:, i] = flipped_vec[:, i] > flipped_thresholds[i]
        case _:
            print("Threshold type is neither max nor percentile.")

    # Filter small mask ROIs and smooth using blur
    if blur:
        print('blurring domains...')
        assert type(blur) is int, 'blur was not valid'
        if blur % 2 != 1:
            blur += 1

        eigenmask = np.zeros(shape, dtype=bool)
        eigenbrain = np.empty(shape)
        eigenbrain[:] = np.nan

        for index in range(mask.shape[1]):

            if roimask is not None:
                eigenmask.flat[maskind] = mask.T[index]
                # Remove small mask objects
                filtered = remove_small_objects(eigenmask, min_size=min_mask_size, connectivity=1)
                filtered_float = filtered.astype(np.float64)
                eigenbrain.flat[maskind] = filtered_float.flat[maskind]
                # Then blur
                blurred = cv2.GaussianBlur(eigenbrain, (blur, blur), 0)
                mask.T[index] = blurred.flat[maskind]
            else:
                eigenbrain.flat = mask.T[index]
                filtered = remove_small_objects(eigenbrain, min_size=min_mask_size, connectivity=1)
                filtered_float = filtered.astype(np.float64)
                eigenbrain.flat[maskind] = filtered_float.flat
                blurred = cv2.GaussianBlur(eigenbrain, (blur, blur), 0)
                mask.T[index] = blurred.flat

    if schematic:
        eigenmask = np.zeros(shape, dtype=np.uint8)
        eigenbrain = np.empty(shape)
        eigenbrain[:] = np.nan

        for i in range(mask.shape[1]):
            event_schematic = np.zeros(shape, dtype=np.uint8)
            eigenmask.flat[maskind] = mask.T[i]
            eigenbrain.flat[maskind] = eig_vec.T[i]
            # print("i is:", i)
            # print(eigenmask)
            if eigenmask.any():
                # tif.imwrite("/home/apluff/dev/test_data/eigenmasks/sub-070_eigenmask"+str(i)+".tif", eigenmask, imagej=True)
                labelled, num_features = ndimage.label(eigenmask)
                # print(labelled)
                # print("labelled contains values:", np.unique(labelled))
                # print("num_features is:", num_features)
                for j in range(1, num_features + 1):
                    centroid = ndimage.center_of_mass(eigenmask, 
                                                      labels = labelled,
                                                      index = j)
                    # print("j is:", j)
                    # print("centroid is:", centroid)
                    int_centroid = tuple(int(x) for x in centroid)
                    event_size = np.sum(labelled, where = labelled == j)/j
                    schem_radius = int(np.sqrt(event_size/np.pi))
                    rr, cc = draw.disk(int_centroid, schem_radius, shape = shape)
                    event_schematic[rr, cc] = 255
                    # print("mask shape is:", mask.shape)
                    # print("event_schematics shape is:", event_schematic.shape)
                    mask.T[i] = event_schematic.flat[maskind]
    
    mask_bool = mask.astype(bool)
    eig_vec[~mask_bool] = 0

    output['thresh_masks'] = mask
    # output['thresh_vec'] = eig_vec
    output['eig_vec'] = eig_vec
    
    return output

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
            eigenbrains[:] = np.nan
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
            eigenbrain[:] = np.nan
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

def dynamic_threshold(components: dict) -> dict:
    # Returns a pySEAS-compatible dictionary entry for the threshold values as
    # calculated per "Dynamic Threshold" method in Weiser et al. 2023. These
    # thresholds are recorded in the polarity relative to the original ICA
    # results (ie; not flipped). 

    eig_vec = components['eig_vec']
    output = {}
    
    # We calculate the bounds of the eig_vec distribution
    min = np.min(eig_vec, axis = 0)
    max = np.max(eig_vec, axis = 0)

    # And check the distribution is centred around zero
    assert np.all(max > 0), "eig_vec distribution is deviant, max is less than 0."
    assert np.all(min < 0), "eig_vec distribution is deviant, min is greater than 0."
    
    # Then we identify return short tail as threshold, adjusting for flipping by ICA
    short_tail = np.where(np.abs(min) > max, max, min)
    flipped = -1 * np.sign(short_tail)
    thresholds = short_tail

    # Good to check our flipped values remain consistent vs other calculations
    if 'flipped' in components.keys():
        print("flipped already exists in components dict.")
        assert np.all(flipped == components['flipped'])
    else:
        output['flipped'] = flipped
    output['component_thresholds'] = thresholds
    return output

def noise_SD_threshold(components: dict, thresh: float = 3) -> dict:
    # Returns a pySEAS-compatible dictionary entry for the threshold values as
    # calculated per "Estimating binary neural activity" method in
    # Suarez et al. 2023. These thresholds are recorded in the polarity 
    # relative to the original ICA results (ie; not flipped).

    timecourses = components['eig_mix'].T
    n_components = timecourses.shape[0]
    output = {}
    
    flipped = np.ones(n_components)
    thresholds = np.zeros(n_components)
    for i in range(n_components):
        counts, bins = np.histogram(timecourses[i], bins = 'fd')
        k = np.argmax(counts)

        # We calculate the peaks and bounds of each timecourse distribution
        peak = (bins[k] + bins[k + 1]) / 2
        min = np.min(timecourses[i])
        max = np.max(timecourses[i])
        assert np.all(max > 0), "Timecourse {i} distribution is deviant, max is less than 0."
        assert np.all(min < 0), "Timecourse {i} distribution is deviant, min is greater than 0."
        
        # And check the polarity of the distribution as returned by ICA
        if np.abs(min) > max:
            short_tail = max
        else:
            short_tail = min
        # We work in normalised polarity now for clarity (assume short tail negative)
        flip = -1 * np.sign(short_tail)
        timecourse = flip * timecourses[i].copy()
        noise_mean = flip * peak
        # And calculate the noise std by extrapolating the short tail
        noise_deltas = np.where(timecourse < noise_mean, 
                                noise_mean - timecourse, 
                                np.nan)
        noise_deltas = noise_deltas[~np.isnan(noise_deltas)]
        print(noise_deltas.shape)
        noise_distr = np.concatenate((noise_deltas, -1 * noise_deltas))
        noise_std = np.std(noise_distr)
        
        flipped[i] = flip
        # Return to original polarity to record threshold
        thresholds[i] = flip * (noise_mean + thresh * noise_std)
    
    # Good to check our flipped values remain consistent vs any previous 
    # calculations from other methods.
    if 'flipped' in components.keys():
        non_artifact_indices = np.where(components['artifact_components'] == False)
        non_noise_indices = np.where(components['noise_components'] == False)
        signal_indices = np.intersect1d(non_artifact_indices, non_noise_indices)
        print("flipped already exists in components dict.")
        assert np.all(flipped == components['flipped'])
    else:
        output['flipped'] = flipped
    
    #output['flipped'] = flipped
    output['timecourse_thresholds'] = thresholds
    return output

def rebuilt_noise_SD_threshold(components: dict, thresh: float = 2) -> dict:
    # Returns a pySEAS-compatible dictionary entry for the threshold values as
    # calculated per "Dynamic Threshold" method in Weiser et al. 2023, but
    # applied to rebuilt timecourses (+ original frame-wise mean). These
    # thresholds are recorded in the polarity of the original signal (ie;
    # all positive). 

    eig_vec = components['eig_vec']
    timecourses = components['eig_mix'].T
    frame_mean = components['mean']
    n_components = timecourses.shape[0]
    output = {}
    
    thresholds = np.zeros(n_components)
    binary_timecourses = np.zeros_like(timecourses, dtype = np.int8)
    # We calculate the max of eig_vec distribution and rebuild our timecourses
    max_weights = np.max(eig_vec, axis = 0)
    rebuilt_timecourses = max_weights[:, np.newaxis] * timecourses + frame_mean
    
    for i in range(n_components):
        timecourse = rebuilt_timecourses[i]
        counts, bins = np.histogram(timecourse, bins = 'fd')
        k = np.argmax(counts)

        # We calculate the peaks and bounds of each timecourse distribution
        noise_mean = (bins[k] + bins[k + 1]) / 2
        min = np.min(timecourse)
        max = np.max(timecourse)
        assert np.all(max > 0), \
            "Timecourse index={i} distribution is deviant, max is less than 0."
        assert np.all(min < noise_mean), \
            "Timecourse index={i} distribution is deviant, min >= noise peak."
        
        # We work in normalised polarity because timecourse has been rebuilt
        # and calculate the noise std by extrapolating the one-sided
        # distribution below the noise mean.
        noise_deltas = np.where(timecourse < noise_mean, 
                                noise_mean - timecourse, 
                                np.nan)
        noise_deltas = noise_deltas[~np.isnan(noise_deltas)]
        noise_distr = np.concatenate((noise_deltas, -1 * noise_deltas))
        noise_std = np.std(noise_distr)
        
        threshold = noise_mean + thresh * noise_std
        binary_timecourses[i] = np.where(timecourse >= threshold, 1, 0)
        timecourse[timecourse >= thresholds[i]]

    output['rebuilt_timecourse_thresholds'] = thresholds
    output['binary_threshold_timecourses'] = binary_timecourses.T
    return output
