from abc import ABC, abstractmethod
from dataclasses import dataclass

from amica import AMICA
import cupy
import numpy as np
from picard import Picard, picard
from scipy import linalg
from sklearn.decomposition import FastICA, NMF
from sklearn.utils.extmath import randomized_svd
from typing import Tuple

from seas.signalanalysis import sort_noise, lag_n_autocorr


@dataclass
class Projection:
    n_components: int
    eig_vec: np.ndarray
    eig_mix: np.ndarray
    lag1_full: np.ndarray
    noise: np.ndarray
    cutoff: float | None
    increased_cutoff: int

    def __post_init__(self) -> None:
        print('components shape:', self.eig_vec.shape)
        assert self.n_components == self.eig_vec.shape[1], \
            'n_components does not match the size of eig_vec, check outputs.'

    def validate_projection(self) -> Tuple[bool, int, int]:
        assert self.noise.size == self.n_components, \
            "Noise length doesn't match n_components, something is wrong."

        # Test signal vs noise to determine if underdecomposed
        frames = self.eig_mix.shape[1]
        underdecomposed = self.check_noise(self.noise, frames)

        # Increase components for next loop if necessary
        if underdecomposed:
            self.n_components += self.n_components // 2
            if self.n_components > frames:
                print('\nComponents maxed out!')
                print('\tAttempted:', self.n_components)
                self.n_components = frames
                print('\tReduced to:', frames)
            self.increased_cutoff += 1

        return (underdecomposed, self.n_components, self.increased_cutoff)

    def check_noise(self, noise: np.ndarray, frames: int) -> bool:
        n_components = noise.size
        p_signal = (1 - noise.sum() / noise.size) * 100
        if noise.size == frames:  # All components are being used.
            return False
        elif p_signal < 75: # Data is sufficiently decomposed.
            print('ICA components were under 75% signal ({0}% signal).'\
                .format(p_signal))
            return False
        elif n_components >= frames: # Data is maximally decomposed.
            print('ICA components were under 75% signal ({0}% signal).'\
                .format(p_signal))
            print('However, number of components is maxed out.')
            print('Using this decomposition...')
            return False
        else: # Data is underdecomposed.
            print('ICA components were over 75% signal ({0}% signal).'\
                .format(p_signal))
            print('Recalculating with more components...')
            return True

def validate_projection(projection: Projection) -> Tuple[bool, int, int]:
        
        n_components = projection.n_components
        noise = projection.noise
        frames = projection.eig_mix.shape[1]
        increased_cutoff = projection.increased_cutoff

        assert noise.size == n_components, \
            "Noise length doesn't match n_components, something is wrong."

        # Test signal vs noise to determine if underdecomposed
        underdecomposed = check_noise(noise, frames)

        # Increase components for next loop if necessary
        if underdecomposed:
            n_components += n_components // 2
            if n_components > frames:
                print('\nComponents maxed out!')
                print('\tAttempted:', n_components)
                n_components = frames
                print('\tReduced to:', frames)
            increased_cutoff += 1

        return (underdecomposed, n_components, increased_cutoff)


def check_noise(noise: np.ndarray, frames: int) -> bool:
            n_components = noise.size
            p_signal = (1 - noise.sum() / noise.size) * 100
            if noise.size == frames:  # All components are being used.
                return False
            elif p_signal < 75: # Data is sufficiently decomposed.
                print('ICA components were under 75% signal ({0}% signal).'\
                    .format(p_signal))
                return False
            elif n_components >= frames: # Data is maximally decomposed.
                print('ICA components were under 75% signal ({0}% signal).'\
                    .format(p_signal))
                print('However, number of components is maxed out.')
                print('Using this decomposition...')
                return False
            else: # Data is underdecomposed.
                print('ICA components were over 75% signal ({0}% signal).'\
                    .format(p_signal))
                print('Recalculating with more components...')
                return True


def estimate_n_components(vector: np.ndarray, 
                          svd_multiplier: float = 5,
                          estimator: str = 'cusvd') -> Tuple[int, np.ndarray]:
    
    match estimator:
        case 'svd':
            calculator = _SVD()
            print('Estimating n_components with SVD...')
        case 'cusvd':
            calculator = _cuSVD()
            print('Estimating n_components with cupy SVD...')
        case 'randomized_svd':
            calculator = _randomizedSVD()
            print('Estimating n_components with randomized SVD...')

    cupy_vector = cupy.asarray(vector)
    u, ev, _ = calculator.decompose(cupy_vector)
    u = cupy.asnumpy(u)
    ev = cupy.asnumpy(ev)
    # components['svd_eigval'] = ev # Not used anywhere, should I store this?

    # Get starting point for decomposition based on svd mutliplier * the 
    # approximate point of transition to linearity in tail of ev components.
    cross_1 = approximate_svd_linearity_transition(ev)
    n_components = cross_1 * svd_multiplier
    w_init = u[:n_components, :n_components].astype('float64')
    
    return n_components, w_init


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


class Projector(ABC):

    @abstractmethod
    def project(self, vector) -> Projection:
        pass


class _FastICA(Projector):

    def __init__(self, 
                 n_components: int | None = None, 
                 svd_multiplier: float | None = 5, 
                 max_iter: int = 1000,
                 estimator: str | None = 'svd') -> None:
        self.n_components = n_components
        self.svd_multiplier = svd_multiplier
        self.max_iter = max_iter
        self.estimator = estimator

    def project(self, vector: np.ndarray) -> Projection:
        '''
        Replicates original FastICA processing in conjunction with the
        top-level function project() (original wrapper) per Weiser et al. 2023.
        '''
        # Estimate n_components if necessary
        if self.n_components is None:
            n_components, w_init = estimate_n_components(vector, 
                                                         self.svd_multiplier,
                                                         self.estimator,
                                                         )
        else:
            n_components = self.n_components
            w_init = None

        underdecomposed = True # To init loop
        increased_cutoff = 0
        while underdecomposed:

            # Calculate ICA
            print('\nCalculating ICA with', n_components, 'components...')
            ica = FastICA(n_components = n_components,
                        max_iter = self.max_iter,
                        random_state = 1000,
                        w_init = w_init,
                        )
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

            # Calculate noise
            timecourses = eig_mix.T
            lag1 = lag_n_autocorr(timecourses, 1)
            noise, cutoff = sort_noise(timecourses, lag1)

            projection = Projection(
                n_components=n_components,
                eig_vec=eig_vec,
                eig_mix=eig_mix,
                lag1_full=lag1,
                noise=noise,
                cutoff=cutoff,
                increased_cutoff=increased_cutoff,
                )

            if self.n_components is None:
                underdecomposed, n_components, increased_cutoff = \
                    validate_projection(projection)
            else:
                underdecomposed = False
        
        return projection


class _FastICA_Testing(Projector):

    def __init__(self, 
                 n_components: int | None = None, 
                 svd_multiplier: float | None = 5, 
                 max_iter: int = 1000,
                 estimator: str | None = 'svd') -> None:
        self.n_components = n_components
        self.svd_multiplier = svd_multiplier
        self.max_iter = max_iter
        self.estimator = estimator

    def project(self, 
                vector: np.ndarray, 
                n_components: int,
                w_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        ica = FastICA(n_components = n_components,
                    max_iter = self.max_iter,
                    random_state = 1000,
                    w_init = w_init,
                    )
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

        return eig_vec, eig_mix


class _PicardICA(Projector):

    def __init__(self, 
                 n_components: int | None = None, 
                 svd_multiplier: float | None = 5, 
                 max_iter: int = 1000,
                 estimator: str | None = 'svd',
                 ortho: bool = False) -> None:
        self.n_components = n_components
        self.svd_multiplier = svd_multiplier
        self.max_iter = max_iter
        self.estimator = estimator
        self.ortho = ortho

    def project(self, vector: np.ndarray) -> Projection:
        '''
        Replicates original FastICA processing in conjunction with the
        top-level function project() (original wrapper) per Weiser et al. 2023,
        using PicardICA's scikit API as a drop-in replacement.
        '''
        # Estimate n_components if necessary
        if self.n_components is None:
            n_components, w_init = estimate_n_components(vector, 
                                                         self.svd_multiplier,
                                                         self.estimator,
                                                        )
        else:
            n_components = self.n_components
            w_init = None

        underdecomposed = True
        increased_cutoff = 0
        while underdecomposed:
            # Calcualte ICA
            print('\nCalculating ICA with', n_components, 'components...')

            ica = Picard(n_components=n_components,
                         max_iter=self.max_iter,
                         random_state=1000,
                         w_init=w_init,
                         ortho=self.ortho,
                         )

            try:
                eig_vec = ica.fit_transform(vector)  # Eigenbrains
            except ValueError:
                print('Calculation exceeded float32 maximum.')
                print('Trying again with float64 vector...')
                # Value error if any value exceeds float32 maximum.
                # Overcome this by converting to float64.
                eig_vec = ica.fit_transform(vector.astype('float64'))
            # print("n_iter:" , ica.n_iter_) # NOT PROVIDED FOR Picard
            eig_mix = ica.mixing_

            # The arrangement of outputs for this is weird. Check in detail
            # if you need to use this implementation rather than sklearn
            # interface above.
            # K, W, Y, n_iter = picard(vector,
            #                          n_components=n_components,
            #                          max_iter=self.max_iter,
            #                          w_init=w_init,
            #                          random_state=1000,
            #                          ortho=self.ortho,
            #                          return_n_iter=True)
            # eig_vec = Y # Y.T???
            # w = np.dot(W, K)
            # A = np.dot(w.T, np.linalg.inv(np.dot(w, w.T)))
            # eig_mix = A
            # print("n_iter:" , n_iter)
            # REMINDER: Returns eig_vec.shape = (n_components, frames), and 
            # eig_mix.shape = (n_components, masked_pixels)

            # Calculate noise
            timecourses = eig_mix.T
            lag1 = lag_n_autocorr(timecourses, 1)
            noise, cutoff = sort_noise(timecourses, lag1)

            projection = Projection(
                n_components=n_components,
                eig_vec=eig_vec,
                eig_mix=eig_mix,
                lag1_full=lag1,
                noise=noise,
                cutoff=cutoff,
                increased_cutoff=increased_cutoff,
                )

            if self.n_components is None:
                underdecomposed, n_components, increased_cutoff = \
                    validate_projection(projection)
            else:
                underdecomposed = False

        #TODO: Does this need to be calculated here?
                # if self.n_components is None:
                #     lag1_full = lag_n_autocorr(eig_mix.T, 1)
                #     svd_cutoff = n_components
                # else: # For compatability with original pySEAS dicts
                #     lag1_full = None # Maybe change this to return all the time.
                #     svd_cutoff = None
        
        return projection
    

class _AMICA(Projector):

    def __init__(self, 
                 n_components: int | None = None, 
                 svd_multiplier: float | None = 5, 
                 max_iter: int = 1000,
                 estimator: str | None = 'svd') -> None:
            self.n_components = n_components
            self.svd_multiplier = svd_multiplier
            self.max_iter = max_iter
            self.estimator = estimator
    
    def project(self, vector: np.ndarray) -> Projection:
        '''
        Replicates original FastICA processing in conjunction with the
        top-level function project() (original wrapper) per Weiser et al. 2023.
        '''
        # Estimate n_components if necessary
        if self.n_components is None:
            n_components, w_init = estimate_n_components(vector, 
                                                         self.svd_multiplier,
                                                         self.estimator,
                                                         )
        else:
            n_components = self.n_components
            w_init = None
        
        underdecomposed = True # To init loop
        increased_cutoff = 0
        while underdecomposed:

            # Calculate ICA
            print('\nCalculating ICA with', n_components, 'components...')
            ica = AMICA(n_components=n_components,
                        max_iter=self.max_iter,
                        random_state=1000,
                        w_init=w_init,
                        device='cuda',
                        do_newton=False,
                        )
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

            # Calculate noise
            timecourses = eig_mix.T
            lag1 = lag_n_autocorr(timecourses, 1)
            noise, cutoff = sort_noise(timecourses, lag1)

            projection = Projection(
                n_components=n_components,
                eig_vec=eig_vec,
                eig_mix=eig_mix,
                lag1_full=lag1,
                noise=noise,
                cutoff=cutoff,
                increased_cutoff=increased_cutoff,
            )

            if self.n_components is None:
                underdecomposed, n_components, increased_cutoff = \
                    validate_projection(projection)
            else:
                underdecomposed = False

            return projection


class _NMF(Projector):

    def __init__(self, 
                 n_components: int | None = None, 
                 svd_multiplier: float | None = 5, 
                 max_iter: int = 1000,
                 estimator: str | None = 'svd') -> None:
        self.n_components = n_components
        self.svd_multiplier = svd_multiplier
        self.max_iter = max_iter
        self.estimator = estimator

    def project(self, vector: np.ndarray) -> Projection:
        '''
        Replicates original FastICA processing in conjunction with the
        top-level function project() (original wrapper) per Weiser et al. 2023.
        '''
        # Estimate n_components if necessary
        if self.n_components is None:
            n_components, _ = estimate_n_components(vector, 
                                                    self.svd_multiplier,
                                                    self.estimator,
                                                    )
        else:
            n_components = self.n_components

        print(f"vector min is: {np.min(vector)}")
        assert not np.any(vector < 0), \
            "Negative values exist in supplied vector for NMF."

        underdecomposed = True # To init loop
        increased_cutoff = 0
        while underdecomposed:

            # Calculate decomposition
            print('\nCalculating NMF with', n_components, 'components...')
            nmf = NMF(n_components=n_components,
                      max_iter=self.max_iter,
                      random_state=1000)
            try:
                eig_vec = nmf.fit_transform(vector)  # Eigenbrains
            except ValueError:
                print('Calculation exceeded float32 maximum.')
                print('Trying again with float64 vector...')
                # Value error if any value exceeds float32 maximum.
                # Overcome this by converting to float64.
                eig_vec = nmf.fit_transform(vector.astype('float64'))
            print("n_iter:" , nmf.n_iter_)
            eig_mix = nmf.components_.T

            # Calculate noise
            timecourses = eig_mix.T
            lag1 = lag_n_autocorr(timecourses, 1)
            noise, cutoff = sort_noise(timecourses, lag1)

            projection = Projection(
                n_components=n_components,
                eig_vec=eig_vec,
                eig_mix=eig_mix,
                lag1_full=lag1,
                noise=noise,
                cutoff=cutoff,
                increased_cutoff=increased_cutoff,
            )

            if self.n_components is None:
                underdecomposed, n_components, increased_cutoff = \
                    validate_projection(projection)
            else:
                underdecomposed = False
        
        return projection


class Estimator(ABC):

    @abstractmethod
    def __init__(self):
            pass

    # Goose method QUACK
    def project(self, n_components: int, w_init: np.ndarray, vector: np.ndarray) -> Projection:
        u, ev, v = self.decompose(vector)
        lag1 = lag_n_autocorr(v.T, 1)
        noise, cutoff = sort_noise(v.T, lag1)
        return Projection(n_components=np.size(ev),
                          eig_vec=u,
                          eig_mix=v.T,
                          lag1_full=lag1,
                          noise=noise,
                          cutoff=cutoff,
                          increased_cutoff=None)


class _SVD(Estimator):

    def __init__(self):
        pass

    def decompose(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                print('PCA run with scipy.linalg.svd and gesvd lapack.')
            except ValueError:
                print('Secondary PCA failed.')
                u, ev, _ = np.linalg.svd(vector,
                                         full_matrices = False)
                print('PCA run with numpy.linalg.svd.')
        
        return u, ev, v


class _cuSVD(Estimator):

    def __init__(self):
            pass
    
    def decompose(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, ev, v = cupy.linalg.svd(vector, full_matrices = False)
        print('PCA run with cupy.linalg.svd and gesvd lapack via CUDA.')

        return u, ev, v
            

class _randomizedSVD(Estimator):

    def __init__(self):
        pass

    def decompose(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_components = np.min(vector.shape)
        u, ev, v = randomized_svd(vector, n_components, n_oversamples = 100)

        return u, ev, v
