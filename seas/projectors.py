from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from picard import Picard, picard
from scipy import linalg
from sklearn.decomposition import FastICA, NMF

from seas.signalanalysis import sort_noise, lag_n_autocorr

class Projector(ABC):

    @abstractmethod
    def preprocess(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def project(self, 
                vector, 
                n_components, 
                w_init) -> Tuple[np.ndarray, np.ndarray]:
        pass


# class _FastICA_Original(Projector):

#     def __init__(self, 
#                  n_components: int | None = None, 
#                  svd_multiplier: float | None = 5, 
#                  max_iter: int = 1000,
#                  estimator: str | None = 'svd') -> None:
#         self.n_components = n_components
#         self.svd_multiplier = svd_multiplier
#         self.max_iter = max_iter
#         self.estimator = estimator

#     def preprocess(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         mean = np.mean(input.vector, 0).flatten()
#         vector = input.vector - mean
#         return mean, vector

#     def project(self, vector: np.ndarray) -> Projection:
#         '''
#         Replicates original FastICA processing in conjunction with the
#         top-level function project() (original wrapper) per Weiser et al. 2023.
#         '''
#         # Estimate n_components if necessary
#         if self.n_components is None:
#             n_components, w_init = estimate_n_components(vector, 
#                                                          self.svd_multiplier,
#                                                          self.estimator,
#                                                          )
#         else:
#             n_components = self.n_components
#             w_init = None

#         underdecomposed = True # To init loop
#         increased_cutoff = 0
#         while underdecomposed:

#             # Calculate ICA
#             print('\nCalculating ICA with', n_components, 'components...')
#             ica = FastICA(n_components = n_components,
#                         max_iter = self.max_iter,
#                         random_state = 1000,
#                         w_init = w_init,
#                         )
#             try:
#                 eig_vec = ica.fit_transform(vector)  # Eigenbrains
#             except ValueError:
#                 print('Calculation exceeded float32 maximum.')
#                 print('Trying again with float64 vector...')
#                 # Value error if any value exceeds float32 maximum.
#                 # Overcome this by converting to float64.
#                 eig_vec = ica.fit_transform(vector.astype('float64'))
#             print("n_iter:" , ica.n_iter_)
#             eig_mix = ica.mixing_

#             # Calculate noise
#             timecourses = eig_mix.T
#             lag1 = lag_n_autocorr(timecourses, 1)
#             noise, cutoff = sort_noise(timecourses, lag1)

#             projection = Projection(
#                 n_components=n_components,
#                 eig_vec=eig_vec,
#                 eig_mix=eig_mix,
#                 lag1_full=lag1,
#                 noise=noise,
#                 cutoff=cutoff,
#                 increased_cutoff=increased_cutoff,
#                 )

#             if self.n_components is None:
#                 underdecomposed, n_components, increased_cutoff = \
#                     validate_projection(projection)
#             else:
#                 underdecomposed = False
        
#         return projection


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

    def preprocess(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            mean = np.mean(input.vector, 0).flatten()
            vector = input.vector - mean

            return mean, vector

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

    def preprocess(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(input.vector, 0).flatten()
        vector = input.vector - mean

        return mean, vector

    def project(self, 
                vector: np.ndarray,
                n_components: int,
                w_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        
            return eig_vec, eig_mix
    

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

    def preprocess(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(input.vector, 0).flatten()
        vector = input.vector + np.abs(np.min(input.vector)) + 1e-8

        return mean, vector

    def project(self, 
                vector: np.ndarray,
                n_components: int,
                w_init: None) -> Tuple[np.ndarray, np.ndarray]:
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

        return eig_vec, eig_mix


class Estimator(ABC):

    @abstractmethod
    def __init__(self):
            pass

    # Goose method QUACK
    # This interface allows estimators (that calculate all components)
    # to act as projectors.
    def project(self, 
                vector: np.ndarray, 
                n_components: None, 
                w_init: None) -> Tuple[np.ndarray, np.ndarray]:
        u, _, v = self.decompose(vector)
        return u, v.T


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