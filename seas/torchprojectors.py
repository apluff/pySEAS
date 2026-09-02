from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
from amica import AMICA
import cupy
import numpy as np
from sklearn.decomposition._nmf import _initialize_nmf
import torch
import torchnmf.nmf

from seas.signalanalysis import sort_noise, lag_n_autocorr


class Projector(ABC):

    @abstractmethod
    def preprocess(self, vector) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def project(self, 
                vector, 
                n_components, 
                w_init) -> Tuple[np.ndarray, np.ndarray]:
        pass


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
            
    def preprocess(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(input.vector, 0).flatten()
        vector = input.vector - mean

        return mean, vector
    
    def project(self, 
                vector: np.ndarray,
                n_components: int,
                w_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        return eig_vec, eig_mix


class _torchNMF(Projector):

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
        
        print(f"vector min is: {np.min(vector)}")
        assert not np.any(vector < 0), \
            "Negative values exist in supplied vector for NMF."

        # Calculate decomposition
        print('\nCalculating NMF on CUDA with', n_components, 'components...')
        W, H = _initialize_nmf(vector.T, n_components, random_state=1000)
        torch_vector = torch.from_numpy(vector)
        torch_vector = torch_vector.t().cuda()
        nmf = torchnmf.nmf.NMF(torch_vector.shape,
                                W=W,
                                H=H,
                                rank=n_components,
                                )
        nmf = nmf.cuda()
        total_iter = nmf.fit(torch_vector)
        W = nmf.W
        eig_vec = W.detach().cpu().numpy()  # Eigenbrains
        print("n_iter:" , total_iter)
        H = nmf.H
        eig_mix = H.detach().cpu().numpy()
        
        return eig_vec, eig_mix


class Estimator(Projector):

    @abstractmethod
    def __init__(self):
            pass

    # Goose method QUACK
    # This interface allows estimators (that calculate all components)
    # to act as projectors.
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


class _torchSVD(Estimator):

    def __init__(self):
            pass
    
    def decompose(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, ev, v = cupy.linalg.svd(vector, full_matrices = False)
        print('PCA run with cupy.linalg.svd and gesvd lapack via CUDA.')

        return u, ev, v
