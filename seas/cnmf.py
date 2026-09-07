from timeit import default_timer as timer

import numpy as np
from seas.ica import Input, rebuild, apply_dynamic_thresholds
from sklearn.decomposition import non_negative_factorization
import matplotlib.pyplot as plt


def run_cnmf(components: dict) -> dict:
    nmf_input = prep_nmf_input(components)
    X = nmf_input.vector
    k_components, W_init, H_init = prep_nmf_init(components)
    print(f"W_init shape is: {W_init.shape}")
    print(f"H_init shape is: {H_init.shape}")
    print(f"roimask area size is {np.sum(nmf_input.roimask)}")
    t0 = timer()
    W, H, n_iter = non_negative_factorization(X.T, 
                                                W=W_init, 
                                                H=H_init, 
                                                n_components=k_components, 
                                                init='custom',
                                                update_H=False, 
                                                tol=1e-7, 
                                                max_iter=1000,
                                                random_state=1000,
                                                verbose=1)
    t = timer() - t0
    print('CNMF took: {0} sec'.format(t))
    print(f'CNMF ran for {n_iter} iterations.')

    mean = zero_mean(components)

    output = {}
    output['W_init'] = W_init
    output['H_init'] = H_init
    output['n_components'] = k_components
    output['eig_vec'] = H.T
    output['eig_mix'] = W
    output['timecourses'] = W.T
    output['mean'] = mean
    output['roimask'] = nmf_input.roimask
    output['artifact_components'] = np.zeros(k_components)
    output['noise_components'] = np.zeros(k_components)

    return output


def zero_mean(components: dict) -> np.ndarray:
    mean = components['mean']
    mean[:] = 0

    return mean


def derive_nmf_mask(components: dict) -> np.ndarray:
    assert 'artifact_components' in components.keys(), \
        "No artifact_components found. Has this dictionary been filtered?"
    try:
        thresh_vec = components['thresh_vec']
    except KeyError:
        thresh = apply_dynamic_thresholds(components)
        components.update(thresh)
        thresh_vec = components['thresh_vec']

    artifact_components = components['artifact_components']
    roimask = components['roimask']

    # Double computation here, abstract?
    signal_components = ~artifact_components.astype(np.bool)
    thresh_masks = thresh_vec[:, signal_components].astype(np.uint8)
    thresh_union = np.sum(thresh_masks, axis=1).astype(np.bool)
    nmf_mask = thresh_union.astype(np.uint8)
    maskind = np.where(roimask.flat == 1)
    eigenbrain = np.empty(roimask.shape)
    eigenbrain[:] = np.nan
    eigenbrain.flat[maskind] = nmf_mask.T

    return eigenbrain


def prep_nmf_input(components: dict) -> Input:
    nmf_mask = derive_nmf_mask(components)
    # Explicit rebuild of ICA decomposition (remove noise manually if needed)
    rebuilt = rebuild(components, apply_mean_filter=False, include_noise=True)
    t, y, x = rebuilt.shape
    rebuiltvec = rebuilt.reshape(t, x*y)
    print(f"rebuiltvec shape is: {rebuiltvec.shape}")
    print(f"rebuild.shape is: {rebuilt.shape}")
    print(f"nmf_mask shape is: {nmf_mask.shape}")
    nmf_input = Input(rebuiltvec.T, rebuilt.shape, nmf_mask)

    return nmf_input


def prep_nmf_init(components: dict) -> tuple:
    # Currently uses artifact status as deciding factor for removal from NMF
    # init. Please select noise components as artifacts if you don't want to
    # include them!

    assert 'artifact_components' in components.keys(), \
        "No artifact_components found. Has this dictionary been filtered?"

    # I could calculate these here...
    assert 'thresh_vec' in components.keys()

    # Crop thresholded components using artifact_components
    artifact_components = components['artifact_components']
    thresh_vec = components['thresh_vec']
    eig_mix = components['eig_mix']
    signal_components = ~artifact_components.astype(np.bool)

    # Init nmf init
    H_init = thresh_vec[:, signal_components]
    H_init = H_init.T # Convention for NMF
    W_init = eig_mix[:, signal_components]
    W_init[W_init<=0] = 1e-8 # Not necessary with hard constraint on H, but not harmful.
    k_components = np.count_nonzero(signal_components)

    union_component = generate_union_component(H_init)

    maskind = np.where(components['roimask'].flat == 1)
    eigenbrain = np.empty(components['roimask'].shape)
    eigenbrain[:] = np.nan
    eigenbrain.flat[maskind] = union_component
    plt.imshow(eigenbrain)
    plt.show()


    k_components, W_init, H_init = append_component(union_component,
                                                    W_init,
                                                    H_init)

    # Constrain H_init to ROIs only
    mask = union_component.astype(np.uint8)
    mask[mask>1] = 1
    print(f"Size of masked area: {np.sum(mask)}")
    nmfind = np.where(mask == 1)
    H_init_sub = H_init[:, nmfind[0]] # Why is nmfind a tupled array?

    return k_components, W_init, H_init_sub


def generate_union_component(H_init: np.ndarray) -> np.ndarray:
    thresh_masks = H_init.astype(np.uint8)
    thresh_union = np.sum(thresh_masks, axis=0).astype(np.bool)
    union_component = thresh_union.astype(np.float32)
    print(f"union_component is: {union_component}")
    print(f"union_component elements include: {np.unique(union_component)}")

    return union_component


def append_component(component: np.ndarray, 
                     W_init: np.ndarray, 
                     H_init: np.ndarray) -> tuple:
    k_components = W_init.shape[1] + 1
    frames = W_init.shape[0]
    pixels = H_init.shape[1]
    H_init_super = np.zeros((k_components, pixels)).astype(np.float32)
    H_init_super[0:k_components-1, :] = H_init
    W_init_super = np.zeros((frames, k_components)).astype(np.float32)
    # Implicit zeros for final timeseries are fine for update_H=False 
    W_init_super[:, 0:k_components-1] = W_init
    print(component.shape)
    print(H_init_super.shape)
    H_init_super[-1, :] = component

    return k_components, W_init_super, H_init_super
