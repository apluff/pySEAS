from typing import List

import matplotlib.pyplot as plt
import numpy as np
from seas.ica import rebuild_eigenbrain, rebuild, Input
from sklearn.metrics import mean_squared_error

def plot_component(components: dict, index: List[int]) -> None:
    # Localise our data
    eig_vec = components['eig_vec']
    timecourses = components['timecourses']
    roimask = components['roimask']

    for i in index:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, 
                                                figsize=(20, 4), 
                                                layout='constrained')
        eigenbrain = rebuild_eigenbrain(eig_vec, 
                                        index=i, 
                                        roimask=roimask)
        ax1.imshow(eigenbrain)
        ax2.hist(eig_vec[:, i], bins='fd', log=False)
        ax3.plot(timecourses[i])
        ax4.hist(timecourses[i], bins='fd')
        ax4.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        fig.suptitle("Component index=" + str(i))
        #plt.show()
        plt.savefig("/home/apluff/dev/test_data/fig_out/sub-201_base1_nmfcomp-" + str(i) + ".png")

def plot_pixel_variance(video_data: np.ndarray, roimask: np.ndarray) -> None:
    boolmask = roimask.astype(np.bool)
    # maskind = np.where(roimask.flat == 1)
    # t, y, x = rebuilt.shape
    # print("Original video shape: (t, y, x) =", (t, y, x))
    # shape = (t, y, x) 
    # print(f"Shape of the original video: {shape} as {type(shape)}") 
    # vidvec = rebuilt.reshape(t, x*y).T
    var_map = np.var(video_data, axis=0, where=boolmask)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4), layout='constrained')
    ax1.imshow(var_map)
    ax2.hist(var_map.flatten(), bins='fd', log=False)
    fig.suptitle("Residual variance map")
    plt.show

def component_MSE(input: Input,
                  components: dict,
                  ) -> dict:

    # This is the draft case for a non-overlapping component.

    dfofvec = input.vector.astype('float64')
    # rebuilt = rebuild(components,
    #                   artifact_components='none',
    #                   apply_mean_filter=False,
    #                   )
    n_components = components['n_components']
    eig_vec = components["eig_vec"]
    eig_mix = components["eig_mix"]
    thresh_vec = components["thresh_vec"]
        
    assert components['shape'] == input.shape, \
        "Original and rebuilt video shapes don't match, are these the same video?"

    # Flat-full to flat-masked
    # maskind = np.where(input.roimask.flat == 1)
    # masked_dfof = vector[maskind, :]
    
    component_errors = np.empty(n_components)
    component_errors[:] = np.nan

    thresh_indices = thresh_vec.astype(np.bool)
    for i in n_components:
        thresh_idx = thresh_indices[i] #CHECK
        #Flat-masked to flat-submasked
        y_true = dfofvec[thresh_idx]
        # Rebuild pixels of interest for single component only
        y_pred = np.dot(eig_vec[i, thresh_idx], eig_mix[:, i])
        # Calculate MSE
        mse = mean_squared_error(y_true, y_pred)
        component_errors[i] = mse

    # We check that MSE was calculated and stored for all components.
    assert np.any(np.isnan(component_errors)) is not True, \
        "At least one component returned NaN for MSE. Check output."

    output = {}
    output["component_MSE"] = component_errors

    return output