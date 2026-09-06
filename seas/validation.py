from typing import List

import matplotlib.pyplot as plt
from seas.ica import rebuild_eigenbrain

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
