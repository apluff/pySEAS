# Import required libraries
import seas
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import h5py # for dealing with the HDF5 objects
import sys

TEST_VIDEO="/home/apluff/Projects/test_data/sub-201_ses-01_age-P34_rec-baseline_run-01_comp-014_video-dfof_scrop-cortex_sbin-4x.tif"
TEST_MASK="/home/apluff/Projects/test_data/sub-201_ses-01_age-P34_rec-baseline_run-01_image-mask_scrop-cortex_sbin-4x.tif"
TEST_MAXITER=1000
TEST_OUTPATH="sub-201_ses-01_age-P34_rec-baseline_run-01_scrop-cortex_sbin-4x_ica-test.hdf5"

class SuccinctPySEAS:
    def __init__(self, video, mask):
        self.video = video # Path to input dfof_processed file
        self.mask = mask # Path to input mask file
    
    def load_data(self):
        # print(f"Video variable is: {video}") # Debug
        try:
            self.video_data = tif.imread(self.video)  # Load the video data using tifffile
            print(f"Video is Type: {type(self.video_data)} \nShape: {self.video_data.shape}") # Check array shape
        except FileNotFoundError:
            print('File not found', flush=True)
        try:
            self.mask_data = tif.imread(self.mask) # Load the mask data using tifffile
            print(f"Mask is Type: {type(self.mask_data)} \nShape: {self.mask_data.shape}") # Check array shape
            self.binary_mask = (self.mask_data > 0).astype(int) # Ensure mask format is binary
            print(f"Unique values: {np.unique(self.binary_mask)}") # make sure no sneaky floats, must be int 0 1
        except FileNotFoundError:
            print('Mask not found', flush=True)

    def run_ica(self, max_iter = 500):
        self.t, self.y, self.x = self.video_data.shape # Extract video dimensions
        print("Original video shape: (t, y, x) =", (self.t, self.y, self.x))
        self.shape = (self.t, self.y, self.x) # Store video dimensions
        print(f"Shape of the original video: {self.shape} as {type(self.shape)}") # Implementation of type() within class?
        self.vidvec = self.video_data.reshape(self.t, self.x*self.y).T # Reshape video
        print("Vector dimensions (should be (x*y, t)): " + str(self.vidvec.shape))
        self.components = seas.ica.project(vector=self.vidvec, 
                                            shape=self.shape, 
                                            roimask=self.binary_mask,
                                            calc_residuals=False,
                                            max_iter=max_iter)
                                            
        # Plot video first frame (unneeded for batch script, maybe save to disk for sanity check?)
        # plt.imshow(video_data[0,:,:])
        # Visualise the mask (unneeded for batch script, maybe save to disk for sanity check?)
        # plt.imshow(binary_mask)

        # you have the option to adding some useful plots, such as the lag-1 autocorrelation
        # saving plot to disk would be a nice feature
        # plt.figure(figsize=(4,4))
        # plt.hist(components['lag1']) # or "lag1_full"
        # plt.axvline(components['cutoff'], color='k',linestyle='dashed', linewidth=1)

        # I think that before visualising we will want to add additional keys from the output of the domain modules
        self.domain_map = seas.domains.get_domain_map(self.components, apply_filter_mean=True, map_only=False) # check the resulting keys, very useful

        # add the domain map to the components dictionary
        self.components.update(self.domain_map)

    def save_data(self, outpath):
        self.outpath = outpath
        self.f = seas.hdf5manager(self.outpath) # Create hdf5 file on disk
        self.f.save(self.components) # Save computed components to hdf5 file

if __name__ == '__main__':
    video = TEST_VIDEO  # Input video file path from command-line argument
    max_iter = TEST_MAXITER # Max iterations for FastICA
    mask = TEST_MASK  # Mask path from command-line argument
    outpath = TEST_OUTPATH  # Output file path from command-line argument
    
    processor = SuccinctPySEAS(video, mask)
    processor.load_data()
    processor.run_ica(max_iter = max_iter)
    processor.save_data(outpath)