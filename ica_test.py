import sys
import seas
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from seas.ica import Input, Config, Components

TEST_VIDEO="/home/apluff/Projects/test_data/sub-201_ses-01_age-P34_rec-baseline_run-01_comp-014_video-dfof_scrop-cortex_sbin-4x.tif"
TEST_MASK="/home/apluff/Projects/test_data/sub-201_ses-01_age-P34_rec-baseline_run-01_image-mask_scrop-cortex_sbin-4x.tif"
TEST_MAXITER=1000
TEST_OUTPATH="sub-201_ses-01_age-P34_rec-baseline_run-01_scrop-cortex_sbin-4x_ica-test.hdf5"

def load_data(video: np.ndarray, mask: np.ndarray) -> Input:
    # Load video and mask data
    try:
        video_data = tif.imread(video)
        print(f"Video is Type: {type(video_data)} \nShape: {video_data.shape}")
    except FileNotFoundError:
        print('File not found', flush=True)
    try:
        mask_data = tif.imread(mask)
        print(f"Mask is Type: {type(mask_data)} \nShape: {mask_data.shape}")
        binary_mask = (mask_data > 0).astype(int)
        print(f"Unique values: {np.unique(binary_mask)}") # NO SNEAKY FLOATS
    except FileNotFoundError:
        print('Mask not found', flush=True)

    # Prep shape and vidvec
    t, y, x = video_data.shape
    print("Original video shape: (t, y, x) =", (t, y, x))
    shape = (t, y, x) 
    print(f"Shape of the original video: {shape} as {type(shape)}") 
    vidvec = video_data.reshape(t, x*y).T
    
    return Input(vidvec, shape, binary_mask)


def run_ica(input: Input, config: Config) -> Components:
        components = seas.ica.project(input, config)
        domain_map = seas.domains.get_domain_map(components, map_only=False)
        components.update(domain_map)

        return components


def save_data(components: Components, outpath: str) -> None:
        f = seas.hdf5manager(outpath)
        f.save(components)


def main(video, max_iter, mask, outpath, test_projector) -> None:
    input = load_data(video, mask)
    config = Config(n_components=None,
                    calc_residuals=False,
                    crop_excess_noise=False, 
                    max_iter=max_iter,
                    projector=test_projector,
                    estimator='cusvd',
                    )
    print(f'Running ICA test with projector: {test_projector}')
    components = run_ica(input, config)
    save_data(components, outpath)


if __name__ == '__main__':
    video = TEST_VIDEO
    max_iter = TEST_MAXITER
    mask = TEST_MASK
    outpath = TEST_OUTPATH
    test_projector=sys.argv[1]
    
    main(video, max_iter, mask, outpath, test_projector)