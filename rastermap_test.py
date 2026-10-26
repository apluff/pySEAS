import os
import numpy as np
import seas.hdf5manager
from seas.cnmf import compile_rastermap_data

TEST_DIR="/home/apluff/dev/test_data/"
TEST_INPATH = (TEST_DIR + 
    "sub-201_ses-01_age-P34_rec-baseline_run-01_comp-014_crop-RH_cnmf-filtered.hdf5")
TEST_TIMEOUT = (TEST_DIR + 
    "sub-201_ses-01_age-P34_rec-baseline_run-01_comp-014_crop-RH_rastermap-timeseries_lpf-1.npy")
TEST_LOCOUT = (TEST_DIR + 
    "sub-201_ses-01_age-P34_rec-baseline_run-01_comp-014_crop-RH_rastermap-locations_lpf-1.npy")


def load_data(inpath: str) -> dict:
    # Load pyseas dictionary
    if os.path.exists(inpath):
        f = seas.hdf5manager(inpath)
    else:
         print(FileNotFoundError)

    return f.load()


def main(inpath, timeseries_out, locations_out) -> None:
    components = load_data(inpath)
    print(f'Running CNMF test using file: {inpath}')
    timeseries, locations = compile_rastermap_data(components)
    print(f'Timeseries and locations look like this:')
    print(timeseries)
    print(locations)
    np.save(timeseries_out, timeseries)
    np.save(locations_out, locations)


if __name__ == '__main__':
    inpath = TEST_INPATH
    timeseries_out = TEST_TIMEOUT
    locations_out = TEST_LOCOUT
    
    main(inpath, timeseries_out, locations_out)
