import os
import seas.hdf5manager
from seas.ica import unflip_components, normalise_components, apply_dynamic_thresholds
from seas.cnmf import run_cnmf

TEST_DIR="/scratch/user/s4296607/"
TEST_INPATH = (TEST_DIR + 
    "sub-201_ses-01_age-P34_rec-baseline_run-01_scrop-normcorre_comp-014_ica-filtered.hdf5")
TEST_OUTPATH = (TEST_DIR + 
    "sub-201_ses-01_age-P34_rec-baseline_run-01_scrop-normcorre_comp-014_cnmf-initial.hdf5")


def load_data(inpath: str) -> dict:
    # Load pyseas dictionary
    if os.path.exists(inpath):
        f = seas.hdf5manager(inpath)
    else:
         print(FileNotFoundError)

    return f.load()


def cnmf_preprocessing(components: dict) -> dict:
     flip = unflip_components(components)
     components.update(flip)
     #normal = normalise_components(components)
     #components.update(normal)
     thresholds = apply_dynamic_thresholds(components)
     components.update(thresholds)

     return components


def save_data(components: dict, outpath: str) -> None:
    f = seas.hdf5manager(outpath)
    f.save(components)


def main(inpath, outpath) -> None:
    components = load_data(inpath)
    print(f'Running CNMF test using file: {inpath}')
    preprocessed_components = cnmf_preprocessing(components)
    print(f'Preprocessed components look like this:')
    print(preprocessed_components)
    cnmf_components = run_cnmf(preprocessed_components)
    save_data(cnmf_components, outpath)


if __name__ == '__main__':
    inpath = TEST_INPATH
    outpath = TEST_OUTPATH
    
    main(inpath, outpath)
