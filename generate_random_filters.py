import numpy as np
import argparse
from utils import parse_ann_benchmarks_hdf5
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "data_filename", help="Path to the HDF5 data file from ANN benchmarks"
)

parser.add_argument("output_dir", help="Path to save the generated qu to")

args = parser.parse_args()

data = parse_ann_benchmarks_hdf5(args.data_filename)[0]

filters = np.random.uniform(size=len(data))

original_path = Path(args.data_filename)

filter_filename = original_path.parent / (original_path.stem + "_filters.npy")

np.save(filter_filename, filters)
