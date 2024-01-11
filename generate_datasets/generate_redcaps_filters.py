from pathlib import Path
from filter_generation_utils import generate_filters
import numpy as np

output_dir = Path("/data/parap/storage/jae/filtered_ann_datasets/")

data = np.load("/data/parap/storage/jae/filtered_ann_datasets/redcaps-512-angular.npy")
queries = np.load(
    "/data/parap/storage/jae/filtered_ann_datasets/redcaps-512-angular_queries.npy"
)
filter_values = np.load(
    "/data/parap/storage/jae/filtered_ann_datasets/redcaps-512-angular_filter-values.npy"
)

generate_filters(output_dir, True, "redcaps-512-angular", data, queries, filter_values)
