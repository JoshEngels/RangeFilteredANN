
import numpy as np
import os
EXPERIMENT_FILTER_WIDTHS = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]

dataset_folder = "/data/parap/storage/jae/filtered_ann_datasets"

for dataset_name in ["glove-100-angular", "deep-image-96-angular", "sift-128-euclidean", "redcaps-512-angular"]:
    
    data = np.load(os.path.join(dataset_folder, f"{dataset_name}.npy"))
    queries = np.load(os.path.join(dataset_folder, f"{dataset_name}_queries.npy"))
    filter_values = np.load(os.path.join(dataset_folder, f"{dataset_name}_filter-values.npy"))

    for filter_width in EXPERIMENT_FILTER_WIDTHS:
        query_filter_ranges = np.load(os.path.join(dataset_folder, f"{dataset_name}_queries_{filter_width}_ranges.npy"))
        query_gt = np.load(os.path.join(dataset_folder, f"{dataset_name}_queries_{filter_width}_gt.npy"))

        