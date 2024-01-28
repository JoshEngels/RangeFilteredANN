import argparse
import numpy as np
import os
import resource
import gc
import wrapper as wp
import csv

# DATASET_FOLDER = "/data/parap/storage/jae/new_filtered_ann_datasets"
DATASET_FOLDER = "/ssd1/anndata/range_filters"
DATASETS = ["sift-128-euclidean"]

def initialize_dataset(dataset_name):
    data = np.load(os.path.join(DATASET_FOLDER, f"{dataset_name}.npy"))
    filter_values = np.load(
        os.path.join(DATASET_FOLDER, f"{dataset_name}_filter-values.npy")
    )
    metric = "mips" if "angular" in dataset_name else "Euclidean"
    return data, filter_values, metric

def get_vamana_tree(data, filter_values, metric, alpha, split_factor):
    vamana_tree_constructor = wp.vamana_range_filter_tree_constructor(metric, "float")
    build_params = wp.BuildParams(64, 500, alpha, f"index_cache/{dataset_name}/")
    gc.disable()
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(vamana_tree_constructor)
    vamana_tree = vamana_tree_constructor(
        data, filter_values, cutoff=1_000, split_factor=split_factor, build_params=build_params
    )
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_memory
    gc.enable()
    return vamana_tree, memory

def write_to_csv(method, branching_factor, memory_usage):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, 'vamana_tree_memory_usage.csv')

    headers = ['method', 'branching_factor', 'memory']
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([method, branching_factor, memory_usage])

parser = argparse.ArgumentParser()
parser.add_argument("--vamana_tree_split_factor", type=int, help="The branching factor for the vamana tree methods", required=True)
parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for all graph based methods")
args = parser.parse_args()

dataset_name = DATASETS[0]  # Using the first dataset for simplicity
data, filter_values, metric = initialize_dataset(dataset_name)
vamana_tree, memory_usage = get_vamana_tree(data, filter_values, metric, args.alpha, args.vamana_tree_split_factor)

write_to_csv('vamana-tree', args.vamana_tree_split_factor, memory_usage)
