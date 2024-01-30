import psutil
import os
import numpy as np
import os
import resource
import gc
import wrapper as wp
import csv
import humanize

DATASET_FOLDER = "/data/parap/storage/jae/new_filtered_ann_datasets"
# DATASET_FOLDER = "/ssd1/anndata/range_filters"
DATASETS = ["sift-128-euclidean", "glove-100-angular", "deep-image-96-angular", "redcaps-512-angular", "adversarial-100-angular"]


def initialize_dataset(dataset_name):
    data = np.load(os.path.join(DATASET_FOLDER, f"{dataset_name}.npy"))
    filter_values = np.load(
        os.path.join(DATASET_FOLDER, f"{dataset_name}_filter-values.npy")
    )
    metric = "mips" if "angular" in dataset_name else "Euclidian"
    return data, filter_values, metric


def get_vamana_tree(data, filter_values, metric, alpha, split_factor):
    vamana_tree_constructor = wp.vamana_range_filter_tree_constructor(metric, "float")
    build_params = wp.BuildParams(64, 500, alpha, f"index_cache/{dataset_name}/")

    start_memory = psutil.Process(os.getpid()).memory_info().rss

    vamana_tree = vamana_tree_constructor(
        data,
        filter_values,
        cutoff=1_000,
        split_factor=split_factor,
        build_params=build_params,
    )

    end_memory = psutil.Process(os.getpid()).memory_info().rss

    memory = humanize.naturalsize(end_memory - start_memory)

    return vamana_tree, memory

def get_postfiltering_index(data, filter_values, metric, alpha):
    build_params = wp.BuildParams(
        64, 500, alpha, f"index_cache/{dataset_name}/unsorted-"
    )
    postfilter_constructor = wp.postfilter_vamana_constructor(metric, "float")

    start_memory = psutil.Process(os.getpid()).memory_info().rss

    postfilter_index = postfilter_constructor(data, filter_values, build_params)

    end_memory = psutil.Process(os.getpid()).memory_info().rss

    memory = humanize.naturalsize(end_memory - start_memory)

    return postfilter_index, memory

def get_super(data, filter_values, metric, alpha, split_factor):
    constructor = wp.super_optimized_postfilter_tree_constructor(metric, "float")
    build_params = wp.BuildParams(
        64, 500, alpha, f"index_cache/{dataset_name}-super_opt_postfiltering/"
    )
    
    start_memory = psutil.Process(os.getpid()).memory_info().rss

    super_optimized_postfilter_tree = constructor(
        data,
        filter_values,
        cutoff=1_000,
        split_factor=split_factor,
        shift_factor=0.5,
        build_params=build_params,
    )

    end_memory = psutil.Process(os.getpid()).memory_info().rss

    memory = humanize.naturalsize(end_memory - start_memory)

    return super_optimized_postfilter_tree, memory

def write_to_csv(method, dataset, memory_usage):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, "memory_usage.csv")

    headers = ["method", "dataset", "memory"]
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([method, dataset, memory_usage])




for dataset_name in DATASETS:
    data, filter_values, metric = initialize_dataset(dataset_name)

    _, memory_usage = get_postfiltering_index(
        data, filter_values, metric, 1
    )

    write_to_csv("postfiltering", dataset_name, memory_usage)

    _, memory_usage = get_vamana_tree(
        data, filter_values, metric, 1, 2
    )

    write_to_csv("vamana-tree", dataset_name, memory_usage)

    _, memory_usage = get_super(
        data, filter_values, metric, 1, 2
    )

    write_to_csv("super postfiltering", dataset_name, memory_usage)