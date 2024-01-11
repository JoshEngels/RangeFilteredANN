import numpy as np
import h5py
import os
import urllib.request
from filter_generation_utils import (
    EXPERIMENT_FILTER_WIDTHS,
    generate_random_query_filter_ranges,
)
from pathlib import Path
from tqdm import tqdm

TOP_K = 10

def parse_ann_benchmarks_hdf5(data_path):
    with h5py.File(data_path, "r") as file:
        gt_neighbors = np.array(file["neighbors"])
        queries = np.array(file["test"])
        data = np.array(file["train"])

        return data, queries, gt_neighbors


download_urls = {
    "deep1b": "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
    "sift": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove": "http://ann-benchmarks.com/glove-100-angular.hdf5",
}

work_dir = Path("temp")


def create_dataset(dataset_name, output_dir):
    os.makedirs(work_dir, exist_ok=True)

    request_url = download_urls[dataset_name]

    file_path = work_dir / Path(request_url).name

    if not file_path.exists():
        urllib.request.urlretrieve(request_url, file_path)

    dataset_friendly_name = file_path.stem

    data, queries, gts = parse_ann_benchmarks_hdf5(file_path)

    if 'angular' in request_url:
        data = data / np.linalg.norm(data, axis=-1)[:, np.newaxis]
        queries = queries / np.linalg.norm(queries, axis=-1)[:, np.newaxis]

    np.save(output_dir / f"{dataset_friendly_name}.npy", data)
    np.save(output_dir / f"{dataset_friendly_name}_queries.npy", queries)

    filter_values = np.random.uniform(size=len(data))

    np.save(output_dir / f"{dataset_friendly_name}_filter_values.npy", filter_values)

    all_filter_ranges = []
    for filter_width in EXPERIMENT_FILTER_WIDTHS:
        filter_ranges = generate_random_query_filter_ranges(
            filter_values=filter_values,
            target_percentage=filter_width,
            num_queries=len(queries),
        )
        all_filter_ranges.append(filter_ranges)
        np.save(
            output_dir / f"{dataset_friendly_name}_queries_{filter_width}_ranges.npy",
            filter_ranges,
        )

    all_gts = [[] for _ in range(len(EXPERIMENT_FILTER_WIDTHS))]
    for query_index, query in tqdm(enumerate(queries)):

        if 'angular' in request_url:
            dot_products = query @ data.T
            sorted_indices = np.argsort(dot_products)[::-1]
        else:
            distances = np.linalg.norm(query - data, axis=-1)
            sorted_indices = np.argsort(distances)

        for experiment_index in range(len(EXPERIMENT_FILTER_WIDTHS)):
            query_filter = all_filter_ranges[experiment_index][query_index]
            query_gt = []
            for data_index in sorted_indices:
                if filter_values[data_index] >= query_filter[0] and filter_values[data_index] <= query_filter[1]:
                    query_gt.append(data_index)
                if len(query_gt) == TOP_K:
                    break

            assert(len(query_gt) == TOP_K)
            
            all_gts[experiment_index].append(query_gt)

    for i in range(len(EXPERIMENT_FILTER_WIDTHS)):
        all_gts[i] = np.array(all_gts[i])
        np.save(
            output_dir / f"{dataset_friendly_name}_queries_{EXPERIMENT_FILTER_WIDTHS[i]}_gt.npy",
            all_gts[i],
        )


output_dir = Path("/data/parap/storage/jae/filtered_ann_datasets/")
create_dataset("sift", output_dir)
create_dataset("glove", output_dir)
create_dataset("deep1b", output_dir)
