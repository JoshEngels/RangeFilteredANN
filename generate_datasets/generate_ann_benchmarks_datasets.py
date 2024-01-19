import numpy as np
import h5py
import os
import urllib.request
from filter_generation_utils import generate_filters
from pathlib import Path


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

    if "angular" in request_url:
        data = data / np.linalg.norm(data, axis=-1)[:, np.newaxis]
        queries = queries / np.linalg.norm(queries, axis=-1)[:, np.newaxis]

    np.save(output_dir / f"{dataset_friendly_name}.npy", data)
    np.save(output_dir / f"{dataset_friendly_name}_queries.npy", queries)

    filter_values = np.random.uniform(size=len(data))

    np.save(output_dir / f"{dataset_friendly_name}_filter-values.npy", filter_values)

    generate_filters(
        output_dir,
        "angular" in request_url,
        dataset_friendly_name,
        data,
        queries,
        filter_values,
    )


output_dir = Path("/data/parap/storage/jae/new_filtered_ann_datasets/")
create_dataset("sift", output_dir)
create_dataset("glove", output_dir)
create_dataset("deep1b", output_dir)
