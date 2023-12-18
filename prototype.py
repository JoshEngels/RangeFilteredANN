import numpy as np
import diskannpy
import os
from math import log2
import argparse
from pathlib import Path
from utils import parse_ann_benchmarks_hdf5

index_directory = "indices"


def create_diskann_index(name, data, alpha, build_complexity, degree, distance_metric):
    if not os.path.exists(index_directory + "/" + name):
        diskannpy.build_memory_index(
            data,
            alpha=alpha,
            complexity=build_complexity,
            graph_degree=degree,
            distance_metric=distance_metric,
            index_directory=index_directory,
            index_prefix=name,
            num_threads=0,
        )

    return diskannpy.StaticMemoryIndex(
        index_directory=index_directory,
        num_threads=0,
        initial_search_complexity=build_complexity,
        index_prefix=name,
    )


class RangeIndex:
    def __init__(self, data, dataset_name, filter_values, cutoff_pow, distance_metric):
        zipped_data = list(zip(filter_values, data))
        zipped_data.sort()
        self.data = np.array([x for _, x in zipped_data])
        self.filter_values = np.array([y for y, _ in zipped_data])
        self.low_pow = cutoff_pow
        self.max_pow = int(log2(len(data) - 1))

        # For now starting at size cutoff_pow and going up (instead of halfing)
        self.indices = {}
        for current_pow in range(self.low_pow, self.max_pow + 1):
            current_bucket_size = 2**current_pow
            for start in range(0, len(data), current_bucket_size):
                self.indices[(current_pow, start)] = create_diskann_index(
                    name=f"{dataset_name}_{current_pow}_{start}",
                    data=self.data[start : start + current_bucket_size],
                    alpha=1.1,
                    build_complexity=64,
                    degree=32,
                    distance_metric=distance_metric,
                )
        pass

    def first_greater_than(self, filter_value):
        start = 0
        end = len(self.filter_values)
        while start + 1 < end:
            mid = (start + end) // 2
            if self.filter_values[mid] > filter_value:
                end = mid
            else:
                start = mid
        return end

    def first_greater_than_or_equal_to(self, filter_value):
        start = 0
        end = len(self.filter_values)
        while start + 1 < end:
            mid = (start + end) // 2
            if self.filter_values[mid] >= filter_value:
                end = mid
            else:
                start = mid
        return end

    # Filter range is exclusive top and bottom
    def query(self, query, top_k, query_complexity, filter_range):
        inclusive_start = self.first_greater_than(filter_range[0])
        exclusive_end = self.first_greater_than_or_equal_to(filter_range[1])

        last_range = None
        for current_pow in range(self.max_pow, self.low_pow - 1, -1):
            current_bucket_size = 2**current_pow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_filename", help="Path to the HDF5 data file from ANN benchmarks"
    )
    # parser.add_argument('filter_values', help='Path to npy file containing one d array of filter values')

    args = parser.parse_args()

    dataset_name, dimension, distance_metric = Path(args.data_filename).stem.split("-")
    distance_metric = {"angular": "cosine", "euclidean": "l2"}[distance_metric]

    data = parse_ann_benchmarks_hdf5(args.data_filename)[0]

    # filter_values = np.load(args.filter_values)
    filter_values = np.random.uniform(size=len(data))

    range_index = RangeIndex(
        data=data,
        dataset_name=dataset_name,
        filter_values=filter_values,
        cutoff_pow=8,
        distance_metric=distance_metric,
    )
