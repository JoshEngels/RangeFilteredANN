import numpy as np
import diskannpy
import os
from math import log2
import argparse
from pathlib import Path
from utils import parse_ann_benchmarks_hdf5
import time

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
        if distance_metric == "mips":
            data = data / np.linalg.norm(data, axis=-1)[:, np.newaxis]

        zipped_data = list(zip(filter_values, data))
        zipped_data.sort()
        self.data = np.array([x for _, x in zipped_data])
        self.filter_values = np.array([y for y, _ in zipped_data])
        self.low_pow = cutoff_pow
        self.max_pow = int(log2(len(data) - 1))
        self.distance_metric = distance_metric

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
        self.indices["full"] = create_diskann_index(
            name=f"{dataset_name}_full",
            data=self.data,
            alpha=1.1,
            build_complexity=64,
            degree=32,
            distance_metric=distance_metric,
        )

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
        start = time.time()

        if self.distance_metric == "mips":
            query /= np.sum(query**2)

        inclusive_start = self.first_greater_than(filter_range[0])
        exclusive_end = self.first_greater_than_or_equal_to(filter_range[1])

        last_range = None
        indices_to_search = []
        for current_pow in range(self.max_pow, self.low_pow - 1, -1):
            current_bucket_size = 2**current_pow

            if last_range == None:
                min_possible_bucket_index_inclusive = (
                    inclusive_start
                ) // current_bucket_size
                max_possible_bucket_index_exclusive = (
                    exclusive_end
                ) // current_bucket_size
                for possible_bucket_index in range(
                    min_possible_bucket_index_inclusive,
                    max_possible_bucket_index_exclusive,
                ):
                    bucket_start = possible_bucket_index * current_bucket_size
                    bucket_end = bucket_start + current_bucket_size
                    if bucket_start >= inclusive_start and bucket_end <= exclusive_end:
                        indices_to_search.append((current_pow, bucket_start))
                        last_range = [bucket_start, bucket_start + current_bucket_size]
                continue

            left_space = last_range[0] - inclusive_start
            right_space = exclusive_end - last_range[1]
            if left_space > current_bucket_size:
                last_range[0] -= current_bucket_size
                indices_to_search.append((current_pow, last_range[0]))
            if right_space > current_bucket_size:
                indices_to_search.append((current_pow, last_range[1]))
                last_range[1] += current_bucket_size

        if len(indices_to_search) == 0:
            return self.prefilter_query(query, top_k, filter_range)

        # print("A", time.time() - start)

        identifiers = []
        distances = []
        for index_pattern in indices_to_search:
            # indexing_start = time.time()
            index = self.indices[index_pattern]
            search_result = index.search(
                query=query, complexity=query_complexity, k_neighbors=top_k
            )
            identifiers += [i + index_pattern[1] for i in search_result.identifiers]
            distances += list(search_result.distances)

        #     print("B", index_pattern, time.time() - indexing_start)
        # print("B_tot", time.time() - start)

        if left_space > 0:
            identifiers += list(range(inclusive_start, last_range[0]))
        if right_space > 0:
            identifiers += list(range(last_range[1], exclusive_end))
        if self.distance_metric == "mips":
            distances += list(self.data[inclusive_start : last_range[0]] @ query)
            distances += list(self.data[last_range[1] : exclusive_end] @ query)
        else:
            raise ValueError("Currently unsupported distance metric in query")

        # print("C", time.time() - start)

        identifiers = np.array(identifiers)
        distances = np.array(distances)

        assert len(identifiers) == len(distances)

        top_indices = np.argpartition(distances, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(distances[top_indices])[::-1]]
        return identifiers[top_indices], distances[top_indices]

    def prefilter_query(self, query, top_k, filter_range):
        if self.distance_metric == "mips":
            query /= np.sum(query**2)

        inclusive_start = self.first_greater_than(filter_range[0])
        exclusive_end = self.first_greater_than_or_equal_to(filter_range[1])

        identifiers = list(range(inclusive_start, exclusive_end))
        if self.distance_metric == "mips":
            distances = list(self.data[inclusive_start:exclusive_end] @ query)
        else:
            raise ValueError("Currently unsupported distance metric in query")

        identifiers = np.array(identifiers)
        distances = np.array(distances)

        assert len(identifiers) == len(distances)

        top_indices = np.argpartition(distances, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(distances[top_indices])[::-1]]
        return identifiers[top_indices], distances[top_indices]

    def postfilter_query(self, query, top_k, filter_range, extra_doubles):
        if self.distance_metric == "mips":
            query /= np.sum(query**2)

        current_complexity = top_k
        while True:
            # TODO: This is a (neccesary) hacky heuristic, todo find a better one
            if current_complexity * pow(2, extra_doubles) > 10 * np.sqrt(
                len(self.data)
            ):
                return self.prefilter_query(query, top_k, filter_range)

            result = self.indices["full"].search(
                query, complexity=current_complexity, k_neighbors=current_complexity
            )
            filtered_identifiers = []
            filtered_distances = []
            for identifier, distance in zip(result.identifiers, result.distances):
                if (
                    self.filter_values[identifier] >= filter_range[0]
                    and self.filter_values[identifier] < filter_range[1]
                ):
                    filtered_identifiers.append(identifier)
                    filtered_distances.append(distance)
            if len(filtered_identifiers) >= top_k:
                if extra_doubles == 0:
                    return filtered_identifiers[:top_k], filtered_distances[:top_k]
                extra_doubles -= 1
            current_complexity *= 2


def create_range_index(data_filename, filters_filename):
    dataset_name, _, distance_metric = Path(data_filename).stem.split("-")
    distance_metric = {"angular": "mips", "euclidean": "l2"}[distance_metric]

    data = parse_ann_benchmarks_hdf5(data_filename)[0]

    filter_values = np.load(filters_filename)

    range_index = RangeIndex(
        data=data,
        dataset_name=dataset_name,
        filter_values=filter_values,
        cutoff_pow=8,
        distance_metric=distance_metric,
    )

    return range_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_filename", help="Path to the HDF5 data file from ANN benchmarks"
    )
    parser.add_argument(
        "filters_filename",
        help="Path to npy file containing one d array of filter values",
    )

    args = parser.parse_args()

    index = create_range_index(args.data_filename, args.filters_filename)
