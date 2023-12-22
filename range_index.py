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
    def query_fenwick(self, query, top_k, query_complexity, filter_range):
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

            if last_range == None:
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
        elif self.distance_metric == "l2":
            distances += list(
                np.sum(
                    (self.data[inclusive_start : last_range[0]] - query) ** 2, axis=-1
                )
            )
            distances += list(
                np.sum((self.data[last_range[1] : exclusive_end] - query) ** 2, axis=-1)
            )
        else:
            raise ValueError("Currently unsupported distance metric in query")

        # print("C", time.time() - start)

        identifiers = np.array(identifiers)
        distances = np.array(distances)

        assert len(identifiers) == len(distances)

        if self.distance_metric == "mips":
            if len(distances) > top_k:
                top_indices = np.argpartition(distances, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(distances))

            top_indices = top_indices[np.argsort(distances[top_indices])[::-1]]

        elif self.distance_metric == "l2":
            if len(distances) > top_k:
                top_indices = np.argpartition(distances, top_k)[:top_k]
            else:
                top_indices = np.arange(len(distances))

            top_indices = top_indices[np.argsort(distances[top_indices])]

        else:
            raise ValueError("Unknown distance metric")

        return identifiers[top_indices], distances[top_indices]

    def query_three_split(
        self, query, top_k, query_complexity, filter_range, extra_side_doubles
    ):
        if self.distance_metric == "mips":
            query /= np.sum(query**2)

        inclusive_start = self.first_greater_than(filter_range[0])
        exclusive_end = self.first_greater_than_or_equal_to(filter_range[1])

        middle_range = None
        indices_to_search = []
        for current_pow in range(self.max_pow, self.low_pow - 1, -1):
            current_bucket_size = 2**current_pow

            if middle_range == None:
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
                        middle_range = [
                            bucket_start,
                            bucket_start + current_bucket_size,
                        ]
                        break

        if middle_range == None:
            return self.prefilter_query(query, top_k, filter_range)

        identifiers = []
        distances = []
        for index_pattern in indices_to_search:
            index = self.indices[index_pattern]
            search_result = index.search(
                query=query, complexity=query_complexity, k_neighbors=top_k
            )
            identifiers += [i + index_pattern[1] for i in search_result.identifiers]
            distances += list(search_result.distances)

        left_space = middle_range[0] - inclusive_start
        right_space = exclusive_end - middle_range[1]

        if left_space > 0:
            left_filter_range = (
                filter_range[0],
                self.filter_values[middle_range[0] - 1],
            )
            search_result = self.postfilter_query(
                query,
                top_k,
                left_filter_range,
                extra_doubles=extra_side_doubles,
                optimize_index_choice=True,
            )
            identifiers += list(search_result[0])
            distances += list(search_result[1])
        if right_space > 0:
            right_filter_range = (
                self.filter_values[middle_range[1]],
                filter_range[1],
            )
            search_result = self.postfilter_query(
                query,
                top_k,
                right_filter_range,
                extra_doubles=extra_side_doubles,
                optimize_index_choice=True,
            )
            identifiers += list(search_result[0])
            distances += list(search_result[1])

        identifiers = np.array(identifiers)
        distances = np.array(distances)

        assert len(identifiers) == len(distances)

        if self.distance_metric == "mips":
            if len(distances) > top_k:
                top_indices = np.argpartition(distances, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(distances))

            top_indices = top_indices[np.argsort(distances[top_indices])[::-1]]

        elif self.distance_metric == "l2":
            if len(distances) > top_k:
                top_indices = np.argpartition(distances, top_k)[:top_k]
            else:
                top_indices = np.arange(len(distances))

            top_indices = top_indices[np.argsort(distances[top_indices])]

        else:
            raise ValueError("Unknown distance metric")

        return identifiers[top_indices], distances[top_indices]

    def prefilter_query(self, query, top_k, filter_range):
        if self.distance_metric == "mips":
            query /= np.sum(query**2)

        inclusive_start = self.first_greater_than(filter_range[0])
        exclusive_end = self.first_greater_than_or_equal_to(filter_range[1])

        if exclusive_end - inclusive_start == 0:
            return [], []

        identifiers = list(range(inclusive_start, exclusive_end))
        if self.distance_metric == "mips":
            distances = list(self.data[inclusive_start:exclusive_end] @ query)
        elif self.distance_metric == "l2":
            squared_distances = np.sum(
                (self.data[inclusive_start:exclusive_end] - query) ** 2, axis=-1
            )
            distances = list(squared_distances)
        else:
            raise ValueError("Unknown distance metric")

        identifiers = np.array(identifiers)
        distances = np.array(distances)

        assert len(identifiers) == len(distances)

        if self.distance_metric == "mips":
            if len(distances) > top_k:
                top_indices = np.argpartition(distances, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(distances))

            top_indices = top_indices[np.argsort(distances[top_indices])[::-1]]

        elif self.distance_metric == "l2":
            if len(distances) > top_k:
                top_indices = np.argpartition(distances, top_k)[:top_k]
            else:
                top_indices = np.arange(len(distances))

            top_indices = top_indices[np.argsort(distances[top_indices])]

        else:
            raise ValueError("Unknown distance metric")

        return identifiers[top_indices], distances[top_indices]

    def postfilter_query(
        self,
        query,
        top_k,
        filter_range,
        extra_doubles,
        index_key="full",
        optimize_index_choice=False,
    ):
        if self.distance_metric == "mips":
            query /= np.sum(query**2)

        inclusive_start = self.first_greater_than(filter_range[0])
        exclusive_end = self.first_greater_than_or_equal_to(filter_range[1])

        if optimize_index_choice:
            for power in range(self.low_pow, self.max_pow + 1):
                bucket_size = 2**power
                start_bucket = inclusive_start // bucket_size
                end_bucket = (exclusive_end - 1) // bucket_size
                if start_bucket == end_bucket:
                    index_key = (power, start_bucket * bucket_size)
                    break

        current_complexity = 2 * top_k
        while True:
            # TODO: This is a (neccesary) hacky heuristic, todo find a better one
            if current_complexity * pow(2, extra_doubles) > 10 * np.sqrt(
                len(self.data) if index_key == "full" else 2 ** index_key[0]
            ):
                # print(optimize_index_choice, current_complexity * pow(2, extra_doubles), np.sqrt(len(self.data) if index_key == "full" else 2**index_key[0]))
                return self.prefilter_query(query, top_k, filter_range)

            result = self.indices[index_key].search(
                query,
                complexity=current_complexity,
                k_neighbors=current_complexity // 2,
            )
            index_offset = 0 if index_key == "full" else index_key[1]
            filtered_identifiers = []
            filtered_distances = []
            for identifier, distance in zip(result.identifiers, result.distances):
                identifier += index_offset
                if (
                    self.filter_values[identifier] >= filter_range[0]
                    and self.filter_values[identifier] < filter_range[1]
                ):
                    filtered_identifiers.append(identifier)
                    filtered_distances.append(distance)

            if len(filtered_identifiers) >= top_k:
                if extra_doubles == 0:
                    # print(optimize_index_choice, current_complexity)
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
