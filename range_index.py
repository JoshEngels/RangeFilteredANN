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
    def query(self, query, top_k, query_complexity, filter_range, postfilter_doubles):
        start = time.time()

        if self.distance_metric == "mips":
            query /= np.sum(query**2)

        inclusive_start = self.first_greater_than(filter_range[0])
        exclusive_end = self.first_greater_than_or_equal_to(filter_range[1])
        # print(inclusive_start, exclusive_end)

        middle_index = None
        middle_range = None
        left_index = None
        right_index = None
        for current_pow in range(self.max_pow, self.low_pow - 1, -1):
            current_bucket_size = 2**current_pow

            if middle_index is None:
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
                        middle_index = (current_pow, bucket_start)
                        middle_range = (bucket_start, bucket_start + current_bucket_size)

            if middle_index is None:
                continue

            if left_index is None:
                left_space = middle_range[0] - inclusive_start
                if current_bucket_size <= left_space * 2:
                    start_left = ((middle_range[0] - 1) // current_bucket_size) * current_bucket_size
                    left_index = (current_pow, start_left) 

            if right_index is None:
                right_space = exclusive_end - middle_range[0]
                if current_bucket_size <= right_space * 2:
                    start_right = ((middle_range[1]) // current_bucket_size) * current_bucket_size
                    right_index = (current_pow, start_right)
                    
        if middle_index is None:
            return self.prefilter_query(query, top_k, filter_range)


        identifiers = []
        distances = []
        middle_search_result = self.indices[middle_index].search(
            query=query, complexity=query_complexity, k_neighbors=top_k
        )
        identifiers += [i + middle_index[1] for i in middle_search_result.identifiers]
        distances += list(middle_search_result.distances)


        # gt_middle_search_result = self.prefilter_query(query, top_k, [self.filter_values[middle_range[0]], self.filter_values[middle_range[1]]])
        # identifiers += list(gt_middle_search_result[0])
        # distances += list(gt_middle_search_result[1])
        # print(gt_middle_search_result, (identifiers, distances))

        if middle_range[0] > 0:
            left_filter_range = [filter_range[0], self.filter_values[middle_range[0]]]

            if left_index is None:
                left_search_result = self.prefilter_query(query, top_k, left_filter_range)
            else:
                left_search_result = self.postfilter_query(query, top_k, left_filter_range, extra_doubles=postfilter_doubles, index_key=left_index)
            
            identifiers += list(left_search_result[0])
            distances += list(left_search_result[1])

        if middle_range[1] < len(self.data):
            right_filter_range = [self.filter_values[middle_range[1]], filter_range[1]]

            if right_index is None:
                right_search_result = self.prefilter_query(query, top_k, right_filter_range)
            else:
                right_search_result = self.postfilter_query(query, top_k, right_filter_range, extra_doubles=postfilter_doubles, index_key=right_index)
            # print(right_search_result)
            # print(list(self.prefilter_query(query, top_k, right_filter_range))[0], list(self.prefilter_query(query, top_k, right_filter_range))[1])
            # intersection_length = len(set(list(right_search_result_1[0]) + list(right_search_result_2[0])))
            # print((inclusive_start, exclusive_end), intersection_length, right_index[1], right_index[1] + 2 ** right_index[0], middle_index[0])
            # if intersection_length > 10:
            #     print(right_search_result_1[1], right_search_result_2[1])
            # print(top_k, right_filter_range, right_index, len(set(list(right_search_result_1[0]) + list(right_search_result_2[0]))))

            identifiers += list(right_search_result[0])
            distances += list(right_search_result[1])
        
        identifiers = np.array(identifiers)
        distances = np.array(distances)

        assert len(identifiers) == len(distances)

        if self.distance_metric == "mips":
            top_indices = np.argpartition(distances, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(distances[top_indices])[::-1]]
        elif self.distance_metric == "l2":
            top_indices = np.argpartition(distances, top_k)[:top_k]
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
            squared_distances = np.sum((self.data[inclusive_start:exclusive_end] - query) ** 2, axis=-1)
            distances = list(squared_distances)
        else:
            raise ValueError("Unknown distance metric")

        identifiers = np.array(identifiers)
        distances = np.array(distances)

        assert len(identifiers) == len(distances)

        if self.distance_metric == "mips":
            top_indices = np.argpartition(distances, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(distances[top_indices])[::-1]]
        elif self.distance_metric == "l2":

            if len(distances) > top_k:
                top_indices = np.argpartition(distances, top_k)
                top_indices = top_indices[:top_k]
            else:
                top_indices = np.arange(len(distances))

            top_indices = top_indices[np.argsort(distances[top_indices])] 
        else:
            raise ValueError("Unknown distance metric")
        
        return identifiers[top_indices], distances[top_indices]

    def postfilter_query(self, query, top_k, filter_range, extra_doubles, index_key="full"):
        if self.distance_metric == "mips":
            query /= np.sum(query**2)

        current_complexity = 2 * top_k
        while True:
            # TODO: This is a (neccesary) hacky heuristic, todo find a better one
            if current_complexity * pow(2, extra_doubles) > 10 * np.sqrt(
                len(self.data) if index_key == "full" else 2**index_key[0]
            ):
                return self.prefilter_query(query, top_k, filter_range)

            result = self.indices[index_key].search(
                query, complexity=current_complexity, k_neighbors=current_complexity // 2
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
