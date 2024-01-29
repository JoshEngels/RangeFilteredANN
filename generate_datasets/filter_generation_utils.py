import numpy as np
from tqdm import tqdm
import os

EXPERIMENT_FILTER_POWERS = range(-16, 1)
TOP_K = 10


def generate_random_query_filter_ranges(
    filter_values, target_percentage, num_queries, follow_data_distribution=True
):
    filter_values = np.sort(filter_values)

    min_filter_value = np.min(filter_values)
    max_filter_value = np.max(filter_values)
    filter_value_range = max_filter_value - min_filter_value

    if target_percentage == 1:
        return np.array(
            [
                (
                    min_filter_value - np.random.randint(1, 100),
                    max_filter_value + np.random.randint(1, 100),
                )
            ]
            * num_queries
        )

    if follow_data_distribution:
        random_ranges = []
        num_in_filter = int(len(filter_values) * target_percentage)

        for _ in range(num_queries):
            starting_index = np.random.randint(0, len(filter_values) - num_in_filter)
            ending_index = starting_index + num_in_filter
            staring_filter_value, ending_filter_value = (
                filter_values[starting_index],
                filter_values[ending_index],
            )
            start_jitter = np.random.uniform() * (
                staring_filter_value - filter_values[starting_index - 1]
                if starting_index > 0
                else 1
            )
            end_jitter = np.random.uniform() * (
                filter_values[ending_index + 1] - ending_filter_value
                if ending_index < len(filter_values) - 1
                else 1
            )
            random_ranges.append(
                (staring_filter_value - start_jitter, ending_filter_value + end_jitter)
            )
            if random_ranges[-1][0] > random_ranges[-1][1]:
                print(
                    "UH OH",
                    staring_filter_value,
                    ending_filter_value,
                    start_jitter,
                    end_jitter,
                    random_ranges[-1],
                )
                exit(0)

    else:
        random_ranges = []
        filter_width = target_percentage * filter_value_range
        max_filter_start = max_filter_value - filter_value_range * target_percentage
        for _ in range(num_queries):
            random_filter_start = np.random.uniform(min_filter_value, max_filter_start)
            random_ranges.append(
                (random_filter_start, random_filter_start + filter_width)
            )

    return np.array(random_ranges)


# def generate_filters(
#     output_dir, is_angular, dataset_friendly_name, data, queries, filter_values
# ):
#     all_filter_ranges = []
#     for filter_width_power in EXPERIMENT_FILTER_POWERS:
#         if (
#             output_dir
#             / f"{dataset_friendly_name}_queries_2pow{filter_width_power}_ranges.npy"
#         ).exists():
#             all_filter_ranges.append(
#                 np.load(
#                     output_dir
#                     / f"{dataset_friendly_name}_queries_2pow{filter_width_power}_ranges.npy"
#                 )
#             )
#             continue

#         filter_width = 2**filter_width_power
#         filter_ranges = generate_random_query_filter_ranges(
#             filter_values=filter_values,
#             target_percentage=filter_width,
#             num_queries=len(queries),
#         )
#         all_filter_ranges.append(filter_ranges)
#         np.save(
#             output_dir
#             / f"{dataset_friendly_name}_queries_2pow{filter_width_power}_ranges.npy",
#             filter_ranges,
#         )

#     all_gts = [[] for _ in range(len(EXPERIMENT_FILTER_POWERS))]
#     for query_index, query in enumerate(tqdm(queries)):
#         if is_angular:
#             dot_products = query @ data.T
#             sorted_indices = np.argsort(dot_products)[::-1]
#         else:
#             distances = np.linalg.norm(query - data, axis=-1)
#             sorted_indices = np.argsort(distances)

#         for experiment_index in range(len(EXPERIMENT_FILTER_POWERS)):
#             query_filter = all_filter_ranges[experiment_index][query_index]
#             query_gt = []
#             for data_index in sorted_indices:
#                 if (
#                     filter_values[data_index] >= query_filter[0]
#                     and filter_values[data_index] <= query_filter[1]
#                 ):
#                     query_gt.append(data_index)
#                 if len(query_gt) == TOP_K:
#                     break

#             assert len(query_gt) == TOP_K

#             all_gts[experiment_index].append(query_gt)

#     for i in range(len(EXPERIMENT_FILTER_POWERS)):
#         all_gts[i] = np.array(all_gts[i])
#         filter_width_power = EXPERIMENT_FILTER_POWERS[i]
#         np.save(
#             output_dir
#             / f"{dataset_friendly_name}_queries_2pow{filter_width_power}_gt.npy",
#             all_gts[i],
#         )


def compute_ground_truths(
    data, queries, filter_ranges, filter_values, top_k, is_angular
):
    num_experiments = len(filter_ranges)
    all_gts = [
        np.empty((len(queries), top_k), dtype=int) for _ in range(num_experiments)
    ]

    for query_index, query in enumerate(tqdm(queries)):
        if is_angular:
            dot_products = query @ data.T
            sorted_indices = np.argsort(dot_products)[::-1]
        else:
            distances = np.linalg.norm(data - query, axis=1)
            sorted_indices = np.argsort(distances)

        for experiment_index in range(num_experiments):
            query_filter = filter_ranges[experiment_index][query_index]
            filtered_indices = sorted_indices[
                (filter_values[sorted_indices] >= query_filter[0])
                & (filter_values[sorted_indices] <= query_filter[1])
            ]
            all_gts[experiment_index][query_index, :] = filtered_indices[:top_k]

            assert len(all_gts[experiment_index][query_index]) == top_k

    return all_gts


def generate_filters(
    output_dir, is_angular, dataset_friendly_name, data, queries, filter_values
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_filter_ranges = []
    for filter_width_power in EXPERIMENT_FILTER_POWERS:
        range_file = os.path.join(
            output_dir,
            f"{dataset_friendly_name}_queries_2pow{filter_width_power}_ranges.npy",
        )
        if os.path.exists(range_file):
            all_filter_ranges.append(np.load(range_file))
        else:
            filter_width = 2**filter_width_power
            filter_ranges = generate_random_query_filter_ranges(
                filter_values=filter_values,
                target_percentage=filter_width,
                num_queries=len(queries),
            )
            all_filter_ranges.append(filter_ranges)
            np.save(range_file, filter_ranges)

    all_gts = compute_ground_truths(
        data, queries, all_filter_ranges, filter_values, TOP_K, is_angular
    )

    for i, gts in enumerate(all_gts):
        filter_width_power = EXPERIMENT_FILTER_POWERS[i]
        gt_file = os.path.join(
            output_dir,
            f"{dataset_friendly_name}_queries_2pow{filter_width_power}_gt.npy",
        )
        np.save(gt_file, gts)
