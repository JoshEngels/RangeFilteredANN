import numpy as np

EXPERIMENT_FILTER_WIDTHS = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]


def generate_random_query_filter_ranges(filter_values, target_percentage, num_queries, follow_data_distribution=True):

    filter_values = np.sort(filter_values)

    min_filter_value = np.min(filter_values)
    max_filter_value = np.max(filter_values)
    filter_value_range = max_filter_value - min_filter_value

    if target_percentage == 1:
        return np.array([(min_filter_value - np.random.randint(1, 100), max_filter_value + np.random.randint(1, 100))] * num_queries)

    if follow_data_distribution:
        random_ranges = []
        num_in_filter = int(len(filter_values) * target_percentage)

        for _ in range(num_queries):
            starting_index = np.random.randint(0, len(filter_values) - num_in_filter)
            ending_index = starting_index + num_in_filter
            staring_filter_value, ending_filter_value = filter_values[starting_index], filter_values[ending_index]
            start_jitter = np.random.uniform() * (staring_filter_value - filter_values[starting_index - 1] if starting_index > 0 else 1)
            end_jitter = np.random.uniform() * (filter_values[ending_index + 1] - ending_filter_value if ending_index < len(filter_values) - 1 else 1)
            random_ranges.append((staring_filter_value - start_jitter, ending_filter_value + end_jitter))
            if random_ranges[-1][0] > random_ranges[-1][1]:
                print("UH OH", staring_filter_value, ending_filter_value, start_jitter, end_jitter, random_ranges[-1])
                exit(0)

    else:
        random_ranges = []
        filter_width = target_percentage * filter_value_range
        max_filter_start = max_filter_value - filter_value_range * target_percentage
        for _ in range(num_queries):
            random_filter_start = np.random.uniform(min_filter_value, max_filter_start)
            random_ranges.append((random_filter_start, random_filter_start + filter_width))
    
    return np.array(random_ranges)
    
