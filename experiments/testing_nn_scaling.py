# %%

from collections import defaultdict
from range_index import create_range_index
from utils import parse_ann_benchmarks_hdf5
import numpy as np
import time
from tqdm import tqdm


data_dir = "/data/scratch/jae/ann_benchmarks_datasets/"
# data_dir = "/ssd1/anndata/ann-benchmarks/"

dataset_name = "sift-128-euclidean"
filter_path = f"{data_dir}/{dataset_name}_filters.npy"
data_path = f"{data_dir}/{dataset_name}.hdf5"

index = create_range_index(data_path, filter_path)

queries = parse_ann_benchmarks_hdf5(data_path)[1]

if "angular" in dataset_name:
    queries = queries / np.linalg.norm(queries, axis=-1)[:, np.newaxis]

top_k = 10
output_file = f"{dataset_name}_experiment.txt"

# %%

import time

times = {}

for q in tqdm(queries[:1000]):
    for beam_search_pow in range(2, 11):
        for n_pow in range(index.low_pow, index.max_pow + 1):
            key = (beam_search_pow, n_pow)
            if key not in times:
                times[key] = []

            beam_size = 2**beam_search_pow
            current_bucket_size = 2**n_pow
            possible_starts = list(range(0, len(index.data), current_bucket_size))
            random_index = index.indices[n_pow, np.random.choice(possible_starts)]

            start = time.time()
            random_index.search(q, top_k, complexity=beam_size)
            times[key].append(time.time() - start)

# %%

average_times = {k: sum(t) / len(t) for k, t in times.items()}

# %%

for n_pow in range(index.low_pow, index.max_pow + 1):
    for beam_search_pow in range(2, 11):
        key = (beam_search_pow, n_pow)
        print(
            beam_search_pow,
            n_pow,
            average_times[key],
            average_times[key] / average_times[(2, n_pow)],
        )

# %%

for beam_search_pow in range(2, 11):
    for n_pow in range(index.low_pow, index.max_pow + 1):
        key = (beam_search_pow, n_pow)
        print(
            beam_search_pow,
            n_pow,
            average_times[key],
            average_times[key] / average_times[(beam_search_pow, index.low_pow)],
        )


# %%
