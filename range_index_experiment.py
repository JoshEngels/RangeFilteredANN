from prototype import create_range_index
from utils import parse_ann_benchmarks_hdf5
import numpy as np
import time
from tqdm import tqdm


filter_path = "/data/scratch/jae/ann_benchmarks_datasets/glove-100-angular_filters.npy"
data_path = "/data/scratch/jae/ann_benchmarks_datasets/glove-100-angular.hdf5"

index = create_range_index(data_path, filter_path)

queries = parse_ann_benchmarks_hdf5(data_path)[1]

queries = queries / np.linalg.norm(queries, axis=-1)[:, np.newaxis]


filter_width = 0.2
our_times = []
brute_force_times = []
our_recalls = []
top_k = 10
query_complexity = 100

for q in tqdm(queries[:1000]):
    random_filter_start = np.random.uniform(0, 1 - filter_width)
    filter_range = (random_filter_start, random_filter_start + filter_width)

    start = time.time()
    result = index.query(
        q, top_k=top_k, query_complexity=query_complexity, filter_range=filter_range
    )
    our_times.append(time.time() - start)

    start = time.time()
    gt = index.naive_query(q, top_k=top_k, filter_range=filter_range)
    brute_force_times.append(time.time() - start)

    recall = len([x for x in gt[0] if x in result[0]])
    our_recalls.append(recall / top_k)

print(np.average(our_recalls))
print(np.average(our_times))
print(np.average(brute_force_times))
