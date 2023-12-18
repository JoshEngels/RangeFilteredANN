from range_index import create_range_index
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
our_recalls = []

# prefilter recall is always 1
prefilter_times = []

postfilter_times = []
postfilter_recalls = []

top_k = 10
query_complexity = 100

for q in tqdm(queries[:1000]):
    random_filter_start = np.random.uniform(0, 1 - filter_width)
    filter_range = (random_filter_start, random_filter_start + filter_width)

    start = time.time()
    our_result = index.query(
        q, top_k=top_k, query_complexity=query_complexity, filter_range=filter_range
    )
    our_times.append(time.time() - start)

    start = time.time()
    gt = index.prefilter_query(q, top_k=top_k, filter_range=filter_range)
    prefilter_times.append(time.time() - start)

    start = time.time()
    postfilter_result = index.postfilter_query(
        q, top_k=top_k, filter_range=filter_range, extra_doubles=2
    )
    postfilter_times.append(time.time() - start)

    our_recall = len([x for x in gt[0] if x in our_result[0]]) / len(gt[0])
    our_recalls.append(our_recall)

    postfilter_recall = len([x for x in gt[0] if x in postfilter_result[0]]) / len(
        gt[0]
    )
    postfilter_recalls.append(postfilter_recall)

print(np.average(our_recalls), np.average(our_times))
print(1, np.average(prefilter_times))
print(np.average(postfilter_recalls), np.average(postfilter_times))
