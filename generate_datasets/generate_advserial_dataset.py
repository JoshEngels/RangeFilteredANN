# %%
import numpy as np
from tqdm import tqdm
from pathlib import Path


def generate_gaussian_mixture(
    num_clusters, points_per_cluster, d, intra_cluster_var=0.01, inter_cluster_var=1
):
    data = []
    filter_values = []
    means = []
    for i in range(num_clusters):
        mean = np.random.multivariate_normal(np.zeros(d), np.eye(d) * inter_cluster_var)
        points = np.random.multivariate_normal(
            mean, np.eye(d) * intra_cluster_var, points_per_cluster
        )
        data.append(points)
        filter_values.append(i - 0.5 + np.random.uniform(size=points_per_cluster))
        means.append(mean)
    return np.concatenate(data), np.concatenate(filter_values), np.vstack(means)


N = 1000000
NUM_CLUSTERS = 100
DIM = 100
INTRA_CLUSTER_VAR = 0.01
INTER_CLUSTER_VAR = 1
TOP_K = 100
OUTPUT_DIR = Path("/data/parap/storage/jae/new_filtered_ann_datasets/")

data, filters, means = generate_gaussian_mixture(
    num_clusters=NUM_CLUSTERS,
    points_per_cluster=N // NUM_CLUSTERS,
    d=DIM,
    intra_cluster_var=INTRA_CLUSTER_VAR,
    inter_cluster_var=INTER_CLUSTER_VAR,
)


# %%

queries = []
query_filters = []

for query_cluster in tqdm(range(NUM_CLUSTERS)):
    for gt_cluster in range(NUM_CLUSTERS):
        if query_cluster == gt_cluster:
            continue
        query = np.random.multivariate_normal(
            means[query_cluster], np.eye(DIM) * INTRA_CLUSTER_VAR
        )
        queries.append(query)
        query_filters.append((gt_cluster - 0.5, gt_cluster + 0.5))

queries = np.array(queries)
query_filters = np.array(query_filters)

# %%

# Normalize data and queries
data /= np.linalg.norm(data, axis=1, keepdims=True)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

# %%
gts = []


for query_index, query in tqdm(enumerate(queries)):
    dot_products = query @ data.T
    sorted_indices = np.argsort(dot_products)[::-1]
    query_filter = query_filters[query_index]
    query_gt = []
    for data_index in sorted_indices:
        if (
            filters[data_index] >= query_filter[0]
            and filters[data_index] <= query_filter[1]
        ):
            query_gt.append(data_index)
        if len(query_gt) == TOP_K:
            break
    gts.append(query_gt)

# %%

gts = np.vstack(gts)


# %%

np.save(OUTPUT_DIR / f"adversarial-{DIM}-angular.npy", data)
np.save(OUTPUT_DIR / f"adversarial-{DIM}-angular_queries.npy", queries)
np.save(OUTPUT_DIR / f"adversarial-{DIM}-angular_filter-values.npy", filters)
np.save(OUTPUT_DIR / f"adversarial-{DIM}-angular_queries_gt.npy", gts)
np.save(OUTPUT_DIR / f"adversarial-{DIM}-angular_queries_ranges.npy", query_filters)
# %%
