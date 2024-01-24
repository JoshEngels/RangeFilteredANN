import time
import os
import sys
import itertools
from math import sqrt
from multiprocessing import Pool, Manager

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from tqdm import tqdm

THREADS = 16  # Adjust this to the desired number of parallel processes
print("THREADS", THREADS)
print(
    "If you want to change the number of process used, OMP_NUM_THREADS and limits:cpus in docker-compose.yml should also both be changed."
)
# Ensure index_cache/postfiler_vamana exists
os.makedirs("results", exist_ok=True)

TOP_K = 10
# Only use a subset of data and query for testing
DATA_SUBSET = None
QUERY_SUBSET = None
DATASETS = [
    "glove-100-angular",
    "deep-image-96-angular",
    "sift-128-euclidean",
    "redcaps-512-angular",
    "adversarial-100-angular",
]
EXPERIMENT_FILTER_WIDTHS = [f"2pow{i}" for i in range(-16, 1)]
# used for ef for HNSW query param
BEAM_SIZES = [10, 20, 40, 80, 160, 320]
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128]
REORDER_KS = [TOP_K * i for i in [1, 2, 4, 8, 16]]
INDEX_TYPES = ["HNSW", "IVF_PQ", "IVF_SQ8", "SCANN", "IVF_FLAT"]

SIZE_OF_QUERY_POOL = 1000

dataset_folder = "/data/parap/storage/jae/new_filtered_ann_datasets"


def compute_recall(gt_neighbors, results, top_k):
    recall = 0
    for i in range(len(gt_neighbors)):  # for each query
        gt = set(gt_neighbors[i])
        res = set(results[i][:top_k])
        recall += len(gt.intersection(res)) / len(gt)
    return recall / len(gt_neighbors)  # average recall per query


def get_index_params(index_type, n, metric, dim):
    m = 10 if dim == 100 else 8  #  Because we need dim mod m == 0
    assert dim % m == 0
    params = {
        "IVF_FLAT": {"nlist": int(sqrt(n))},
        "IVF_SQ8": {"nlist": int(sqrt(n))},
        "IVF_PQ": {"nlist": int(sqrt(n)), "m": m},
        "SCANN": {"nlist": int(sqrt(n))},
        "HNSW": {"M": 64, "efConstruction": 500},
    }
    if index_type not in params:
        raise Exception(f"Invalid index type {index_type}")

    index = {
        "index_type": index_type,
        "metric_type": metric,
        "params": params[index_type],
    }
    return index


def get_query_params(index_type, n):
    params = {
        "IVF_FLAT": [{"nprobe": i} for i in NPROBES],
        "IVF_SQ8": [{"nprobe": i} for i in NPROBES],
        "IVF_PQ": [{"nprobe": i} for i in NPROBES],
        "SCANN": [
            {
                "nprobe": i,
                "reorder_k": j,
            }
            for i, j in itertools.product(NPROBES, REORDER_KS)
        ],
        "HNSW": [{"ef": i} for i in BEAM_SIZES],
        "DISKANN": [{"search_list": i} for i in BEAM_SIZES],
    }
    if index_type not in params:
        raise Exception("Invalid index type")
    return params[index_type]


# connect to milvus
connections.connect("default", host="localhost", port="19530")
print("Connected to Milvus.")


# remove all existing data
collections = utility.list_collections()
for collection in collections:
    utility.drop_collection(collection)


for dataset_name in DATASETS:
    dim = int(dataset_name.split("-")[1])

    output_file = f"results/{dataset_name}_milvus_results.csv"

    # only write header if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "a") as f:
            f.write("filter_width,method,recall,average_time,qps,threads\n")

    print("Loading data.")
    data = np.load(os.path.join(dataset_folder, f"{dataset_name}.npy"))
    if DATA_SUBSET is not None:
        data = data[:DATA_SUBSET]
    num_entities, dim = data.shape
    print("data.shape, ", data.shape)

    print("Loading queries.")
    queries = np.load(os.path.join(dataset_folder, f"{dataset_name}_queries.npy"))
    if QUERY_SUBSET is not None:
        queries = queries[:QUERY_SUBSET]
    print("queries.shape, ", queries.shape)

    print("Loading filter values.")
    filter_values = np.load(
        os.path.join(dataset_folder, f"{dataset_name}_filter-values.npy")
    )
    if DATA_SUBSET is not None:
        filter_values = filter_values[:DATA_SUBSET]
    print("filter.shape, ", filter_values.shape)

    metric = "IP" if "angular" in dataset_name else "L2"

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
            max_length=100,
        ),
        FieldSchema(name="priority", dtype=DataType.FLOAT),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, "points with id and priority")
    points = Collection("points", schema, consistency_level="Strong")

    print("Inserting points to database.")
    entities = [
        [i for i in range(num_entities)],
        filter_values.tolist(),
        data,
    ]
    insert_result = points.insert(entities)
    points.flush()
    print("insert_result:\n", insert_result)

    # use different indices
    for index in INDEX_TYPES:
        print("Building index: ", index, flush=True)
        index_params = get_index_params(index, num_entities, metric, dim)
        search_params_list = get_query_params(index, num_entities)
        print("index_params: ", index_params)
        formatted_index_params = "_".join(
            [f"{key}_{value}" for key, value in index_params["params"].items()]
        )

        start_time = time.time()
        points.create_index("embeddings", index_params)
        points.load()
        end_time = time.time()
        print("build index time = {:.4f}s".format(end_time - start_time), flush=True)
        utility.index_building_progress("points")

        dataset_experiment_filter_widths = (
            [""]
            if dataset_name == "adversarial-100-angular"
            else EXPERIMENT_FILTER_WIDTHS
        )
        for filter_width in dataset_experiment_filter_widths:
            filter_width = "_" if filter_width == "" else f"_{filter_width}_"
            print("==filter width: 2pow", filter_width)
            run_results = []
            query_filter_ranges = np.load(
                os.path.join(
                    dataset_folder,
                    f"{dataset_name}_queries{filter_width}ranges.npy",
                )
            )
            query_gt = np.load(
                os.path.join(
                    dataset_folder, f"{dataset_name}_queries{filter_width}gt.npy"
                )
            )
            if QUERY_SUBSET is not None:
                query_filter_ranges = query_filter_ranges[:QUERY_SUBSET]
                query_gt = query_gt[:QUERY_SUBSET]
            print("==query_filter_ranges.shape", query_filter_ranges.shape)
            print("==query_gt.shape", query_gt.shape)

            # Perform the search
            manager = Manager()
            search_results = manager.list([None] * len(query_filter_ranges))

            def search_point(tup):
                ith_point, search_params = tup
                filter_start = query_filter_ranges[ith_point][0]
                filter_end = query_filter_ranges[ith_point][1]
                expr = "(priority > %s) && (priority < %s)" % (
                    filter_start,
                    filter_end,
                )

                result = points.search(
                    [queries[ith_point]],
                    "embeddings",
                    search_params,
                    limit=TOP_K,
                    expr=expr,
                )

                assert len(result) == 1
                hits = result[0]
                result_i = []
                distance = {}
                for hit in hits:
                    result_i.append(
                        hit.id
                    )  # , priority field: {hit.entity.get('priority')}
                    distance[hit.id] = hit.distance
                # sort results by distance
                result_i.sort(key=lambda k: distance[k])
                search_results[ith_point] = result_i

            with Pool(SIZE_OF_QUERY_POOL) as query_pool:
                for search_params in search_params_list:
                    print("=search_params: ", search_params)
                    formatted_search_params = "_".join(
                        [f"{key}_{value}" for key, value in search_params.items()]
                    )

                    # Use tqdm to track progress
                    start_time = (
                        time.time()
                    )  # Don't count the pool start time in the query time
                    for _ in tqdm(
                        query_pool.imap_unordered(
                            search_point,
                            zip(
                                range(len(query_filter_ranges)),
                                [search_params] * len(query_filter_ranges),
                            ),
                        )
                    ):
                        pass
                    end_time = time.time()

                    total_time = end_time - start_time
                    average_recall = compute_recall(query_gt, search_results, TOP_K)
                    print("=average recall = {:.4f}".format(average_recall))
                    print("=search latency = {:.4f}s".format(total_time))

                    run_results.append(
                        (
                            f"milvus_{index}_{formatted_index_params}_{formatted_search_params}",
                            average_recall,
                            total_time,
                        )
                    )

            with open(output_file, "a") as f:
                for name, recall, total_time in run_results:
                    f.write(
                        f"{filter_width},{name},{recall},{total_time/len(queries)},{len(queries)/total_time},{THREADS}\n"
                    )
        # delete index
        points.release()
        points.drop_index()

    # remove points from database
    utility.drop_collection("points")
