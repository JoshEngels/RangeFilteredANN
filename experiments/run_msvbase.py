import time
import os
import sys
import itertools
from math import sqrt
from multiprocessing import Pool, Manager

import numpy as np
from tqdm import tqdm
import psycopg2

THREADS = 16  # Adjust this to the desired number of parallel processes
print("THREADS", THREADS)
print(
    "If you want to change the number of process used, OMP_NUM_THREADS in docker-compose.yml should also be changed."
)
# Ensure index_cache/postfiler_vamana exists
os.makedirs("results", exist_ok=True)

TOP_K = 10
# Only use a subset of data and query for testing
# TODO: change to None
DATA_SUBSET = 100
QUERY_SUBSET = 100
DATASETS = [
    "glove-100-angular",
    # "deep-image-96-angular",
    # "sift-128-euclidean",
    # "redcaps-512-angular",
]
EXPERIMENT_FILTER_WIDTHS = [str(-i) for i in range(17)]

dataset_folder = "/data/parap/storage/jae/new_filtered_ann_datasets"


def compute_recall(gt_neighbors, results, top_k):
    recall = 0
    for i in range(len(gt_neighbors)):  # for each query
        gt = set(gt_neighbors[i])
        res = set(results[i][:top_k])
        recall += len(gt.intersection(res)) / len(gt)
    return recall / len(gt_neighbors)  # average recall per query


# Set the connection parameters
db_params = {
    "dbname": "vectordb",
    "user": "vectordb",
    "password": "vectordb",
    "host": "172.17.0.2",  # Use the host machine's IP address
    "port": "5432",  # The default PostgreSQL port
}

# Establish the database connection
conn = psycopg2.connect(**db_params)
print("Connected to MSVBASE.")
# Create a cursor
cursor = conn.cursor()

# remove all existing data
drop_all_tables(cursor)
cursor.execute("SET max_parallel_workers = %s;" % THREADS)
cursor.execute("SET max_parallel_workers_per_gather = %s;" % THREADS)
cursor.execute("CREATE EXTENSION vectordb")


for dataset_name in DATASETS:
    output_file = f"results/{dataset_name}_msvbase_results.csv"

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

    # When calculating distances, the '<->' operator represents the L2 distance, while '<*>' represents the inner product distance.
    metric = "<*>" if "angular" in dataset_name else "<->"

    # data must be in list, not numpy array
    data = data.to_list()
    # filter_values = filter_values.to_list()
    ids = list(range(len(data)))

    values_to_insert = [
        (id_value, filter_value, vector_value)
        for id_value, filter_value, vector_value in zip(ids, filter_values, data)
    ]

    print("Inserting points to database.")
    table_name = dataset_name
    cursor.execute(f"create table {table_name}(id int, filter REAL, vector_1 REAL[3]);")
    insert_query = (
        f"INSERT INTO {table_name}(id, filter, vector_1) VALUES (%s, %s, %s);"
    )
    cursor.executemany(insert_query, values_to_insert)

    # use different indices
    print("Building index: ", index)
    index_params = get_index_params(index, num_entities, metric)
    search_params_list = get_query_params(index, num_entities)
    print("index_params: ", index_params)
    formatted_index_params = "_".join(
        [f"{key}_{value}" for key, value in index_params["params"].items()]
    )

    start_time = time.time()
    points.create_index("embeddings", index_params)
    points.load()
    end_time = time.time()
    print("build index latency = {:.4f}s".format(end_time - start_time))
    utility.index_building_progress("points")

    for filter_width in EXPERIMENT_FILTER_WIDTHS:
        print("==filter width: 2pow", filter_width)
        run_results = []
        query_filter_ranges = np.load(
            os.path.join(
                dataset_folder,
                f"{dataset_name}_queries_2pow{filter_width}_ranges.npy",
            )
        )
        query_gt = np.load(
            os.path.join(
                dataset_folder, f"{dataset_name}_queries_2pow{filter_width}_gt.npy"
            )
        )
        if QUERY_SUBSET is not None:
            query_filter_ranges = query_filter_ranges[:QUERY_SUBSET]
            query_gt = query_gt[:QUERY_SUBSET]
        print("==query_filter_ranges.shape", query_filter_ranges.shape)
        print("==query_gt.shape", query_gt.shape)

        start_time = time.time()
        search_results = []
        # TODO: change to multithreading
        for query_id in range(len(queries)):
            filter_start = query_filter_ranges[query_id][0]
            filter_end = query_filter_ranges[query_id][1]
            query_str = f"{{{','.join(map(str,  queries[ith_point]))}}}"
            cursor.execute(
                "select * from %s where filter > %s and filter < %s order by vector_1 %s '%s' limit %s;"
                % (table_name, filter_start, filter_end, metric, query_str, TOP_K)
            )
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append(row[0])
            search_restuls.append(result)
        end_time = time.time()

        total_time = end_time - start_time
        average_recall = compute_recall(query_gt, search_results, TOP_K)
        print("=average recall = {:.4f}".format(average_recall))
        print("=search latency = {:.4f}s".format(total_time))

        with open(output_file, "a") as f:
            for name, recall, total_time in run_results:
                f.write(
                    f"{filter_width},{name},{recall},{total_time/len(queries)},{len(queries)/total_time},{THREADS}\n"
                )
    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")


# remove points from database
drop_all_tables(cursor)
cursor.close()
conn.close()
