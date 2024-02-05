import time
import os
from tqdm import tqdm

import numpy as np
import psycopg2

from psycopg2.extensions import register_adapter, AsIs

register_adapter(np.int64, AsIs)

THREADS = 16  # Adjust this to the desired number of parallel processes
print("THREADS", THREADS)
# Ensure index_cache/postfiler_vamana exists
os.makedirs("results", exist_ok=True)

TOP_K = 10
# Only use a subset of data and query for testing
DATA_SUBSET = None
QUERY_SUBSET = None
DATASETS = [
    "glove-100-angular",
    "sift-128-euclidean",
    "adversarial-100-angular",
    "deep-image-96-angular",
    "redcaps-512-angular",
]
EXPERIMENT_FILTER_WIDTHS = [f"2pow{i}" for i in range(-16, 1)]

dataset_folder = "/data/parap/storage/jae/new_filtered_ann_datasets"


def compute_recall(gt_neighbors, results, top_k):
    recall = 0
    for i in range(len(gt_neighbors)):  # for each query
        gt = set(gt_neighbors[i])
        res = set(results[i][:top_k])
        recall += len(gt.intersection(res)) / len(gt)
    return recall / len(gt_neighbors)  # average recall per query


def drop_all_tables(cursor):
    # Get a list of all table names in the database
    cursor.execute(
        "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';"
    )
    table_names = cursor.fetchall()

    # Generate and execute DROP TABLE statements for each table
    for table_name in table_names:
        table_name = table_name[0]  # Extract the table name from the result tuple
        drop_table_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
        cursor.execute(drop_table_query)
        print(f"Dropped table: {table_name}")

    # Commit the transaction
    conn.commit()


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
conn.autocommit = True
print("Connected to MSVBASE.")
# Create a cursor
cursor = conn.cursor()

# remove all existing data
try:
    cursor.execute("DROP EXTENSION vectordb")
except:
    pass
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
    print()
    print("Loading data, ", dataset_name)
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
    # data = data.tolist()
    # filter_values = filter_values.tolist()
    ids = list(range(len(data)))

    print("Inserting points to database.")
    table_name = dataset_name.replace("-", "_")
    cursor.execute(f"create table {table_name}(id int, filter REAL, vector_1 REAL[3]);")
    insert_query = (
        f"INSERT INTO {table_name}(id, filter, vector_1) VALUES (%s, %s, %s);"
    )
    start_time = time.time()
    batch_size = 1000
    for i in tqdm(list(range(0, len(data), batch_size))):
        data_subset = data[i : i + batch_size].tolist()
        values_to_insert = [
            (id_value, filter_value, vector_value)
            for id_value, filter_value, vector_value in zip(
                ids[i : i + batch_size], filter_values[i : i + batch_size], data_subset
            )
        ]
        cursor.executemany(insert_query, values_to_insert)

    end_time = time.time()
    print("build index latency = {:.4f}s".format(end_time - start_time))

    cursor.execute(f"CREATE INDEX {table_name}_filter_idx ON {table_name} (filter);")

    dataset_experiment_filter_widths = (
        [""] if dataset_name == "adversarial-100-angular" else EXPERIMENT_FILTER_WIDTHS
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
            os.path.join(dataset_folder, f"{dataset_name}_queries{filter_width}gt.npy")
        )
        if QUERY_SUBSET is not None:
            query_filter_ranges = query_filter_ranges[:QUERY_SUBSET]
            query_gt = query_gt[:QUERY_SUBSET]
        print("==query_filter_ranges.shape", query_filter_ranges.shape)
        print("==query_gt.shape", query_gt.shape)

        start_time = time.time()
        search_results = []
        # cannot use multithreading due to psycopg2 synchronization.
        for query_id in range(len(queries)):
            filter_start = query_filter_ranges[query_id][0]
            filter_end = query_filter_ranges[query_id][1]
            query_str = f"{{{','.join(map(str,  queries[query_id]))}}}"
            cursor.execute(
                "select * from %s where filter > %s and filter < %s order by vector_1 %s '%s' limit %s;"
                % (table_name, filter_start, filter_end, metric, query_str, TOP_K)
            )
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append(row[0])
            search_results.append(result)
        end_time = time.time()

        total_time = end_time - start_time
        average_recall = compute_recall(query_gt, search_results, TOP_K)
        print("=average recall = {:.4f}".format(average_recall))
        print("=search latency = {:.4f}s".format(total_time))

        with open(output_file, "a") as f:
            f.write(
                f"{filter_width},msvbase,{average_recall},{total_time/len(queries)},{len(queries)/total_time},{THREADS}\n"
            )
    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")


# remove points from database
cursor.execute("DROP EXTENSION vectordb")
drop_all_tables(cursor)
cursor.close()
conn.close()
