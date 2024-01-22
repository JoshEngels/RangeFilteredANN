import time
import os
import sys

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
from math import sqrt 
from multiprocessing import Pool

# Ensure index_cache/postfiler_vamana exists
os.makedirs("index_cache/postfilter_vamana", exist_ok=True)
os.makedirs("results", exist_ok=True)

TOP_K = 10
EXPERIMENT_FILTER_WIDTHS = [str(-i) for i in range(1)] # 17
INDEX_TYPES = ["IVF_FLAT"]  # , "IVF_PQ", "IVF_SQ8", "HNSW", "SCANN", "DISKANN", "FLAT"

dataset_folder = "/data/parap/storage/jae/new_filtered_ann_datasets"

def compute_recall(gt_neighbors, results, top_k):
    recall = 0
    for i in range(len(gt_neighbors)):  # for each query
        gt = set(gt_neighbors[i])
        res = set(results[i][:top_k])
        recall += len(gt.intersection(res)) / len(gt)
    return recall / len(gt_neighbors)  # average recall per query

def get_index_params(index_type, n):
    params = {
        "FLAT": {},
        "IVF_FLAT": {"nlist": int(sqrt(n))},  # TODO: add more parameters
        "IVF_SQ8": {"nlist": int(sqrt(n))},  # TODO: add more parameters
        "IVF_PQ": {"nlist": int(sqrt(n)), "m": 10},  # TODO: add more parameters. dim mod m == 0
        "SCANN": {"nlist": int(sqrt(n))},  # TODO: add more parameters.
        "HNSW": {"M": 128, "efConstruction": 128},
        "DISKANN": {},
    }
    if index_type not in params:
        raise Exception("Invalid index type")

    # print(fmt.format("Start Creating index IVF_FLAT"))
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": params[index_type],
    }
    return index


def get_query_params(index_type, n):
    params = {
        "FLAT": {},
        "IVF_FLAT": {"nprobe": int(sqrt(n))},  # range [1, nlist] TODO: add more parameters
        "IVF_SQ8": {"nprobe": int(sqrt(n))},  # range [1, nlist] TODO: add more parameters
        "IVF_PQ": {"nprobe": int(sqrt(n))},  # range [1, nlist] TODO: add more parameters
        "SCANN": {
            "nprobe": int(sqrt(n)),
            "reorder_k": TOP_K,
        },  # range [1, nlist] TODO: add more parameters
        "HNSW": {"ef": 50},
        "DISKANN": {"search_list": 50},
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

for dataset_name in [
    "glove-100-angular",
    # "deep-image-96-angular",
    # "sift-128-euclidean",
    # "redcaps-512-angular",
]:
    output_file = f"results/{dataset_name}_milvus_results.csv"

    # only write header if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "a") as f:
            f.write("filter_width,method,recall,average_time,qps,threads\n")

    print("Loading data.")
    data = np.load(os.path.join(dataset_folder, f"{dataset_name}.npy"))
    num_entities, dim = data.shape
    print("data.shape, ", data.shape)

    print("Loading queries.")
    queries = np.load(os.path.join(dataset_folder, f"{dataset_name}_queries.npy"))
    print("queries.shape, ", queries.shape)
    # TODO: remove
    queries = queries[:1000]

    print("Loading filter values.")
    filter_values = np.load(
        os.path.join(dataset_folder, f"{dataset_name}_filter-values.npy")
    )
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
        print("Building index ", index)
        index_params = get_index_params(index, num_entities)
        search_params = get_query_params(index, num_entities)
        print("index_params: ", index_params)
        print("search_params: ", search_params)


        start_time = time.time()
        points.create_index("embeddings", index_params)
        points.load()
        end_time = time.time()
        print("build index latency = {:.4f}s".format(end_time - start_time))
        utility.index_building_progress("points")

        for filter_width in EXPERIMENT_FILTER_WIDTHS:
            print("filter width: 2pow", filter_width)
            run_results = []
            query_filter_ranges = np.load(
                os.path.join(
                    dataset_folder, f"{dataset_name}_queries_2pow{filter_width}_ranges.npy"
                )
            )
            query_gt = np.load(
                os.path.join(
                    dataset_folder, f"{dataset_name}_queries_2pow{filter_width}_gt.npy"
                )
            )
            print("query_filter_ranges.shape", query_filter_ranges.shape)
            print("query_gt.shape", query_gt.shape)
            times = np.zeros(len(query_filter_ranges))
            # Perform the search
            def search_point(ith_point):
                filter_start = query_filter_ranges[ith_point][0]
                filter_end = query_filter_ranges[ith_point][1]
                expr = "(priority > %s) && (priority < %s)" % (filter_start, filter_end)

                start_time = time.time()
                result = points.search(
                    [queries[ith_point]], "embeddings", search_params, limit=TOP_K, expr=expr
                )
                end_time = time.time()
                times[ith_point] = end_time - start_time
                return times[ith_point]

            num_processes = 4  # Adjust this to the desired number of parallel processes
            
            # Create a Pool of worker processes
            with Pool(num_processes) as pool:
                # Use tqdm to track progress
                for time_taken in tqdm(pool.imap_unordered(search_point, range(len(query_filter_ranges)))):
                    # times.append(time_taken)
                    pass
                
                # results = []
                # for hits in result:
                #     result_i = []
                #     for hit in hits:
                #         # possible to return less than TOP_K results
                #         result_i.append(hit.id) # , priority field: {hit.entity.get('priority')}
                #     results.append(result_i)
                # TODO (shangdi): compute recall and write csv file
            total_time = sum(times)
            print("search latency = {:.4f}s".format(total_time ))
        # delete index
        points.release()
        points.drop_index()

    # remove points from database
    utility.drop_collection("points")
