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

# Ensure index_cache/postfiler_vamana exists
os.makedirs("index_cache/postfilter_vamana", exist_ok=True)
os.makedirs("results", exist_ok=True)

TOP_K = 10
EXPERIMENT_FILTER_WIDTHS = [0.0001] #, 0.001, 0.01, 0.1, 0.2, 0.5, 1
INDEX_TYPES = ["IVF_FLAT"]  # , "IVF_PQ", "IVF_SQ8", "HNSW", "SCANN", "DISKANN", "FLAT"

dataset_folder = "/data/parap/storage/jae/filtered_ann_datasets"


def get_index_params(index_type):
    params = {
        "FLAT": {},
        "IVF_FLAT": {"nlist": 128},  # TODO: add more parameters
        "IVF_SQ8": {"nlist": 128},  # TODO: add more parameters
        "IVF_PQ": {"nlist": 128, "m": 10},  # TODO: add more parameters. dim mod m == 0
        "SCANN": {"nlist": 128},  # TODO: add more parameters.
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


def get_query_params(index_type):
    params = {
        "FLAT": {},
        "IVF_FLAT": {"nprobe": 10},  # range [1, nlist] TODO: add more parameters
        "IVF_SQ8": {"nprobe": 10},  # range [1, nlist] TODO: add more parameters
        "IVF_PQ": {"nprobe": 10},  # range [1, nlist] TODO: add more parameters
        "SCANN": {
            "nprobe": 10,
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
# remove all existing data
collections = utility.list_collections()
for collection in collections:
    utility.drop_collection(collection)

for dataset_name in [
    "glove-100-angular",
    "deep-image-96-angular",
    "sift-128-euclidean",
    "redcaps-512-angular",
]:
    output_file = f"results/{dataset_name}_milvus_results.csv"

    # only write header if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "a") as f:
            f.write("filter_width,method,recall,average_time,qps,threads\n")

    data = np.load(os.path.join(dataset_folder, f"{dataset_name}.npy"))
    num_entities, dim = data.shape

    queries = np.load(os.path.join(dataset_folder, f"{dataset_name}_queries.npy"))
    queries = queries[:1000]

    filter_values = np.load(
        os.path.join(dataset_folder, f"{dataset_name}_filter-values.npy")
    )

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
        index_params = get_index_params(index)
        search_params = get_query_params(index)

        start_time = time.time()
        points.create_index("embeddings", index_params)
        points.load()
        end_time = time.time()
        print("build index latency = {:.4f}s".format(end_time - start_time))
        utility.index_building_progress("points")

        for filter_width in EXPERIMENT_FILTER_WIDTHS:
            run_results = []
            query_filter_ranges = np.load(
                os.path.join(
                    dataset_folder, f"{dataset_name}_queries_{filter_width}_ranges.npy"
                )
            )
            query_gt = np.load(
                os.path.join(
                    dataset_folder, f"{dataset_name}_queries_{filter_width}_gt.npy"
                )
            )
            print(query_filter_ranges.shape)
            print(query_gt.shape)


            start_time = time.time()
            # TODO (shangdi): use out filtering range 
            result = points.search(
                queries, "embeddings", search_params, limit=TOP_K, expr="priority > 0.5"
            )  # , output_fields=["priority"]
            end_time = time.time()
            print("search latency = {:.4f}s".format(end_time - start_time))


        # delete index
        points.drop_index()

    # remove points from database
    points.release()
    utility.drop_collection("points")


# for hits in result:
#     for hit in hits:
#         print(f"hit: {hit}")  # , priority field: {hit.entity.get('priority')}
