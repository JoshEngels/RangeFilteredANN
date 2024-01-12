import numpy as np
import os
import wrapper as wp
import time
import sys


# Ensure index_cache/postfiler_vamana exists
os.makedirs("index_cache/postfilter_vamana", exist_ok=True)
os.makedirs("results", exist_ok=True)


EXPERIMENT_FILTER_WIDTHS = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]

dataset_folder = "/data/parap/storage/jae/filtered_ann_datasets"


if len(sys.argv) > 1:
    THREADS = int(sys.argv[1])
else:
    THREADS = 80
os.environ["PARLAY_NUM_THREADS"] = str(THREADS)


# run experiment
TOP_K = 10
BEAM_SIZES = [10, 20, 40, 80, 160, 320]
FINAL_MULTIPLIES = [1, 2, 4, 8, 16]


def compute_recall(gt_neighbors, results, top_k):
    recall = 0
    for i in range(len(gt_neighbors)):  # for each query
        gt = set(gt_neighbors[i])
        res = set(results[i][:top_k])
        recall += len(gt.intersection(res)) / len(gt)
    return recall / len(gt_neighbors)  # average recall per query


for dataset_name in ["glove-100-angular", "deep-image-96-angular", "sift-128-euclidean", "redcaps-512-angular"]:
    output_file = f"results/{dataset_name}_results.csv"

    # only write header if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "a") as f:
            f.write("filter_width,method,recall,average_time,qps,threads\n")

    data = np.load(os.path.join(dataset_folder, f"{dataset_name}.npy"))
    queries = np.load(os.path.join(dataset_folder, f"{dataset_name}_queries.npy"))
    filter_values = np.load(
        os.path.join(dataset_folder, f"{dataset_name}_filter-values.npy")
    )

    metric = "mips" if "angular" in dataset_name else "Euclidian"

    # Build prefilter index
    prefilter_constructor = wp.prefilter_index_constructor(metric, "float")
    prefilter_build_start = time.time()
    prefilter_index = prefilter_constructor(data, filter_values)
    prefilter_build_end = time.time()
    prefilter_build_time = prefilter_build_end - prefilter_build_start
    print(f"Prefiltering index build time: {prefilter_build_time:.3f}s")

    # Build postfilter index
    build_params = wp.BuildParams(64, 500, 1.35)
    postfilter_constructor = wp.postfilter_vamana_constructor(metric, "float")
    postfilter_build_start = time.time()
    postfilter = postfilter_constructor(
        data,
        filter_values,
        build_params,
        f"index_cache/postfilter_vamana/{dataset_name}_",
    )
    postfilter_build_time = time.time() - postfilter_build_start
    print(f"Naive postfilter build time: {postfilter_build_time:.3f}s")

    # Build Vamana tree index
    vamana_tree_constructor = wp.vamana_range_filter_tree_constructor(metric, "float")
    vamana_tree_build_start = time.time()
    vamana_tree = vamana_tree_constructor(data, filter_values, 1_000)
    vamana_tree_build_end = time.time()
    vamana_tree_build_time = vamana_tree_build_end - vamana_tree_build_start
    print(f"Vamana tree build time: {vamana_tree_build_time:.3f}s")

    continue

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

        start = time.time()
        prefilter_results = prefilter_index.batch_query(
            queries, query_filter_ranges, queries.shape[0], TOP_K
        )
        run_results.append((
            f"prefiltering",
            compute_recall(prefilter_results[0], query_gt, TOP_K),
            time.time() - start,
        ))
        print(run_results[-1])

        for beam_size in BEAM_SIZES:
            start = time.time()
            vamana_tree_results = vamana_tree.batch_filter_search(
                queries, query_filter_ranges, queries.shape[0], TOP_K
            )
            run_results.append((
                f"vamana_tree_{beam_size}",
                compute_recall(vamana_tree_results[0], query_gt, TOP_K),
                time.time() - start,
            ))
            print(run_results[-1])

        for beam_size in BEAM_SIZES:
            for final_multiply in FINAL_MULTIPLIES:
                query_params = wp.build_query_params(k=TOP_K, beam_size=100)
                start = time.time()
                postfilter_results = postfilter.batch_query(
                    queries,
                    query_filter_ranges,
                    queries.shape[0],
                    TOP_K,
                    query_params,
                    final_multiply,
                )
                run_results.append((
                    f"postfiltering_{beam_size}_{final_multiply}",
                    compute_recall(postfilter_results[0], query_gt, TOP_K),
                    time.time() - start,
                ))
                print(run_results[-1])


        with open(output_file, "a") as f:
            for name, recall, total_time in run_results:
                f.write(
                    f"{filter_width},{name},{recall / len(queries)},{total_time/len(queries)},{len(queries)/total_time}{THREADS}\n"
                )
