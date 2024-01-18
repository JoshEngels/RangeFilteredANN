import argparse
import numpy as np
import os
import wrapper as wp
import time
import sys
import multiprocessing

# Ensure index_cache/postfiler_vamana exists so indices are saved correctly
os.makedirs("index_cache/postfilter_vamana", exist_ok=True)
os.makedirs("results", exist_ok=True)

DATASET_FOLDER = "/data/parap/storage/jae/filtered_ann_datasets"
DATASETS = [
    "glove-100-angular",
    "deep-image-96-angular",
    "sift-128-euclidean",
    "redcaps-512-angular",
]

EXPERIMENT_FILTER_WIDTHS = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]

TOP_K = 10
BEAM_SIZES = [10, 20, 40, 80, 160, 320, 640, 1280]
FINAL_MULTIPLIES = [1, 2, 3, 4, 8, 16, 32]


parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=None, help="Number of threads")
parser.add_argument(
    "--postfiltering", action="store_true", help="Run postfiltering method"
)
parser.add_argument(
    "--optimized-postfiltering",
    action="store_true",
    help="Run optimized postfiltering method",
)
parser.add_argument(
    "--vamana-tree", action="store_true", help="Run Vamana tree method"
)
parser.add_argument(
    "--prefiltering", action="store_true", help="Run prefiltering method"
)
parser.add_argument(
    "--smart-combined", action="store_true", help="Run smart combined method"
)
parser.add_argument(
    "--three-split", action="store_true", help="Run three split method"
)
parser.add_argument("--all_methods", action="store_true", help="Run all methods")
parser.add_argument(
    "--results_file_prefix",
    help="Optional prefix to prepend to results files",
    default="",
)
parser.add_argument(
    "--beam_search_size",
    type=int,
    help=f"Optional beam size to use for experiments. Default of None corresponds to do all of {BEAM_SIZES}",
    default=None,
)
parser.add_argument(
    "--experiment_filter_width",
    type=str,
    help=f"Optional experiment filter size to use for experiments. Default of None corresponds to do all of {EXPERIMENT_FILTER_WIDTHS}",
    default=None,
)
parser.add_argument(
    "--num_final_multiplies",
    type=int,
    help=f"Optional number of final multiplies to use for experiments. Default of None corresponds to do all of {FINAL_MULTIPLIES}",
    default=None,
)
parser.add_argument(
    "--dataset",
    type=str,
    help=f"Optional dataset to use for experiments. Default of None corresponds to do all of {DATASETS}",
    default=None,
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Whether to run queries with the verbose flag",
)
parser.add_argument(
    "--dont_write_to_results_file",
    action="store_true",
    help="If included, won't write to a results file and will just print results",
)
parser.add_argument(
    "--tree_split_factor",
    type=int,
    default=2,
    help="The branching factor for the vamana tree methods",
)
args = parser.parse_args()

num_threads = args.threads
if num_threads is None:
    print("NOTE: No number of threads specified, so using all available threads")
    num_threads = multiprocessing.cpu_count()
os.environ["PARLAY_NUM_THREADS"] = str(num_threads)

if args.beam_search_size is not None:
    BEAM_SIZES = [args.beam_search_size]
if args.experiment_filter_width is not None:
    EXPERIMENT_FILTER_WIDTHS = [args.experiment_filter_width]
if args.num_final_multiplies is not None:
    FINAL_MULTIPLIES = [args.num_final_multiplies]
if args.dataset is not None:
    DATASETS = [args.dataset]

run_postfiltering = args.postfiltering or args.all_methods
run_optimized_postfiltering = args.optimized_postfiltering or args.all_methods
run_vamana_tree = args.vamana_tree or args.all_methods
run_prefiltering = args.prefiltering or args.all_methods
run_smart_combined = args.smart_combined or args.all_methods
run_three_split = args.three_split or args.all_methods

if not (
    run_postfiltering
    or run_optimized_postfiltering
    or run_vamana_tree
    or run_prefiltering
    or run_smart_combined
    or run_three_split
):
    print("NOTE: No experiments specified, so aborting")
    parser.print_help()
    sys.exit(0)

VERBOSE = args.verbose


def compute_recall(gt_neighbors, results, top_k):
    recall = 0
    for i in range(len(gt_neighbors)):  # for each query
        gt = set(gt_neighbors[i])
        res = set(results[i][:top_k])
        recall += len(gt.intersection(res)) / len(gt)
    return recall / len(gt_neighbors)  # average recall per query


# Returns true if the last two results at this number of multiplies had the exact same recall,
# or the last result had a recall of 1, or the last result took longer than the last prefiltering
# step, if such a step exists.
# We might remove this function for final experiments.
def should_break(run_results):
    if len(run_results) == 0:
        return False
    if run_results[-1][1] == 1.0:
        return True
    if len(run_results) == 1:
        return False

    prefiltering_results = [x for x in run_results if x[0] == "prefiltering"]
    if len(prefiltering_results) == 0:
        return False
    last_prefilter_time = prefiltering_results[-1][2]

    recalls_equal = run_results[-1][1] == run_results[-2][1]
    one_multiply = run_results[-1][0].split("_")[-1] == "1"
    slower_than_prefilter = run_results[-1][2] > last_prefilter_time

    return (recalls_equal and not one_multiply) or slower_than_prefilter


for dataset_name in DATASETS:
    output_file = f"results/{args.results_file_prefix}{dataset_name}_results.csv"

    # only write header if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "a") as f:
            f.write("filter_width,method,recall,average_time,qps,threads\n")

    data = np.load(os.path.join(DATASET_FOLDER, f"{dataset_name}.npy"))
    queries = np.load(os.path.join(DATASET_FOLDER, f"{dataset_name}_queries.npy"))

    # TODO: Remove for final experiments
    queries = queries[:1000]

    filter_values = np.load(
        os.path.join(DATASET_FOLDER, f"{dataset_name}_filter-values.npy")
    )

    metric = "mips" if "angular" in dataset_name else "Euclidian"

    # Build prefilter index
    if run_prefiltering:
        prefilter_constructor = wp.prefilter_index_constructor(metric, "float")
        prefilter_build_start = time.time()
        prefilter_index = prefilter_constructor(data, filter_values)
        prefilter_build_end = time.time()
        prefilter_build_time = prefilter_build_end - prefilter_build_start
        print(f"Prefiltering index build time: {prefilter_build_time:.3f}s")

    # Build postfilter index
    if run_postfiltering:
        # TODO: Add different alpha in build params
        build_params = wp.BuildParams(64, 500, 1.175)
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
    if (
        run_vamana_tree
        or run_optimized_postfiltering
        or run_smart_combined
        or run_three_split
    ):
        vamana_tree_constructor = wp.vamana_range_filter_tree_constructor(
            metric, "float"
        )
        vamana_tree_build_start = time.time()
        vamana_tree = vamana_tree_constructor(data, filter_values, cutoff=1_000, split_factor=args.tree_split_factor)
        vamana_tree_build_end = time.time()
        vamana_tree_build_time = vamana_tree_build_end - vamana_tree_build_start
        print(f"Vamana tree build time: {vamana_tree_build_time:.3f}s")

    for filter_width in EXPERIMENT_FILTER_WIDTHS:
        run_results = []
        query_filter_ranges = np.load(
            os.path.join(
                DATASET_FOLDER, f"{dataset_name}_queries_{filter_width}_ranges.npy"
            )
        )
        query_gt = np.load(
            os.path.join(
                DATASET_FOLDER, f"{dataset_name}_queries_{filter_width}_gt.npy"
            )
        )

        if run_prefiltering:
            start = time.time()
            query_params = wp.build_query_params(k=TOP_K, beam_size=0, verbose=VERBOSE)
            prefilter_results = prefilter_index.batch_search(
                queries, query_filter_ranges, queries.shape[0], query_params
            )
            run_results.append(
                (
                    f"prefiltering",
                    compute_recall(prefilter_results[0], query_gt, TOP_K),
                    time.time() - start,
                )
            )
            print(run_results[-1])

        if run_vamana_tree:
            for beam_size in BEAM_SIZES:
                start = time.time()
                query_params = wp.build_query_params(
                    k=TOP_K, beam_size=beam_size, verbose=VERBOSE
                )
                vamana_tree_results = vamana_tree.batch_search(
                    queries,
                    query_filter_ranges,
                    queries.shape[0],
                    "fenwick",
                    query_params,
                )
                run_results.append(
                    (
                        f"vamana-tree_{beam_size}",
                        compute_recall(vamana_tree_results[0], query_gt, TOP_K),
                        time.time() - start,
                    )
                )
                print(run_results[-1])

        if run_postfiltering:
            for beam_size in BEAM_SIZES:
                for final_beam_multiply in FINAL_MULTIPLIES:
                    query_params = wp.build_query_params(
                        k=TOP_K,
                        beam_size=beam_size,
                        final_beam_multiply=final_beam_multiply,
                        verbose=VERBOSE,
                    )
                    start = time.time()
                    postfilter_results = postfilter.batch_search(
                        queries,
                        query_filter_ranges,
                        queries.shape[0],
                        query_params,
                    )
                    run_results.append(
                        (
                            f"postfiltering_{beam_size}_{final_beam_multiply}",
                            compute_recall(postfilter_results[0], query_gt, TOP_K),
                            time.time() - start,
                        )
                    )
                    print(run_results[-1])
                    if should_break(run_results):
                        break

        if run_optimized_postfiltering:
            for beam_size in BEAM_SIZES:
                for final_beam_multiply in FINAL_MULTIPLIES:
                    query_params = wp.build_query_params(
                        k=TOP_K,
                        beam_size=beam_size,
                        final_beam_multiply=final_beam_multiply,
                        verbose=VERBOSE,
                    )
                    start = time.time()
                    optimized_postfilter_results = vamana_tree.batch_search(
                        queries,
                        query_filter_ranges,
                        queries.shape[0],
                        "optimized_postfilter",
                        query_params,
                    )
                    run_results.append(
                        (
                            f"optimized-postfiltering_{beam_size}_{final_beam_multiply}",
                            compute_recall(
                                optimized_postfilter_results[0], query_gt, TOP_K
                            ),
                            time.time() - start,
                        )
                    )
                    print(run_results[-1])
                    if should_break(run_results):
                        break

        if run_smart_combined:
            for beam_size in BEAM_SIZES:
                for final_beam_multiply in FINAL_MULTIPLIES:
                    query_params = wp.build_query_params(
                        k=TOP_K,
                        beam_size=beam_size,
                        final_beam_multiply=final_beam_multiply,
                        min_query_to_bucket_ratio=0.05,
                        verbose=VERBOSE,
                    )

                    start = time.time()
                    smart_combined_results = vamana_tree.batch_search(
                        queries,
                        query_filter_ranges,
                        queries.shape[0],
                        "smart_combined",
                        query_params,
                    )
                    run_results.append(
                        (
                            f"smart-combined_{beam_size}_{final_beam_multiply}",
                            compute_recall(
                                smart_combined_results[0], query_gt, TOP_K
                            ),
                            time.time() - start,
                        )
                    )
                    print(run_results[-1])
                    if should_break(run_results):
                        break

        if run_three_split:
            for beam_size in BEAM_SIZES:
                for final_beam_multiply in FINAL_MULTIPLIES:
                    start = time.time()
                    query_params = wp.build_query_params(
                        k=TOP_K,
                        beam_size=beam_size,
                        final_beam_multiply=final_beam_multiply,
                        min_query_to_bucket_ratio=0.05,
                        verbose=VERBOSE,
                    )
                    three_split_tree_results = vamana_tree.batch_search(
                        queries,
                        query_filter_ranges,
                        queries.shape[0],
                        "three_split",
                        query_params,
                    )
                    run_results.append(
                        (
                            f"three-split_{beam_size}_{final_beam_multiply}",
                            compute_recall(three_split_tree_results[0], query_gt, TOP_K),
                            time.time() - start,
                        )
                    )
                    print(run_results[-1])
                    if should_break(run_results):
                        break

        if not args.dont_write_to_results_file:
            with open(output_file, "a") as f:
                for name, recall, total_time in run_results:
                    f.write(
                        f"{filter_width},{name},{recall},{total_time/len(queries)},{len(queries)/total_time},{num_threads}\n"
                    )
