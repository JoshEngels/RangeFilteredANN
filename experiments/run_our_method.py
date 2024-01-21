import argparse
import numpy as np
import os
import wrapper as wp
import time
import sys
import multiprocessing

DATASET_FOLDER = "/data/parap/storage/jae/new_filtered_ann_datasets"
DATASETS = [
    "sift-128-euclidean",
    "glove-100-angular",
    "deep-image-96-angular",
    "redcaps-512-angular",
]


# Ensure results and index caches exist so indices are saved correctly
os.makedirs("results", exist_ok=True)
for dataset in DATASETS:
    os.makedirs(f"index_cache/{dataset}/", exist_ok=True)
    os.makedirs(f"index_cache/{dataset}-super_opt_postfiltering/", exist_ok=True)

EXPERIMENT_FILTER_WIDTHS = [f"2pow{i}" for i in range(-16, 1)]

TOP_K = 10
BEAM_SIZES = [10, 20, 40, 80, 160, 320, 640, 1280]
FINAL_MULTIPLIES = [1, 2, 3, 4, 8, 16, 32]

ALPHAS = [1]
VAMANA_TREE_SPLIT_FACTORS = [2]

SUPER_POSTFILTERING_SPLIT_FACTORS = [2]
SUPER_POSTFILTERING_SHIFT_FACTORS = [0.5]

# TODO: Change to 10000 for final experiments
NUM_QUERIES = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=None, help="Number of threads")
parser.add_argument(
    "--postfiltering", action="store_true", help="Run postfiltering method"
)
parser.add_argument(
    "--optimized_postfiltering",
    action="store_true",
    help="Run optimized postfiltering method",
)
parser.add_argument("--vamana_tree", action="store_true", help="Run Vamana tree method")
parser.add_argument(
    "--prefiltering", action="store_true", help="Run prefiltering method"
)
parser.add_argument(
    "--smart_combined", action="store_true", help="Run smart combined method"
)
parser.add_argument("--three_split", action="store_true", help="Run three split method")
parser.add_argument(
    "--super_opt_postfiltering",
    action="store_true",
    help="Run super optimized postfiltering method",
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
    "--vamana_tree_split_factor",
    type=int,
    help="The branching factor for the vamana tree methods",
)
parser.add_argument(
    "--alpha",
    type=float,
    help="Alpha for all graph based methods",
)
parser.add_argument(
    "--super_opt_postfiltering_split_factor",
    type=float,
    help="Split factor for super optimized postfiltering",
)
parser.add_argument(
    "--super_opt_postfiltering_shift_factor",
    type=float,
    help="Shift factor for super optimized postfiltering",
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
if args.alpha is not None:
    ALPHAS = [args.alpha]
if args.vamana_tree_split_factor is not None:
    VAMANA_TREE_SPLIT_FACTORS = [args.tree_split_factor]
if args.super_opt_postfiltering_split_factor is not None:
    SUPER_POSTFILTERING_SPLIT_FACTORS = [args.super_opt_postfiltering_split_factor]
if args.super_opt_postfiltering_shift_factor is not None:
    SUPER_POSTFILTERING_SHIFT_FACTORS = [args.super_opt_postfiltering_shift_factor]

run_postfiltering = args.postfiltering or args.all_methods
run_optimized_postfiltering = args.optimized_postfiltering or args.all_methods
run_vamana_tree = args.vamana_tree or args.all_methods
run_prefiltering = args.prefiltering or args.all_methods
run_smart_combined = args.smart_combined or args.all_methods
run_three_split = args.three_split or args.all_methods
run_super_opt_postfiltering = args.super_opt_postfiltering or args.all_methods

if not (
    run_postfiltering
    or run_optimized_postfiltering
    or run_vamana_tree
    or run_prefiltering
    or run_smart_combined
    or run_three_split
    or run_super_opt_postfiltering
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
    if run_results[-1][2] == 1.0:
        return True
    if len(run_results) == 1:
        return False

    recall_not_better = run_results[-1][2] <= run_results[-2][2]
    one_multiply = run_results[-1][1].split("_")[-1] == "1"
    if (recall_not_better and not one_multiply):
        return True

    prefiltering_results = [x for x in run_results if x[1] == "prefiltering"]
    if len(prefiltering_results) == 0:
        return False
    last_prefilter_time = prefiltering_results[-1][3]
    slower_than_prefilter = run_results[-1][3] > last_prefilter_time

    return slower_than_prefilter


def initialize_dataset(dataset_name):
    data = np.load(os.path.join(DATASET_FOLDER, f"{dataset_name}.npy"))
    queries = np.load(os.path.join(DATASET_FOLDER, f"{dataset_name}_queries.npy"))

    queries = queries[:NUM_QUERIES]

    filter_values = np.load(
        os.path.join(DATASET_FOLDER, f"{dataset_name}_filter-values.npy")
    )

    metric = "mips" if "angular" in dataset_name else "Euclidian"

    return data, queries, filter_values, metric


def get_queries_and_gt(dataset_name, filter_width):
    query_filter_ranges = np.load(
        os.path.join(
            DATASET_FOLDER, f"{dataset_name}_queries_{filter_width}_ranges.npy"
        )
    )
    query_gt = np.load(
        os.path.join(DATASET_FOLDER, f"{dataset_name}_queries_{filter_width}_gt.npy")
    )

    return query_filter_ranges, query_gt


def run_prefiltering_experiment(all_results, dataset_name, filter_width):
    data, queries, filter_values, metric = initialize_dataset(dataset_name)

    prefilter_constructor = wp.prefilter_index_constructor(metric, "float")
    prefilter_build_start = time.time()
    prefilter_index = prefilter_constructor(data, filter_values)
    prefilter_build_end = time.time()
    prefilter_build_time = prefilter_build_end - prefilter_build_start
    print(f"Prefiltering index build time: {prefilter_build_time:.3f}s", flush=True)

    query_filter_ranges, query_gt = get_queries_and_gt(dataset_name, filter_width)

    start = time.time()
    query_params = wp.build_query_params(k=TOP_K, beam_size=0, verbose=VERBOSE)
    prefilter_results = prefilter_index.batch_search(
        queries, query_filter_ranges, queries.shape[0], query_params
    )
    all_results.append(
        (
            filter_width,
            f"prefiltering",
            compute_recall(prefilter_results[0], query_gt, TOP_K),
            time.time() - start,
        )
    )
    print(all_results[-1])


def run_postfiltering_experiment(all_results, dataset_name, filter_width, alpha):
    data, queries, filter_values, metric = initialize_dataset(dataset_name)

    build_params = wp.BuildParams(64, 500, alpha, f"index_cache/{dataset_name}/unsorted-")
    postfilter_constructor = wp.postfilter_vamana_constructor(metric, "float")
    postfilter_build_start = time.time()
    postfilter = postfilter_constructor(
        data,
        filter_values,
        build_params
    )
    postfilter_build_time = time.time() - postfilter_build_start
    print(f"Naive postfilter build time: {postfilter_build_time:.3f}s", flush=True)

    query_filter_ranges, query_gt = get_queries_and_gt(dataset_name, filter_width)

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
            all_results.append(
                (
                    filter_width,
                    f"postfiltering_{alpha}_{beam_size}_{final_beam_multiply}",
                    compute_recall(postfilter_results[0], query_gt, TOP_K),
                    time.time() - start,
                )
            )
            print(all_results[-1])
            if should_break(all_results):
                break


def get_vamana_tree(data, filter_values, metric, alpha, split_factor):
    vamana_tree_constructor = wp.vamana_range_filter_tree_constructor(metric, "float")
    vamana_tree_build_start = time.time()
    build_params = wp.BuildParams(64, 500, alpha, f"index_cache/{dataset_name}/")
    vamana_tree = vamana_tree_constructor(
        data,
        filter_values,
        cutoff=1_000,
        split_factor=split_factor,
        build_params=build_params,
    )
    vamana_tree_build_end = time.time()
    vamana_tree_build_time = vamana_tree_build_end - vamana_tree_build_start
    print(f"Vamana tree build time: {vamana_tree_build_time:.3f}s", flush=True)

    return vamana_tree


# Runs tree based methods if their booleans are true
def run_tree_experiments(all_results, dataset_name, filter_width, alpha, split_factor):
    if not (
        run_vamana_tree
        or run_optimized_postfiltering
        or run_smart_combined
        or run_three_split
    ):
        return

    data, queries, filter_values, metric = initialize_dataset(dataset_name)

    vamana_tree = get_vamana_tree(data, filter_values, metric, alpha, split_factor)

    query_filter_ranges, query_gt = get_queries_and_gt(dataset_name, filter_width)

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
            all_results.append(
                (
                    filter_width,
                    f"vamana-tree_{alpha:.3f}_{split_factor}_{beam_size}",
                    compute_recall(vamana_tree_results[0], query_gt, TOP_K),
                    time.time() - start,
                )
            )
            print(all_results[-1])

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
                all_results.append(
                    (
                        filter_width,
                        f"optimized-postfiltering_{alpha:.3f}_{split_factor}_{beam_size}_{final_beam_multiply}",
                        compute_recall(
                            optimized_postfilter_results[0], query_gt, TOP_K
                        ),
                        time.time() - start,
                    )
                )
                print(all_results[-1])
                if should_break(all_results):
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
                all_results.append(
                    (
                        filter_width,
                        f"smart-combined_{alpha:.3f}_{split_factor}_{beam_size}_{final_beam_multiply}",
                        compute_recall(smart_combined_results[0], query_gt, TOP_K),
                        time.time() - start,
                    )
                )
                print(all_results[-1])
                if should_break(all_results):
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
                all_results.append(
                    (
                        filter_width,
                        f"three-split_{alpha:.3f}_{split_factor}_{beam_size}_{final_beam_multiply}",
                        compute_recall(three_split_tree_results[0], query_gt, TOP_K),
                        time.time() - start,
                    )
                )
                print(all_results[-1])
                if should_break(all_results):
                    break



def run_super_optimized_postfiltering_experiment(all_results, dataset_name, filter_width, alpha, split_factor, shift_factor):
    data, queries, filter_values, metric = initialize_dataset(dataset_name)

    constructor = wp.super_optimized_postfilter_tree_constructor(metric, "float")
    build_start = time.time()
    build_params = wp.BuildParams(64, 500, alpha, f"index_cache/{dataset_name}-super_opt_postfiltering/")
    super_optimized_postfilter_tree = constructor(
        data,
        filter_values,
        cutoff=1_000,
        split_factor=split_factor,
        shift_factor=shift_factor,
        build_params=build_params,
    )
    build_time = time.time() - build_start
    print(f"Super optimized postfilter tree build time: {build_time:.3f}s", flush=True)

    query_filter_ranges, query_gt = get_queries_and_gt(dataset_name, filter_width)

    for beam_size in BEAM_SIZES:
        for final_beam_multiply in FINAL_MULTIPLIES:
            query_params = wp.build_query_params(
                k=TOP_K,
                beam_size=beam_size,
                final_beam_multiply=final_beam_multiply,
                verbose=VERBOSE,
            )
            start = time.time()
            postfilter_results = super_optimized_postfilter_tree.batch_search(
                queries,
                query_filter_ranges,
                queries.shape[0],
                query_params,
            )
            all_results.append(
                (
                    filter_width,
                    f"super-postfiltering_{split_factor}_{shift_factor}_{alpha}_{beam_size}_{final_beam_multiply}",
                    compute_recall(postfilter_results[0], query_gt, TOP_K),
                    time.time() - start,
                )
            )
            print(all_results[-1])
            if should_break(all_results):
                break




def save_results(all_results, dataset_name):
    output_file = f"results/{args.results_file_prefix}{dataset_name}_results.csv"

    # only write header if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "a") as f:
            f.write("filter_width,method,recall,average_time,qps,threads\n")

    if not args.dont_write_to_results_file:
        with open(output_file, "a") as f:
            for filter_width, name, recall, total_time in all_results:
                f.write(
                    f"{filter_width},{name},{recall},{total_time/NUM_QUERIES},{NUM_QUERIES/total_time},{num_threads}\n"
                )


for dataset_name in DATASETS:
    for experiment_filter_width in EXPERIMENT_FILTER_WIDTHS:
        all_results = []
        if run_prefiltering:
            run_prefiltering_experiment(all_results, dataset_name, experiment_filter_width)
        for alpha in ALPHAS:
            if run_postfiltering:
                run_postfiltering_experiment(
                    all_results, dataset_name, experiment_filter_width, alpha
                )
            for split_factor in VAMANA_TREE_SPLIT_FACTORS:
                run_tree_experiments(
                    all_results,
                    dataset_name,
                    experiment_filter_width,
                    alpha,
                    split_factor,
                )
            if run_super_opt_postfiltering:
                for split_factor in SUPER_POSTFILTERING_SPLIT_FACTORS:
                    for shift_factor in SUPER_POSTFILTERING_SHIFT_FACTORS:
                        run_super_optimized_postfiltering_experiment(
                            all_results,
                            dataset_name,
                            experiment_filter_width,
                            alpha,
                            split_factor,
                            shift_factor,
                        )
        save_results(all_results, dataset_name)
