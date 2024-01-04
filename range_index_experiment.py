from collections import defaultdict
from range_index import create_range_index
from utils import parse_ann_benchmarks_hdf5
import numpy as np
import time
from tqdm import tqdm


data_dir = "/data/scratch/jae/ann_benchmarks_datasets/"
# data_dir = "/ssd1/anndata/ann-benchmarks/"

# TODO: Should also vary index quality
# TODO: Should also vary cutoff level
# TODO: Try on larger datasets

for dataset_name in ["glove-100-angular", "sift-128-euclidean"]:
    filter_path = f"{data_dir}/{dataset_name}_filters.npy"
    data_path = f"{data_dir}/{dataset_name}.hdf5"

    index = create_range_index(data_path, filter_path)

    queries = parse_ann_benchmarks_hdf5(data_path)[1]

    if "angular" in dataset_name:
        queries = queries / np.linalg.norm(queries, axis=-1)[:, np.newaxis]

    top_k = 10
    output_file = f"{dataset_name}_experiment.txt"

    with open(output_file, "a") as f:
        f.write("filter_width,method,recall,average_time\n")

    for filter_width in [0.001, 0.01, 0.1]:
    # for filter_width in [0.01]:
        run_results = defaultdict(list)
        multipliers = []
        for q in tqdm(queries[:100]):

            # Filter to the "easy" queries
            while True:
                random_filter_start = np.random.uniform(0, 1 - filter_width)
                filter_range = (random_filter_start, random_filter_start + filter_width)
                multiplier = index.get_optimized_multiplier(filter_range)
                if multiplier < 4:
                    break

            multipliers.append(index.get_optimized_multiplier(filter_range))

            start = time.time()
            gt = index.prefilter_query(q, top_k=top_k, filter_range=filter_range)
            run_results["prefiltering"].append((1, time.time() - start, multiplier))


            for query_complexity in [10, 20, 40, 80, 160, 320]:
                start = time.time()
                our_result = index.query_fenwick(
                    q,
                    top_k=top_k,
                    query_complexity=query_complexity,
                    filter_range=filter_range,
                )
                run_results[f"fenwick_{query_complexity}"].append(
                    (
                        len([x for x in gt[0] if x in our_result[0]]) / len(gt[0]),
                        time.time() - start,
                        multiplier
                    )
                )

            for query_complexity in [10, 20, 40, 80, 160, 320]:
                for extra_side_doubles in range(6):
                    start = time.time()
                    our_result = index.query_three_split(
                        q,
                        top_k=top_k,
                        query_complexity=query_complexity,
                        filter_range=filter_range,
                        extra_side_doubles=extra_side_doubles,
                    )
                    run_results[
                        f"three-split_{query_complexity}_{extra_side_doubles}"
                    ].append(
                        (
                            len([x for x in gt[0] if x in our_result[0]]) / len(gt[0]),
                            time.time() - start,
                            multiplier
                        )
                    )

            for optimize_index_choice in [True, False]:
                for starting_complexity in [10, 20, 40, 80, 160, 320]:
                    for extra_doubles in range(6):
                        start = time.time()
                        postfilter_result = index.postfilter_query(
                            q,
                            top_k=top_k,
                            filter_range=filter_range,
                            extra_doubles=extra_doubles,
                            optimize_index_choice=optimize_index_choice,
                            starting_complexity=starting_complexity
                        )
                        run_results[
                            f"postfiltering{'-optimized' if optimize_index_choice else ''}_{extra_doubles}_{starting_complexity}"
                        ].append(
                            (
                                len([x for x in gt[0] if x in postfilter_result[0]])
                                / len(gt[0]),
                                time.time() - start,
                                multiplier
                            )
                        )

        # Create histogram from multipliers variable:
        # import matplotlib.pyplot as plt
        # plt.hist(multipliers)
        # plt.savefig(f"figs/{dataset_name}_{filter_width}_multipliers.png")
        # plt.clf()
        print(np.percentile(multipliers, 50))

        with open(output_file, "a") as f:
            for name, zipped_recalls_times_multipliers in run_results.items():
                recalls = [r for r, _, _ in zipped_recalls_times_multipliers]
                times = [t for _, t, _ in zipped_recalls_times_multipliers]
                multipliers = [m for _, _, m in zipped_recalls_times_multipliers]                
                f.write(f"{filter_width},{name},{np.mean(recalls)},{np.mean(times)}\n")