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

    for filter_width in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]:
        run_results = defaultdict(list)
        for q in tqdm(queries[:1000]):
            random_filter_start = np.random.uniform(0, 1 - filter_width)
            filter_range = (random_filter_start, random_filter_start + filter_width)

            start = time.time()
            gt = index.prefilter_query(q, top_k=top_k, filter_range=filter_range)
            run_results["prefiltering"].append((1, time.time() - start))

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
                        )
                    )

            for optimize_index_choice in [True, False]:
                for extra_doubles in range(6):
                    start = time.time()
                    postfilter_result = index.postfilter_query(
                        q,
                        top_k=top_k,
                        filter_range=filter_range,
                        extra_doubles=extra_doubles,
                        optimize_index_choice=optimize_index_choice,
                    )
                    run_results[
                        f"postfiltering{'-optimized' if optimize_index_choice else ''}_{extra_doubles}"
                    ].append(
                        (
                            len([x for x in gt[0] if x in postfilter_result[0]])
                            / len(gt[0]),
                            time.time() - start,
                        )
                    )

        with open(output_file, "a") as f:
            for name, zipped_recalls_times in run_results.items():
                recalls = [r for r, _ in zipped_recalls_times]
                times = [t for _, t in zipped_recalls_times]
                f.write(f"{filter_width},{name},{np.mean(recalls)},{np.mean(times)}\n")
