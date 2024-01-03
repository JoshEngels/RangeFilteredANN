from collections import defaultdict
from range_index import create_range_index
from utils import parse_ann_benchmarks_hdf5
import numpy as np
import time
from tqdm import tqdm
import os


data_dir = "/data/scratch/jae/ann_benchmarks_datasets/"

for dataset_name in ["glove-100-angular", "sift-128-euclidean"]:

    output_file = f"{dataset_name}_beamsearch_experiment.txt"

    with open(output_file, "a") as f:
        f.write("filter_width,method,recall,average_time\n")

    filter_path = f"{data_dir}/{dataset_name}_filters.npy"
    data_path = f"{data_dir}/{dataset_name}.hdf5"

    index = create_range_index(data_path, filter_path)

    queries = parse_ann_benchmarks_hdf5(data_path)[1]

    if "angular" in dataset_name:
        queries = queries / np.linalg.norm(queries, axis=-1)[:, np.newaxis]

    top_k = 10

    for filter_width in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]:
        run_results = defaultdict(list)
        for q in tqdm(queries[:10]):
            random_filter_start = np.random.uniform(0, 1 - filter_width)
            filter_range = (random_filter_start, random_filter_start + filter_width)

            start = time.time()
            gt = index.prefilter_query(q, top_k=top_k, filter_range=filter_range)
            run_results["prefiltering"].append((1, time.time() - start))

            for use_newbeam in [True, False]:
                os.environ["TRY_NEW_BEAMSEARCH"] = "1" if use_newbeam else "0"
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
                                starting_complexity=starting_complexity,
                                use_newbeam=use_newbeam
                            )
                            run_results[
                                f"postfiltering{'-newbeam' if use_newbeam else ''}{'-optimized' if optimize_index_choice else ''}_{extra_doubles}_{starting_complexity}"
                            ].append(
                                (
                                    len([x for x in gt[0] if x in postfilter_result[0]])
                                    / len(gt[0]),
                                    time.time() - start,
                                )
                            )

        with open(output_file, "a") as f:
            for name, zipped_recalls_times in run_results.items():
                if "post" not in name:
                    continue
                recalls = [r for r, _ in zipped_recalls_times]
                times = [t for _, t in zipped_recalls_times]
                f.write(f"{filter_width},{name},{np.mean(recalls)},{np.mean(times)}\n")
