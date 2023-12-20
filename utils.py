import h5py
import numpy as np
import os
import diskannpy

def parse_ann_benchmarks_hdf5(data_path):
    with h5py.File(data_path, "r") as file:
        gt_neighbors = np.array(file["neighbors"])
        queries = np.array(file["test"])
        data = np.array(file["train"])

        return data, queries, gt_neighbors


def pareto_front(x, y):
    sorted_indices = sorted(range(len(y)), key=lambda k: -y[k])
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]

    for i in range(1, len(x_sorted)):
        if x_sorted[i] > pareto_front_x[-1]:
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])

    return pareto_front_x, pareto_front_y



def create_diskann_index(name, data, alpha, build_complexity, degree, distance_metric, filter_labels=None, filter_complexity=0, index_directory="indices"):
    if not os.path.exists(index_directory + "/" + name):
        diskannpy.build_memory_index(
            data,
            alpha=alpha,
            complexity=build_complexity,
            graph_degree=degree,
            distance_metric=distance_metric,
            index_directory=index_directory,
            index_prefix=name,
            num_threads=0,
            filter_labels=filter_labels,
            filter_complexity=filter_complexity
        )

    return diskannpy.StaticMemoryIndex(
        index_directory=index_directory,
        num_threads=0,
        initial_search_complexity=build_complexity,
        index_prefix=name,
        enable_filters=(filter_labels is not None)
    )