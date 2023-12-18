import h5py
import numpy as np


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
