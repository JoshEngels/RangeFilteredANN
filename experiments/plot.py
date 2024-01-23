import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


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


SMALL_SIZE = 14
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc(
    "axes", labelsize=MEDIUM_SIZE
)  # fontsize of the x and y labels for the small plots
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc(
    "figure", labelsize=BIGGER_SIZE
)  # fontsize of the x and y labels for the big plots

cmap = plt.get_cmap("tab10")
next_unused_cmap_index = 0
cmap_colors = {}


filter_out_map = {
    "sift-128-euclidean": [-16, -15, -14, -13, -12],
    "glove-100-angular": [-16, -15, -14, -13, -12],
    "deep-image-96-angular": [-1, -3, -5, -7, -9, -11, -13, -15, -16],
    "redcaps-512-angular": [-1, -3, -5, -7, -9, -11, -13, -15, -16],
}


def plot(dataset_name):
    global next_unused_cmap_index

    df = pd.read_csv(f"results/{dataset_name}_results.csv")

    filter_out = [f"2pow{i}" for i in filter_out_map[dataset_name]]
    df = df[~df["filter_width"].isin(filter_out)]

    df["method"] = df["method"].str.split("_").str[0]

    grouped_data = df.groupby(["filter_width", "method"])

    num_plots = len(df["filter_width"].unique())
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols
    plot_scaler = 6

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(plot_scaler * num_cols, plot_scaler * num_rows),
        tight_layout=True,
    )
    axes = axes.reshape(-1)

    for (filter_width, method), group in grouped_data:
        if method not in cmap_colors:
            cmap_colors[method] = cmap(next_unused_cmap_index)
            next_unused_cmap_index += 1
        color = cmap_colors[method]

        ax = axes[df["filter_width"].unique().tolist().index(filter_width)]
        sorted_group = group.sort_values(by="recall", ascending=False)

        x, y = pareto_front(
            np.array(sorted_group["recall"]),
            1.0 / np.array(sorted_group["average_time"]),
        )

        if len(x) == 1:
            ax.plot(x, y, label=method, markersize=20, marker="x", color=color)
        else:
            ax.plot(x, y, label=method, color=color)
        ax.set_title(f"Filter Width: {filter_width.replace('2pow', '2^')}")

    fig.supxlabel("Recall")
    fig.supylabel("Queries Per Second")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=5)

    fig.suptitle(f"Pareto Fronts by Filter Width on {dataset_name}")

    plt.tight_layout()
    plt.savefig(f"results/plots/{dataset_name}_results.pdf", bbox_inches="tight")


if not os.path.exists("results/plots"):
    os.makedirs("results/plots")

plot("sift-128-euclidean")
plot("glove-100-angular")
plot("deep-image-96-angular")
plot("redcaps-512-angular")
