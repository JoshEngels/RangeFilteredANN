from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from plot import pareto_front
import glob


BASELINE_METHODS = ["postfiltering", "prefiltering", "milvus", "msvbase"]

cmap = plt.get_cmap("tab10")
next_unused_cmap_index = 0
cmap_colors = {}


def plot(dataset_name):
    global next_unused_cmap_index

    paths = glob.glob(f"results/*{dataset_name}*.csv")
    dfs = [pd.read_csv(path) for path in paths]
    df = pd.concat(dfs)

    df["method"] = df["method"].str.split("_").str[0]
    df = df[df["method"] != "smart-combined"]

    grouped_data = df.groupby(["method"])

    for method, group in grouped_data:
        method = method[0]
        if method not in cmap_colors:
            cmap_colors[method] = cmap(next_unused_cmap_index)
            next_unused_cmap_index += 1
        color = cmap_colors[method]

        if method in BASELINE_METHODS:
            method = "Baseline: " + method.capitalize()
            marker = "o"
        else:
            marker = "x"

        sorted_group = group.sort_values(by="recall", ascending=False)

        x, y = pareto_front(
            np.array(sorted_group["recall"]),
            1.0 / np.array(sorted_group["average_time"]),
        )

        plt.plot(
            x, y, label=method, color=color, marker=marker, markersize=10, linewidth=1
        )
    ax = plt.gca()

    alpha = 10

    def fun(x):
        return 1 - (1 - x) ** (1 / alpha)

    def inv_fun(x):
        return 1 - (1 - x) ** alpha

    ax.set_xscale("function", functions=(fun, inv_fun))

    from matplotlib import ticker

    ax.xaxis.set_major_formatter(ticker.LogitFormatter())

    ticks = [
        0,
        1 / 2,
        1 - 1e-1,
        1 - 1e-2,
        1 - 1e-3,
        1 - 1e-4,
    ]

    ax.set_xticks(ticks)
    ax.set_xlim(-0.3, max(ticks))

    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    ax.tick_params(axis="x", labelsize=11, rotation=40)

    ax.grid(visible=True, which="major", color="0.85", linestyle="-")

    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Queries Per Second", fontsize=16)

    swaps = {
        "Baseline: Msvbase": "Baseline: VBASE",
        "three-split": "Three Split",
        "vamana-tree": "DiskANN WST",
        "super-postfiltering": "Super Postfiltering",
        "optimized-postfiltering": "Optimized Postfiltering",
    }

    handles, labels = ax.get_legend_handles_labels()
    order = [
        "Baseline: Prefiltering",
        "Baseline: Postfiltering",
        "Baseline: Milvus",
        "Baseline: Msvbase",
        "vamana-tree",
        "three-split",
        "optimized-postfiltering",
        "super-postfiltering",
    ]
    handles = [handles[labels.index(method)] for method in order]
    labels = [swaps[label] if label in swaps else label for label in order]

    plt.legend(handles, labels, loc="upper right")

    # plt.title(f"Pareto Fronts on Adversarial Data")

    plt.tight_layout()
    plt.savefig(f"results/plots/{dataset_name}_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    if not os.path.exists("results/plots"):
        os.makedirs("results/plots")
    plot("adversarial-100-angular")
