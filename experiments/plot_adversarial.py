import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from plot import pareto_front


cmap = plt.get_cmap("tab10")
next_unused_cmap_index = 0
cmap_colors = {}


def plot(dataset_name):
    global next_unused_cmap_index

    df = pd.read_csv(f"results/{dataset_name}_results.csv")

    df["method"] = df["method"].str.split("_").str[0]

    grouped_data = df.groupby(["method"])

    for method, group in grouped_data:
        if method not in cmap_colors:
            cmap_colors[method] = cmap(next_unused_cmap_index)
            next_unused_cmap_index += 1
        color = cmap_colors[method]

        sorted_group = group.sort_values(by="recall", ascending=False)

        x, y = pareto_front(
            np.array(sorted_group["recall"]),
            1.0 / np.array(sorted_group["average_time"]),
        )

        if len(x) == 1:
            plt.plot(x, y, label=method[0], markersize=20, marker="x", color=color)
        else:
            plt.plot(x, y, label=method[0], color=color)

    plt.xlabel("Recall")
    plt.ylabel("Queries Per Second")
    # plt.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=5)
    plt.legend(loc="center left")

    plt.title(f"Pareto Fronts on Adversarial Data")

    plt.tight_layout()
    plt.savefig(f"results/plots/{dataset_name}_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    if not os.path.exists("results/plots"):
        os.makedirs("results/plots")
    plot("adversarial-100-angular")
