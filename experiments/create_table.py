import pandas as pd
import glob


def speedup_of_our_best_method(dataset_name, filter_width, recall_threshold):
    paths = glob.glob(f"results/*{dataset_name}*.csv")
    dfs = [pd.read_csv(path) for path in paths]
    df = pd.concat(dfs)
    df["filter_width"] = df["filter_width"].str.strip("_")

    df = df[df["filter_width"] == filter_width]

    df["method"] = df["method"].str.split("_").str[0]

    our_methods = [
        "vamana-tree",
        "three-split",
        "super-postfiltering",
        "optimized-postfiltering",
    ]
    their_methods = ["milvus", "vbase", "postfiltering", "prefiltering"]

    our_best_qps = df[
        df["method"].isin(our_methods) & (df["recall"] > recall_threshold)
    ]["qps"].max()
    their_best_qps = df[
        df["method"].isin(their_methods) & (df["recall"] > recall_threshold)
    ]["qps"].max()

    return our_best_qps / their_best_qps


# pows = [-14, -12, -10, -9, -8, -7 ,-6, -5, -4, -2, 0]
pows = range(-11, 1)

if __name__ == "__main__":
    rows = []
    for dataset in [
        "deep-image-96-angular",
        "sift-128-euclidean",
        "glove-100-angular",
        "redcaps-512-angular",
    ]:
        for recall_threshold in [0.99]:
            pow_speedups = []
            for pow in pows:
                pow_speedups.append(
                    speedup_of_our_best_method(dataset, f"2pow{pow}", recall_threshold)
                )
            rows.append([dataset] + pow_speedups)

    df = pd.DataFrame(rows, columns=["Dataset"] + [f"$2^{{{pow}}}$" for pow in pows])
    print(df.to_latex(index=False, float_format="{:0.3g}".format))
