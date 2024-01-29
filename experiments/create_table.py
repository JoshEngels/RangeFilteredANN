import pandas as pd
import glob


def create_latex_table(
    dataset_name, filter_width, recall_thresholds=(0.9, 0.99, 0.999)
):
    table_rows = []

    # Read in all csvs containing dataset_name as a substring
    paths = glob.glob(f"results/*{dataset_name}*.csv")
    dfs = [pd.read_csv(path) for path in paths]
    df = pd.concat(dfs)
    df["filter_width"] = df["filter_width"].str.strip("_")

    df = df[df["filter_width"] == filter_width]

    df["method"] = df["method"].str.split("_").str[0]

    for method, group in df.groupby("method"):
        qpss = []
        for recall_threshold in recall_thresholds:
            filtered_df = group[group["recall"] > recall_threshold]
            max_qps = filtered_df["qps"].max()
            qpss.append(max_qps)

        table_rows.append([method] + qpss)

    result_df = pd.DataFrame(
        table_rows,
        columns=["Method"]
        + [f"Max QPS, Recall = {recall}" for recall in recall_thresholds],
    )

    # bold_df = df_auc.apply(lambda x: ['\\textbf{{ {:.2f} }}'.format(val) if val == x.max() else '{:.2f}'.format(val) for val in x])
    # memory, build time, varying b

    latex_table = result_df.to_latex(index=False)

    return latex_table


if __name__ == "__main__":
    print(create_latex_table("sift-128-euclidean", "2pow-8"))
