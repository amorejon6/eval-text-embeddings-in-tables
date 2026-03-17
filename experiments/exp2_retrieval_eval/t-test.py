import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

DATA_DIR = Path("res_final")
BASE_METRICS = ["nDCG@10", "P@1", "P@5", "P@10"]
APPROACHES = ["colnames_query", "cluster10", "random10", "baseline"]

files = list(DATA_DIR.glob("perquery_disjoint_*.csv"))


def parse_filename(path: Path):
    """
    Returns: model, dataset, method.
    Expected pattern: perquery_disjoint_<run_id>_<approach>.csv
    where run_id is parsed as <model>_<dataset...>.
    """
    name = path.stem.replace("perquery_disjoint_", "")
    method = None
    run_id = name

    for approach in sorted(APPROACHES, key=len, reverse=True):
        suffix = f"_{approach}"
        if name.endswith(suffix):
            method = approach
            run_id = name[:-len(suffix)]
            break

    # Fallback for unknown approach names
    if method is None:
        parts = name.split("_")
        method = parts[-1]
        run_id = "_".join(parts[:-1])

    if "_" in run_id:
        model, dataset = run_id.split("_", 1)
    else:
        model, dataset = run_id, "default"
    return model, dataset, method


def ordered_metrics(columns):
    """Return metric names in a stable order, including dynamic MRR@K columns."""
    mrr_cols = [c for c in columns if c.startswith("MRR@")]

    def mrr_key(col):
        m = re.search(r"MRR@(\d+)", col)
        return int(m.group(1)) if m else 10**9

    mrr_cols = sorted(set(mrr_cols), key=mrr_key)
    base_cols = [m for m in BASE_METRICS if m in columns]
    extra_cols = sorted([c for c in columns if c not in set(mrr_cols) and c not in set(base_cols)])
    return mrr_cols + base_cols + extra_cols


# Load all CSVs into memory
# Structure: data[(dataset, method)][model] = df_perquery
data = {}
for f in files:
    model, dataset, method = parse_filename(f)
    df = pd.read_csv(f).copy()

    if "query_table_id" not in df.columns:
        raise ValueError(f"Missing 'query_table_id' in {f.name}")

    metric_cols = []
    for col in df.columns:
        if col in {"query_table_id", "approach"}:
            continue
        col_num = pd.to_numeric(df[col], errors="coerce")
        if col_num.notna().any():
            df[col] = col_num
            metric_cols.append(col)

    df = df[["query_table_id"] + metric_cols].sort_values("query_table_id").reset_index(drop=True)
    data.setdefault((dataset, method), {})[model] = df

results = []

for (dataset, method), model_dfs in data.items():
    models = list(model_dfs.keys())
    if len(models) < 2:
        # Cannot compare best vs second if there is only one model
        continue

    group_columns = set()
    for df in model_dfs.values():
        group_columns.update([c for c in df.columns if c != "query_table_id"])

    # For each metric, rank models by mean and compare top1 vs top2
    for metric in ordered_metrics(group_columns):
        available = []
        for model, df in model_dfs.items():
            if metric in df.columns:
                available.append((model, df))

        if len(available) < 2:
            continue

        means = []
        for model, df in available:
            means.append((model, df[metric].mean()))

        means.sort(key=lambda x: x[1], reverse=True)
        best_model, _ = means[0]
        second_model, _ = means[1]

        df_best = model_dfs[best_model]
        df_second = model_dfs[second_model]

        # Align queries defensively (in case some file is missing queries)
        merged = df_best[["query_table_id", metric]].merge(
            df_second[["query_table_id", metric]],
            on="query_table_id",
            suffixes=("_best", "_second"),
            how="inner"
        )

        if merged.empty or len(merged) < 2:
            continue

        best_vals = merged[f"{metric}_best"].astype(float).to_numpy()
        second_vals = merged[f"{metric}_second"].astype(float).to_numpy()

        if np.allclose(best_vals, second_vals):
            p_t = 1.0
            p_w = 1.0
        else:
            _, p_t = ttest_rel(best_vals, second_vals)
            p_t = None if np.isnan(p_t) else float(p_t)
            try:
                _, p_w = wilcoxon(best_vals, second_vals)
                p_w = float(p_w)
            except ValueError:
                p_w = None

        results.append({
            "dataset": dataset,
            "method": method,
            "metric": metric,
            "best_model": best_model,
            "second_model": second_model,
            "n_queries": int(len(merged)),
            "best_mean": float(best_vals.mean()),
            "second_mean": float(second_vals.mean()),
            "delta": float(best_vals.mean() - second_vals.mean()),
            "p_ttest": p_t,
            "p_wilcoxon": p_w,
        })

if results:
    results_df = pd.DataFrame(results).sort_values(["dataset", "method", "metric"])
else:
    results_df = pd.DataFrame(columns=[
        "dataset", "method", "metric", "best_model", "second_model", "n_queries",
        "best_mean", "second_mean", "delta", "p_ttest", "p_wilcoxon"
    ])

out_path = "best_vs_second_stats.csv"
results_df.to_csv(out_path, index=False)

print("✔ Comparisons (best vs second) completed")
print(f"Saved to: {out_path}")
if results_df.empty:
    print("No comparable model pairs were found (at least two models per dataset/method are required).")
else:
    print(results_df.head(10))
