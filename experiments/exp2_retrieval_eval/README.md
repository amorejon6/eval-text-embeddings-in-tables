# Table Retrieval Experiment with Row Embeddings

This project generates row-level embeddings for CSV tables and evaluates different table representation methods for a table-to-table retrieval task.

## 1. Requirements

- Python 3.9+ (recommended: 3.10 or 3.11)
- A `pip` environment
- Optional GPU (speeds up `index.py` and large models)

Install core dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas torch sentence-transformers scikit-learn scipy chardet tqdm joblib
```

Optional dependency:

- `faiss-cpu` or `faiss-gpu` to accelerate top-k search in `eval2.py`

```bash
python3 -m pip install faiss-cpu
```

## 2. Expected Data Structure

- A directory with CSV files, where **each CSV represents one table**.
- Example:

```text
data/
  table_0001.csv
  table_0002.csv
  ...
```

## 3. Step 1: Generate Row Embeddings

Script: `index.py`

Example:

```bash
python3 index.py \
  --input_dir data \
  --out_pickle outputs/qwen_tables.pkl \
  --model qwen \
  --batch_size 64 \
  --include_header \
  --amp
```

Useful options:

- `--float16`: reduces pickle size
- `--no_table_vector`: do not save aggregated table embedding
- `--limit_files N`: process only the first `N` tables
- `--overwrite`: recompute tables already present in the pickle
- `--save_every N`: incremental save every `N` tables

Main output:

- A `.pickle` file with metadata and per-table/per-row embeddings

## 4. Step 2: Run Experiment Evaluation

Script: `eval2.py` (single evaluation entrypoint)

```bash
python3 eval2.py \
  --pickle outputs/qwen_tables.pkl \
  --out_dir res_final \
  --approaches baseline random10 cluster10 colnames_query \
  --seed 42 \
  --q_tables_ratio 0.10 \
  --q_rows_ratio 0.10 \
  --top_k 50 \
  --save_ranks
```

Useful options:

- `--cluster_k_cap 8`: controls `cluster10` cost
- `--jobs -1`: parallel prebuild (if `joblib` is available)
- `--no_faiss`: disable FAISS and use NumPy fallback
- `--jobs 1 --no_faiss`: compatibility mode (slower, fewer dependencies)

## 5. Step 3: Statistical Comparison (best vs second)

Script: `t-test.py`

By default, it reads `perquery_disjoint_*.csv` files from `res_final/`.

```bash
python3 t-test.py
```

Output:

- `best_vs_second_stats.csv`
- Includes, per dataset/method/metric: best model, second-best model, delta, paired t-test p-value, and Wilcoxon p-value

Important:

- `best_vs_second_stats.csv` can be empty if there are not enough model runs to compare.
- The script needs at least **2 different models for the same dataset+method group** (for example, two files for `baseline` on the same dataset, but produced with different embedding models).
- If only one model is available in each group, there is no "best vs second" pair and no rows will be produced.

## 6. Evaluation Output Files

In the `--out_dir` directory:

- `perquery_disjoint_<pickle_stem>_<approach>.csv`
- `all_per_query_disjoint_<pickle_stem>.csv`
- `summary_disjoint_<pickle_stem>.csv`
- `ranks_disjoint_<pickle_stem>_<approach>.jsonl` (if `--save_ranks` is enabled)

## 7. Reported Metrics

- `P@1`, `P@5`, `P@10`: used here with **Hit@k** semantics (1 if the correct table appears in top-k, else 0)
- `nDCG@10`
- `MRR@K` (where `K` is defined by `--top_k`; with `--top_k 50`, this is `MRR@50`)

## 8. Reproducibility

- Fix `--seed`
- Keep `--q_tables_ratio` and `--q_rows_ratio` fixed
- Use the same `--approaches` list
- Avoid mixing outputs from different runs in the same `out_dir`
