# CSV Table Embeddings Robustness Experiment

This project extracts base embeddings from CSV tables using text embeddings models and performs robustness tests: random column reordering, column deletion, random string comparison, and header-only vectors.

## 1. Requirements

- Python 3.9+ (recommended: 3.10 or 3.11)
- A `pip` environment
- Optional GPU (speeds up `index.py` and large models)

Install core dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas torch sentence-transformers scikit-learn scipy chardet tqdm joblib
```

## 2. Expected Data Structure

- A directory with CSV files, where **each CSV represents one table**.
- Example:

```text
datasets/sensors/
  table_0001.csv
  table_0002.csv
  ...
```

## 3. Generate Row Embeddings and Run Evaluation

Script: `index.py`

Example:

```bash
python3 index.py \
  --input sensors \
  --model all-mini \
  --test all \
  --embeddings yes
```

## 4. Evaluation Output Files

In the `results` directory you can find the results csv files of the tests, ordered by test, dataset and model.

