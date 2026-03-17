
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_row_embeddings_pickle_optimized.py

Generates one embedding per ROW for each table (CSV) and stores it in a reusable
.pickle file, optimized to leverage GPU:
- Uses convert_to_tensor=True and normalizes on GPU.
- Uses torch.inference_mode() to reduce overhead.
- Auto-tunes the maximum batch size without OOM (binary search).
- Retries on OOM by halving the batch size.
- Optional --amp for autocast (fp16/bf16) during forward pass, with final output in fp32.

Pickle structure:
{
  "meta": {
    "model_name": "...",
    "embedding_dim": 768,
    "updated_at": "2025-...Z",
    "args": {...},
    "schema": "row_embeddings_v1"
  },
  "tables": {
    "table.csv": {
      "table_id": "table",
      "file_name": "table.csv",
      "n_rows": 123,
      "n_cols": 8,
      "columns": [...],
      "row_text_mode": {"include_header": true, "sep_cell": " | "},
      "row_embeddings": np.ndarray (n_rows, d) float32 or float16,
      "embedding_table_mean_rows": np.ndarray (d,) optional
    },
    ...
  }
}
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import traceback
import pickle
import chardet
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List, Optional

from experiments.shared.utils import get_model  # should return (model, dimensions)


# -----------------------------
# Global GPU optimizations
# -----------------------------

torch.backends.cudnn.benchmark = True  # better kernels for variable input sizes
try:
    torch.set_float32_matmul_precision("high")  # faster matmul on modern GPUs
except Exception:
    pass


# -----------------------------
# Text utilities
# -----------------------------

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully empty columns and rows."""
    df = df.copy()
    df.dropna(axis='columns', how='all', inplace=True)
    df.dropna(how='all', inplace=True)
    return df


def row_to_text(row_vals: List[str], sep_cell: str = " ") -> str:
    return sep_cell.join(map(str, row_vals))


def table_rows_to_texts(
    df: pd.DataFrame,
    include_header: bool = False,
    sep_cell: str = " "
) -> List[str]:
    """
    Convert each row to one text line. If include_header=True, prepend the header.
    Implemented with itertuples to minimize Python-side copies.
    """
    df = df.astype("string").fillna("")
    header = sep_cell.join(map(str, df.columns.tolist())) if include_header else ""
    texts = []
    if include_header:
        for row in df.itertuples(index=False, name=None):
            base = sep_cell.join(row)
            texts.append((header + " " + base).strip())
    else:
        for row in df.itertuples(index=False, name=None):
            texts.append(sep_cell.join(row))
    return texts


# -----------------------------
# Normalization and batch auto-tune helpers
# -----------------------------

def _torch_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _auto_tune_batch(model, sample_texts, target_device, base_bs=256, amp=False):
    """
    Find the largest batch_size that fits without OOM for model/text.
    Binary search in [1, base_bs].
    """
    if len(sample_texts) == 0:
        return 1
    bs = min(base_bs, len(sample_texts))
    lo, hi, best = 1, bs, 1

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            with torch.inference_mode():
                if amp and target_device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _ = model.encode(sample_texts[:mid],
                                         batch_size=mid,
                                         convert_to_tensor=True,
                                         device=target_device,
                                         show_progress_bar=False,
                                         normalize_embeddings=False)
                else:
                    _ = model.encode(sample_texts[:mid],
                                     batch_size=mid,
                                     convert_to_tensor=True,
                                     device=target_device,
                                     show_progress_bar=False,
                                     normalize_embeddings=False)
            best = mid
            lo = mid + 1
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                hi = mid - 1
            else:
                hi = mid - 1
    return max(1, best)


# -----------------------------
# Optimized and safe encoding
# -----------------------------

def encode_with_memory_safety(
    model,
    texts: List[str],
    batch_size: int = 8,
    normalize: bool = True,
    amp: bool = False
) -> np.ndarray:
    """
    - Process batches on GPU (convert_to_tensor=True), normalize on GPU.
    - Auto-tune max batch_size without OOM.
    - Retry with batch/2 when OOM appears.
    - Return float32 (full precision). With amp=True, forward uses fp16/bf16 autocast,
      but output is normalized and stored in fp32.
    """
    if len(texts) == 0:
        return np.zeros((0, 1), dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tuned_bs = _auto_tune_batch(model, texts, device, base_bs=max(batch_size, 32), amp=amp)
    bs = max(batch_size, tuned_bs)

    out_list = []
    start = time.time()
    total = len(texts)
    i = 0

    with torch.inference_mode():
        while i < total:
            j = min(i + bs, total)
            batch = texts[i:j]
            try:
                if amp and device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        emb = model.encode(
                            batch,
                            batch_size=len(batch),
                            convert_to_tensor=True,
                            device=device,
                            show_progress_bar=False,
                            normalize_embeddings=False
                        )
                else:
                    emb = model.encode(
                        batch,
                        batch_size=len(batch),
                        convert_to_tensor=True,
                        device=device,
                        show_progress_bar=False,
                        normalize_embeddings=False
                    )
                # Normalize on GPU
                if normalize:
                    emb = _torch_l2norm(emb, dim=1)
                out_list.append(emb.detach().cpu())

                i = j
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and bs > 1:
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    continue
                raise

            pct = int(100 * i / total)
            #print(f"\r🔄 Encoding rows: {pct}% complete (bs={bs})", end="", flush=True)

    elapsed = time.time() - start
    #print(f"\n✅ Encoding completed in {elapsed:.2f} s ({elapsed/60:.2f} min)")

    X = torch.cat(out_list, dim=0).contiguous()  # CPU
    return X.numpy().astype(np.float32)


# -----------------------------
# Pickle helpers
# -----------------------------

def load_existing_pickle(pickle_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(pickle_path):
        return None
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Dict[str, Any], pickle_path: str):
    print(f"💾 Saving pickle to {pickle_path} ...")
    tmp = pickle_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, pickle_path)


def ensure_meta(container: Dict[str, Any], model_name: str, emb_dim: int, args: Optional[argparse.Namespace] = None):
    if "meta" not in container:
        container["meta"] = {}
    container["meta"].update({
        "model_name": model_name,
        "embedding_dim": int(emb_dim),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "args": vars(args) if args is not None else container["meta"].get("args", {}),
        "schema": "row_embeddings_v1"
    })
    if "tables" not in container:
        container["tables"] = {}


# -----------------------------
# Core processing
# -----------------------------

def process_tables_to_row_embeddings(
    input_dir: str,
    files: List[str],
    model_name: str,
    batch_size: int,
    out_pickle: str,
    include_header: bool = False,
    sep_cell: str = " ",
    store_float16: bool = False,
    add_table_vector: bool = True,
    overwrite: bool = False,
    save_every: int = 10,
    amp: bool = False,
) -> Dict[str, Any]:

    # Model
    model, _, emb_dim = get_model(model_name)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #try:
    #    model = model.to(device)
    #except Exception:
    #    pass

    # Resume from existing pickle if available
    container = load_existing_pickle(out_pickle) or {}
    ensure_meta(container, model_name, emb_dim)

    processed, skipped = 0, 0
    t0 = time.time()

    for file in tqdm(files, desc=f"Rows->Embeddings ({model_name})"):
        file_path = os.path.join(input_dir, file)
        table_key = file

        if not overwrite and table_key in container.get("tables", {}):
            skipped += 1
            continue

        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                encoding = result['encoding'] or 'utf-8'
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            df = clean_table(df)
            if df.empty:
                continue

            row_texts = table_rows_to_texts(df, include_header=include_header, sep_cell=sep_cell)
            if len(row_texts) == 0:
                continue

            row_embs = encode_with_memory_safety(
                model,
                row_texts,
                batch_size=batch_size,
                normalize=True,
                amp=amp
            )
            table_vec = (row_embs.mean(axis=0) / (np.linalg.norm(row_embs.mean(axis=0)) + 1e-12)).astype(np.float32) if add_table_vector else None

            # Reduce size if requested
            if store_float16:
                row_embs_to_store = row_embs.astype(np.float16)
                table_vec_to_store = None if table_vec is None else table_vec.astype(np.float16)
            else:
                row_embs_to_store = row_embs.astype(np.float32)
                table_vec_to_store = table_vec

            entry = {
                "table_id": os.path.splitext(file)[0],
                "file_name": file,
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "columns": df.columns.tolist(),
                "row_text_mode": {"include_header": include_header, "sep_cell": sep_cell},
                "row_embeddings": row_embs_to_store,   # (n_rows, d)
            }
            if add_table_vector:
                entry["embedding_table_mean_rows"] = table_vec_to_store  # (d,)

            container["tables"][table_key] = entry
            processed += 1

            if processed % save_every == 0:
                save_pickle(container, out_pickle)

        except Exception:
            print(f"\n❌ Error processing: {file}")
            traceback.print_exc()

    save_pickle(container, out_pickle)
    elapsed = time.time() - t0
    print(f"\n✅ Done. Processed: {processed}, skipped (resume): {skipped}. Time: {elapsed:.2f}s")
    return container


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate one embedding per ROW for each table and store it in a .pickle (GPU-optimized)")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with CSVs (each CSV = 1 table)')
    parser.add_argument('--out_pickle', type=str, required=True, help='Output .pickle')
    parser.add_argument('-m', '--model', default='all-MiniLM-L12-v2',
                        choices=['all-MiniLM-L12-v2', 'qwen', 'snowflake', 'jasper', 'bge-large', 'all-mpnet-base-v2'],
                        help='Model via utils.get_model')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Initial batch size (then auto-tuned to the largest possible)')
    parser.add_argument('--include_header', action='store_true', help='Prepend header to each row text')
    parser.add_argument('--sep_cell', type=str, default=' ', help='Separator between cells when converting rows to text')
    parser.add_argument('--float16', action='store_true', help='Store embeddings as float16 (smaller size)')
    parser.add_argument('--no_table_vector', action='store_true', help='Do NOT store aggregated table embedding')
    parser.add_argument('--overwrite', action='store_true', help='Recompute even if table already exists in pickle')
    parser.add_argument('--limit_files', type=int, default=0, help='Limit number of files (0=all)')
    parser.add_argument('--save_every', type=int, default=10, help='Save to disk every N tables')
    parser.add_argument('--amp', action='store_true', help='Use autocast (fp16/bf16) to speed up forward pass (final output in fp32)')
    args = parser.parse_args()

    # File list
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.csv')]
    files.sort()
    if args.limit_files and args.limit_files > 0:
        files = files[:args.limit_files]

    # Write metadata upfront
    container = load_existing_pickle(args.out_pickle) or {}
    model_for_meta, _, emb_dim = get_model(args.model)
    ensure_meta(container, args.model, emb_dim, args=args)
    save_pickle(container, args.out_pickle)

    process_tables_to_row_embeddings(
        input_dir=args.input_dir,
        files=files,
        model_name=args.model,
        batch_size=args.batch_size,
        out_pickle=args.out_pickle,
        include_header=args.include_header,
        sep_cell=args.sep_cell,
        store_float16=args.float16,
        add_table_vector=(not args.no_table_vector),
        overwrite=args.overwrite,
        save_every=args.save_every,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
