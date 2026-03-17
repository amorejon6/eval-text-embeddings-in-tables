import argparse
import random
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

# Optional: joblib for parallel prebuild
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# Optional: faiss for very fast exact top-k IP search
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from experiments.shared.utils import get_model


# -----------------------------
# Utils
# -----------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def stable_int(s: str) -> int:
    """Stable (reproducible) hash -> int."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n


def encode_texts(model, texts: List[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            e = model.encode(batch, batch_size=batch_size, normalize_embeddings=normalize)
        except TypeError:
            e = model.encode(batch, batch_size=batch_size)
            e = np.asarray(e)
            if normalize:
                e = l2norm(e, axis=1)
        out.append(np.asarray(e))
    X = np.vstack(out).astype(np.float32)
    if normalize:
        X = l2norm(X, axis=1)
    return X


# -----------------------------
# Metrics (Hit@k semantics + nDCG@10 + MRR@K)
# -----------------------------

def dcg_at_k(relevances: List[int], k: int) -> float:
    rel = np.asarray(relevances[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return 0.0 if ideal == 0.0 else float(dcg / ideal)


def mrr_at_k(relevances: List[int], k: int) -> float:
    """MRR@K: first relevant item inside top-K, otherwise 0."""
    for i, r in enumerate(relevances[:k], start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def hit_at_k(relevances: List[int], k: int) -> float:
    return 1.0 if any(r > 0 for r in relevances[:k]) else 0.0


# -----------------------------
# Approaches operating on a given set of rows
# -----------------------------

def vec_mean(rows: np.ndarray) -> np.ndarray:
    v = rows.mean(axis=0)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def vec_random10(rows: np.ndarray, rng: random.Random) -> np.ndarray:
    n = rows.shape[0]
    k = max(1, int(round(0.10 * n)))
    idx = rng.sample(range(n), k) if n > 1 else [0]
    v = rows[idx].mean(axis=0)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def vec_cluster10_fast(
    rows: np.ndarray,
    seed: int,
    k_cap: int = 8,
    batch_cap: int = 256,
    max_iter: int = 20
) -> np.ndarray:
    """
    Fast version of 'cluster10':
    - MiniBatchKMeans
    - aggressive cluster cap
    - n_init=1
    - low max_iter
    """
    n = rows.shape[0]
    k = max(1, int(round(0.10 * n)))
    k = min(k, n, k_cap)

    if k == 1:
        v = rows.mean(axis=0)
        return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(batch_cap, n),
        n_init=1,
        max_iter=max_iter
    )
    km.fit(rows)
    v = km.cluster_centers_.mean(axis=0)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def build_vector(
    rows: np.ndarray,
    approach: str,
    seed: int,
    rng: Optional[random.Random] = None,
    cluster_k_cap: int = 8
) -> np.ndarray:
    if approach == "baseline":
        return vec_mean(rows)
    elif approach == "random10":
        rng = rng or random.Random(seed)
        return vec_random10(rows, rng)
    elif approach == "cluster10":
        return vec_cluster10_fast(rows, seed=seed, k_cap=cluster_k_cap)
    else:
        raise ValueError(f"Unknown approach: {approach}")


# -----------------------------
# Row-split (disjoint) helpers
# -----------------------------

def disjoint_row_split(n_rows: int, seed: int, key: str, q_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic (reproducible) split using a stable hash.
    """
    rng = random.Random(seed + stable_int(key))
    idxs = list(range(n_rows))
    rng.shuffle(idxs)
    q_count = max(1, int(round(q_ratio * n_rows)))
    q_set = set(idxs[:q_count])
    q_idx = np.array(sorted(q_set), dtype=int)
    i_idx = np.array(sorted(set(idxs) - q_set), dtype=int)
    if i_idx.size == 0:
        i_idx = np.array([idxs[-1]], dtype=int)
        if i_idx[0] in q_set and len(idxs) > 1:
            i_idx = np.array([idxs[-2]], dtype=int)
    return q_idx, i_idx


# -----------------------------
# Fast retrieval helpers (FAISS + fallback)
# -----------------------------

def build_faiss_ip_index(index_mat: np.ndarray):
    """
    Very fast exact top-k with IndexFlatIP.
    """
    dim = index_mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(index_mat.astype(np.float32))
    return index


def search_topk(
    index_mat: np.ndarray,
    qmat: np.ndarray,
    top_k: int,
    use_faiss: bool = True,
    block_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (I, D):
      - I: top_k indices per query (shape: [nq, top_k])
      - D: corresponding scores (shape: [nq, top_k])

    If FAISS is available and use_faiss=True -> exact FAISS IP search.
    Otherwise -> block-based fallback with argpartition.
    """
    qmat = qmat.astype(np.float32)
    index_mat = index_mat.astype(np.float32)
    top_k = max(1, min(top_k, index_mat.shape[0]))

    if use_faiss and _HAS_FAISS:
        idx = build_faiss_ip_index(index_mat)
        D, I = idx.search(qmat, top_k)
        return I, D

    # Block-based fallback (slower, but avoids a huge matrix)
    nq = qmat.shape[0]
    n = index_mat.shape[0]
    I_out = np.empty((nq, top_k), dtype=np.int64)
    D_out = np.empty((nq, top_k), dtype=np.float32)

    # Process query-by-query, but still using BLAS matmul against index_mat.T.
    # This can be further batched if needed.
    for qi in range(nq):
        q = qmat[qi]
        sims = index_mat @ q  # (n,)

        # Fast top-k
        top_idx = np.argpartition(-sims, kth=top_k - 1)[:top_k]
        top_sorted = top_idx[np.argsort(-sims[top_idx])]

        I_out[qi] = top_sorted
        D_out[qi] = sims[top_sorted].astype(np.float32)

    return I_out, D_out


# -----------------------------
# Main evaluation
# -----------------------------

def evaluate_from_row_pickle_disjoint_fast(
    pickle_path: Path,
    out_dir: Path,
    approaches: List[str],
    seed: int = 42,
    limit_tables: int = 0,
    q_tables_ratio: float = 0.10,
    q_rows_ratio: float = 0.10,
    colnames_model_override: Optional[str] = None,
    save_ranks: bool = False,
    top_k: int = 50,
    cluster_k_cap: int = 8,
    jobs: int = -1,
    use_faiss: bool = True,
):
    """
    Optimized version:
    - Top-k retrieval (FAISS recommended)
    - Metrics computed on top-k: P@1/5/10 (Hit@k), nDCG@10, MRR@K
    - Faster cluster10
    - Parallel prebuild (if joblib is available)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    all_items = list(data["tables"].items())
    all_items.sort(key=lambda kv: kv[0])  # by filename/key
    if limit_tables and limit_tables > 0:
        all_items = all_items[:limit_tables]

    set_all_seeds(seed)

    n_tables = len(all_items)
    tbl_idxs = list(range(n_tables))
    random.shuffle(tbl_idxs)
    qtbl_count = max(1, int(round(q_tables_ratio * n_tables)))
    qtbl_set = set(tbl_idxs[:qtbl_count])

    # Model for colnames_query
    model_for_cols = None
    if "colnames_query" in approaches:
        model_name = colnames_model_override or data.get("meta", {}).get("model_name", None)
        if not model_name:
            raise RuntimeError("No model name in pickle meta. Use --colnames_model_override.")
        model_for_cols, _, _ = get_model(model_name)

    # Cache for column-name embeddings (avoids recomputation)
    col_cache: Dict[str, np.ndarray] = {}

    # Helper: build index vector for one table
    def _build_index_vec(ap: str, i: int, fname: str, entry: dict) -> Tuple[str, np.ndarray]:
        rows = np.asarray(entry["row_embeddings"], dtype=np.float32)

        if i in qtbl_set:
            q_idx, i_idx = disjoint_row_split(rows.shape[0], seed, entry["table_id"], q_rows_ratio)
            idx_rows = rows[i_idx] if i_idx.size > 0 else rows
        else:
            idx_rows = rows

        if ap == "colnames_query":
            v = vec_mean(idx_rows)  # baseline-like
        else:
            rng = random.Random(seed + stable_int(entry["table_id"]))
            v = build_vector(idx_rows, ap, seed=seed, rng=rng, cluster_k_cap=cluster_k_cap)

        return entry["table_id"], v

    # Precompute index vectors per approach
    prebuilt_index: Dict[str, Dict[str, np.ndarray]] = {ap: {} for ap in approaches}

    for ap in approaches:
        if _HAS_JOBLIB and jobs != 1:
            n_jobs = jobs
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_build_index_vec)(ap, i, fname, entry)
                for i, (fname, entry) in enumerate(all_items)
            )
            prebuilt_index[ap] = dict(results)
        else:
            dct = {}
            for i, (fname, entry) in enumerate(all_items):
                tid, v = _build_index_vec(ap, i, fname, entry)
                dct[tid] = v
            prebuilt_index[ap] = dct

    # IDs in fixed order (index rows)
    ids = [entry["table_id"] for _, entry in all_items]
    id_to_pos = {tid: idx for idx, tid in enumerate(ids)}

    # Evaluate each approach
    for ap in approaches:
        index_mat = np.vstack([prebuilt_index[ap][tid] for tid in ids]).astype(np.float32)
        index_mat = l2norm(index_mat, axis=1)

        per_query_rows = []
        ranks_rows = []

        # Build ALL query vectors (so we can search in batches)
        q_ids = []
        q_vecs = []

        for i in range(n_tables):
            if i not in qtbl_set:
                continue
            _, entry = all_items[i]
            rows = np.asarray(entry["row_embeddings"], dtype=np.float32)

            if ap == "colnames_query":
                cols_txt = " ".join(entry.get("columns", [])) if entry.get("columns") else ""
                cols_txt = cols_txt.strip()

                if cols_txt:
                    if cols_txt in col_cache:
                        qvec = col_cache[cols_txt]
                    else:
                        qvec = encode_texts(model_for_cols, [cols_txt], batch_size=1, normalize=True)[0]
                        qvec = qvec.astype(np.float32)
                        col_cache[cols_txt] = qvec
                else:
                    q_idx, _ = disjoint_row_split(rows.shape[0], seed, entry["table_id"], q_rows_ratio)
                    qvec = vec_mean(rows[q_idx])
            else:
                q_idx, _ = disjoint_row_split(rows.shape[0], seed, entry["table_id"], q_rows_ratio)
                rng = random.Random(seed + 7 + stable_int(entry["table_id"]))
                qvec = build_vector(rows[q_idx], ap, seed=seed, rng=rng, cluster_k_cap=cluster_k_cap)

            qvec = l2norm(qvec.reshape(1, -1), axis=1)[0].astype(np.float32)

            q_ids.append(entry["table_id"])
            q_vecs.append(qvec)

        if not q_vecs:
            continue

        qmat = np.vstack(q_vecs).astype(np.float32)

        # top-k search (FAISS if available)
        # If save_ranks -> enforce top_k>=50 to store top-50
        requested_top_k = max(top_k, 50) if save_ranks else top_k
        effective_top_k = max(1, min(requested_top_k, index_mat.shape[0]))

        I, D = search_topk(
            index_mat=index_mat,
            qmat=qmat,
            top_k=effective_top_k,
            use_faiss=use_faiss,
        )

        # Per-query metrics (on top-K)
        for qi, qid in enumerate(q_ids):
            top_idx = I[qi].tolist()
            ranked_ids = [ids[j] for j in top_idx]
            ranked_scores = D[qi].astype(float).tolist()

            relevances = [1 if rid == qid else 0 for rid in ranked_ids]

            mrr = mrr_at_k(relevances, effective_top_k)  # MRR@K
            ndcg10 = ndcg_at_k(relevances, 10)
            p1 = hit_at_k(relevances, 1)
            p5 = hit_at_k(relevances, 5)
            p10 = hit_at_k(relevances, 10)

            per_query_rows.append({
                "approach": ap,
                "query_table_id": qid,
                f"MRR@{effective_top_k}": mrr,
                "nDCG@10": ndcg10,
                "P@1": p1,
                "P@5": p5,
                "P@10": p10,
            })

            if save_ranks:
                ranks_rows.append({
                    "approach": ap,
                    "query_table_id": qid,
                    "ranked_ids": ranked_ids[:50],
                    "ranked_scores": ranked_scores[:50],
                })

        df = pd.DataFrame(per_query_rows)
        df.to_csv(out_dir / f"perquery_disjoint_{Path(pickle_path).stem}_{ap}.csv", index=False)

        if save_ranks and ranks_rows:
            df_r = pd.DataFrame(ranks_rows)
            df_r.to_json(
                out_dir / f"ranks_disjoint_{Path(pickle_path).stem}_{ap}.jsonl",
                orient="records",
                lines=True
            )

    # Merge and summarize
    perquery_files = [out_dir / f"perquery_disjoint_{Path(pickle_path).stem}_{ap}.csv" for ap in approaches]
    frames = [pd.read_csv(p) for p in perquery_files if p.exists()]
    if frames:
        big = pd.concat(frames, ignore_index=True)
        big.to_csv(out_dir / f"all_per_query_disjoint_{Path(pickle_path).stem}.csv", index=False)

        # Detect MRR@K column (depends on effective_top_k; usually top_k or 50)
        mrr_cols = [c for c in big.columns if c.startswith("MRR@")]
        agg_dict = {
            "nDCG@10": "mean",
            "P@1": "mean",
            "P@5": "mean",
            "P@10": "mean",
        }
        for c in mrr_cols:
            agg_dict[c] = "mean"

        summary = big.groupby("approach").agg(agg_dict).reset_index()
        summary.to_csv(out_dir / f"summary_disjoint_{Path(pickle_path).stem}.csv", index=False)


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fast disjoint evaluation (top-k) for large table collections: FAISS + optimized cluster10 + @K metrics."
    )
    p.add_argument("--pickle", type=Path, required=True, help="Path to the .pickle with row embeddings")
    p.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--approaches", nargs="+",
        default=["baseline", "random10", "cluster10", "colnames_query"],
        choices=["baseline", "random10", "cluster10", "colnames_query"],
        help="Approaches to evaluate"
    )
    p.add_argument("--seed", type=int, default=42, help="Global seed")
    p.add_argument("--limit_tables", type=int, default=0, help="Limit number of tables for quick tests")
    p.add_argument("--q_tables_ratio", type=float, default=0.10, help="Fraction of tables used as queries")
    p.add_argument("--q_rows_ratio", type=float, default=0.10, help="Fraction of rows used to build QUERY vectors (disjoint)")
    p.add_argument("--colnames_model_override", type=str, default=None, help="Model for 'colnames_query'")
    p.add_argument("--save_ranks", action="store_true", help="Save top-50 rankings per query in JSONL")

    # Optimizations
    p.add_argument("--top_k", type=int, default=50, help="Top-K for evaluation (MRR@K and rankings)")
    p.add_argument("--cluster_k_cap", type=int, default=8, help="Cluster cap for cluster10 (recommended 6-10)")
    p.add_argument("--jobs", type=int, default=-1, help="n_jobs for prebuild (joblib). Use 1 to disable parallelism.")
    p.add_argument("--no_faiss", action="store_true", help="Disable FAISS even if installed (use fallback)")

    return p.parse_args()


def main():
    args = parse_args()
    evaluate_from_row_pickle_disjoint_fast(
        pickle_path=args.pickle,
        out_dir=args.out_dir,
        approaches=args.approaches,
        seed=args.seed,
        limit_tables=args.limit_tables,
        q_tables_ratio=args.q_tables_ratio,
        q_rows_ratio=args.q_rows_ratio,
        colnames_model_override=args.colnames_model_override,
        save_ranks=args.save_ranks,
        top_k=args.top_k,
        cluster_k_cap=args.cluster_k_cap,
        jobs=args.jobs,
        use_faiss=(not args.no_faiss),
    )


if __name__ == "__main__":
    main()
