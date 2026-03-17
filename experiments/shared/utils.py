import csv
import os
import joblib
import numpy as np
import pandas as pd
import torch
import chardet
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

__all__ = [
    "random_words",
    "get_model",
    "find_delimiter",
    "load_clean_csv",
    "content_embeddings",
    "load_embeddings",
]

random_words = [
    "ability",
    "absence",
    "abundance",
    "acceptance",
    "accomplishment",
    "achievement",
    "adventure",
    "attention",
    "automation",
    "balance",
    "beauty",
    "belonging",
    "bravery",
    "calmness",
    "celebration",
    "clarity",
    "compassion",
    "confidence",
    "creativity",
    "curiosity",
    "determination",
    "discipline",
    "diversity",
    "dream",
    "education",
    "empathy",
    "energy",
    "enthusiasm",
    "freedom",
    "friendship",
    "growth",
    "harmony",
    "hope",
    "humility",
    "imagination",
    "innovation",
    "integrity",
    "justice",
    "kindness",
    "knowledge",
    "leadership",
    "liberty",
    "love",
    "loyalty",
    "motivation",
    "passion",
    "patience",
    "perseverance",
    "positivity",
    "resilience",
    "respect",
    "responsibility",
    "service",
    "success",
    "sustainability",
    "trust",
    "unity",
    "wisdom",
    "wonder",
]

_CACHE_FOLDER = "models"


def _load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, cache_dir=_CACHE_FOLDER)
    except Exception:
        return None


def _finalize_model(model, tokenizer, name_hint=None):
    if tokenizer is None and hasattr(model, "tokenizer"):
        tokenizer = getattr(model, "tokenizer", None)
    try:
        dimensions = int(model.get_sentence_embedding_dimension())
    except Exception:
        dimensions = None
    if hasattr(model, "max_seq_length") and dimensions:
        model.max_seq_length = max(dimensions, getattr(model, "max_seq_length", dimensions))
    return model, tokenizer, dimensions


def get_model(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = None
    model = None

    if name == "jasper":
        model = SentenceTransformer(
            "NovaSearch/jasper_en_vision_language_v1",
            trust_remote_code=True,
            cache_folder=_CACHE_FOLDER,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            },
            config_kwargs={"is_text_encoder": True, "vector_dim": 1024},
        )
    elif name in ("qwen2", "qwen"):
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True, cache_folder=_CACHE_FOLDER)
        tokenizer = _load_tokenizer(model_name)
        model.max_seq_length = 8192
    elif name in ("all-mini", "all-MiniLM-L12-v2"):
        model_name = "sentence-transformers/all-MiniLM-L12-v2"
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True, cache_folder=_CACHE_FOLDER)
        tokenizer = _load_tokenizer(model_name)
        model.max_seq_length = 512
    elif name in ("all-mpnet", "all-mpnet-base-v2"):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True, cache_folder=_CACHE_FOLDER)
        tokenizer = _load_tokenizer(model_name)
    elif name == "snowflake":
        model_name = "Snowflake/snowflake-arctic-embed-m-v2.0"
        model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder=_CACHE_FOLDER)
        tokenizer = _load_tokenizer(model_name)
    elif name == "bge-large":
        model_name = "BAAI/bge-large-en-v1.5"
        model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder=_CACHE_FOLDER)
        tokenizer = _load_tokenizer(model_name)
    else:
        raise ValueError(f"Unknown model name: {name}")

    return _finalize_model(model, tokenizer, name_hint=name)


def find_delimiter(filename: str):
    sniffer = csv.Sniffer()
    with open(filename, "r", encoding="utf-8", errors="ignore") as fp:
        sample = fp.read(5000)
        delimiter = sniffer.sniff(sample).delimiter
    return delimiter


def load_clean_csv(file_path: str):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(10000))
        encoding = result.get("encoding") or "utf-8"

    try:
        df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
    except Exception:
        return None

    df.dropna(axis="columns", how="all", inplace=True)
    df.dropna(how="all", inplace=True)

    if df.empty:
        return None

    return df


def _encode_text(model, text):
    emb = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    return emb


def content_embeddings(model, df, size, model_name, tokenizer=None, header=False, randow_row=False):
    all_embs = []
    for _, row in df.iterrows():
        if header:
            text = ", ".join(map(str, df.columns.tolist()))
        else:
            text = ", ".join(map(str, row.values.flatten().tolist()))

        if randow_row:
            text = ", ".join(map(str, row.values.flatten().tolist()))

        emb = _encode_text(model, text)
        all_embs.append(emb.reshape(1, -1))

    if not all_embs:
        return np.empty((0, size), dtype=np.float32)

    stacked = np.vstack(all_embs).astype(np.float32)
    return stacked


def load_embeddings(model_name, dataset, path, file_base):
    file_path = os.path.join(path, f"{file_base}.pkl")
    with open(file_path, "rb") as f:
        return joblib.load(f)
