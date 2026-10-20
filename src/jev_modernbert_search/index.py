from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .search import ModernBertSearch


def _fingerprint(model_name: str, documents: list[str]) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for document in documents:
        digest.update(b"\0")
        digest.update(document.encode("utf-8"))
    return digest.hexdigest()


def load_or_build_index(
    model: ModernBertSearch,
    model_name: str,
    documents: list[str],
    cache_path: Path,
) -> np.ndarray:
    fingerprint = _fingerprint(model_name, documents)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_fingerprint = str(cached["fingerprint"].item())
            cached_documents = cached["documents"].tolist()
            embeddings = cached["embeddings"]
        if cached_fingerprint == fingerprint and cached_documents == documents:
            if embeddings.ndim == 2 and embeddings.shape[0] == len(documents):
                return embeddings

    embeddings = model.encode(documents)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("wb") as destination:
        np.savez_compressed(
            destination,
            fingerprint=np.asarray(fingerprint),
            documents=np.asarray(documents, dtype=np.str_),
            embeddings=embeddings,
        )
    temporary_path.replace(cache_path)
    return embeddings
