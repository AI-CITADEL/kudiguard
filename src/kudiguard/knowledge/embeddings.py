"""Optional dense embeddings for semantic search (install extra: ``kudiguard[semantic]``).

Queries are English pivot text, so a small English model suffices. Vectors are L2-normalised so a dot
product is cosine similarity; packs are small enough that brute-force numpy beats an ANN index.
Embedding is synchronous CPU work: async callers must run it in a thread (see ``KnowledgeBase``).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_THREADS = 4  # leave CPU for speech recognition, which also runs on CPU


class Embedder(Protocol):
    """Rows are L2-normalised (unit) vectors, so a dot product is cosine similarity."""

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Matrix: ...


def l2_normalize(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


class CachedEmbedder:
    """Remembers document vectors (pack facts repeat on every turn); queries pass straight through.

    Bounded LRU, safe to share between worker threads. Cache only shared content (pack facts, fixed text), never user
    messages or replies: embed those with ``inner`` directly.
    """

    def __init__(self, embedder: Embedder, max_items: int = 4096) -> None:
        if max_items < 1:
            raise ValueError(f"max_items must be positive, got {max_items}")
        self._embedder = embedder
        self._max_items = max_items
        self._vectors: OrderedDict[str, Matrix] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def inner(self) -> Embedder:
        return self._embedder

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            with self._lock:
                dim = next((len(v) for v in self._vectors.values()), 0)
            return np.zeros((0, dim), dtype=np.float32)
        with self._lock:
            found = {t: self._vectors[t] for t in texts if t in self._vectors}
            for text in found:
                self._vectors.move_to_end(text)
        missing = list(dict.fromkeys(t for t in texts if t not in found))
        if missing:  # embedded outside the lock: other threads keep reading
            found.update(zip(missing, self._embedder.embed_documents(missing), strict=True))
            with self._lock:
                for text in missing:
                    self._vectors[text] = found[text]
                    self._vectors.move_to_end(text)
                while len(self._vectors) > self._max_items:
                    self._vectors.popitem(last=False)
        return np.stack([found[t] for t in texts]).astype(np.float32, copy=False)

    def embed_query(self, text: str) -> Matrix:
        return self._embedder.embed_query(text)


class FastEmbedder:
    """ONNX (CPU) embeddings via ``fastembed``; no PyTorch, keeps the GPU free for N-ATLAS.

    In production pass ``local_files_only=True`` with a pre-populated ``cache_dir`` so startup never downloads.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        *,
        threads: int = DEFAULT_THREADS,
        local_files_only: bool = False,
    ) -> None:
        from fastembed import TextEmbedding  # optional dependency

        self.model_name = model_name  # checked against artefacts trained on it (safety.guard_classifier)
        self._model = TextEmbedding(
            model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
            threads=threads,
            local_files_only=local_files_only,
        )

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.asarray(list(self._model.embed(list(texts))), dtype=np.float32))

    def embed_query(self, text: str) -> Matrix:
        return l2_normalize(np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32))
