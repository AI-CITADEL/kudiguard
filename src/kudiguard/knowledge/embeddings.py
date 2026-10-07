"""Optional dense embeddings for semantic search (install extra: ``kudiguard[semantic]``).

Queries are English pivot text, so a small English model suffices. Vectors are L2-normalised so a dot
product is cosine similarity; packs are small enough that brute-force numpy beats an ANN index.
Embedding is synchronous CPU work: async callers must run it in a thread (see ``KnowledgeBase``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_THREADS = 4  # leave CPU for speech recognition, which also runs on CPU


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Matrix: ...


def l2_normalize(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


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
