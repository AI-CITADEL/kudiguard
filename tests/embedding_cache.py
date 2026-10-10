"""The real embedding model for live tests, from a local cache only: tests never download models."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kudiguard.knowledge.embeddings import FastEmbedder

ROOT = Path(__file__).resolve().parents[1]


def local_fast_embedder() -> FastEmbedder:
    """``$KUDIGUARD_EMBEDDINGS_CACHE`` (as in production settings) or ``models/fastembed``; skips if absent."""
    cache = Path(os.environ.get("KUDIGUARD_EMBEDDINGS_CACHE") or ROOT / "models" / "fastembed")
    if not cache.is_dir():
        pytest.skip(f"no local embedding cache at {cache} (set KUDIGUARD_EMBEDDINGS_CACHE)")
    return FastEmbedder(cache_dir=cache, local_files_only=True)
