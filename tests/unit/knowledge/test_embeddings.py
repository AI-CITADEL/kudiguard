"""FastEmbedder against a stand-in ``fastembed`` module: no model download, just our wrapper's contract."""

import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from kudiguard.knowledge.embeddings import DEFAULT_MODEL, DEFAULT_THREADS, CachedEmbedder, FastEmbedder


class FakeTextEmbedding:
    instances: ClassVar[list["FakeTextEmbedding"]] = []

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.documents: list[list[str]] = []
        FakeTextEmbedding.instances.append(self)

    def embed(self, texts: list[str]) -> Iterator[np.ndarray]:
        self.documents.append(texts)
        return (np.array([3.0, 4.0]) * (i + 1) for i, _ in enumerate(texts))  # a generator, like fastembed

    def query_embed(self, text: str) -> Iterator[np.ndarray]:
        return iter([np.array([0.0, 2.0])])


@pytest.fixture(autouse=True)
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeTextEmbedding.instances = []
    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)


def test_defaults_passed_to_fastembed() -> None:
    FastEmbedder()
    (model,) = FakeTextEmbedding.instances
    assert model.model_name == DEFAULT_MODEL
    assert model.kwargs == {"cache_dir": None, "threads": DEFAULT_THREADS, "local_files_only": False}


def test_offline_cache_dir_is_passed_as_string(tmp_path: Path) -> None:
    FastEmbedder("other/model", tmp_path, threads=1, local_files_only=True)
    (model,) = FakeTextEmbedding.instances
    assert model.model_name == "other/model"
    assert model.kwargs == {"cache_dir": str(tmp_path), "threads": 1, "local_files_only": True}


def test_documents_are_float32_unit_vectors_in_order() -> None:
    embedder = FastEmbedder()
    matrix = embedder.embed_documents(("a", "b"))
    assert matrix.dtype == np.float32 and matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-6)
    assert FakeTextEmbedding.instances[0].documents == [["a", "b"]]  # sequence converted to a list


def test_no_documents_skips_the_model() -> None:
    embedder = FastEmbedder()
    matrix = embedder.embed_documents([])
    assert matrix.shape == (0, 0) and matrix.dtype == np.float32
    assert FakeTextEmbedding.instances[0].documents == []


def test_query_is_a_normalised_vector() -> None:
    vector = FastEmbedder().embed_query("hello")
    assert vector.dtype == np.float32 and vector.shape == (2,)
    np.testing.assert_allclose(vector, [0.0, 1.0])


# --- CachedEmbedder ---------------------------------------------------------------------------------------------


class Counting:
    """Embeds "a", "b", ... as distinct unit vectors and records each batch."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.batches.append(list(texts))
        return np.stack([np.eye(8, dtype=np.float32)[ord(t[0]) - ord("a")] for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return np.ones(8, dtype=np.float32)


def test_cache_embeds_only_new_texts_once_and_keeps_order() -> None:
    inner = Counting()
    cached = CachedEmbedder(inner)
    first = cached.embed_documents(["a", "b", "a"])
    assert inner.batches == [["a", "b"]]  # deduplicated
    np.testing.assert_array_equal(first, np.eye(8, dtype=np.float32)[[0, 1, 0]])
    second = cached.embed_documents(["c", "b"])
    assert inner.batches[-1] == ["c"] and second.dtype == np.float32
    np.testing.assert_array_equal(second, np.eye(8, dtype=np.float32)[[2, 1]])
    cached.embed_documents(["a", "b", "c"])
    assert len(inner.batches) == 2  # all cached
    np.testing.assert_array_equal(cached.embed_query("x"), np.ones(8))  # queries pass through


def test_cache_evicts_least_recently_used() -> None:
    inner = Counting()
    cached = CachedEmbedder(inner, max_items=2)
    cached.embed_documents(["a", "b"])
    cached.embed_documents(["a"])  # a is now the most recent
    cached.embed_documents(["c"])  # evicts b
    cached.embed_documents(["a", "b"])
    assert inner.batches == [["a", "b"], ["c"], ["b"]]
    assert cached.embed_documents(["d", "e", "f"]).shape == (
        3,
        8,
    )  # a batch larger than the cache still works


def test_cache_edge_cases() -> None:
    inner = Counting()
    assert CachedEmbedder(inner).embed_documents([]).shape == (0, 0) and inner.batches == []
    with pytest.raises(ValueError, match="max_items"):
        CachedEmbedder(inner, max_items=0)
