"""FastEmbedder against a stand-in ``fastembed`` module: no model download, just our wrapper's contract."""

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from kudiguard.knowledge.embeddings import DEFAULT_MODEL, DEFAULT_THREADS, FastEmbedder


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
