"""Retrieval accuracy and latency on the shipped core pack.

BM25-only runs in the default suite; the hybrid (real embedding model) run is ``-m live``.
Thresholds are regression guards: raise them as content improves, never lower them silently.
"""

import time
from pathlib import Path
from statistics import median

import pytest
import yaml

from kudiguard.knowledge.embeddings import Embedder
from kudiguard.knowledge.file_pack import FilePackSource
from kudiguard.knowledge.models import Query

ROOT = Path(__file__).resolve().parents[2]
CASES = yaml.safe_load((ROOT / "data" / "eval" / "retrieval.yaml").read_text(encoding="utf-8"))


async def evaluate(embedder: Embedder | None) -> tuple[float, float, float]:
    from kudiguard.knowledge.service import KnowledgeBase, Limits

    core = FilePackSource.load(ROOT / "data" / "packs" / "kudiguard-core", embedder=embedder)
    kb = KnowledgeBase([core], embedder=embedder, limits=Limits(flag_entries=0, hits=3))
    top1 = top3 = 0
    timings = []
    for case in CASES:
        start = time.perf_counter()
        hits = [s.entry.id for s in (await kb.search(Query(text=case["q"], limit=3))).entries]
        timings.append(time.perf_counter() - start)
        top1 += bool(hits) and hits[0] in case["want"]
        top3 += any(h in case["want"] for h in hits)
        if not (hits and hits[0] in case["want"]):
            print(f"  miss: {case['q']!r} -> {hits} (want {case['want']})")
    n = len(CASES)
    return top1 / n, top3 / n, median(timings) * 1000


async def test_bm25_accuracy_and_latency() -> None:
    top1, top3, p50_ms = await evaluate(None)
    print(f"\nBM25: top1={top1:.0%} top3={top3:.0%} p50={p50_ms:.2f}ms")
    assert top1 >= 0.78
    assert top3 >= 0.85
    assert p50_ms < 5


@pytest.mark.live
async def test_hybrid_accuracy_and_latency() -> None:
    from kudiguard.knowledge.embeddings import FastEmbedder

    top1, top3, p50_ms = await evaluate(FastEmbedder(cache_dir=ROOT / "models" / "fastembed"))
    print(f"\nHybrid: top1={top1:.0%} top3={top3:.0%} p50={p50_ms:.2f}ms")
    assert top1 >= 0.85
    assert top3 >= 0.90
    assert p50_ms < 30


async def evaluate_pipeline(embedder: Embedder | None) -> tuple[float, float]:
    """As production runs it: deterministic rules flag the message, then the knowledge base answers."""
    from kudiguard.knowledge.service import KnowledgeBase
    from kudiguard.languages import LanguageRegistry
    from kudiguard.safety.scam_rules import assess

    lexicon = LanguageRegistry.load(ROOT / "data" / "languages").lexicon_for("en")
    kb = KnowledgeBase(
        [FilePackSource.load(ROOT / "data" / "packs" / "kudiguard-core", embedder=embedder)],
        embedder=embedder,
    )
    top1 = top3 = 0
    for case in CASES:
        flags = assess([case["q"]], lexicon).flags
        ids = [s.entry.id for s in (await kb.search(Query(text=case["q"], flags=flags))).entries]
        top1 += bool(ids) and ids[0] in case["want"]
        top3 += any(i in case["want"] for i in ids[:3])
    return top1 / len(CASES), top3 / len(CASES)


async def test_pipeline_accuracy_bm25() -> None:
    top1, top3 = await evaluate_pipeline(None)
    print(f"\nPipeline (rules + BM25): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.78
    assert top3 >= 0.90


@pytest.mark.live
async def test_pipeline_accuracy_hybrid() -> None:
    from kudiguard.knowledge.embeddings import FastEmbedder

    top1, top3 = await evaluate_pipeline(FastEmbedder(cache_dir=ROOT / "models" / "fastembed"))
    print(f"\nPipeline (rules + hybrid): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.85
    assert top3 >= 0.92
