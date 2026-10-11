"""Retrieval accuracy and latency on the shipped core pack.

BM25-only runs in the default suite; the hybrid (real embedding model) run is ``-m live``.
Thresholds are regression guards: raise them as content improves, never lower them silently.
"""

import time
from pathlib import Path
from statistics import median

import pytest
import yaml

from embedding_cache import local_fast_embedder
from kudiguard.knowledge.embeddings import Embedder
from kudiguard.knowledge.file_pack import FilePackSource
from kudiguard.knowledge.models import Query

ROOT = Path(__file__).resolve().parents[2]
CASES = yaml.safe_load((ROOT / "data" / "eval" / "retrieval.yaml").read_text(encoding="utf-8"))
# Written blind by an independent author (no access to pack content) and frozen: never tune content on it.
BLIND = yaml.safe_load((ROOT / "data" / "eval" / "retrieval_blind.yaml").read_text(encoding="utf-8"))
MONEY = yaml.safe_load((ROOT / "data" / "eval" / "retrieval_money.yaml").read_text(encoding="utf-8"))
CORE = ("kudiguard-core",)
# What production loads by default: every enabled, non-sample pack.
PRODUCTION = tuple(
    d.name
    for d in sorted((ROOT / "data" / "packs").iterdir())
    if (d / "pack.yaml").is_file()
    and (m := yaml.safe_load((d / "pack.yaml").read_text(encoding="utf-8"))).get("enabled", True)
    and not m.get("sample", False)
)
# The pack set the blind labels were frozen with (2026-10-10): its floors stay measured on it, since the labels
# predate genuine-notices and situations (all-pack numbers are reported only).
FROZEN_PACKS = ("kudiguard-core", "money-skills")


async def evaluate(embedder: Embedder | None, cases: list[dict] = CASES) -> tuple[float, float, float]:  # type: ignore[type-arg]
    from kudiguard.knowledge.service import KnowledgeBase, Limits

    core = FilePackSource.load(ROOT / "data" / "packs" / "kudiguard-core", embedder=embedder)
    kb = KnowledgeBase([core], embedder=embedder, limits=Limits(flag_entries=0, hits=3))
    top1 = top3 = 0
    timings = []
    for case in cases:
        start = time.perf_counter()
        hits = [s.entry.id for s in (await kb.search(Query(text=case["q"], limit=3))).entries]
        timings.append(time.perf_counter() - start)
        top1 += bool(hits) and hits[0] in case["want"]
        top3 += any(h in case["want"] for h in hits)
        if not (hits and hits[0] in case["want"]):
            print(f"  miss: {case['q']!r} -> {hits} (want {case['want']})")
    n = len(cases)
    return top1 / n, top3 / n, median(timings) * 1000


async def test_bm25_accuracy_and_latency() -> None:
    top1, top3, p50_ms = await evaluate(None)
    print(f"\nBM25: top1={top1:.0%} top3={top3:.0%} p50={p50_ms:.2f}ms")
    assert top1 >= 0.78
    assert top3 >= 0.85
    assert p50_ms < 5


@pytest.mark.live
async def test_hybrid_accuracy_and_latency() -> None:

    top1, top3, p50_ms = await evaluate(local_fast_embedder())
    print(f"\nHybrid: top1={top1:.0%} top3={top3:.0%} p50={p50_ms:.2f}ms")
    assert top1 >= 0.85
    assert top3 >= 0.90
    assert p50_ms < 30


async def evaluate_pipeline(
    embedder: Embedder | None,
    cases: list[dict] = CASES,  # type: ignore[type-arg]
    packs: tuple[str, ...] = CORE,
) -> tuple[float, float]:
    """As production runs it: deterministic rules flag the message, then the knowledge base answers."""
    from kudiguard.knowledge.service import KnowledgeBase
    from kudiguard.languages import LanguageRegistry
    from kudiguard.safety.scam_rules import assess

    lexicon = LanguageRegistry.load(ROOT / "data" / "languages").lexicon_for("en")
    kb = KnowledgeBase(
        [FilePackSource.load(ROOT / "data" / "packs" / pack, embedder=embedder) for pack in packs],
        embedder=embedder,
    )
    top1 = top3 = 0
    for case in cases:
        flags = assess([case["q"]], lexicon).flags
        ids = [s.entry.id for s in (await kb.search(Query(text=case["q"], flags=flags))).entries]
        top1 += bool(ids) and ids[0] in case["want"]
        top3 += any(i in case["want"] for i in ids[:3])
    return top1 / len(cases), top3 / len(cases)


async def test_pipeline_accuracy_bm25() -> None:
    top1, top3 = await evaluate_pipeline(None)
    print(f"\nPipeline (rules + BM25): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.78
    assert top3 >= 0.90


@pytest.mark.live
async def test_pipeline_accuracy_hybrid() -> None:

    top1, top3 = await evaluate_pipeline(local_fast_embedder())
    print(f"\nPipeline (rules + hybrid): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.85
    assert top3 >= 0.92


async def test_blind_set_pipeline_bm25() -> None:
    """Frozen blind set (independent author): measured 67% top-1 / 79% top-3 on 2026-10-08. Guard, never tune."""
    top1, top3 = await evaluate_pipeline(None, BLIND)
    print(f"\nBlind pipeline (BM25): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.6
    assert top3 >= 0.75


@pytest.mark.live
async def test_blind_set_pipeline_hybrid() -> None:
    """Measured 67% top-1 / 96% top-3 on 2026-10-08 (compose is grounded on the top 3)."""

    top1, top3 = await evaluate_pipeline(local_fast_embedder(), BLIND)
    print(f"\nBlind pipeline (hybrid): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.6
    assert top3 >= 0.9


# --- production pack set (core + money-skills) ----------------------------------------------------------


def test_production_pack_set() -> None:
    assert PRODUCTION == ("genuine-notices", "kudiguard-core", "money-skills", "situations")


async def test_money_skills_reachable_bm25() -> None:
    """Dev set (written with the pack, not blind): every entry must stay reachable."""
    top1, top3 = await evaluate_pipeline(None, MONEY, PRODUCTION)
    print(f"\nMoney skills (rules + BM25): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.72  # measured 77% (17/22); one miss of headroom (BM25 is the fallback)
    assert top3 >= 0.95


@pytest.mark.live
async def test_money_skills_reachable_hybrid() -> None:

    top1, top3 = await evaluate_pipeline(local_fast_embedder(), MONEY, PRODUCTION)
    print(f"\nMoney skills (rules + hybrid): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.8
    assert top3 >= 0.95


async def test_blind_set_with_production_packs_bm25() -> None:
    """Extra packs must not crowd out core answers. Measured 67% / 75% on 2026-10-10 (core alone: 67% / 79%).

    Accepted: one blind question now gets a money-skills entry in the top 3 instead of a core one. The blind set is
    frozen, so content is never tuned on it; the floors leave one miss of headroom."""
    top1, top3 = await evaluate_pipeline(None, BLIND, FROZEN_PACKS)
    print(f"\nBlind, frozen pack set (BM25): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.6
    assert top3 >= 0.7


@pytest.mark.live
async def test_blind_set_with_production_packs_hybrid() -> None:
    """Measured 62% top-1 / 96% top-3 on 2026-10-10 (core alone: 67% / 96%; top 3 is what grounds the answer).

    Accepted -1 top-1 versus core: "keep small small money aside every week" now ranks irregular-income above
    saving-basics (both in the top 3)."""

    top1, top3 = await evaluate_pipeline(local_fast_embedder(), BLIND, FROZEN_PACKS)
    print(f"\nBlind, frozen pack set (hybrid): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.58  # one miss below the measured 15/24
    assert top3 >= 0.9


async def test_blind_set_with_all_packs_reported_bm25() -> None:
    """Reported only: the blind labels predate genuine-notices and situations, so a better sit-*/notice-* answer
    counts as a miss here. Measured on 2026-10-11 (docs/SPIKES.md); never tune on it."""
    top1, top3 = await evaluate_pipeline(None, BLIND, PRODUCTION)
    print(f"\nBlind, all production packs (BM25, reported): top1={top1:.0%} top3={top3:.0%}")
    assert 0 <= top1 <= top3 <= 1


@pytest.mark.live
async def test_blind_set_with_all_packs_reported_hybrid() -> None:
    """Reported only (see the BM25 twin)."""
    top1, top3 = await evaluate_pipeline(local_fast_embedder(), BLIND, PRODUCTION)
    print(f"\nBlind, all production packs (hybrid, reported): top1={top1:.0%} top3={top3:.0%}")
    assert 0 <= top1 <= top3 <= 1


async def test_pipeline_accuracy_bm25_production() -> None:
    """Dev set with the production pack set (money-skills ids accepted where they answer better)."""
    top1, top3 = await evaluate_pipeline(None, CASES, PRODUCTION)
    print(f"\nPipeline, production packs (rules + BM25): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.75
    assert top3 >= 0.9


@pytest.mark.live
async def test_pipeline_accuracy_hybrid_production() -> None:

    top1, top3 = await evaluate_pipeline(local_fast_embedder(), CASES, PRODUCTION)
    print(f"\nPipeline, production packs (rules + hybrid): top1={top1:.0%} top3={top3:.0%}")
    assert top1 >= 0.85
    assert top3 >= 0.92


async def test_extra_packs_never_displace_core_red_flag_guidance() -> None:
    """Two red flags: both red-flag slots go to core safety entries, whatever else is loaded."""
    from kudiguard.domain import Flag
    from kudiguard.knowledge.service import KnowledgeBase

    kb = KnowledgeBase([FilePackSource.load(ROOT / "data" / "packs" / pack) for pack in PRODUCTION])
    flags = (Flag.GUARANTEED_RETURNS, Flag.UPFRONT_FEE)
    result = await kb.search(
        Query(text="pay a fee to join my forex trading and get double profit", flags=flags)
    )
    first, second = result.entries[:2]
    assert (first.entry.id, second.entry.id) == ("flag-guaranteed-returns", "flag-upfront-fee")
    assert {first.source_id, second.source_id} == {"kudiguard-core"}


# --- probes for the situations and genuine-notices packs (production pack set, rules flags as in production) --------


async def _top(query: str, embedder: Embedder | None) -> list[str]:
    from kudiguard.knowledge.service import KnowledgeBase
    from kudiguard.languages import LanguageRegistry
    from kudiguard.safety.scam_rules import assess

    lexicon = LanguageRegistry.load(ROOT / "data" / "languages").lexicon_for("en")
    kb = KnowledgeBase(
        [FilePackSource.load(ROOT / "data" / "packs" / pack, embedder=embedder) for pack in PRODUCTION],
        embedder=embedder,
    )
    result = await kb.search(Query(text=query, flags=assess([query], lexicon).flags))
    return [item.entry.id for item in result.entries if item.relevance is not None]  # search hits, best first


PROBES_FOUND = [
    ("my card was stolen", "sit-card-stolen"),
    (
        "I just made a transfer in my app and got an OTP SMS for it, is it normal?",
        "notice-otp-own-transaction",
    ),
    ("a twitter support account asked for my pin in DM after my complaint", "sit-fake-customer-care"),
    ("the woman collecting our daily contribution has disappeared", "sit-ajo-collector-disappears"),
]
CUSTOMER_CARE = "what is my bank's customer care number?"


@pytest.mark.parametrize(("query", "entry"), PROBES_FOUND)
async def test_situation_and_notice_probes_bm25(query: str, entry: str) -> None:
    assert entry in await _top(query, None)


@pytest.mark.live
@pytest.mark.parametrize(("query", "entry"), PROBES_FOUND)
async def test_situation_and_notice_probes_hybrid(query: str, entry: str) -> None:
    assert entry in await _top(query, local_fast_embedder())


async def test_customer_care_number_is_not_answered_with_the_fake_care_scam_bm25() -> None:
    assert (await _top(CUSTOMER_CARE, None))[0] != "sit-fake-customer-care"


@pytest.mark.live
async def test_customer_care_number_is_not_answered_with_the_fake_care_scam_hybrid() -> None:
    assert (await _top(CUSTOMER_CARE, local_fast_embedder()))[0] != "sit-fake-customer-care"


PARAPHRASE_PROBES_BM25 = [  # review of 53bdeed: inflected wording must still cue the situational entry
    ("Someone stole my ATM card at the market", "sit-card-stolen"),
    ("thieves took my bank card from my bag", "sit-card-stolen"),
    ("I was defrauded of 200k yesterday", "sit-lost-money-now"),
]


@pytest.mark.parametrize(("query", "entry"), PARAPHRASE_PROBES_BM25)
async def test_paraphrase_probes_bm25(query: str, entry: str) -> None:
    assert entry in (await _top(query, None))[:3]
