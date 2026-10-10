"""Grounding check with the real embedding model (CPU, -m live) on data/eval/grounding_claims.yaml.

"Positive" = an unsupported (invented) claim, the thing the check must catch. The threshold is chosen on the dev split
only; held-out numbers come from the test split. Thresholds below are regression guards: raise them as the check
improves, never lower them silently. Set KUDIGUARD_EMBEDDINGS_CACHE to the model cache outside the main checkout.
"""

import asyncio
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from statistics import median
from typing import Any

import httpx
import numpy as np
import pytest
import yaml

from kudiguard.domain import Flag, Verdict
from kudiguard.knowledge.embeddings import CachedEmbedder
from kudiguard.knowledge.file_pack import FilePackSource
from kudiguard.knowledge.models import KnowledgeResult, Query, SourcedEntry
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.grounding import DEFAULT_THRESHOLD, GroundingChecker, evidence_for, score_claims
from kudiguard.safety.output_guard import OutputGuard

ROOT = Path(__file__).resolve().parents[2]
DATA: dict[str, list[dict[str, Any]]] = yaml.safe_load(
    (ROOT / "data" / "eval" / "grounding_claims.yaml").read_text(encoding="utf-8")
)
CACHE = Path(os.environ.get("KUDIGUARD_EMBEDDINGS_CACHE", ROOT / "models" / "fastembed"))
CORE = {e.id: e for e in FilePackSource.load(ROOT / "data" / "packs" / "kudiguard-core").entries}
REPLY = (
    "🔴 This looks like a scam. Your bank will never ask for your OTP.\n"
    "1. Do not share the code. End the call.\n"
    "2. If you already shared it, call your bank using the number on the back of your card.\n"
    "3. Change your PIN and password."
)


def evidence(case: dict[str, Any]) -> list[str]:
    entries = tuple(SourcedEntry(CORE[i], "kudiguard-core", "KudiGuard") for i in case["entries"])
    verdict = Verdict(case["verdict"]) if "verdict" in case else None
    return evidence_for(KnowledgeResult(entries=entries), verdict=verdict)


def claims(split: str) -> list[dict[str, Any]]:
    return [c for c in DATA["claims"] if c["set"] == split and c["lang"] == "en"]


@pytest.fixture(scope="module")
def embedder() -> CachedEmbedder:
    from kudiguard.knowledge.embeddings import FastEmbedder

    return CachedEmbedder(FastEmbedder(cache_dir=CACHE))


@pytest.fixture(scope="module")
def checker(registry: LanguageRegistry, embedder: CachedEmbedder) -> GroundingChecker:
    return GroundingChecker(None, embedder, registry)


@pytest.fixture
async def knowledge(embedder: CachedEmbedder) -> AsyncIterator[KnowledgeBase]:
    async with httpx.AsyncClient() as client:
        kb = knowledge_base_from_packs(ROOT / "data" / "packs", client=client, embedder=embedder.inner)
        await kb.refresh_all()
        yield kb


def tally(
    checker: GroundingChecker, cases: list[dict[str, Any]], threshold: float
) -> tuple[int, int, int, int]:
    """(caught, missed, false flags, supported)"""
    caught = missed = false_flags = supported = 0
    for case in cases:
        (claim,) = score_claims(
            [case["claim"]], [*evidence(case), *checker.framing], checker._embedder, threshold=threshold
        )
        if case["supported"]:
            supported += 1
            false_flags += not claim.supported
        else:
            caught += not claim.supported
            missed += claim.supported
    return caught, missed, false_flags, supported


@pytest.mark.live
def test_threshold_is_the_dev_choice_and_holds_out(checker: GroundingChecker) -> None:
    """Dev: the highest threshold with no false flags (a false flag replaces a good answer). Test: report only."""
    sweep = {round(t, 2): tally(checker, claims("dev"), t) for t in np.arange(0.70, 0.86, 0.01)}
    for t, (caught, missed, ff, sup) in sweep.items():
        print(f"  dev t={t:.2f}: caught {caught}/{caught + missed}, false flags {ff}/{sup}")
    safe = [t for t, (_, _, ff, _) in sweep.items() if ff == 0]
    print(f"dev choice: {max(safe):.2f} (shipped {DEFAULT_THRESHOLD})")
    assert max(safe) >= DEFAULT_THRESHOLD
    for split in ("dev", "test"):
        caught, missed, ff, sup = tally(checker, claims(split), DEFAULT_THRESHOLD)
        precision = caught / max(caught + ff, 1)
        print(
            f"{split} @ {DEFAULT_THRESHOLD}: precision {precision:.0%} recall {caught / (caught + missed):.0%} "
            f"(caught {caught}/{caught + missed}, false flags {ff}/{sup})"
        )
    caught, missed, ff, sup = tally(checker, claims("test"), DEFAULT_THRESHOLD)
    assert caught / (caught + missed) >= 0.5  # held out, incl. polarity inversions (measured 56%)
    assert ff / sup <= 0.1  # measured 0/14


@pytest.mark.live
async def test_replies_scam_echoes_blocked_and_realistic_replies_measured(
    registry: LanguageRegistry, checker: GroundingChecker, knowledge: KnowledgeBase
) -> None:
    """Production evidence (knowledge search on the message). Unacceptable replies must all be blocked by the
    grounding check or the output guard; the false-replacement rate of acceptable replies is reported (enforce mode
    would replace them; that is why the default is shadow)."""
    guard = OutputGuard(registry)
    blocked_bad = flagged_good = good = 0
    for case in DATA["replies"]:
        flags = tuple(Flag(f) for f in case["flags"])
        found = await knowledge.search(Query(text=case["message"], flags=flags, language="en"))
        ev = evidence_for(found, verdict=Verdict(case["verdict"]), flags=flags)
        report = await checker.check(case["reply"], "en", ev, message=case["message"])
        guarded = bool(guard.check(case["reply"], "en"))
        if case["acceptable"]:
            good += 1
            flagged_good += not report.grounded
            assert not guarded, case["reply"]
            if not report.grounded:
                print(f"  false flag: {report.unsupported}")
        else:
            blocked = guarded or not report.grounded
            blocked_bad += blocked
            assert blocked, case["reply"]
    print(f"\nreplies: {blocked_bad} unacceptable blocked; acceptable flagged {flagged_good}/{good}")


@pytest.mark.live
async def test_attributed_restatement_passes_and_unattributed_echo_does_not(
    registry: LanguageRegistry, checker: GroundingChecker, knowledge: KnowledgeBase
) -> None:
    message = "They said my account will be blocked today unless I send them the OTP they sent me."
    flags = (Flag.ACCOUNT_THREAT, Flag.ASKS_CREDENTIALS)
    ev = evidence_for(
        await knowledge.search(Query(text=message, flags=flags)), verdict=Verdict.LIKELY_SCAM, flags=flags
    )
    said = await checker.check(
        "They said your account will be blocked unless you send the OTP.", "en", ev, message=message
    )
    echo = await checker.check(
        "Your account will be blocked today unless you send them the OTP.", "en", ev, message=message
    )
    guard = OutputGuard(registry)
    assert said.grounded and not guard.check(said.english, "en")
    assert not echo.grounded or guard.check(echo.english, "en")  # blocked by one layer or the other


async def _latencies(checker: GroundingChecker, message_chars: int, runs: int = 30) -> list[float]:
    case = {"entries": ["flag-asks-credentials"], "verdict": "likely_scam"}
    base = (
        "Dear customer, your account has been restricted due to an incomplete BVN update, click the link now."
    )
    timings = []
    for n in range(runs):
        message = " ".join(f"{base} ({n}.{i})" for i in range(message_chars // len(base) + 1))[:message_chars]
        reply = f"{REPLY}\nTell someone you trust before you act, call {n}."
        start = time.perf_counter()
        await checker.check(reply, "en", evidence(case), message=message)
        timings.append((time.perf_counter() - start) * 1000)
    return sorted(timings)


@pytest.mark.live
async def test_latency_by_message_length(checker: GroundingChecker) -> None:
    """Added wall time per English reply as a turn sees it: pack facts pre-embedded (as at startup); each turn embeds
    the reply's sentences and the user's message (at most 10 sentences of 300 chars) off the event loop."""
    start = time.perf_counter()
    warmed = checker.warm(t for e in CORE.values() for t in (*e.facts, *e.actions))
    print(f"\nwarm-up: {warmed} core sentences in {time.perf_counter() - start:.2f} s")
    await checker.check(REPLY, "en", [], message="warm")  # starts the worker thread
    for chars in (80, 500, 1000, 2000):
        timings = await _latencies(checker, chars)
        p50, p95 = median(timings), timings[int(0.95 * len(timings)) - 1]
        print(f"  message {chars:>4} chars: p50 {p50:.1f} ms, p95 {p95:.1f} ms")
        assert p95 < 150, chars  # the live budget


@pytest.mark.live
async def test_concurrent_turns_never_queue(checker: GroundingChecker) -> None:
    case = {"entries": ["flag-asks-credentials"], "verdict": "likely_scam"}

    async def one(n: int) -> str:
        try:
            await checker.check(f"{REPLY} {n}", "en", evidence(case), message=f"message {n}")
        except Exception as exc:
            return type(exc).__name__
        return "ok"

    start = time.perf_counter()
    results = await asyncio.gather(*(one(n) for n in range(8)))
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\n8 concurrent: {results} in {elapsed:.0f} ms")
    assert "ok" in results and elapsed < 150
