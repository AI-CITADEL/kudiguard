"""Grounding check calibration with the real embedding model (CPU, -m live): data/eval/grounding_claims.yaml.

"Positive" = an unsupported (invented) claim, the thing the check must catch. Thresholds are regression guards:
raise them as the check improves, never lower them silently. Set KUDIGUARD_EMBEDDINGS_CACHE to the model cache when
running outside the main checkout.
"""

import os
import time
from pathlib import Path
from statistics import median
from typing import Any

import pytest
import yaml

from kudiguard.domain import Verdict
from kudiguard.knowledge.file_pack import FilePackSource
from kudiguard.knowledge.models import KnowledgeResult, SourcedEntry
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.grounding import DEFAULT_THRESHOLD, GroundingChecker, evidence_for

ROOT = Path(__file__).resolve().parents[2]
CASES: list[dict[str, Any]] = yaml.safe_load(
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


@pytest.fixture(scope="module")
def checker(registry: LanguageRegistry) -> GroundingChecker:
    from kudiguard.knowledge.embeddings import FastEmbedder

    return GroundingChecker(None, FastEmbedder(cache_dir=CACHE), registry)


@pytest.mark.live
async def test_english_claims_precision_and_recall(checker: GroundingChecker) -> None:
    caught = missed = false_flags = supported = 0
    for case in (c for c in CASES if c["lang"] == "en"):
        report = await checker.check(case["claim"], "en", evidence(case))
        assert len(report.claims) == 1, case["claim"]
        flagged = not report.grounded
        claim = report.claims[0]
        if case["supported"]:
            supported += 1
            false_flags += flagged
        else:
            caught += flagged
            missed += not flagged
        if flagged == case["supported"]:
            print(f"  wrong: {claim.support:.3f} {claim.unmatched} {case['claim']!r}")
    precision = caught / (caught + false_flags)
    recall = caught / (caught + missed)
    print(
        f"\nthreshold {DEFAULT_THRESHOLD}: precision {precision:.0%} recall {recall:.0%} "
        f"(caught {caught}/{caught + missed} inventions, false flags {false_flags}/{supported})"
    )
    assert recall >= 0.85
    assert false_flags / supported <= 0.05  # each false flag replaces a good answer with fixed text


@pytest.mark.live
async def test_english_reply_latency(checker: GroundingChecker) -> None:
    """Added wall time per reply as a turn sees it: pack facts pre-embedded (``warm``, as at startup); each turn embeds
    its reply sentences and the user's new message in one batch."""
    case = {"entries": ["flag-asks-credentials"], "verdict": "likely_scam"}
    start = time.perf_counter()
    warmed = checker.warm(t for e in CORE.values() for t in (*e.facts, *e.actions))
    print(f"\nstartup: {warmed} core evidence sentences embedded in {time.perf_counter() - start:.2f} s")
    report = await checker.check(REPLY, "en", evidence(case))  # first call also starts the worker thread
    assert report.grounded, report.unsupported
    timings = []
    for n in range(40):
        message = f"Someone called me {n} times and asked for the code they sent to my phone."
        reply = f"{REPLY}\nTell someone you trust before you act, call {n}."
        start = time.perf_counter()
        await checker.check(reply, "en", [*evidence(case), message])
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    p50, p95 = median(timings), timings[int(0.95 * len(timings)) - 1]
    print(f"\nEnglish reply ({len(report.claims) + 1} claims + message): p50 {p50:.1f} ms, p95 {p95:.1f} ms")
    assert p50 < 50 and p95 < 50
    assert checker.budget_s is None or p95 < checker.budget_s * 1000
