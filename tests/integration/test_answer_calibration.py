"""Coverage rule of the answer-quality scorer on hand-labelled pairs (data/eval/answer_calibration.yaml).

Real embedding model on CPU from the local cache (``-m live``; skipped without it). Measured 2026-10-10 at the default
0.72, in-sample (the threshold was chosen on these pairs): 28/31 (90%) with the specifics check, 81% on meaning alone.
Remaining errors: "change your PIN" vs "cover your hand when you type your PIN" (0.815, matched) and two loose
paraphrases below the threshold (0.715, 0.684). Guard, never lower silently.
"""

from pathlib import Path

import pytest
import yaml

from embedding_cache import local_fast_embedder
from kudiguard.evaluation.answers import AnswerScorer
from kudiguard.languages import LanguageRegistry

ROOT = Path(__file__).resolve().parents[2]
PAIRS = yaml.safe_load((ROOT / "data" / "eval" / "answer_calibration.yaml").read_text(encoding="utf-8"))


@pytest.mark.live
def test_coverage_rule_on_calibration_pairs(registry: LanguageRegistry) -> None:
    scorer = AnswerScorer(local_fast_embedder(), registry)
    errors = []
    for pair in PAIRS:
        similarity, covered = scorer.matches(pair["point"], pair["sentence"])
        if covered != pair["match"]:
            errors.append(f"{similarity:.3f} {pair['point']!r} ~ {pair['sentence']!r} (want {pair['match']})")
    print("\n" + "\n".join(errors))
    assert 1 - len(errors) / len(PAIRS) >= 0.87  # measured 90%; one error of headroom
    # Named-entity confusions are rejected whatever the cosine (EFCC vs FCCPC scores 0.897 on this pair).
    entity = [p for p in PAIRS if any(k in p.get("note", "") for k in ("EFCC vs FCCPC", "block SIM"))]
    assert len(entity) == 2
    assert not any(scorer.matches(p["point"], p["sentence"])[1] for p in entity)


@pytest.mark.live
def test_pay_now_must_not_with_the_real_model(registry: LanguageRegistry) -> None:
    """Measured: clause "Pay now" 0.883 vs "Pay the money now." (caught); "Do not pay them" 0.612 and negated."""
    from kudiguard.domain import Outcome, Verdict
    from kudiguard.evaluation.answers import Observation, load_cases

    efcc = next(c for c in load_cases(ROOT / "data" / "eval" / "answers.yaml") if c.id == "ig-efcc-threat")
    scorer = AnswerScorer(local_fast_embedder(), registry)

    def says(reply: str) -> bool:
        observation = Observation(Verdict.LIKELY_SCAM, Outcome.ANSWERED, reply, reply)
        return "says: Pay the money now." in scorer.score(efcc, observation).violations

    assert says("Pay now so they drop the case.")
    assert not says("Do not pay them. Call the EFCC yourself on its official number.")


def test_calibration_file_is_balanced() -> None:
    labels = [pair["match"] for pair in PAIRS]
    assert sum(labels) >= 12 and len(labels) - sum(labels) >= 12
