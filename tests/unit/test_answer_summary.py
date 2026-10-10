"""Answer-quality records, results files, summaries and before/after comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from answer_helpers import case
from kudiguard.domain import Flag, Outcome, Verdict
from kudiguard.engine import Answer, AnswerKind
from kudiguard.evaluation.answers import Observation, Score
from kudiguard.evaluation.runner import CaseResult
from kudiguard.evaluation.stats import mean_or_none, percentile
from kudiguard.evaluation.summary import (
    ALL,
    Record,
    ResultsWriter,
    Summary,
    changed_cases,
    format_comparison,
    format_summary,
    macro_coverage,
    read_meta,
    read_records,
    reproducibility,
    summarize,
)
from kudiguard.prompts import Intent
from kudiguard.safety.output_guard import OutputIssue


def rec(id: str = "a", lang: str = "en", run: int = 0, **kw: Any) -> Record:
    data: dict[str, Any] = {
        "verdict": "likely_scam",
        "expected": ["likely_scam"],
        "verdict_ok": True,
        "outcome": "answered",
        "body": "reply",
        "english": "reply",
        "coverage": 1.0,
        "covered": [True],
        "grounded": True,
        "seconds": 1.0,
    }
    return Record(id=id, lang=lang, kind="scam", run=run, **(data | kw))


def test_record_from_result_and_failure() -> None:
    answer = Answer(
        "ha", AnswerKind.CHECK, Outcome.ANSWERED, Verdict.CAUTION, flags=(Flag.URGENCY,), intent=Intent.SCAM_CHECK,
        body="jiki", sources=("x",), issues=(OutputIssue.REFUSAL,), llm_ms=900,
    )  # fmt: skip
    observation = Observation(Verdict.CAUTION, Outcome.ANSWERED, "jiki", "body")
    score = Score(
        True, 0.33333, (True, False, False), 0.5, ("number",), ("link: x",), ("*346#",), ("u",), False
    )
    result = CaseResult(answer, observation, score, 1.23456, 40, True)
    record = Record.from_result(case(lang="ha"), result, run=2)
    assert record == Record(
        id="c1", lang="ha", kind="scam", run=2, verdict="caution", expected=["likely_scam", "caution"],
        verdict_ok=True, outcome="answered", intent="scam_check", flags=["urgency"], issues=["refusal"], body="jiki",
        english="body", coverage=0.3333, specific_coverage=0.5, covered=[True, False, False], violations=["number"],
        translation_hits=["link: x"], example_copied=["*346#"], unsupported=["u"], grounded=False,
        evidence_mismatch=True, seconds=1.235, llm_ms=900, grounding_ms=40, sources=["x"],
    )  # fmt: skip
    assert record.scored
    unscored = Score(False, None, (), None, (), (), (), (), None)
    plain = Answer("en", AnswerKind.GREETING, Outcome.FIXED_REPLY)
    bare = Record.from_result(case(), CaseResult(plain, observation, unscored, 0.0), run=0)
    assert (
        bare.coverage is None and bare.specific_coverage is None and bare.intent is None and not bare.scored
    )
    assert bare.grounding_ms is None and bare.evidence_mismatch is None
    assert Record.failed(case(), 1, "boom") == Record(id="c1", lang="en", kind="scam", run=1, error="boom")


def test_results_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "r.jsonl"
    writer = ResultsWriter(path, {"sampling": "temperature=0 seed=42", "url": "u"})
    writer.add(rec("é"))
    assert read_records(path) == [rec("é")]  # flushed as it goes: an interrupted run keeps its records
    writer.add(rec("b", error="x"))
    writer.close()
    assert read_meta(path) == {"sampling": "temperature=0 seed=42", "url": "u"}
    assert [r.id for r in read_records(path)] == ["é", "b"]
    bare = tmp_path / "bare.jsonl"
    bare.write_text(rec().to_json() + "\n\n", encoding="utf-8")
    assert read_meta(bare) == {} and len(read_records(bare)) == 1


def test_stats() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert percentile([1.0, 2.0], 0.95) == 2.0
    assert mean_or_none([]) is None and mean_or_none([1, 0]) == 0.5


def test_summary_of_nothing_and_of_errors_only() -> None:
    empty = Summary.of([])
    assert (empty.n, empty.scored, empty.coverage, empty.p50, empty.reproducible) == (0, 0, None, 0.0, None)
    errors = Summary.of([rec(error="boom")])
    assert (errors.n, errors.errors, errors.verdict_accuracy) == (0, 1, None)
    assert "-" in errors.cells()


def test_populations() -> None:
    """Verdicts and fallbacks over answered records; coverage and English-side rates over scored ones only."""
    records = [
        rec("a", "en", seconds=1.0, example_copied=["*346#"], specific_coverage=1.0),
        rec(
            "b",
            "en",
            verdict_ok=False,
            coverage=0.0,
            violations=["number"],
            grounded=None,
            outcome="fallback",
        ),
        rec(
            "c",
            "ha",
            grounded=False,
            seconds=3.0,
            coverage=None,
            violations=["link removed"],
            example_copied=["x"],
        ),
        rec("d", "ha", error="boom"),
    ]
    summaries = summarize(records)
    assert list(summaries) == ["en", "ha", ALL]
    en, ha, every = summaries["en"], summaries["ha"], summaries[ALL]
    assert (en.n, en.scored, en.verdict_accuracy, en.coverage, en.specific) == (2, 2, 0.5, 0.5, 1.0)
    assert (en.violation_rate, en.copied_rate, en.grounded_rate, en.grounded_n) == (0.5, 0.5, 1.0, 1)
    assert en.fallback_rate == 0.5
    # ha: the unscored answer is left out of coverage and the English-side rates, not counted as clean.
    assert (ha.n, ha.scored, ha.errors, ha.coverage, ha.violation_rate, ha.copied_rate) == (
        1,
        0,
        1,
        None,
        None,
        None,
    )
    assert (ha.grounded_rate, ha.grounded_n) == (0.0, 1)
    assert (every.grounded_rate, every.grounded_n, every.p95) == (0.5, 2, 3.0)
    assert macro_coverage(summaries) == 0.5
    assert macro_coverage({ALL: every}) is None


def test_reproducibility_counts_model_answers_only() -> None:
    assert reproducibility([rec("a"), rec("b")]) == (None, 0)  # one run
    runs = [
        rec("a", run=0), rec("a", run=1),  # identical
        rec("b", run=0, body="x"), rec("b", run=1, body="y"),  # changed
        rec("c", run=0, outcome="fallback"), rec("c", run=1, outcome="fallback"),  # fallback only
        rec("d", run=0, outcome="fallback"), rec("d", run=1),  # mixed: in neither
    ]  # fmt: skip
    assert reproducibility(runs) == (0.5, 1)


def test_format_summary() -> None:
    records = [
        rec("a", run=0),
        rec("a", run=1, grounded=None, evidence_mismatch=True),
        rec("b", "yo", grounded=None),
    ]
    text = format_summary(records, {"grounding": "FallbackGrounding"})
    lines = text.splitlines()
    assert "coverage" in lines[0] and "same" in lines[0] and "g.n" in lines[0] and "scored" in lines[0]
    assert [line.split()[0] for line in lines[1:4]] == ["en", "yo", ALL]
    assert "100%" in lines[1] and "1.0s" in lines[1]  # en: identical replies across both runs
    assert lines[-2] == "  runs: 2  macro coverage (mean of languages): 1.00  evidence rebuilt differently: 1"
    assert "stand-in" in lines[-1]
    assert format_summary(records).endswith("grounding: unknown")
    assert format_summary(records, {"grounding": "CheckerGrounding"}).endswith("grounding: CheckerGrounding")


META_A = {"sampling": "temperature=0.1 seed=none", "grounding": "FallbackGrounding", "cover_threshold": 0.72,
          "forbid_threshold": 0.85, "commit": "aaa"}  # fmt: skip


def test_comparison_on_pairs_scored_in_both() -> None:
    before = [rec("a"), rec("b"), rec("c", "ig"), rec("gone"), rec("x", coverage=None), rec("y")]
    after = [
        rec("a"),
        rec("b", coverage=0.5, violations=["link: x"], verdict="caution", body="new"),
        rec("c", "ig", grounded=None),
        rec("new"),
        rec("x"),
        rec("y", error="boom"),
    ]
    assert changed_cases(before, after)[0].startswith("b: verdict likely_scam -> caution")
    meta_b = META_A | {"sampling": "temperature=0 seed=42", "cover_threshold": 0.7, "commit": "bbb"}
    text = format_comparison(before, after, ("t01", "t0"), (META_A, meta_b))
    lines = text.splitlines()
    assert lines[0].startswith("t01: temperature=0.1 seed=none; grounding: FallbackGrounding (stand-in")
    assert "cover 0.72 forbid 0.85; commit aaa" in lines[0] and "cover 0.7 " in lines[1]
    assert lines[2:4] == [
        "WARNING: cover_threshold differs (0.72 vs 0.7)",
        "note: different commits (aaa vs bbb)",
    ]
    assert lines[4:7] == [
        "only in t01: gone",
        "only in t0: new",
        "compared: 3 (case, run) pairs scored in both; 2 left out",
    ]
    assert "delta" in text and "-0.17" in text  # all: coverage 1.00 -> 0.83 over a, b, c
    assert "changed cases (2):" in text
    assert (
        "b: verdict likely_scam -> caution, coverage 1.00 -> 0.50, violations 0.00 -> 1.00, reply text changed"
        in text
    )
    assert "c: grounded 1.00 -> -" in text


def test_comparison_without_meta_changes_or_common_cases() -> None:
    same = format_comparison([rec("a")], [rec("a")], metas=(META_A, META_A))
    assert same.endswith("changed cases (0):\n  none") and "+0.0s" in same and "+0pp" in same and "+0" in same
    assert "WARNING" not in same and "note:" not in same
    disjoint = format_comparison([rec("a")], [rec("b")])
    assert disjoint.startswith("a: (no meta)\nb: (no meta)\nonly in a: a\nonly in b: b\ncompared: 0")
    assert "changed cases (0)" in disjoint
