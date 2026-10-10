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
    score = Score(True, 0.33333, (True, False, False), ("number",), ("link: x",), ("*346#",), ("u",), False)
    record = Record.from_result(case(lang="ha"), CaseResult(answer, observation, score, 1.23456, 40), run=2)
    assert record == Record(
        id="c1", lang="ha", kind="scam", run=2, verdict="caution", expected=["likely_scam", "caution"],
        verdict_ok=True, outcome="answered", intent="scam_check", flags=["urgency"], issues=["refusal"], body="jiki",
        english="body", coverage=0.3333, covered=[True, False, False], violations=["number"],
        translation_hits=["link: x"], example_copied=["*346#"], unsupported=["u"], grounded=False, seconds=1.235,
        llm_ms=900, grounding_ms=40, sources=["x"],
    )  # fmt: skip
    unscored = Score(False, None, (), (), (), (), (), None)
    plain = Answer("en", AnswerKind.GREETING, Outcome.FIXED_REPLY)
    bare = Record.from_result(case(), CaseResult(plain, observation, unscored, 0.0), run=0)
    assert bare.coverage is None and bare.intent is None and bare.grounding_ms is None
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
    assert empty.n == 0 and empty.coverage is None and empty.p50 == 0.0
    errors = Summary.of([rec(error="boom")])
    assert (errors.n, errors.errors, errors.verdict_accuracy) == (0, 1, None)
    assert "-" in errors.cells()


def test_summarize_per_language_and_overall() -> None:
    records = [
        rec("a", "en", seconds=1.0, example_copied=["*346#"]),
        rec(
            "b",
            "en",
            verdict_ok=False,
            coverage=0.0,
            violations=["number"],
            grounded=None,
            outcome="fallback",
        ),
        rec("c", "ha", grounded=False, seconds=3.0, coverage=None),
        rec("d", "ha", error="boom"),
    ]
    summaries = summarize(records)
    assert list(summaries) == ["en", "ha", ALL]
    en, ha = summaries["en"], summaries["ha"]
    assert (en.n, en.verdict_accuracy, en.coverage, en.violation_rate) == (2, 0.5, 0.5, 0.5)
    assert (en.grounded_rate, en.fallback_rate, en.copied_rate) == (1.0, 0.5, 0.5)
    assert (ha.n, ha.errors, ha.coverage) == (1, 1, None)  # failed back-translation: left out of coverage
    assert summaries[ALL].grounded_rate == 0.5 and summaries[ALL].p95 == 3.0
    assert macro_coverage(summaries) == 0.5
    assert macro_coverage({ALL: summaries[ALL]}) is None


def test_reproducibility() -> None:
    assert reproducibility([rec("a"), rec("b")]) is None  # one run
    runs = [rec("a", run=0), rec("a", run=1), rec("b", run=0, body="x"), rec("b", run=1, body="y")]
    assert reproducibility(runs) == 0.5


def test_format_summary() -> None:
    text = format_summary([rec("a", run=0), rec("a", run=1, grounded=None), rec("b", "yo", grounded=None)])
    lines = text.splitlines()
    assert "coverage" in lines[0] and "same" in lines[0]
    assert [line.split()[0] for line in lines[1:4]] == ["en", "yo", ALL]
    assert "100%" in lines[1] and "1.0s" in lines[1]  # en: identical replies across both runs
    assert lines[-1] == "  runs: 2  macro coverage (mean of languages): 1.00"


def test_comparison_on_common_cases() -> None:
    before = [rec("a"), rec("b"), rec("c", "ig"), rec("gone")]
    after = [
        rec("a"),
        rec("b", coverage=0.5, violations=["link: x"], verdict="caution", body="new"),
        rec("c", "ig", grounded=None),
        rec("new"),
    ]
    assert changed_cases(before, after) == [
        "b: verdict likely_scam -> caution, coverage 1.00 -> 0.50, violations 0.00 -> 1.00, reply text changed",
        "c: grounded 1.00 -> -",
    ]
    text = format_comparison(before, after, ("t01", "t0"), ("temperature=0.1 seed=none", ""))
    lines = text.splitlines()
    assert lines[:4] == [
        "t01: temperature=0.1 seed=none",
        "t0: (no meta)",
        "only in t01: gone",
        "only in t0: new",
    ]
    assert "delta" in text and "-50pp" not in text and "-0.17" in text  # all: coverage 1.00 -> 0.83
    assert "changed cases (2):" in text
    assert "  -  " in text  # delta of an unmeasured metric


def test_comparison_without_changes_or_common_cases() -> None:
    same = format_comparison([rec("a")], [rec("a")])
    assert same.endswith("changed cases (0):\n  none") and "+0.0s" in same and "+0pp" in same and "+0" in same
    disjoint = format_comparison([rec("a")], [rec("b")])
    assert "only in a: a" in disjoint and "only in b: b" in disjoint and "changed cases (0)" in disjoint
