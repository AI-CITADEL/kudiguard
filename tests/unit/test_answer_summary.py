"""Answer-quality records, summaries and before/after comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kudiguard.domain import Outcome, Verdict
from kudiguard.engine import Answer, AnswerKind
from kudiguard.evaluation.answers import AnswerCase, CaseResult, Observation, Score
from kudiguard.evaluation.summary import (
    ALL,
    Record,
    Summary,
    changed_cases,
    format_comparison,
    format_summary,
    percentile,
    read_records,
    reproducibility,
    summarize,
    write_records,
)


def rec(id: str = "a", lang: str = "en", run: int = 0, **kw: Any) -> Record:
    data: dict[str, Any] = {
        "id": id,
        "lang": lang,
        "kind": "scam",
        "run": run,
        "verdict": "likely_scam",
        "expected": ["likely_scam"],
        "verdict_ok": True,
        "outcome": "answered",
        "body": "reply",
        "english": "reply",
        "coverage": 1.0,
        "covered": [True],
        "violations": [],
        "unsupported": [],
        "grounded": True,
        "seconds": 1.0,
        "sources": ["flag-urgency"],
    }
    return Record(**(data | kw))


def test_record_from_result_and_jsonl_round_trip(tmp_path: Path) -> None:
    case = AnswerCase.model_validate(
        {
            "id": "c1",
            "lang": "ha",
            "kind": "scam",
            "text": "t",
            "verdicts": ["caution"],
            "must": [{"point": "abc", "from": "x"}],
        }
    )
    answer = Answer("ha", AnswerKind.CHECK, Outcome.ANSWERED, Verdict.CAUTION, body="jiki", sources=("x",))
    observation = Observation(Verdict.CAUTION, Outcome.ANSWERED, "jiki", "body")
    score = Score(True, 0.33333, (True, False, False), ("number",), ("u",), False)
    record = Record.from_result(case, CaseResult(answer, observation, score, 1.23456), run=2)
    assert record == rec(
        "c1", "ha", 2, verdict="caution", expected=["caution"], body="jiki", english="body", coverage=0.3333,
        covered=[True, False, False], violations=["number"], unsupported=["u"], grounded=False, seconds=1.235,
        sources=["x"],
    )  # fmt: skip
    path = tmp_path / "out" / "r.jsonl"
    write_records(path, [record, rec("é")])
    assert read_records(path) == [record, rec("é")]


def test_percentile() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert percentile([1.0, 2.0], 0.95) == 2.0


def test_summarize_per_language_and_overall() -> None:
    records = [
        rec("a", "en", seconds=1.0),
        rec(
            "b",
            "en",
            verdict_ok=False,
            coverage=0.0,
            violations=["number"],
            grounded=None,
            outcome="fallback",
        ),
        rec("c", "ha", grounded=False, seconds=3.0),
    ]
    summaries = summarize(records)
    assert list(summaries) == ["en", "ha", ALL]
    en = summaries["en"]
    assert (en.n, en.verdict_accuracy, en.coverage, en.violation_rate) == (2, 0.5, 0.5, 0.5)
    assert (en.grounded_rate, en.fallback_rate) == (1.0, 0.5)
    assert summaries[ALL].grounded_rate == 0.5 and summaries[ALL].p95 == 3.0
    assert Summary.of([rec(grounded=None)]).grounded_rate is None


def test_reproducibility() -> None:
    assert reproducibility([rec("a"), rec("b")]) is None  # one run
    runs = [rec("a", run=0), rec("a", run=1), rec("b", run=0, body="x"), rec("b", run=1, body="y")]
    assert reproducibility(runs) == 0.5


def test_format_summary() -> None:
    text = format_summary([rec("a", run=0), rec("a", run=1, grounded=None), rec("b", "yo", grounded=None)])
    lines = text.splitlines()
    assert "coverage" in lines[0] and lines[1].split()[0] == "en" and lines[3].split()[0] == ALL
    assert "  -  " in lines[2]  # yo: grounding not measured
    assert lines[-1].endswith("identical replies across 2 runs: 100%")
    assert format_summary([rec()]).endswith("1 runs: -")


def test_comparison_lists_changed_cases() -> None:
    before = [rec("a"), rec("b"), rec("c", "ig"), rec("gone")]
    after = [rec("a"), rec("b", coverage=0.5, violations=["link: x"]), rec("c", "ig", grounded=None)]
    assert changed_cases(before, after) == [
        "b: coverage 1.00 -> 0.50, violations 0.00 -> 1.00",
        "c: grounded 1.00 -> -",
    ]
    text = format_comparison(before, after, ("t01", "t0"))
    assert "t01" in text and "t0 " in text and "changed cases (2):" in text
    assert [line.split()[0] for line in text.splitlines()[1:6:2]] == ["en", "ig", ALL]
    assert "identical replies across runs: - -> -" in text
    assert format_comparison(before, before).endswith("changed cases (0):\n  none")
