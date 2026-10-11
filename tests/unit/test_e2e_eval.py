"""Scoring for the end-to-end test set T (rules from the cases.yaml v2.1 header), on a synthetic mini fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kudiguard.domain import Flag, Outcome, Verdict
from kudiguard.engine import Answer, AnswerKind
from kudiguard.evaluation.e2e import (
    ALL,
    FLAG_MAP,
    JUDGE_FIELDS,
    SCORED_FLAGS,
    UNSCORED_FLAGS,
    CountingLLM,
    diagnose_miss,
    expected_flags,
    format_summary,
    is_answered,
    judge_item,
    load_cases,
    outcome_matches,
    passes,
    record,
    resume_ids,
    summarize,
    too_many_errors,
)
from kudiguard.llm.base import ChatRequest, Completion

MINI = Path(__file__).resolve().parents[1] / "fixtures" / "e2e_mini.yaml"


@pytest.fixture(scope="module")
def cases() -> dict[str, dict[str, Any]]:
    _, items = load_cases(MINI)
    return {case["id"]: case for case in items}


def check(verdict: Verdict, *flags: Flag, outcome: Outcome = Outcome.ANSWERED, **kw: Any) -> Answer:
    return Answer("en", AnswerKind.CHECK, outcome, verdict=verdict, flags=flags, body="reply", **kw)


def test_flag_map_covers_vocabulary_and_harassment_is_no_match() -> None:
    # every system flag is either in the frozen vocabulary (one-to-one) or explicitly unscored
    assert {flag.value for flag in Flag} == SCORED_FLAGS | UNSCORED_FLAGS
    assert {Flag.THREAT.value} == UNSCORED_FLAGS
    assert all(FLAG_MAP[flag.value] is flag for flag in Flag if flag.value in SCORED_FLAGS)
    assert FLAG_MAP["harassment"] is None


def test_load_cases_reads_meta_and_rejects_unknown_flags(tmp_path: Path) -> None:
    meta, items = load_cases(MINI)
    assert meta["name"] == "e2e-mini" and len(items) == 8
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "cases:\n  - {id: x, lang: en, label: scam, text: t, red_flags: [made_up]}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="made_up"):
        load_cases(bad)
    bare = tmp_path / "bare.yaml"
    bare.write_text("cases:\n  - {id: y, lang: en, label: question, text: t}\n", encoding="utf-8")
    assert load_cases(bare)[0] == {}


def test_expected_flags_drop_harassment(cases: dict[str, dict[str, Any]]) -> None:
    assert expected_flags(cases["t-pcm-01"]) == ["urgency"]
    assert expected_flags(cases["t-en-02"]) == []
    assert expected_flags({"red_flags": ["urgency", "urgency"]}) == ["urgency"]


def test_outcome_values() -> None:
    assert outcome_matches("scam", flagged=True, answered=False)
    assert outcome_matches("caution", flagged=True, answered=False)
    assert not outcome_matches("scam", flagged=False, answered=True)
    assert outcome_matches("genuine", flagged=False, answered=False)
    assert not outcome_matches("genuine", flagged=True, answered=True)
    assert outcome_matches("question", flagged=True, answered=True)
    assert not outcome_matches("question", flagged=False, answered=False)
    with pytest.raises(ValueError):
        outcome_matches("maybe", flagged=True, answered=True)


def test_answered_needs_a_composed_check_reply() -> None:
    assert is_answered("check", "answered")
    assert not is_answered("check", "fallback")
    assert not is_answered("greeting", "fixed_reply")


@pytest.mark.parametrize(
    ("case_id", "answer", "passed"),
    [
        ("t-en-01", check(Verdict.CAUTION), True),  # orange counts as flagged
        ("t-en-01", check(Verdict.NO_FLAGS), False),
        ("t-en-02", check(Verdict.NO_FLAGS), True),
        ("t-en-02", check(Verdict.LIKELY_SCAM), False),
        ("t-en-03", check(Verdict.LIKELY_SCAM), True),  # colour ignored for a question that is answered
        ("t-en-03", check(Verdict.NOT_A_CHECK, outcome=Outcome.FALLBACK), False),
        ("t-ha-01", check(Verdict.CAUTION, outcome=Outcome.FALLBACK), True),  # accept: caution = flagged
        ("t-ha-01", check(Verdict.NOT_A_CHECK, outcome=Outcome.FALLBACK), False),
        ("t-en-04", Answer("en", AnswerKind.GREETING, Outcome.FIXED_REPLY), True),
    ],
)
def test_verdict_rules(cases: dict[str, dict[str, Any]], case_id: str, answer: Answer, passed: bool) -> None:
    row = record(cases[case_id], answer, reply="shown")
    assert row["passed"] is passed and passes(row) is passed


def test_record_keeps_judge_fields_translation_and_counts(cases: dict[str, dict[str, Any]]) -> None:
    answer = check(
        Verdict.CAUTION, Flag.URGENCY, english="pay now", reply_english="r", translation_rejections=("steps",)
    )
    row = record(cases["t-pcm-01"], answer, reply="shown", latency_s=1.23456, model_calls=3)
    assert row["accept"] == ["scam", "caution"] and row["subtype"] == "unlawful_threat"
    assert row["lang"] == "pcm" and row["engine_lang"] == "en"  # Pidgin runs as English, as in the product
    assert record(cases["t-yo-01"], None, error="x")["engine_lang"] == "yo"
    assert row["must_advise"] == ["Report to FCCPC"] and row["must_not"] == ["Pay to stop the shaming"]
    assert row["english"] == "pay now" and row["reply_english"] == "r"
    assert (
        row["translation_rejections"] == ["steps"] and row["latency_s"] == 1.235 and row["model_calls"] == 3
    )
    assert row["miss_diagnosis"] == "" and row["english_rule_flags"] == []
    assert judge_item(row) == {name: row[name] for name in JUDGE_FIELDS}
    assert "label" not in judge_item(row) and "verdict" not in judge_item(row)


def test_error_row(cases: dict[str, dict[str, Any]]) -> None:
    row = record(cases["t-en-01"], None, latency_s=5.0, error="timeout")
    assert row["error"] == "timeout" and row["passed"] is False and row["outcome"] == "error"


def test_yoruba_igbo_scam_misses_are_diagnosed(cases: dict[str, dict[str, Any]]) -> None:
    survived = record(
        cases["t-yo-01"], check(Verdict.NO_FLAGS, english="send OTP"), english_rule_flags=["asks_credentials"]
    )
    assert survived["miss_diagnosis"] == "cue_survived_translation"
    assert survived["english_rule_flags"] == ["asks_credentials"]
    lost = record(cases["t-ig-01"], check(Verdict.NO_FLAGS, english="hello"))
    assert lost["miss_diagnosis"] == "no_rule_cue_in_translation"
    caught = record(
        cases["t-ig-01"], check(Verdict.LIKELY_SCAM, english="pay fee"), english_rule_flags=["upfront_fee"]
    )
    assert caught["miss_diagnosis"] == "" and caught["english_rule_flags"] == []
    english_miss = record(cases["t-en-01"], check(Verdict.NO_FLAGS), english_rule_flags=["urgency"])
    assert english_miss["miss_diagnosis"] == ""
    assert diagnose_miss("  ", ["urgency"]) == "no_translation"


def test_summary_per_language_and_overall(cases: dict[str, dict[str, Any]]) -> None:
    rows = [
        record(
            cases["t-en-01"],
            check(Verdict.LIKELY_SCAM, Flag.URGENCY, Flag.SECRECY),
            latency_s=2.0,
            model_calls=2,
        ),
        record(cases["t-en-02"], check(Verdict.CAUTION, Flag.URGENCY), latency_s=4.0),
        record(cases["t-en-03"], check(Verdict.LIKELY_SCAM, Flag.URGENCY), latency_s=6.0),
        record(cases["t-en-04"], check(Verdict.NO_FLAGS, outcome=Outcome.FALLBACK), latency_s=1.0),
        record(cases["t-pcm-01"], check(Verdict.NO_FLAGS), latency_s=1.0),  # first run: superseded below
        record(cases["t-pcm-01"], check(Verdict.CAUTION, translation_rejections=("steps",)), latency_s=3.0),
        record(cases["t-yo-01"], check(Verdict.NO_FLAGS, english="x"), latency_s=8.0),
        record(cases["t-ig-01"], None, error="timeout"),
    ]
    summary = summarize(rows)
    assert list(summary) == ["en", "ig", "pcm", "yo", ALL]
    en, total = summary["en"], summary[ALL]
    assert en["n"] == 4 and en["scam_recall"] == 1.0 and en["genuine_false_positive_rate"] == 0.5
    assert en["question_answered_rate"] == 1.0 and en["question_shown_red"] == 1
    assert en["flag_recall"] == 0.25 and en["flag_precision_lower_bound"] == 0.5
    assert en["fallback_rate"] == 0.25 and en["voice_cases"] == 1 and en["model_calls"] == 2
    assert (
        summary["ig"]["errors"] == 1
        and summary["ig"]["scam_recall"] is None
        and summary["ig"]["pass_rate"] is None
    )
    assert summary["pcm"]["translation_rejection_rate"] == 1.0 and summary["pcm"]["n"] == 1
    assert summary["yo"]["miss_diagnosis"] == {"no_rule_cue_in_translation": 1}
    assert total["n"] == 7 and total["errors"] == 1 and total["voice_cases"] == 2
    assert total["scam_recall"] == 2 / 3 and total["latency_p50_s"] == 4.0 and total["latency_p95_s"] == 8.0
    assert total["counts"]["scam"] == [2, 3]
    text = format_summary(summary, mode="rules-only")
    assert "(rules-only)" in text and "yo scam misses" in text and "2 voice cases were run as text" in text
    assert "  -  " in text  # a rate with nothing to measure
    assert format_summary(summary).startswith("End-to-end test set T\n")
    assert total["timeouts"] == 1 and total["worst_case_pass_rate"] == 4 / 7
    assert "ERRORS: 1 of 7 cases (timeouts 1) excluded from all rates" in text
    assert "WARNING: error rate above 2%" in text and too_many_errors(summary)
    assert "Worst-case pass rate (errors = failures): 57%" in text and "lower bound" in text


def test_no_errors_no_warning(cases: dict[str, dict[str, Any]]) -> None:
    summary = summarize([record(cases["t-en-01"], check(Verdict.CAUTION))])
    text = format_summary(summary)
    assert "ERRORS" not in text and "WARNING" not in text and not too_many_errors(summary)
    assert not too_many_errors({ALL: {"error_rate": None}})


def test_flag_precision_only_on_cases_with_a_mapped_expected_flag() -> None:
    case = {"id": "s", "lang": "en", "label": "scam", "text": "t", "red_flags": ["harassment"]}
    summary = summarize([record(case, check(Verdict.LIKELY_SCAM, Flag.URGENCY))])
    assert summary[ALL]["flag_precision_lower_bound"] is None and summary[ALL]["flag_recall"] is None


def test_a_flag_outside_the_frozen_vocabulary_is_not_counted_wrong() -> None:
    case = {"id": "s", "lang": "en", "label": "scam", "text": "t", "red_flags": ["urgency"]}
    summary = summarize([record(case, check(Verdict.LIKELY_SCAM, Flag.URGENCY, Flag.THREAT))])
    assert summary[ALL]["flag_precision_lower_bound"] == 1.0 and summary[ALL]["flag_recall"] == 1.0


def test_resume_retries_errors_and_refuses_another_mode(cases: dict[str, dict[str, Any]]) -> None:
    kw = {"mode": "engine", "url": "http://x"}
    rows = [
        record(cases["t-en-01"], check(Verdict.CAUTION), **kw),
        record(cases["t-en-02"], None, error="timeout", **kw),
        record(cases["t-en-03"], check(Verdict.NO_FLAGS), **kw),
        record(cases["t-en-03"], None, error="timeout", **kw),  # a later error: retried again
        record(cases["t-en-04"], None, error="timeout", **kw),
        record(cases["t-en-04"], check(Verdict.NO_FLAGS), **kw),  # retried and done
    ]
    assert rows[0]["mode"] == "engine" and rows[0]["url"] == "http://x"
    assert resume_ids(rows, **kw) == {"t-en-01", "t-en-04"}
    with pytest.raises(ValueError, match="cannot resume"):
        resume_ids(rows, mode="rules-only", url="")
    with pytest.raises(ValueError, match="cannot resume"):
        resume_ids(rows, mode="engine", url="http://other")
    assert resume_ids([], **kw) == set()


class _Inner:
    async def complete(self, request: ChatRequest) -> Completion:
        return Completion(text="ok", finish_reason="stop")

    async def ready(self) -> bool:
        return True


async def test_counting_llm_counts_calls() -> None:
    llm = CountingLLM(_Inner())
    request = ChatRequest(messages=({"role": "user", "content": "hi"},), max_tokens=8)
    assert (await llm.complete(request)).text == "ok"
    await llm.complete(request)
    assert llm.calls == 2 and await llm.ready()
