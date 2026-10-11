"""Scoring for the frozen end-to-end test set T (``data/eval/e2e/cases.yaml``, v2.1).

Follows the scoring rules in the cases file header exactly:

* Verdict colour: red (``likely_scam``) and orange (``caution``) count as *flagged*; green (``no_flags`` and every
  non-check reply) counts as *not flagged*.
* ``label: scam`` passes when flagged; ``label: genuine`` passes when not flagged; ``label: question`` passes when
  the system answered the question (a model-composed reply). The colour shown with a question is ignored here: whether
  a red verdict "calls the situation a scam with no basis" needs reading, so such rows are counted
  (``question_shown_red``) and left to the independent judge.
* ``accept`` values also pass: ``scam`` and ``caution`` = flagged (colour-only scoring), ``genuine`` = not flagged,
  ``question`` = answered.
* ``red_flags`` are compared with the system's flags through :data:`FLAG_MAP` (``harassment`` = no match).
* ``must_advise`` / ``must_not`` are copied into every row for an independent judge and are never keyword-scored.

Rows whose run failed (``error`` set: timeout or exception) are excluded from every rate and counted separately.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kudiguard.domain import Flag, Outcome, Verdict
from kudiguard.engine import Answer, AnswerKind
from kudiguard.evaluation.stats import percentile
from kudiguard.llm.base import LLM, ChatRequest, Completion

ALL = "all"
FLAGGED_VERDICTS = frozenset({Verdict.LIKELY_SCAM.value, Verdict.CAUTION.value})

# Test vocabulary (cases.yaml ``red_flags_vocab`` + ``extra_red_flags``) -> the system's ``Flag`` enum. The 13 shared
# names are the same concept on both sides (the test author used the published flag names); ``harassment`` has no
# system flag and maps to None: it never counts as a hit or a miss (header rule v2.1).
FLAG_MAP: Mapping[str, Flag | None] = {
    "asks_credentials": Flag.ASKS_CREDENTIALS,
    "remote_access": Flag.REMOTE_ACCESS,
    "account_threat": Flag.ACCOUNT_THREAT,
    "upfront_fee": Flag.UPFRONT_FEE,
    "prize_lottery": Flag.PRIZE_LOTTERY,
    "guaranteed_returns": Flag.GUARANTEED_RETURNS,
    "wrong_transfer": Flag.WRONG_TRANSFER,
    "fake_payment": Flag.FAKE_PAYMENT,
    "money_request": Flag.MONEY_REQUEST,
    "suspicious_link": Flag.SUSPICIOUS_LINK,
    "impersonation": Flag.IMPERSONATION,
    "urgency": Flag.URGENCY,
    "secrecy": Flag.SECRECY,
    "harassment": None,
}

# Languages whose scam misses get a translation diagnosis (the English pivot the engine used is stored).
DIAGNOSED_LANGUAGES = frozenset({"yo", "ig"})

# Case language -> the language the engine is asked to answer in. KudiGuard has no separate Pidgin language: Pidgin
# messages arrive as English (the pipeline and the output guard detect Pidgin inside "en"). Rows keep the case's
# own ``lang`` for per-language scoring.
ENGINE_LANGUAGE: Mapping[str, str] = {"pcm": "en"}

# Fields an independent answer judge sees: no label, verdict, flags or notes.
JUDGE_FIELDS = ("id", "lang", "text", "reply", "must_advise", "must_not")


def load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(meta, cases) from a cases file. Raises ValueError on a red flag outside :data:`FLAG_MAP`."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = data["cases"]
    for case in cases:
        unknown = [name for name in case.get("red_flags") or () if name not in FLAG_MAP]
        if unknown:
            raise ValueError(f"{case['id']}: red flags outside the mapped vocabulary: {unknown}")
    return data.get("meta") or {}, cases


def expected_flags(case: Mapping[str, Any]) -> list[str]:
    """The case's red flags mapped to system flag values (unmatched vocabulary dropped), in order, no repeats."""
    mapped = (FLAG_MAP[name] for name in case.get("red_flags") or ())
    return list(dict.fromkeys(flag.value for flag in mapped if flag is not None))


def engine_language(lang: str) -> str:
    return ENGINE_LANGUAGE.get(lang, lang)


def is_flagged(verdict: str) -> bool:
    return verdict in FLAGGED_VERDICTS


def is_answered(kind: str, outcome: str) -> bool:
    """A question is answered when the reply is a model-composed answer (not fixed or fallback text)."""
    return kind == AnswerKind.CHECK.value and outcome == Outcome.ANSWERED.value


def outcome_matches(expected: str, *, flagged: bool, answered: bool) -> bool:
    """Whether one label/accept value is satisfied (colour-only scoring: ``caution`` is the same as flagged)."""
    if expected in ("scam", "caution"):
        return flagged
    if expected == "genuine":
        return not flagged
    if expected == "question":
        return answered
    raise ValueError(f"unknown outcome {expected!r}")


def passes(row: Mapping[str, Any]) -> bool:
    flagged = is_flagged(row["verdict"])
    answered = is_answered(row["kind"], row["outcome"])
    wanted = [row["label"], *(row.get("accept") or ())]
    return any(outcome_matches(value, flagged=flagged, answered=answered) for value in wanted)


def diagnose_miss(english: str, english_rule_flags: Sequence[str]) -> str:
    """Why a scam was missed after translation: ``no_translation`` (no usable English pivot),
    ``cue_survived_translation`` (the rules find a red flag in the English pivot, so the cue was not lost in
    translation) or ``no_rule_cue_in_translation`` (the rules find nothing in the English pivot: either the
    translation lost the cue or the rules miss it in English too; read ``english`` to tell which)."""
    if not english.strip():
        return "no_translation"
    return "cue_survived_translation" if english_rule_flags else "no_rule_cue_in_translation"


def record(
    case: Mapping[str, Any],
    answer: Answer | None,
    *,
    reply: str = "",
    latency_s: float = 0.0,
    model_calls: int = 0,
    error: str = "",
    english_rule_flags: Sequence[str] = (),
) -> dict[str, Any]:
    """One results row. ``answer`` None = the run failed (``error`` says why)."""
    row: dict[str, Any] = {
        "id": case["id"],
        "lang": case["lang"],
        "engine_lang": engine_language(case["lang"]),
        "label": case["label"],
        "accept": list(case.get("accept") or ()),
        "subtype": case.get("subtype", ""),
        "channel": case.get("channel", ""),
        "voice": bool(case.get("voice", False)),
        "text": case["text"],
        "expected_flags": expected_flags(case),
        "must_advise": list(case.get("must_advise") or ()),
        "must_not": list(case.get("must_not") or ()),
        "latency_s": round(latency_s, 3),
        "model_calls": model_calls,
        "error": error,
    }
    if answer is None:
        row.update(verdict="", kind="", outcome=Outcome.ERROR.value, flags=[], reply="", body="", english="")
        row.update(reply_english="", translation_rejections=[], llm_ms=0, passed=False, miss_diagnosis="")
        return row
    row.update(
        verdict=answer.verdict.value,
        kind=answer.kind.value,
        outcome=answer.outcome.value,
        flags=[flag.value for flag in answer.flags],
        reply=reply,
        body=answer.body,
        english=answer.english,
        reply_english=answer.reply_english,
        translation_rejections=list(answer.translation_rejections),
        llm_ms=answer.llm_ms,
    )
    row["passed"] = passes(row)
    missed = row["label"] == "scam" and not is_flagged(row["verdict"])
    diagnosed = missed and row["lang"] in DIAGNOSED_LANGUAGES
    row["english_rule_flags"] = list(english_rule_flags) if diagnosed else []
    row["miss_diagnosis"] = diagnose_miss(answer.english, english_rule_flags) if diagnosed else ""
    return row


@dataclass
class Group:
    """Counts for one language (or all)."""

    n: int = 0
    errors: int = 0
    passed: int = 0
    scam: int = 0
    scam_flagged: int = 0
    genuine: int = 0
    genuine_flagged: int = 0
    question: int = 0
    question_answered: int = 0
    question_shown_red: int = 0
    flag_hits: int = 0
    flags_expected: int = 0
    flags_predicted: int = 0
    fallbacks: int = 0
    rejected: int = 0
    voice: int = 0
    model_calls: int = 0
    latencies: list[float] = field(default_factory=list)
    diagnoses: Counter[str] = field(default_factory=Counter)

    def add(self, row: Mapping[str, Any]) -> None:
        self.n += 1
        self.voice += bool(row.get("voice"))
        if row.get("error"):
            self.errors += 1
            return
        flagged = is_flagged(row["verdict"])
        self.passed += passes(row)
        self.latencies.append(float(row["latency_s"]))
        self.model_calls += int(row.get("model_calls", 0))
        self.fallbacks += row["outcome"] == Outcome.FALLBACK.value
        self.rejected += bool(row.get("translation_rejections"))
        label = row["label"]
        if label == "scam":
            self.scam += 1
            self.scam_flagged += flagged
            expected, predicted = set(row["expected_flags"]), set(row["flags"])
            self.flag_hits += len(expected & predicted)
            self.flags_expected += len(expected)
            self.flags_predicted += len(predicted)
            if row.get("miss_diagnosis"):
                self.diagnoses[row["miss_diagnosis"]] += 1
        elif label == "genuine":
            self.genuine += 1
            self.genuine_flagged += flagged
        else:
            self.question += 1
            self.question_answered += is_answered(row["kind"], row["outcome"])
            self.question_shown_red += row["verdict"] == Verdict.LIKELY_SCAM.value

    @property
    def scored(self) -> int:
        return self.n - self.errors


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def metrics(group: Group) -> dict[str, Any]:
    return {
        "n": group.n,
        "errors": group.errors,
        "voice_cases": group.voice,
        "pass_rate": _rate(group.passed, group.scored),
        "scam_recall": _rate(group.scam_flagged, group.scam),
        "genuine_false_positive_rate": _rate(group.genuine_flagged, group.genuine),
        "question_answered_rate": _rate(group.question_answered, group.question),
        "question_shown_red": group.question_shown_red,
        "flag_recall": _rate(group.flag_hits, group.flags_expected),
        "flag_precision": _rate(group.flag_hits, group.flags_predicted),
        "latency_p50_s": percentile(group.latencies, 0.5),
        "latency_p95_s": percentile(group.latencies, 0.95),
        "fallback_rate": _rate(group.fallbacks, group.scored),
        "translation_rejection_rate": _rate(group.rejected, group.scored),
        "model_calls": group.model_calls,
        "counts": {
            "scam": [group.scam_flagged, group.scam],
            "genuine_flagged": [group.genuine_flagged, group.genuine],
            "question_answered": [group.question_answered, group.question],
        },
        "miss_diagnosis": dict(sorted(group.diagnoses.items())),
    }


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Metrics per language plus ``all``. A case id seen twice (a resumed run) counts once: the last row wins."""
    latest = {row["id"]: row for row in rows}
    groups: dict[str, Group] = defaultdict(Group)
    for row in latest.values():
        groups[row["lang"]].add(row)
        groups[ALL].add(row)
    return {lang: metrics(groups[lang]) for lang in [*sorted(set(groups) - {ALL}), ALL]}


def _pct(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.0%}"


def format_summary(summary: Mapping[str, Mapping[str, Any]], *, mode: str = "") -> str:
    head = f"End-to-end test set T{f' ({mode})' if mode else ''}"
    lines = [
        head,
        "  lang    n  err  pass  scam-recall  genuine-FPR  q-answered  flag-R  flag-P  fallback  tr-reject"
        "  p50 s  p95 s",
    ]
    for lang, m in summary.items():
        lines.append(
            f"  {lang:4} {m['n']:4d} {m['errors']:4d} {_pct(m['pass_rate'])}   {_pct(m['scam_recall'])}"
            f"        {_pct(m['genuine_false_positive_rate'])}       {_pct(m['question_answered_rate'])}"
            f"     {_pct(m['flag_recall'])}   {_pct(m['flag_precision'])}    {_pct(m['fallback_rate'])}"
            f"     {_pct(m['translation_rejection_rate'])}  {m['latency_p50_s']:5.1f}  {m['latency_p95_s']:5.1f}"
        )
    total = summary[ALL]
    counts = total["counts"]
    lines.append(
        f"  all counts: scam flagged {counts['scam'][0]}/{counts['scam'][1]}, genuine flagged "
        f"{counts['genuine_flagged'][0]}/{counts['genuine_flagged'][1]}, questions answered "
        f"{counts['question_answered'][0]}/{counts['question_answered'][1]} "
        f"(shown red: {total['question_shown_red']}, basis left to the judge), model calls {total['model_calls']}"
    )
    for lang, m in summary.items():
        if lang != ALL and m["miss_diagnosis"]:
            lines.append(f"  {lang} scam misses: {m['miss_diagnosis']}")
    lines.append(
        f"  {total['voice_cases']} voice cases were run as text (no TTS step). must_advise / must_not are not"
        " scored here: they go to the independent judge."
    )
    return "\n".join(lines)


def judge_item(row: Mapping[str, Any]) -> dict[str, Any]:
    """What a blind answer judge sees: the user's message, the reply shown, and the case's advice criteria."""
    return {name: row[name] for name in JUDGE_FIELDS}


class CountingLLM:
    """Wraps an LLM and counts the calls made through it (per case: read and reset ``calls``)."""

    def __init__(self, inner: LLM) -> None:
        self.inner = inner
        self.calls = 0

    async def complete(self, request: ChatRequest) -> Completion:
        self.calls += 1
        return await self.inner.complete(request)

    async def ready(self) -> bool:
        return await self.inner.ready()
