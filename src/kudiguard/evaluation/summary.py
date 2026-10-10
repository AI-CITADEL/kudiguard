"""Answer-quality results: per-item records (JSONL with a meta line), summaries per language, before/after comparison.

A results file starts with one ``{"meta": {...}}`` line (settings, grounding, thresholds, commit, url, time) and then
has one record per case and run, appended as they are produced, so an interrupted run keeps what it measured.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from kudiguard.domain import Outcome
from kudiguard.evaluation.answers import AnswerCase
from kudiguard.evaluation.runner import CaseResult
from kudiguard.evaluation.stats import mean_or_none, percentile

ALL = "all"


@dataclass(frozen=True, slots=True)
class Record:
    id: str
    lang: str
    kind: str
    run: int
    error: str | None = None  # the case crashed: excluded from every metric but the error count
    verdict: str = ""
    expected: list[str] = field(default_factory=list)
    verdict_ok: bool = False
    outcome: str = ""
    intent: str | None = None
    flags: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)  # output-guard findings that forced a retry or fallback
    body: str = ""
    english: str | None = None  # back-translation; None = failed (unscored)
    coverage: float | None = None
    covered: list[bool] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    translation_hits: list[str] = field(default_factory=list)
    example_copied: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    grounded: bool | None = None
    seconds: float = 0.0
    llm_ms: int = 0
    grounding_ms: int | None = None
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, case: AnswerCase, result: CaseResult, run: int) -> Record:
        score, answer = result.score, result.answer
        return cls(
            id=case.id,
            lang=case.lang,
            kind=case.kind.value,
            run=run,
            verdict=answer.verdict.value,
            expected=[v.value for v in case.verdicts],
            verdict_ok=score.verdict_ok,
            outcome=answer.outcome.value,
            intent=None if answer.intent is None else answer.intent.value,
            flags=[f.value for f in answer.flags],
            issues=[i.value for i in answer.issues],
            body=answer.body,
            english=result.observation.english,
            coverage=None if score.coverage is None else round(score.coverage, 4),
            covered=list(score.covered),
            violations=list(score.violations),
            translation_hits=list(score.translation_hits),
            example_copied=list(score.example_copied),
            unsupported=list(score.unsupported),
            grounded=score.grounded,
            seconds=round(result.seconds, 3),
            llm_ms=answer.llm_ms,
            grounding_ms=result.grounding_ms,
            sources=list(answer.sources),
        )

    @classmethod
    def failed(cls, case: AnswerCase, run: int, error: str) -> Record:
        return cls(id=case.id, lang=case.lang, kind=case.kind.value, run=run, error=error)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


# --- files ------------------------------------------------------------------------------------------------


class ResultsWriter:
    """Writes the meta line, then appends records one by one (flushed)."""

    def __init__(self, path: Path, meta: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = path.open("w", encoding="utf-8")
        self._write(json.dumps({"meta": dict(meta)}, ensure_ascii=False))

    def add(self, record: Record) -> None:
        self._write(record.to_json())

    def close(self) -> None:
        self._file.close()

    def _write(self, line: str) -> None:
        self._file.write(line + "\n")
        self._file.flush()


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_records(path: Path) -> list[Record]:
    return [Record(**item) for item in _lines(path) if "meta" not in item]


def read_meta(path: Path) -> dict[str, Any]:
    return next((dict(item["meta"]) for item in _lines(path) if "meta" in item), {})


# --- summaries --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Summary:
    n: int  # answered cases (errors excluded)
    errors: int
    verdict_accuracy: float | None
    coverage: (
        float | None
    )  # mean over scored answers (fallback text scores 0; failed back-translation: left out)
    violation_rate: float | None  # answers with any violation
    grounded_rate: float | None  # fully grounded, of the model-composed answers where grounding was measured
    fallback_rate: float | None
    copied_rate: float | None  # answers containing a worked-example phrase
    reproducible: float | None  # cases with identical reply text in every run
    p50: float
    p95: float

    @classmethod
    def of(cls, records: Sequence[Record]) -> Summary:
        ok = [r for r in records if r.error is None]
        seconds = [r.seconds for r in ok]
        return cls(
            n=len(ok),
            errors=len(records) - len(ok),
            verdict_accuracy=mean_or_none(r.verdict_ok for r in ok),
            coverage=mean_or_none(r.coverage for r in ok if r.coverage is not None),
            violation_rate=mean_or_none(bool(r.violations) for r in ok),
            grounded_rate=mean_or_none(r.grounded for r in ok if r.grounded is not None),
            fallback_rate=mean_or_none(r.outcome == Outcome.FALLBACK for r in ok),
            copied_rate=mean_or_none(bool(r.example_copied) for r in ok),
            reproducible=reproducibility(ok),
            p50=percentile(seconds, 0.5),
            p95=percentile(seconds, 0.95),
        )

    def values(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def cells(self) -> list[str]:
        return [_cell(k, v) for k, v in self.values().items()]


HEADER = [
    "n",
    "errors",
    "verdict",
    "coverage",
    "violating",
    "grounded",
    "fallback",
    "copied",
    "same",
    "p50",
    "p95",
]
_COUNTS = {"n", "errors"}
_SECONDS = {"p50", "p95"}


def _cell(key: str, value: float | None) -> str:
    if value is None:
        return "-"
    if key in _COUNTS:
        return str(value)
    if key in _SECONDS:
        return f"{value:.1f}s"
    if key == "coverage":
        return f"{value:.2f}"
    return f"{value:.0%}"


def summarize(records: Sequence[Record]) -> dict[str, Summary]:
    """Per language (sorted) and overall (``ALL``)."""
    by_lang: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_lang[record.lang].append(record)
    summaries = {lang: Summary.of(items) for lang, items in sorted(by_lang.items())}
    summaries[ALL] = Summary.of(records)
    return summaries


def macro_coverage(summaries: Mapping[str, Summary]) -> float | None:
    """Mean of the per-language coverages: each language counts the same, whatever its number of cases."""
    return mean_or_none(s.coverage for k, s in summaries.items() if k != ALL and s.coverage is not None)


def reproducibility(records: Sequence[Record]) -> float | None:
    """Share of cases whose reply text is identical in every run; None with fewer than two runs."""
    replies: dict[str, list[str]] = defaultdict(list)
    for record in records:
        replies[record.id].append(record.body)
    return mean_or_none(len(set(b)) == 1 for b in replies.values() if len(b) > 1)


def _table(rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) for row in rows)


def format_summary(records: Sequence[Record]) -> str:
    summaries = summarize(records)
    rows = [["", *HEADER], *([key, *s.cells()] for key, s in summaries.items())]
    runs = len({r.run for r in records})
    macro = _cell("coverage", macro_coverage(summaries))
    return f"{_table(rows)}\n  runs: {runs}  macro coverage (mean of languages): {macro}"


# --- comparison -------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseView:
    verdicts: tuple[str, ...]
    coverage: float | None
    violations: float | None
    grounded: float | None
    bodies: frozenset[str]

    @classmethod
    def of(cls, records: Sequence[Record]) -> CaseView:
        ok = [r for r in records if r.error is None]
        return cls(
            verdicts=tuple(sorted({r.verdict for r in ok})),
            coverage=mean_or_none(r.coverage for r in ok if r.coverage is not None),
            violations=mean_or_none(len(r.violations) for r in ok),
            grounded=mean_or_none(r.grounded for r in ok if r.grounded is not None),
            bodies=frozenset(r.body for r in ok),
        )


def _group(records: Iterable[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.id].append(record)
    return grouped


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def changed_cases(a: Sequence[Record], b: Sequence[Record]) -> list[str]:
    """Cases in both files whose verdicts, mean scores or reply texts differ: ``id: what a -> b, ...``."""
    before, after = _group(a), _group(b)
    lines = []
    for case in sorted(before.keys() & after.keys()):
        old, new = CaseView.of(before[case]), CaseView.of(after[case])
        diffs = []
        if old.verdicts != new.verdicts:
            diffs.append(f"verdict {'/'.join(old.verdicts)} -> {'/'.join(new.verdicts)}")
        for metric in ("coverage", "violations", "grounded"):
            x, y = _fmt(getattr(old, metric)), _fmt(getattr(new, metric))
            if x != y:
                diffs.append(f"{metric} {x} -> {y}")
        if old.bodies != new.bodies:
            diffs.append("reply text changed")
        if diffs:
            lines.append(f"{case}: " + ", ".join(diffs))
    return lines


def _delta(a: Summary, b: Summary) -> list[str]:
    cells = []
    for key, x in a.values().items():
        y = b.values()[key]
        if x is None or y is None:
            cells.append("-")
        elif key in _COUNTS:
            cells.append(f"{y - x:+g}")
        elif key in _SECONDS:
            cells.append(f"{y - x:+.1f}s")
        elif key == "coverage":
            cells.append(f"{y - x:+.2f}")
        else:
            cells.append(f"{(y - x) * 100:+.0f}pp")
    return cells


def format_comparison(
    a: Sequence[Record],
    b: Sequence[Record],
    labels: tuple[str, str] = ("a", "b"),
    settings: tuple[str, str] = ("", ""),
) -> str:
    """Side by side on the cases both files contain (one-sided ids are listed), with a delta row per language."""
    common = {r.id for r in a} & {r.id for r in b}
    only = [
        f"only in {label}: {', '.join(sorted({r.id for r in rs} - common))}"
        for label, rs in zip(labels, (a, b), strict=True)
    ]
    a, b = [r for r in a if r.id in common], [r for r in b if r.id in common]
    sa, sb = summarize(a), summarize(b)
    rows: list[list[str]] = [["", "", *HEADER]]
    for key in [k for k in sa if k in sb and k != ALL] + ([ALL] if common else []):
        rows += [[key, labels[0], *sa[key].cells()], ["", labels[1], *sb[key].cells()]]
        rows.append(["", "delta", *_delta(sa[key], sb[key])])
    lines = [f"{label}: {setting or '(no meta)'}" for label, setting in zip(labels, settings, strict=True)]
    lines += [line for line in only if not line.endswith(": ")]
    lines.append(_table(rows))
    changes = changed_cases(a, b)
    lines.append(f"changed cases ({len(changes)}):")
    lines += [f"  {line}" for line in changes] or ["  none"]
    return "\n".join(lines)
