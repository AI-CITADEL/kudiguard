"""Answer-quality results: per-item records (JSONL with a meta line), summaries per language, before/after comparison.

A results file starts with one ``{"meta": {...}}`` line (settings, grounding, thresholds, commit, url, time) and then
has one record per case and run, appended as they are produced, so an interrupted run keeps what it measured.

Populations: verdict accuracy and fallback rate count every answered record (errors excluded); coverage and the
English-side rates (violations, copied examples) count *scored* records only (fixed text scores 0; a failed
back-translation is unscored); the grounded rate counts the records where grounding was measured (``g.n``).
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
STAND_IN = "FallbackGrounding"  # grounding stand-in until workstream B merges: its numbers are not comparable
COMPARED_META = ("grounding", "cover_threshold", "forbid_threshold")  # must match for a fair comparison


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
    specific_coverage: float | None = None
    covered: list[bool] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    translation_hits: list[str] = field(default_factory=list)
    example_copied: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    grounded: bool | None = None
    evidence_mismatch: bool | None = None
    seconds: float = 0.0
    llm_ms: int = 0
    grounding_ms: int | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.error is None and self.coverage is not None

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
            specific_coverage=None if score.specific_coverage is None else round(score.specific_coverage, 4),
            covered=list(score.covered),
            violations=list(score.violations),
            translation_hits=list(score.translation_hits),
            example_copied=list(score.example_copied),
            unsupported=list(score.unsupported),
            grounded=score.grounded,
            evidence_mismatch=result.evidence_mismatch,
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
    n: int  # records without an error (each case x run)
    scored: int  # of those, with a coverage score
    errors: int
    verdict_accuracy: float | None
    coverage: float | None
    specific: float | None  # coverage of the case-specific points (not in the worked examples)
    violation_rate: float | None  # scored answers with any violation
    grounded_rate: float | None  # fully grounded, of the ``grounded_n`` answers where grounding was measured
    grounded_n: int
    fallback_rate: float | None
    copied_rate: float | None  # scored answers containing a worked-example phrase
    reproducible: float | None  # cases answered by the model in every run whose reply text is identical
    fallback_only: int  # cases with fixed fallback text in every run (left out of ``reproducible``)
    p50: float
    p95: float

    @classmethod
    def of(cls, records: Sequence[Record]) -> Summary:
        ok = [r for r in records if r.error is None]
        scored = [r for r in ok if r.scored]
        measured = [r.grounded for r in ok if r.grounded is not None]
        seconds = [r.seconds for r in ok]
        same, fallback_only = reproducibility(ok)
        return cls(
            n=len(ok),
            scored=len(scored),
            errors=len(records) - len(ok),
            verdict_accuracy=mean_or_none(r.verdict_ok for r in ok),
            coverage=mean_or_none(r.coverage for r in scored if r.coverage is not None),
            specific=mean_or_none(r.specific_coverage for r in scored if r.specific_coverage is not None),
            violation_rate=mean_or_none(bool(r.violations) for r in scored),
            grounded_rate=mean_or_none(measured),
            grounded_n=len(measured),
            fallback_rate=mean_or_none(r.outcome == Outcome.FALLBACK for r in ok),
            copied_rate=mean_or_none(bool(r.example_copied) for r in scored),
            reproducible=same,
            fallback_only=fallback_only,
            p50=percentile(seconds, 0.5),
            p95=percentile(seconds, 0.95),
        )

    def values(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def cells(self) -> list[str]:
        return [_cell(k, v) for k, v in self.values().items()]


HEADER = [
    "n", "scored", "errors", "verdict", "coverage", "specific", "violating", "grounded", "g.n", "fallback", "copied",
    "same", "fb-only", "p50", "p95",
]  # fmt: skip
_COUNTS = {"n", "scored", "errors", "grounded_n", "fallback_only"}
_SECONDS = {"p50", "p95"}
_SCORES = {"coverage", "specific"}


def _cell(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if key in _COUNTS:
        return str(value)
    if key in _SECONDS:
        return f"{value:.1f}s"
    if key in _SCORES:
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


def reproducibility(records: Sequence[Record]) -> tuple[float | None, int]:
    """(share of cases with identical reply text in every run, number of fallback-only cases).

    Only cases the model answered in every run count (fixed text is identical by construction); cases with fixed
    fallback text in every run are counted separately; mixed cases are in neither. None with fewer than two runs."""
    grouped = _group(records)
    repeated = [items for items in grouped.values() if len(items) > 1]
    answered = [items for items in repeated if all(r.outcome == Outcome.ANSWERED for r in items)]
    fallback_only = sum(all(r.outcome != Outcome.ANSWERED for r in items) for items in repeated)
    return mean_or_none(len({r.body for r in items}) == 1 for items in answered), fallback_only


def _table(rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) for row in rows)


def grounding_note(meta: Mapping[str, Any]) -> str:
    grounding = meta.get("grounding", "unknown")
    if grounding == STAND_IN:
        return f"grounding: {grounding} (stand-in: grounded rates not comparable with workstream B's check)"
    return f"grounding: {grounding}"


def format_summary(records: Sequence[Record], meta: Mapping[str, Any] | None = None) -> str:
    summaries = summarize(records)
    rows = [["", *HEADER], *([key, *s.cells()] for key, s in summaries.items())]
    runs = len({r.run for r in records})
    macro = _cell("coverage", macro_coverage(summaries))
    mismatches = sum(bool(r.evidence_mismatch) for r in records)
    return (
        f"{_table(rows)}\n  runs: {runs}  macro coverage (mean of languages): {macro}"
        f"  evidence rebuilt differently: {mismatches}\n  {grounding_note(meta or {})}"
    )


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
        return cls(
            verdicts=tuple(sorted({r.verdict for r in records})),
            coverage=mean_or_none(r.coverage for r in records if r.coverage is not None),
            violations=mean_or_none(len(r.violations) for r in records),
            grounded=mean_or_none(r.grounded for r in records if r.grounded is not None),
            bodies=frozenset(r.body for r in records),
        )


def _group(records: Iterable[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.id].append(record)
    return grouped


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def changed_cases(a: Sequence[Record], b: Sequence[Record]) -> list[str]:
    """Cases in both lists whose verdicts, mean scores or reply texts differ: ``id: what a -> b, ...``."""
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
            cells.append(f"{y - x:+d}")
        elif key in _SECONDS:
            cells.append(f"{y - x:+.1f}s")
        elif key in _SCORES:
            cells.append(f"{y - x:+.2f}")
        else:
            cells.append(f"{(y - x) * 100:+.0f}pp")
    return cells


def _settings(labels: tuple[str, str], metas: tuple[Mapping[str, Any], Mapping[str, Any]]) -> list[str]:
    lines = []
    for label, meta in zip(labels, metas, strict=True):
        if not meta:
            lines.append(f"{label}: (no meta)")
            continue
        thresholds = f"cover {meta.get('cover_threshold')} forbid {meta.get('forbid_threshold')}"
        lines.append(
            f"{label}: {meta.get('sampling')}; {grounding_note(meta)}; {thresholds}; commit {meta.get('commit')}"
        )
    a, b = metas
    if a and b:
        lines += [
            f"WARNING: {k} differs ({a.get(k)} vs {b.get(k)})" for k in COMPARED_META if a.get(k) != b.get(k)
        ]
        if a.get("commit") != b.get("commit"):
            lines.append(f"note: different commits ({a.get('commit')} vs {b.get('commit')})")
    return lines


def format_comparison(
    a: Sequence[Record],
    b: Sequence[Record],
    labels: tuple[str, str] = ("a", "b"),
    metas: tuple[Mapping[str, Any], Mapping[str, Any]] = ({}, {}),
) -> str:
    """Side by side on the (case, run) pairs scored in both files, with a delta row per language.

    One-sided case ids are listed; pairs scored on one side only (an error or failed back-translation) are counted."""
    ids = {r.id for r in a} & {r.id for r in b}
    only = [
        f"only in {label}: {', '.join(sorted({r.id for r in rs} - ids))}"
        for label, rs in zip(labels, (a, b), strict=True)
    ]
    pairs = {(r.id, r.run) for r in a if r.scored} & {(r.id, r.run) for r in b if r.scored}
    in_both = {(r.id, r.run) for r in a if r.id in ids} & {(r.id, r.run) for r in b if r.id in ids}
    a, b = [r for r in a if (r.id, r.run) in pairs], [r for r in b if (r.id, r.run) in pairs]
    sa, sb = summarize(a), summarize(b)
    rows: list[list[str]] = [["", "", *HEADER]]
    for key in [k for k in sa if k in sb and k != ALL] + ([ALL] if pairs else []):
        rows += [[key, labels[0], *sa[key].cells()], ["", labels[1], *sb[key].cells()]]
        rows.append(["", "delta", *_delta(sa[key], sb[key])])
    lines = _settings(labels, metas)
    lines += [line for line in only if not line.endswith(": ")]
    lines.append(f"compared: {len(pairs)} (case, run) pairs scored in both; {len(in_both - pairs)} left out")
    lines.append(_table(rows))
    changes = changed_cases(a, b)
    lines.append(f"changed cases ({len(changes)}):")
    lines += [f"  {line}" for line in changes] or ["  none"]
    return "\n".join(lines)
