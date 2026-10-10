"""Answer-quality results: per-item records (JSONL), summaries per language, run-to-run comparison."""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from kudiguard.domain import Outcome
from kudiguard.evaluation.answers import AnswerCase, CaseResult

ALL = "all"


@dataclass(frozen=True, slots=True)
class Record:
    id: str
    lang: str
    kind: str
    run: int
    verdict: str
    expected: list[str]
    verdict_ok: bool
    outcome: str
    body: str
    english: str
    coverage: float
    covered: list[bool]
    violations: list[str]
    unsupported: list[str]
    grounded: bool | None
    seconds: float
    sources: list[str]

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
            body=answer.body,
            english=result.observation.english,
            coverage=round(score.coverage, 4),
            covered=list(score.covered),
            violations=list(score.violations),
            unsupported=list(score.unsupported),
            grounded=score.grounded,
            seconds=round(result.seconds, 3),
            sources=list(answer.sources),
        )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def read_records(path: Path) -> list[Record]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Record(**json.loads(line)) for line in lines if line.strip()]


def write_records(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (as in ``scripts/eval_scenarios.py``); 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


@dataclass(frozen=True, slots=True)
class Summary:
    n: int
    verdict_accuracy: float
    coverage: float
    violation_rate: float  # answers with any violation
    grounded_rate: float | None  # fully grounded answers, of those where grounding was measured
    fallback_rate: float
    p50: float
    p95: float

    @classmethod
    def of(cls, records: Sequence[Record]) -> Summary:
        measured = [r.grounded for r in records if r.grounded is not None]
        seconds = [r.seconds for r in records]
        return cls(
            n=len(records),
            verdict_accuracy=fmean(r.verdict_ok for r in records),
            coverage=fmean(r.coverage for r in records),
            violation_rate=fmean(bool(r.violations) for r in records),
            grounded_rate=fmean(measured) if measured else None,
            fallback_rate=fmean(r.outcome == Outcome.FALLBACK for r in records),
            p50=percentile(seconds, 0.5),
            p95=percentile(seconds, 0.95),
        )

    def cells(self) -> list[str]:
        grounded = "-" if self.grounded_rate is None else f"{self.grounded_rate:.0%}"
        return [
            str(self.n),
            f"{self.verdict_accuracy:.0%}",
            f"{self.coverage:.2f}",
            f"{self.violation_rate:.0%}",
            grounded,
            f"{self.fallback_rate:.0%}",
            f"{self.p50:.1f}s",
            f"{self.p95:.1f}s",
        ]


HEADER = ["n", "verdict", "coverage", "violating", "grounded", "fallback", "p50", "p95"]


def summarize(records: Sequence[Record]) -> dict[str, Summary]:
    """Per language (sorted) and overall (``ALL``). ``records`` must not be empty."""
    by_lang: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_lang[record.lang].append(record)
    summaries = {lang: Summary.of(items) for lang, items in sorted(by_lang.items())}
    summaries[ALL] = Summary.of(records)
    return summaries


def reproducibility(records: Sequence[Record]) -> float | None:
    """Share of cases whose reply text is identical in every run; None with fewer than two runs."""
    replies: dict[str, list[str]] = defaultdict(list)
    for record in records:
        replies[record.id].append(record.body)
    repeated = [bodies for bodies in replies.values() if len(bodies) > 1]
    if not repeated:
        return None
    return fmean(len(set(bodies)) == 1 for bodies in repeated)


def _table(rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) for row in rows)


def format_summary(records: Sequence[Record]) -> str:
    summaries = summarize(records)
    rows = [["", *HEADER], *([key, *s.cells()] for key, s in summaries.items())]
    repro = reproducibility(records)
    runs = len({r.run for r in records})
    tail = f"identical replies across {runs} runs: {_fmt_rate(repro)}"
    return _table(rows) + "\n  " + tail


@dataclass(frozen=True, slots=True)
class CaseMeans:
    verdict_ok: float
    coverage: float
    violations: float
    grounded: float | None

    @classmethod
    def of(cls, records: Sequence[Record]) -> CaseMeans:
        measured = [r.grounded for r in records if r.grounded is not None]
        return cls(
            verdict_ok=fmean(r.verdict_ok for r in records),
            coverage=fmean(r.coverage for r in records),
            violations=fmean(len(r.violations) for r in records),
            grounded=fmean(measured) if measured else None,
        )


def _per_case(records: Sequence[Record]) -> dict[str, CaseMeans]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.id].append(record)
    return {case: CaseMeans.of(items) for case, items in grouped.items()}


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def changed_cases(a: Sequence[Record], b: Sequence[Record]) -> list[str]:
    """Cases (present in both) whose mean scores differ, one line each: ``id: metric a -> b, ...``."""
    before, after = _per_case(a), _per_case(b)
    lines = []
    for case in sorted(before.keys() & after.keys()):
        old, new = dataclasses.asdict(before[case]), dataclasses.asdict(after[case])
        diffs = [f"{k} {_fmt(old[k])} -> {_fmt(new[k])}" for k in old if _fmt(old[k]) != _fmt(new[k])]
        if diffs:
            lines.append(f"{case}: " + ", ".join(diffs))
    return lines


def format_comparison(a: Sequence[Record], b: Sequence[Record], labels: tuple[str, str] = ("a", "b")) -> str:
    sa, sb = summarize(a), summarize(b)
    rows: list[list[str]] = [["", "", *HEADER]]
    for key in [*(k for k in sa if k in sb and k != ALL), ALL]:
        rows += [[key, labels[0], *sa[key].cells()], ["", labels[1], *sb[key].cells()]]
    ra, rb = reproducibility(a), reproducibility(b)
    repro = f"identical replies across runs: {_fmt_rate(ra)} -> {_fmt_rate(rb)}"
    changes = changed_cases(a, b)
    body = "\n".join(f"  {line}" for line in changes) if changes else "  none"
    return f"{_table(rows)}\n  {repro}\nchanged cases ({len(changes)}):\n{body}"


def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"
