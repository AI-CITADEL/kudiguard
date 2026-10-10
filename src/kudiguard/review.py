"""Native-speaker review kit: sample real pilot replies for grading, then score the graded sheets.

Export (``kudiguard review export``): a stratified random sample of model-written check replies from the evidence
export (``Store.export_interactions``), balanced across languages and between scam checks and questions, written as
a UTF-8 (BOM) CSV that opens cleanly in Excel and Google Sheets. Each row carries the redacted message, its English
pivot, the reply and the knowledge-base facts it was grounded on. ``Answer.sources`` is not stored, so the facts are
*re-derived* by repeating the deterministic knowledge search on the stored English and flags: packs edited since the
reply was sent (or redaction of the stored English) can make them differ slightly from what the model saw.

Score (``kudiguard review score``): validates the grades in one or more sheets (one per reviewer, or a ``reviewer``
column) and reports per language. An item counts as unfaithful or harmful when *any* reviewer marked it so (safety
first); scales are averaged per item, then across items. Agreement is measured on items graded by two or more
reviewers: exact agreement for the yes/no questions, mean absolute difference for the 1-5 scales. See
docs/NATIVE_REVIEW.md for the rubric.
"""

from __future__ import annotations

import csv
import io
import random
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from kudiguard.domain import Flag, Outcome, Verdict
from kudiguard.knowledge.models import Query
from kudiguard.knowledge.service import KnowledgeBase
from kudiguard.privacy import UserHasher
from kudiguard.prompts import facts_block
from kudiguard.reporting import USER_CHANNELS
from kudiguard.storage.export import spreadsheet_cell

DEFAULT_SAMPLE = 50
DEFAULT_LANGUAGES = ("ha", "yo", "ig")
CHECK, QUESTION = "check", "question"  # verdict groups the sample is balanced across

CONTEXT_COLUMNS = (
    "review_id",
    "language",
    "channel",
    "verdict",
    "user_message",
    "english_pivot",
    "reply",
    "facts_rederived",
)
GRADE_COLUMNS = ("accuracy", "fluency", "faithful", "harmful_or_misleading")
COLUMNS = (*CONTEXT_COLUMNS, *GRADE_COLUMNS, "reviewer", "notes")
SCALE = range(1, 6)
_YES, _NO = frozenset({"yes", "y"}), frozenset({"no", "n"})


# --- export ---------------------------------------------------------------------------------------------------


def verdict_group(verdict: str) -> str:
    return QUESTION if verdict == Verdict.NOT_A_CHECK else CHECK


def eligible(
    rows: Iterable[Mapping[str, Any]],
    *,
    languages: Collection[str] = DEFAULT_LANGUAGES,
    channels: Collection[str] = USER_CHANNELS,
    exclude_users: Collection[str] = (),
    since: date | None = None,
) -> list[Mapping[str, Any]]:
    """Model-written replies to real checks and questions.

    Left out: fixed text (greetings, commands), fallbacks (the model failed or was blocked: nothing to grade) and
    button follow-ups (they restate the knowledge base's steps or the user's own message, not an answer)."""
    return [
        row
        for row in rows
        if row.get("outcome") == Outcome.ANSWERED
        and row.get("verdict")
        and row.get("reply")
        and row.get("english")
        and not str(row.get("intent") or "").startswith("button:")
        and row.get("language") in languages
        and row.get("channel") in channels
        and row.get("user") not in exclude_users
        and (since is None or str(row.get("created_at", ""))[:10] >= since.isoformat())
    ]


def stratified_sample(
    rows: Sequence[Mapping[str, Any]], n: int, *, seed: int, languages: Sequence[str] = DEFAULT_LANGUAGES
) -> list[Mapping[str, Any]]:
    """Up to ``n`` rows, round-robin over (language, verdict group) strata in random order within each stratum.

    Strata with fewer rows than their share leave it to the others. The same rows and seed give the same sample."""
    rng = random.Random(seed)
    strata: list[list[Mapping[str, Any]]] = []
    for language in languages:
        for group in (CHECK, QUESTION):
            members = sorted(
                (r for r in rows if r["language"] == language and verdict_group(r["verdict"]) == group),
                key=lambda r: int(r["id"]),
            )
            rng.shuffle(members)
            strata.append(members)
    sample: list[Mapping[str, Any]] = []
    while len(sample) < n and (live := [s for s in strata if s]):
        rng.shuffle(live)  # which strata get the remainder when n does not divide evenly
        for stratum in live[: n - len(sample)]:
            sample.append(stratum.pop())
    return sample


def format_facts(facts: Sequence[str], actions: Sequence[str]) -> str:
    lines = [f"{number}. {fact}" for number, fact in enumerate(facts, start=1)] or ["(no facts found)"]
    if actions:
        lines += ["Steps:", *(f"- {action}" for action in actions)]
    return "\n".join(lines)


async def grounding(knowledge: KnowledgeBase, row: Mapping[str, Any]) -> str:
    """Re-derive the facts and steps a reply was grounded on (the engine's search on the stored English + flags)."""
    query = Query(
        text=str(row["english"]),
        flags=tuple(Flag(f) for f in row.get("flags") or ()),
        language=row["language"],
    )
    facts, actions = facts_block(await knowledge.search(query))
    return format_facts(facts, actions)


async def review_rows(
    sample: Iterable[Mapping[str, Any]], *, hasher: UserHasher, knowledge: KnowledgeBase
) -> list[dict[str, str]]:
    """Sheet rows: an opaque review id instead of any database or user id; grading columns left empty."""
    rows = [
        {
            "review_id": hasher.review_id(int(row["id"])),
            "language": str(row["language"]),
            "channel": str(row["channel"]),
            "verdict": str(row["verdict"]),
            "user_message": str(row.get("transcript") or ""),
            "english_pivot": str(row["english"]),
            "reply": str(row["reply"]),
            "facts_rederived": await grounding(knowledge, row),
        }
        for row in sample
    ]
    return sorted(rows, key=lambda r: (r["language"], r["review_id"]))  # one block per language; groups mixed


def write_sheet(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    """UTF-8 with BOM (Excel otherwise mangles ẹ, ọ, ṣ, ɓ, ƙ…), CRLF rows, formula-safe cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: spreadsheet_cell(row.get(column, "")) for column in COLUMNS})


# --- score ----------------------------------------------------------------------------------------------------


class GradeError(ValueError):
    """A sheet cannot be scored; the message names the file and row."""


@dataclass(frozen=True, slots=True)
class Grade:
    review_id: str
    language: str
    reviewer: str
    accuracy: int
    fluency: int
    faithful: bool
    harmful: bool


@dataclass(frozen=True, slots=True)
class Sheet:
    grades: tuple[Grade, ...]
    items: Mapping[str, str]  # review id -> language, graded or not


def _text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # Excel's plain "CSV" save uses the ANSI code page; grades are ASCII anyway
        return raw.decode("cp1252", errors="replace")


def _scale(value: str, column: str, where: str) -> int:
    try:
        number = float(value)
    except ValueError:
        number = 0.0
    if not number.is_integer() or int(number) not in SCALE:
        raise GradeError(f"{where}: {column} must be a whole number from 1 to 5, got {value!r}")
    return int(number)


def _yes_no(value: str, column: str, where: str) -> bool:
    word = value.lower()
    if word not in _YES | _NO:
        raise GradeError(f"{where}: {column} must be yes or no, got {value!r}")
    return word in _YES


def read_sheet(path: Path) -> Sheet:
    """Parse one graded sheet. Rows with no grades are ungraded; a partly graded or malformed row is an error."""
    try:
        text = _text(path.read_bytes())
    except OSError as exc:
        raise GradeError(f"{path}: cannot read ({exc.strerror})") from exc
    delimiter = ";" if text.partition("\n")[0].count(";") > text.partition("\n")[0].count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if missing := [
        c for c in ("review_id", "language", *GRADE_COLUMNS) if c not in (reader.fieldnames or ())
    ]:
        raise GradeError(f"{path.name}: not a review sheet (missing columns {', '.join(missing)})")
    grades: list[Grade] = []
    items: dict[str, str] = {}
    for line, row in enumerate(reader, start=2):  # row 1 is the header, as numbered in a spreadsheet
        cells = {k: (v or "").strip().lstrip("'") for k, v in row.items() if k is not None}
        where = f"{path.name} row {line}"
        review_id = cells["review_id"]
        if not review_id:
            continue
        if review_id in items:
            raise GradeError(f"{where}: {review_id} appears twice")
        items[review_id] = cells["language"]
        values = [cells[c] for c in GRADE_COLUMNS]
        if not any(values):
            continue
        if blank := [c for c, v in zip(GRADE_COLUMNS, values, strict=True) if not v]:
            raise GradeError(f"{where}: {review_id} is partly graded (missing {', '.join(blank)})")
        grades.append(
            Grade(
                review_id=review_id,
                language=cells["language"],
                reviewer=cells.get("reviewer") or path.stem,
                accuracy=_scale(cells["accuracy"], "accuracy", where),
                fluency=_scale(cells["fluency"], "fluency", where),
                faithful=_yes_no(cells["faithful"], "faithful", where),
                harmful=_yes_no(cells["harmful_or_misleading"], "harmful_or_misleading", where),
            )
        )
    return Sheet(tuple(grades), items)


@dataclass(frozen=True, slots=True)
class LanguageScore:
    language: str  # "all" for the overall row
    graded: int
    ungraded: int
    accuracy: float | None
    fluency: float | None
    faithful_rate: float | None
    harmful_rate: float | None


@dataclass(frozen=True, slots=True)
class Agreement:
    items: int  # graded by two or more reviewers
    faithful: float | None  # share of reviewer pairs giving the same answer
    harmful: float | None
    accuracy_mad: float | None  # mean absolute difference between reviewer pairs, in scale points
    fluency_mad: float | None


@dataclass(frozen=True, slots=True)
class Scorecard:
    languages: tuple[LanguageScore, ...]  # per language, then "all"
    reviewers: tuple[str, ...]
    agreement: Agreement
    harmful: tuple[str, ...]  # review ids any reviewer flagged, for follow-up
    unfaithful: tuple[str, ...]


def _mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _language_score(language: str, by_item: Mapping[str, Sequence[Grade]], ungraded: int) -> LanguageScore:
    graded = list(by_item.values())
    return LanguageScore(
        language=language,
        graded=len(graded),
        ungraded=ungraded,
        accuracy=_mean([mean(g.accuracy for g in item) for item in graded]),
        fluency=_mean([mean(g.fluency for g in item) for item in graded]),
        faithful_rate=_mean([float(all(g.faithful for g in item)) for item in graded]),
        harmful_rate=_mean([float(any(g.harmful for g in item)) for item in graded]),
    )


def _agreement(by_item: Mapping[str, Sequence[Grade]]) -> Agreement:
    pairs = [pair for item in by_item.values() for pair in combinations(item, 2)]
    return Agreement(
        items=sum(1 for item in by_item.values() if len(item) > 1),
        faithful=_mean([float(a.faithful == b.faithful) for a, b in pairs]),
        harmful=_mean([float(a.harmful == b.harmful) for a, b in pairs]),
        accuracy_mad=_mean([abs(a.accuracy - b.accuracy) for a, b in pairs]),
        fluency_mad=_mean([abs(a.fluency - b.fluency) for a, b in pairs]),
    )


def score(sheets: Sequence[Sheet]) -> Scorecard:
    """Combine sheets; a reviewer may grade an item only once (across all sheets)."""
    by_item: dict[str, list[Grade]] = {}
    for grade in (g for sheet in sheets for g in sheet.grades):
        if any(g.reviewer == grade.reviewer for g in by_item.get(grade.review_id, ())):
            raise GradeError(f"{grade.review_id} graded twice by reviewer {grade.reviewer!r}")
        by_item.setdefault(grade.review_id, []).append(grade)
    items = {rid: language for sheet in sheets for rid, language in sheet.items.items()}
    ungraded = [rid for rid in items if rid not in by_item]
    languages = list(dict.fromkeys(items.values()))

    def of(language: str) -> dict[str, list[Grade]]:
        return {rid: grades for rid, grades in by_item.items() if items[rid] == language}

    return Scorecard(
        languages=(
            *(_language_score(lang, of(lang), sum(items[r] == lang for r in ungraded)) for lang in languages),
            _language_score("all", by_item, len(ungraded)),
        ),
        reviewers=tuple(sorted({g.reviewer for grades in by_item.values() for g in grades})),
        agreement=_agreement(by_item),
        harmful=tuple(sorted(rid for rid, grades in by_item.items() if any(g.harmful for g in grades))),
        unfaithful=tuple(
            sorted(rid for rid, grades in by_item.items() if not all(g.faithful for g in grades))
        ),
    )


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _ids(ids: Sequence[str]) -> str:
    return ", ".join(ids) or "none"


def format_scorecard(card: Scorecard) -> str:
    """Human-readable summary (CLI)."""
    lines = [f"Reviewers: {', '.join(card.reviewers) or 'none'}"]
    for s in card.languages:
        lines.append(
            f"{s.language}: graded {s.graded} (ungraded {s.ungraded})  accuracy {_num(s.accuracy)}  "
            f"fluency {_num(s.fluency)}  faithful {_pct(s.faithful_rate)}  harmful {_pct(s.harmful_rate)}"
        )
    a = card.agreement
    lines += [
        f"Agreement on {a.items} items graded by 2+ reviewers: faithful {_pct(a.faithful)}, harmful "
        f"{_pct(a.harmful)} exact; accuracy {_num(a.accuracy_mad)}, fluency {_num(a.fluency_mad)} points apart "
        "on average",
        f"Harmful or misleading (follow up): {_ids(card.harmful)}",
        f"Not faithful to the facts (follow up): {_ids(card.unfaithful)}",
    ]
    return "\n".join(lines)


def markdown_scorecard(card: Scorecard) -> str:
    """A table for docs/VALIDATION.md."""
    lines = [
        "| Language | Replies graded | Accuracy (1-5) | Fluency (1-5) | Faithful to facts | Harmful or misleading |",
        "|---|---|---|---|---|---|",
    ]
    for s in card.languages:
        name = "**All**" if s.language == "all" else s.language
        lines.append(
            f"| {name} | {s.graded} | {_num(s.accuracy)} | {_num(s.fluency)} | {_pct(s.faithful_rate)} "
            f"| {_pct(s.harmful_rate)} |"
        )
    a = card.agreement
    lines += [
        "",
        f"{len(card.reviewers)} native-speaker reviewers. Inter-rater agreement on {a.items} double-graded replies: "
        f"faithful {_pct(a.faithful)} and harmful {_pct(a.harmful)} exact agreement; accuracy and fluency differ "
        f"by {_num(a.accuracy_mad)} and {_num(a.fluency_mad)} points on average. A reply counts as unfaithful or "
        "harmful if any reviewer marked it so.",
    ]
    return "\n".join(lines)
