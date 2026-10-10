"""Native-speaker review kit: sample real pilot replies for grading, then score the graded sheets.

Export (``kudiguard review export``): a stratified random sample of model-written check replies from the evidence
export (``Store.export_interactions``), balanced across languages and between scam checks and questions, written as
one UTF-8 (BOM) CSV per language (a reviewer only receives their own language) that opens cleanly in Excel and Google
Sheets. Each row carries the redacted message, its English pivot, the reply and the knowledge-base facts it was
grounded on. ``Answer.sources`` is not stored, so the facts are *re-derived* by repeating the deterministic knowledge
search on the stored English and flags: packs edited since the reply was sent (or redaction of the stored English)
can make them differ slightly from what the model saw.

Score (``kudiguard review score``): validates the grades in one or more sheets (one per reviewer, or a shared sheet
with a ``reviewer`` column) and reports per language. A reply counts as unfaithful or harmful when *any* reviewer
marked it so (safety first); scales are averaged per reply, then across replies. Agreement uses the replies graded by
two or more reviewers: for the yes/no questions the observed pairwise agreement, Gwet's AC1 (chance-corrected, and
unlike kappa stable when nearly every answer is the same, as "harmful = no" will be) and the number of replies on
which reviewers split; for the 1-5 scales the mean absolute difference between reviewers. See docs/NATIVE_REVIEW.md.
"""

from __future__ import annotations

import csv
import io
import os
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
LANGUAGE_ORDER = ("ha", "yo", "ig", "en")  # report order; other codes follow alphabetically
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
_YES, _NO = (
    frozenset({"yes", "y", "true"}),
    frozenset({"no", "n", "false"}),
)  # TRUE/FALSE: spreadsheet booleans
_DELIMITERS = (",", ";", "\t")  # comma, Excel in comma-decimal locales, Excel "Unicode Text"
SHEET_MODE = 0o600  # owner read/write only (POSIX; on Windows only the read-only bit exists)


def language_key(code: str) -> tuple[int, str]:
    return (LANGUAGE_ORDER.index(code) if code in LANGUAGE_ORDER else len(LANGUAGE_ORDER), code)


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

    Strata sizes differ by at most one unless a stratum runs out, whose share then goes to the others. The same rows
    (in any order) and seed give the same sample."""
    rng = random.Random(seed)
    strata: list[list[Mapping[str, Any]]] = []
    for language in dict.fromkeys(languages):
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
    return sorted(rows, key=lambda r: (language_key(r["language"]), r["review_id"]))  # verdict groups mixed


def write_sheet(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    """UTF-8 with BOM (Excel otherwise mangles ẹ, ọ, ṣ, ɓ, ƙ…), CRLF rows, formula-safe cells, owner-only file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, SHEET_MODE)
    os.chmod(path, SHEET_MODE)  # also when the file already existed with wider permissions
    with open(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: spreadsheet_cell(row.get(column, "")) for column in COLUMNS})


def write_sheets(directory: Path, rows: Sequence[Mapping[str, str]], *, stem: str) -> dict[str, Path]:
    """One sheet per language (``<stem>-<language>.csv``): reviewers only see their own language's messages."""
    paths: dict[str, Path] = {}
    for language in sorted({r["language"] for r in rows}, key=language_key):
        paths[language] = directory / f"{stem}-{language}.csv"
        write_sheet(paths[language], [r for r in rows if r["language"] == language])
    return paths


def known_review_ids(interaction_ids: Iterable[int], hasher: UserHasher) -> frozenset[str]:
    """Review ids of every interaction still stored (``Store.interaction_ids``; erased or expired ones are gone)."""
    return frozenset(hasher.review_id(i) for i in interaction_ids)


# --- read graded sheets ---------------------------------------------------------------------------------------


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
    where: str = ""  # "file.csv row 7", for error messages


@dataclass(frozen=True, slots=True)
class Sheet:
    name: str
    grades: tuple[Grade, ...]
    items: Mapping[str, str]  # review id -> language, graded or not


@dataclass(frozen=True, slots=True)
class _Row:
    where: str
    review_id: str
    language: str
    reviewer: str
    grades: tuple[str, ...]  # raw GRADE_COLUMNS values; all empty = ungraded


def _decode(raw: bytes, name: str) -> str:
    try:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # Excel "Unicode Text" / UTF-16 saves
            return raw.decode("utf-16")
        try:
            return raw.decode("utf-8-sig")
        except (
            UnicodeDecodeError
        ):  # Excel's plain "CSV" save uses the ANSI code page; grades are ASCII anyway
            return raw.decode("cp1252", errors="replace")
    except UnicodeDecodeError as exc:
        raise GradeError(f"{name}: unreadable text ({exc.reason}); save it as CSV UTF-8") from exc


def _records(text: str, name: str) -> list[list[str]]:
    header = text.split("\n", 1)[0]
    reader = csv.reader(io.StringIO(text), delimiter=max(_DELIMITERS, key=header.count))
    try:
        return list(reader)
    except csv.Error as exc:
        raise GradeError(f"{name} line {reader.line_num}: unreadable CSV ({exc})") from exc


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


def _rows(records: list[list[str]], name: str) -> list[_Row]:
    header = [cell.strip() for cell in records[0]] if records else []
    if missing := [c for c in ("review_id", "language", *GRADE_COLUMNS) if c not in header]:
        raise GradeError(f"{name}: not a review sheet (missing columns {', '.join(missing)})")
    rows: list[_Row] = []
    for number, record in enumerate(
        records[1:], start=2
    ):  # spreadsheet numbering: header is row 1, blanks count
        cells = {column: value.strip().lstrip("'") for column, value in zip(header, record, strict=False)}
        row = _Row(
            where=f"{name} row {number}",
            review_id=cells.get("review_id", ""),
            language=cells.get("language", ""),
            reviewer=cells.get("reviewer", ""),
            grades=tuple(cells.get(c, "") for c in GRADE_COLUMNS),
        )
        if not row.review_id:
            if any(row.grades):
                raise GradeError(f"{row.where}: grades without a review_id")
            continue
        rows.append(row)
    return rows


def _reviewer_for(rows: Sequence[_Row], stem: str) -> Mapping[str, str]:
    """Reviewer per row: its own name; else the sheet's only name; else (no names at all) the file name."""
    names = sorted({r.reviewer for r in rows if r.reviewer})
    fallback = names[0] if len(names) == 1 else stem if not names else None
    resolved: dict[str, str] = {}
    for row in rows:
        if row.reviewer or fallback is not None:
            resolved[row.where] = row.reviewer or str(fallback)
        elif any(row.grades):
            raise GradeError(
                f"{row.where}: no reviewer name, and the sheet names several ({', '.join(names)})"
            )
    return resolved


def _grade(row: _Row, reviewer: str) -> Grade:
    if blank := [c for c, v in zip(GRADE_COLUMNS, row.grades, strict=True) if not v]:
        raise GradeError(f"{row.where}: {row.review_id} is partly graded (missing {', '.join(blank)})")
    accuracy, fluency, faithful, harmful = row.grades
    return Grade(
        review_id=row.review_id,
        language=row.language,
        reviewer=reviewer,
        accuracy=_scale(accuracy, "accuracy", row.where),
        fluency=_scale(fluency, "fluency", row.where),
        faithful=_yes_no(faithful, "faithful", row.where),
        harmful=_yes_no(harmful, "harmful_or_misleading", row.where),
        where=row.where,
    )


def _read_records(path: Path) -> list[list[str]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GradeError(f"{path}: cannot read ({exc.strerror})") from exc
    return _records(_decode(raw, path.name), path.name)


def read_review_ids(path: Path) -> frozenset[str]:
    """Only the review ids of a sheet (grades are not checked: a half-graded working copy is fine)."""
    records = _read_records(path)
    header = [cell.strip() for cell in records[0]] if records else []
    if "review_id" not in header:
        raise GradeError(f"{path.name}: not a review sheet (no review_id column)")
    column = header.index("review_id")
    ids = (record[column].strip().lstrip("'") for record in records[1:] if len(record) > column)
    return frozenset(rid for rid in ids if rid)


def read_sheet(path: Path) -> Sheet:
    """Parse one graded sheet. Rows with no grades are ungraded; a partly graded or malformed row is an error."""
    rows = _rows(_read_records(path), path.name)
    reviewers = _reviewer_for(rows, path.stem)
    grades: dict[tuple[str, str], Grade] = {}
    items: dict[str, str] = {}
    for row in rows:
        if items.setdefault(row.review_id, row.language) != row.language:
            raise GradeError(
                f"{row.where}: {row.review_id} is {row.language!r} here but {items[row.review_id]!r} above"
            )
        if not any(row.grades):
            continue
        grade = _grade(row, reviewers[row.where])
        if (previous := grades.get((grade.review_id, grade.reviewer))) is not None:
            raise GradeError(
                f"{row.where}: {grade.review_id} already graded by {grade.reviewer!r} ({previous.where})"
            )
        grades[(grade.review_id, grade.reviewer)] = grade
    return Sheet(path.name, tuple(grades.values()), items)


# --- score ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class YesNoAgreement:
    observed: float  # share of reviewer pairs giving the same answer, averaged per reply
    ac1: float  # Gwet's AC1: 1 = perfect, 0 = what chance alone would give
    splits: int  # replies on which the reviewers disagreed


@dataclass(frozen=True, slots=True)
class Agreement:
    items: int  # graded by two or more reviewers
    faithful: YesNoAgreement | None
    harmful: YesNoAgreement | None
    accuracy_gap: float | None  # mean absolute difference between reviewer pairs, in scale points
    fluency_gap: float | None


@dataclass(frozen=True, slots=True)
class LanguageScore:
    language: str  # "all" for the overall row
    graded: int
    ungraded: int
    accuracy: float | None
    fluency: float | None
    faithful_rate: float | None
    harmful_rate: float | None
    agreement: Agreement


@dataclass(frozen=True, slots=True)
class Scorecard:
    languages: tuple[LanguageScore, ...]  # per language (LANGUAGE_ORDER), then "all"
    reviewers: tuple[str, ...]
    harmful: tuple[str, ...]  # review ids any reviewer flagged, for follow-up
    unfaithful: tuple[str, ...]

    @property
    def overall(self) -> LanguageScore:
        return self.languages[-1]


def _mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def yes_no_agreement(items: Iterable[Sequence[bool]]) -> YesNoAgreement | None:
    """Gwet's AC1 for two categories and any number of reviewers per reply (Gwet 2008, eq. for multiple raters).

    ``items``: every graded reply's answers. p_a (observed): per reply graded by two or more, agreeing ordered pairs
    / all ordered pairs, averaged over those replies; π: share of "yes" answers per reply, averaged over *all*
    graded replies (single-rated ones too, as Gwet 2008/2014 specify); chance agreement p_e = 2π(1-π) (at most 0.5,
    so AC1 is always defined); AC1 = (p_a-p_e)/(1-p_e). None when no reply has two answers."""
    rated = [votes for votes in items if votes]
    multi = [votes for votes in rated if len(votes) > 1]
    if not multi:
        return None
    observed = mean(
        sum(k * (k - 1) for k in (sum(v), len(v) - sum(v))) / (len(v) * (len(v) - 1)) for v in multi
    )
    share = mean(sum(v) / len(v) for v in rated)
    chance = 2 * share * (1 - share)
    return YesNoAgreement(observed, (observed - chance) / (1 - chance), sum(len(set(v)) > 1 for v in multi))


def _agreement(by_item: Mapping[str, Sequence[Grade]]) -> Agreement:
    multi = [item for item in by_item.values() if len(item) > 1]
    pairs = [pair for item in multi for pair in combinations(item, 2)]
    return Agreement(
        items=len(multi),
        faithful=yes_no_agreement([g.faithful for g in item] for item in by_item.values()),
        harmful=yes_no_agreement([g.harmful for g in item] for item in by_item.values()),
        accuracy_gap=_mean([abs(a.accuracy - b.accuracy) for a, b in pairs]),
        fluency_gap=_mean([abs(a.fluency - b.fluency) for a, b in pairs]),
    )


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
        agreement=_agreement(by_item),
    )


def score(sheets: Sequence[Sheet]) -> Scorecard:
    """Combine sheets; a reviewer may grade a reply only once, and a reply has one language, across all sheets."""
    items: dict[str, str] = {}
    for sheet in sheets:
        for rid, language in sheet.items.items():
            if items.setdefault(rid, language) != language:
                raise GradeError(
                    f"{sheet.name}: {rid} is {language!r} here but {items[rid]!r} in another sheet"
                )
    by_item: dict[str, list[Grade]] = {}
    for grade in (g for sheet in sheets for g in sheet.grades):
        if twice := [g for g in by_item.get(grade.review_id, ()) if g.reviewer == grade.reviewer]:
            raise GradeError(
                f"{grade.review_id} graded twice by reviewer {grade.reviewer!r} ({twice[0].where}, {grade.where})"
            )
        by_item.setdefault(grade.review_id, []).append(grade)
    ungraded = [rid for rid in items if rid not in by_item]

    def of(language: str) -> dict[str, list[Grade]]:
        return {rid: grades for rid, grades in by_item.items() if items[rid] == language}

    return Scorecard(
        languages=(
            *(
                _language_score(lang, of(lang), sum(items[r] == lang for r in ungraded))
                for lang in sorted(set(items.values()), key=language_key)
            ),
            _language_score("all", by_item, len(ungraded)),
        ),
        reviewers=tuple(sorted({g.reviewer for grades in by_item.values() for g in grades})),
        harmful=tuple(sorted(rid for rid, grades in by_item.items() if any(g.harmful for g in grades))),
        unfaithful=tuple(
            sorted(rid for rid, grades in by_item.items() if not all(g.faithful for g in grades))
        ),
    )


# --- formatting -----------------------------------------------------------------------------------------------


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _ids(ids: Sequence[str]) -> str:
    return ", ".join(ids) or "none"


def _yes_no_text(a: YesNoAgreement | None) -> str:
    return "n/a" if a is None else f"{a.observed:.0%} (AC1 {a.ac1:.2f}, {a.splits} split)"


def format_scorecard(card: Scorecard) -> str:
    """Human-readable summary (CLI)."""
    lines = [f"Reviewers: {', '.join(card.reviewers) or 'none'}"]
    for s in card.languages:
        lines.append(
            f"{s.language}: graded {s.graded} (ungraded {s.ungraded})  accuracy {_num(s.accuracy)}  "
            f"fluency {_num(s.fluency)}  faithful {_pct(s.faithful_rate)}  harmful {_pct(s.harmful_rate)}"
        )
        if (a := s.agreement).items:
            lines.append(
                f"  agreement on {a.items} replies graded by 2+ reviewers: faithful {_yes_no_text(a.faithful)}, "
                f"harmful {_yes_no_text(a.harmful)}; accuracy {_num(a.accuracy_gap)}, fluency "
                f"{_num(a.fluency_gap)} points apart on average"
            )
    lines += [
        f"Harmful or misleading (follow up): {_ids(card.harmful)}",
        f"Not faithful to the facts (follow up): {_ids(card.unfaithful)}",
    ]
    return "\n".join(lines)


def _name(language: str) -> str:
    return "**All**" if language == "all" else language.replace("|", "\\|")  # codes come from the sheets


def markdown_scorecard(card: Scorecard) -> str:
    """Tables for docs/VALIDATION.md: scores, then inter-rater agreement."""
    lines = [
        "| Language | Replies graded | Accuracy (1-5) | Fluency (1-5) | Faithful to facts | Harmful or misleading |",
        "|---|---|---|---|---|---|",
        *(
            f"| {_name(s.language)} | {s.graded} | {_num(s.accuracy)} | {_num(s.fluency)} | "
            f"{_pct(s.faithful_rate)} | {_pct(s.harmful_rate)} |"
            for s in card.languages
        ),
        "",
        "| Language | Graded by 2+ | Faithful: agreement (AC1, splits) | Harmful: agreement (AC1, splits) "
        "| Accuracy gap | Fluency gap |",
        "|---|---|---|---|---|---|",
        *(
            f"| {_name(s.language)} | {s.agreement.items} | {_yes_no_text(s.agreement.faithful)} | "
            f"{_yes_no_text(s.agreement.harmful)} | {_num(s.agreement.accuracy_gap)} | "
            f"{_num(s.agreement.fluency_gap)} |"
            for s in card.languages
        ),
        "",
        f"{len(card.reviewers)} native-speaker reviewers. A reply counts as unfaithful or harmful if any reviewer "
        "marked it so. Agreement: share of reviewer pairs giving the same answer; AC1 corrects it for chance "
        "(1 = perfect, 0 = chance); gaps are mean differences between reviewers in scale points.",
    ]
    return "\n".join(lines)
