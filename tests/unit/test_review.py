"""Native-speaker review kit: sampling, the reviewer sheets, grade validation, scoring and the CLI.

``hypothesis`` is not a dependency, so the property tests are seeded randomised loops."""

import asyncio
import csv
import io
import os
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from kudiguard import cli, review
from kudiguard.domain import Flag, Interaction, MessageKind, Outcome, Verdict
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.export import spreadsheet_cell
from kudiguard.storage.sqlite import SQLiteStore

ROOT = Path(__file__).resolve().parents[2]
SECRET = "s" * MIN_SECRET_BYTES
HASHER = UserHasher(SECRET.encode())
DIACRITICS = "Ẹ kú àárọ̀, ṣé dáadáa ni? Ɓarawo ƙarya ne. Ọ́ dị́ mma, ụ́lọ̀ akụ̀"


def row(
    id: int, language: str = "ha", verdict: str | None = "likely_scam", **overrides: Any
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": id,
        "created_at": "2026-10-09T10:00:00+00:00",
        "user": f"user{id}",
        "channel": "whatsapp",
        "language": language,
        "kind": "text",
        "outcome": "answered",
        "intent": "scam_check",
        "verdict": verdict,
        "flags": ["asks_credentials"],
        "transcript": "Ka turo min lambar OTP",
        "english": "Send me your OTP code",
        "reply": "🔴 Wannan zamba ne.",
    }
    return base | overrides


# --- export: selection and sampling --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"outcome": "fallback"},  # nothing model-written to grade
        {"outcome": "fixed_reply", "verdict": None},  # greeting
        {"intent": "button:steps"},  # follow-up
        {"reply": None},
        {"english": None},
        {"verdict": None},
        {"language": "en"},  # not requested
        {"channel": "api"},  # business callers, not people
        {"user": "team"},
        {"created_at": "2026-09-30T23:59:59+00:00"},  # before --since
    ],
)
def test_only_model_written_check_replies_are_eligible(overrides: dict[str, Any]) -> None:
    kept = review.eligible([row(1), row(2, **overrides)], exclude_users={"team"}, since=date(2026, 10, 1))
    assert [r["id"] for r in kept] == [1]


def test_sample_is_balanced_across_languages_and_verdict_groups() -> None:
    rows = [
        row(n, language, verdict)
        for n, (language, verdict) in enumerate(
            [(lang, v) for lang in ("ha", "yo", "ig") for v in ("likely_scam", "caution", "not_a_check")] * 10
        )
    ]
    sample = review.stratified_sample(rows, 12, seed=1)
    strata = Counter((r["language"], review.verdict_group(r["verdict"])) for r in sample)
    assert set(strata.values()) == {2} and len(strata) == 6
    assert review.stratified_sample(rows, 12, seed=1) == sample  # reproducible
    assert review.stratified_sample(rows, 12, seed=2) != sample
    assert review.stratified_sample(rows, 12, seed=1, languages=("ha", "ha", "yo", "ig")) == sample  # deduped


def test_small_strata_leave_their_share_to_others() -> None:
    rows = [row(1, "ha", "not_a_check")] + [row(n, "yo") for n in range(2, 30)]
    sample = review.stratified_sample(rows, 10, seed=0)
    assert len(sample) == 10 and sum(r["language"] == "ha" for r in sample) == 1
    assert len(review.stratified_sample(rows[:3], 10, seed=0)) == 3  # fewer than asked: all of them


def test_sampling_properties_hold_for_random_inputs() -> None:
    verdicts = [v.value for v in Verdict]
    for trial in range(300):
        rng = random.Random(trial)
        rows = [
            row(n, rng.choice(("ha", "yo", "ig", "en")), rng.choice(verdicts))
            for n in rng.sample(range(10_000), rng.randint(0, 80))
        ]
        languages = tuple(rng.sample(("ha", "yo", "ig", "en"), rng.randint(1, 4)))
        n = rng.randint(1, 90)
        sample = review.stratified_sample(rows, n, seed=trial, languages=languages)
        pool = [r for r in rows if r["language"] in languages]
        ids = [r["id"] for r in sample]
        assert len(ids) == len(set(ids)) == min(n, len(pool))  # no duplicates, as many as possible
        assert all(r in pool for r in sample)  # a subset of the requested languages
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert review.stratified_sample(shuffled, n, seed=trial, languages=languages) == sample  # order-free

        def key(r: Mapping[str, Any]) -> tuple[str, str]:
            return r["language"], review.verdict_group(r["verdict"])

        available, taken = Counter(map(key, pool)), Counter(map(key, sample))
        strata = [(lang, g) for lang in languages for g in (review.CHECK, review.QUESTION)]
        for s in strata:
            for t in strata:  # balanced within one, unless the smaller stratum ran out
                assert taken[s] >= taken[t] - 1 or taken[s] == available[s], (trial, s, t)


# --- export: the sheet ---------------------------------------------------------------------------------------


def test_spreadsheet_cells_cannot_run_formulas() -> None:
    assert spreadsheet_cell('=HYPERLINK("http://x","Click")') == '\'=HYPERLINK("http://x","Click")'
    risky = (
        "+234",
        "-x",
        "@a",
        "\tz",
        "\rz",
        " =1+1",
        "\n@SUM(A1)",
        "\r\n -2",
        "\u3000=1",
        "\uff1d1",
        "\uff0b1",
    )
    assert [spreadsheet_cell(v) for v in risky] == ["'" + v for v in risky]
    assert [spreadsheet_cell(v) for v in ("\uff0dx", "\uff20x", "\t plain")] == [
        "'\uff0dx",
        "'\uff20x",
        "'\t plain",
    ]
    assert [spreadsheet_cell(v) for v in (DIACRITICS, " plain", "a=b", "", "1-2")] == [
        DIACRITICS, " plain", "a=b", "", "1-2"
    ]  # fmt: skip


def test_review_ids_are_opaque_stable_and_domain_separated() -> None:
    first = HASHER.review_id(7)
    assert first == HASHER.review_id(7) != HASHER.review_id(8)
    assert re.fullmatch(r"R-[0-9a-f]{10}", first)
    assert UserHasher(b"t" * MIN_SECRET_BYTES).review_id(7) != first
    others = (HASHER.user_id("7"), HASHER.user_id("review:7"), HASHER.message_key("review", "7"))
    assert first[2:] not in {other[:10] for other in others}  # its own HMAC namespace


def test_facts_format() -> None:
    assert review.format_facts([], []) == "(no facts found)"
    assert review.format_facts(["A.", "B."], ["Do x."]) == "1. A.\n2. B.\nSteps:\n- Do x."


@pytest.fixture
async def knowledge() -> Any:
    async with httpx.AsyncClient() as client:
        yield knowledge_base_from_packs(ROOT / "data" / "packs", client=client)


def read_cells(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8-sig"))))


async def test_sheet_rows_carry_no_ids_and_rederive_the_facts(
    knowledge: KnowledgeBase, tmp_path: Path
) -> None:
    sample = [
        row(5, "yo", transcript="=cmd|' /C calc'!A0"),
        row(3, "ha", "not_a_check", flags=[]),
        row(4, "en"),
        row(6, "ig"),
    ]
    rows = await review.review_rows(sample, hasher=HASHER, knowledge=knowledge)
    assert [r["language"] for r in rows] == ["ha", "yo", "ig", "en"]
    otp = rows[1]
    assert otp["review_id"] == HASHER.review_id(5) and "OTP" in otp["facts_rederived"]
    assert otp["facts_rederived"].startswith("1. ")

    path = tmp_path / "out" / "sheet.csv"
    review.write_sheet(path, rows)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw  # BOM for Excel
    text = raw.decode("utf-8-sig")
    assert "user5" not in text and ',"5",' not in text
    parsed = read_cells(path)
    assert list(parsed[0]) == list(review.COLUMNS)
    assert parsed[1]["user_message"] == "'=cmd|' /C calc'!A0"
    assert all(parsed[1][c] == "" for c in (*review.GRADE_COLUMNS, "reviewer", "notes"))


async def test_grounding_matches_the_engine_query(knowledge: KnowledgeBase) -> None:
    from kudiguard.knowledge.models import Query
    from kudiguard.prompts import facts_block

    facts, actions = facts_block(
        await knowledge.search(
            Query(text="Send me your OTP code", flags=(Flag.ASKS_CREDENTIALS,), language="ha")
        )
    )
    assert await review.grounding(knowledge, row(1)) == review.format_facts(facts, actions)


def test_one_owner_only_sheet_per_language(tmp_path: Path) -> None:
    rows = [{"review_id": f"R-{n}", "language": lang} for n, lang in enumerate(("yo", "ha", "yo", "ig"))]
    stale = tmp_path / "review-x-yo.csv"
    stale.write_text("old", encoding="utf-8")
    if os.name == "posix":  # permissions are POSIX-only
        stale.chmod(0o644)
    paths = review.write_sheets(tmp_path, rows, stem="review-x")
    assert list(paths) == ["ha", "yo", "ig"] and paths["yo"] == stale
    assert [r["review_id"] for r in read_cells(paths["yo"])] == [
        "R-0",
        "R-2",
    ]  # overwritten, own language only
    assert {r["language"] for r in read_cells(paths["ig"])} == {"ig"}
    if os.name == "posix":  # Windows only has a read-only bit
        assert all(p.stat().st_mode & 0o777 == review.SHEET_MODE for p in paths.values())


async def test_no_identifier_reaches_any_cell(knowledge: KnowledgeBase, tmp_path: Path) -> None:
    """Property over random real-looking turns stored through the real store (which redacts)."""
    rng = random.Random(7)
    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    start = datetime(2026, 10, 1, tzinfo=UTC)
    for n in range(40):
        phone = f"+234{rng.randint(7000000000, 9099999999)}"
        account = "".join(rng.choices("0123456789", k=rng.randint(10, 16)))
        await store.add_interaction(
            Interaction(
                channel=rng.choice(("whatsapp", "web")), message_id=f"wamid.{phone}.{n}",
                user_hash=HASHER.user_id(phone), language=rng.choice(("ha", "yo", "ig")), kind=MessageKind.TEXT,
                outcome=Outcome.ANSWERED, verdict=rng.choice(list(Verdict)), flags=(Flag.ASKS_CREDENTIALS,),
                transcript=f"{DIACRITICS} call {phone} acct {account}", english=f"Send OTP to {account}",
                reply=f"{DIACRITICS}\nDon't call {phone}", created_at=start + timedelta(minutes=n * 7),
            )
        )  # fmt: skip
    exported = await store.export_interactions()
    await store.close()
    sample = review.stratified_sample(review.eligible(exported), 30, seed=1)
    paths = review.write_sheets(
        tmp_path, await review.review_rows(sample, hasher=HASHER, knowledge=knowledge), stem="r"
    )
    sheet_rows = [r for path in paths.values() for r in read_cells(path)]
    assert all(
        re.fullmatch(r"R-[0-9a-f]{10}", r.pop("review_id")) for r in sheet_rows
    )  # hex may be all digits
    cells = [cell for r in sheet_rows for cell in r.values()]
    secrets = {str(r[k]) for r in exported for k in ("user", "created_at")}
    ids = {str(r["id"]) for r in exported}
    assert len(cells) == 30 * (len(review.COLUMNS) - 1)
    for cell in cells:
        assert not re.search(r"\d{10,}", cell), cell
        assert not any(secret in cell for secret in secrets) and cell not in ids
    assert any(DIACRITICS in cell for cell in cells)


def test_known_review_ids() -> None:
    assert review.known_review_ids([1, 2], HASHER) == {HASHER.review_id(1), HASHER.review_id(2)}


def test_more_formula_prefixes() -> None:
    risky = ("|cmd /C calc", "\u00a0=1+1", " \u00a0@SUM(A1)")
    assert [spreadsheet_cell(v) for v in risky] == ["'" + v for v in risky]
    assert spreadsheet_cell("\u00a0plain") == "\u00a0plain"


# --- reading graded sheets -----------------------------------------------------------------------------------

HEADER = list(review.COLUMNS)


def sheet(
    path: Path, rows: Sequence[Mapping[str, str]], *, delimiter: str = ",", encoding: str = "utf-8-sig"
) -> Path:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEADER, delimiter=delimiter, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows({c: r.get(c, "") for c in HEADER} for r in rows)
    path.write_bytes(buffer.getvalue().encode(encoding, errors="replace"))
    return path


def graded(
    rid: str, lang: str, acc: str, flu: str, faithful: str, harmful: str, **extra: str
) -> dict[str, str]:
    return {
        "review_id": rid,
        "language": lang,
        "accuracy": acc,
        "fluency": flu,
        "faithful": faithful,
        "harmful_or_misleading": harmful,
        **extra,
    }


def test_score_single_reviewer(tmp_path: Path) -> None:
    path = sheet(
        tmp_path / "ade.csv",
        [
            graded("R-1", "yo", "5", "4", "yes", "no"),
            graded("R-2", "yo", "3.0", "2", "No", "Y", notes="'- wrong number"),
            graded("R-3", "ha", "", "", "", ""),  # ungraded: counted
            graded("R-4", "en", "4", "4", "TRUE", "FALSE"),  # spreadsheet booleans
            {"review_id": "", "language": ""},  # blank line in a spreadsheet
        ],
    )
    card = review.score([review.read_sheet(path)])
    ha, yo, en, overall = card.languages  # report order, not file order
    assert (yo.language, yo.graded, yo.ungraded, yo.accuracy, yo.fluency) == ("yo", 2, 0, 4.0, 3.0)
    assert (yo.faithful_rate, yo.harmful_rate) == (0.5, 0.5)
    assert (ha.graded, ha.ungraded, ha.accuracy, ha.faithful_rate) == (0, 1, None, None)
    assert (en.faithful_rate, en.harmful_rate) == (1.0, 0.0)
    assert card.overall is overall and (overall.language, overall.graded, overall.ungraded) == ("all", 3, 1)
    assert card.reviewers == ("ade",) and card.harmful == ("R-2",) and card.unfaithful == ("R-2",)
    assert overall.agreement == review.Agreement(0, None, None, None, None)
    text = review.format_scorecard(card)
    assert "yo: graded 2 (ungraded 0)  accuracy 4.0" in text and "(follow up): R-2" in text
    assert "ha: graded 0 (ungraded 1)  accuracy n/a" in text and "agreement" not in text


def test_score_reviewers_in_separate_sheets(tmp_path: Path) -> None:
    a = sheet(  # the name only on the first row applies to the whole sheet
        tmp_path / "ig-chi.csv",
        [
            graded("R-1", "ig", "5", "5", "yes", "no", reviewer="Chi"),
            graded("R-2", "ig", "4", "4", "yes", "no"),
        ],
    )
    b = sheet(
        tmp_path / "ig-obi.csv",
        [graded("R-1", "ig", "3", "4", "no", "no"), graded("R-2", "ig", "4", "2", "yes", "yes")],
    )
    card = review.score([review.read_sheet(a), review.read_sheet(b)])
    ig = card.languages[0]
    assert ig.graded == 2 and ig.accuracy == 4.0 and ig.fluency == 3.75
    assert ig.faithful_rate == 0.5 and ig.harmful_rate == 0.5  # any reviewer's flag counts
    assert card.reviewers == ("Chi", "ig-obi")  # no name in the sheet: the file name
    agreement = ig.agreement
    assert (agreement.items, agreement.accuracy_gap, agreement.fluency_gap) == (2, 1.0, 1.5)
    assert agreement.faithful == review.YesNoAgreement(0.5, pytest.approx(0.2), 1)
    assert agreement.harmful == review.YesNoAgreement(0.5, pytest.approx(0.2), 1)
    md = review.markdown_scorecard(card)
    assert md.splitlines()[2] == "| ig | 2 | 4.0 | 3.8 | 50% | 50% |"
    assert "| ig | 2 | 50% (AC1 0.20, 1 split) | 50% (AC1 0.20, 1 split) | 1.0 | 1.5 |" in md
    assert "2 native-speaker reviewers" in md
    assert "agreement on 2 replies graded by 2+ reviewers: faithful 50% (AC1 0.20" in review.format_scorecard(
        card
    )


def test_shared_sheet_with_three_reviewers_on_one_reply(tmp_path: Path) -> None:
    path = sheet(
        tmp_path / "shared.csv",
        [
            graded("R-1", "ha", "5", "5", "yes", "no", reviewer="Ade"),
            graded("R-1", "ha", "4", "5", "yes", "no", reviewer="Bola"),
            graded("R-1", "ha", "3", "5", "no", "no", reviewer="Chi"),
            graded("R-1", "ha", "", "", "", ""),  # an ungraded copy of the row
        ],
    )
    card = review.score([review.read_sheet(path)])
    a = card.overall.agreement
    assert card.reviewers == ("Ade", "Bola", "Chi") and a.items == 1
    assert a.faithful == review.YesNoAgreement(pytest.approx(1 / 3), pytest.approx(-0.2), 1)
    assert a.harmful == review.YesNoAgreement(1.0, 1.0, 0)  # all "no": AC1 stays defined
    assert a.accuracy_gap == pytest.approx(4 / 3) and a.fluency_gap == 0


def test_ac1_examples() -> None:
    assert review.yes_no_agreement([[True]]) is None  # single ratings say nothing about agreement
    perfect = review.yes_no_agreement([[True, True], [False, False]])
    assert perfect == review.YesNoAgreement(1.0, 1.0, 0)
    opposite = review.yes_no_agreement([[True, False], [False, True]])
    assert opposite == review.YesNoAgreement(0.0, -1.0, 2)


def test_ac1_chance_term_uses_single_rated_replies_too() -> None:
    # p_a from the double-rated reply only (0); pi over both replies: (1 + 1/2) / 2 = 0.75, p_e = 0.375
    mixed = review.yes_no_agreement([[True], [True, False], []])
    assert mixed == review.YesNoAgreement(0.0, pytest.approx(-0.6), 1)
    restricted = review.yes_no_agreement([[True, False]])  # pi = 0.5, p_e = 0.5
    assert restricted == review.YesNoAgreement(0.0, pytest.approx(-1.0), 1)
    assert review.yes_no_agreement([[True], [False]]) is None


def test_score_agreement_counts_single_rated_replies_in_chance(tmp_path: Path) -> None:
    path = sheet(
        tmp_path / "shared.csv",
        [
            graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Ade"),
            graded("R-1", "ha", "4", "4", "no", "no", reviewer="Bola"),
            graded("R-2", "ha", "4", "4", "yes", "no", reviewer="Ade"),
        ],
    )
    agreement = review.score([review.read_sheet(path)]).overall.agreement
    assert agreement.items == 1 and agreement.faithful == review.YesNoAgreement(0.0, pytest.approx(-0.6), 1)


def test_cp1252_and_utf16_sheets_are_read(tmp_path: Path) -> None:
    path = tmp_path / "excel.csv"
    path.write_bytes(
        "review_id,language,accuracy,fluency,faithful,harmful_or_misleading,notes\r\n"
        "R-1,ha,4,4,yes,no,caf\xe9 \x81\r\n".encode("latin-1")
    )
    assert review.read_sheet(path).grades[0].accuracy == 4
    unicode_text = tmp_path / "unicode.txt"  # Excel "Unicode Text": UTF-16 with BOM, tab-separated
    unicode_text.write_bytes(
        "review_id\tlanguage\taccuracy\tfluency\tfaithful\tharmful_or_misleading\r\nR-2\tyo\t5\t5\tyes\tno\r\n"
        .encode("utf-16")
    )  # fmt: skip
    assert review.read_sheet(unicode_text).grades[0].review_id == "R-2"


async def test_excel_round_trip_keeps_ids_and_grades(knowledge: KnowledgeBase, tmp_path: Path) -> None:
    sample = [
        row(n, lang, transcript=f"{DIACRITICS}\nline two", reply=f"{DIACRITICS}\n\n- step", english="OTP?")
        for n, lang in enumerate(("ha", "yo", "ig"), start=1)
    ]
    exported = tmp_path / "export.csv"
    review.write_sheet(exported, await review.review_rows(sample, hasher=HASHER, knowledge=knowledge))
    rows = read_cells(exported)
    assert rows[0]["reply"] == f"{DIACRITICS}\n\n- step" and "\n" in rows[0]["facts_rederived"]
    for n, r in enumerate(rows):
        r.update(accuracy=str(n + 3), fluency="4", faithful="yes", harmful_or_misleading="no", reviewer="Ọlá")
    resaved = [
        sheet(
            tmp_path / "semicolon-ansi.csv", rows, delimiter=";", encoding="cp1252"
        ),  # diacritics become "?"
        sheet(tmp_path / "utf8.csv", rows),
    ]
    for path in resaved:
        parsed = review.read_sheet(path)
        assert sorted(g.review_id for g in parsed.grades) == sorted(HASHER.review_id(n) for n in (1, 2, 3))
        assert sorted(g.accuracy for g in parsed.grades) == [3, 4, 5]
    assert review.read_sheet(resaved[1]).grades[0].reviewer == "Ọlá"


@pytest.mark.parametrize(
    ("grades", "message"),
    [
        (("6", "4", "yes", "no"), "row 2: accuracy must be a whole number from 1 to 5, got '6'"),
        (("4", "3.5", "yes", "no"), "fluency must be a whole number"),
        (("4", "good", "yes", "no"), "fluency must be a whole number"),
        (("4", "4", "maybe", "no"), "faithful must be yes or no, got 'maybe'"),
        (("4", "4", "yes", "1"), "harmful_or_misleading must be yes or no"),
        (("4", "", "yes", ""), "R-1 is partly graded (missing fluency, harmful_or_misleading)"),
    ],
)
def test_malformed_grades_name_the_row(tmp_path: Path, grades: tuple[str, ...], message: str) -> None:
    path = sheet(tmp_path / "x.csv", [graded("R-1", "ha", *grades)])
    with pytest.raises(review.GradeError, match=re.escape(message)):
        review.read_sheet(path)


def test_row_numbers_count_blank_lines_and_multiline_cells(tmp_path: Path) -> None:
    path = tmp_path / "gaps.csv"
    path.write_text(
        "review_id,language,accuracy,fluency,faithful,harmful_or_misleading,notes\n"
        'R-1,ha,4,4,yes,no,"two\nlines"\n'
        "\n"
        "R-2,ha,9,4,yes,no,\n",
        encoding="utf-8",
    )
    with pytest.raises(review.GradeError, match=re.escape("gaps.csv row 4: accuracy")):
        review.read_sheet(path)


def write(path: Path, text: str | bytes) -> Path:
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("id,text\n1,x\n", "not a review sheet (missing columns review_id"),
        ("", "not a review sheet"),
        (b"\xff\xfeA", "unreadable text (truncated data)"),  # odd-length UTF-16
        ("review_id,language,accuracy,fluency,faithful,harmful_or_misleading\n\"" + "x" * 200_000, "line 2: unreadable CSV"),
        (",ha,4,4,yes,no\n".join(["review_id,language,accuracy,fluency,faithful,harmful_or_misleading\n", ""]),
         "row 2: grades without a review_id"),
    ],
    ids=["other-csv", "empty", "utf16-truncated", "huge-field", "grades-without-id"],
)  # fmt: skip
def test_unreadable_sheets(tmp_path: Path, content: str | bytes, message: str) -> None:
    with pytest.raises(review.GradeError, match=re.escape(message)):
        review.read_sheet(write(tmp_path / "s.csv", content))


def test_sheet_level_errors(tmp_path: Path) -> None:
    with pytest.raises(review.GradeError, match="cannot read"):
        review.read_sheet(tmp_path / "missing.csv")
    dup = sheet(
        tmp_path / "dup.csv",
        [
            graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Ade"),
            graded("R-1", "ha", "5", "4", "yes", "no"),
        ],
    )
    with pytest.raises(
        review.GradeError, match=re.escape("dup.csv row 3: R-1 already graded by 'Ade' (dup.csv row 2)")
    ):
        review.read_sheet(dup)
    several = sheet(
        tmp_path / "several.csv",
        [
            graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Ade"),
            graded("R-2", "ha", "", "", "", ""),  # unnamed but ungraded: fine
            graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Bola"),
            graded("R-3", "ha", "4", "4", "yes", "no"),
        ],
    )
    with pytest.raises(
        review.GradeError,
        match=re.escape("several.csv row 5: no reviewer name, and the sheet names several (Ade, Bola)"),
    ):
        review.read_sheet(several)
    mixed = sheet(
        tmp_path / "mixed.csv", [graded("R-1", "ha", "", "", "", ""), graded("R-1", "yo", "", "", "", "")]
    )
    with pytest.raises(
        review.GradeError, match=re.escape("mixed.csv row 3: R-1 is 'yo' here but 'ha' above")
    ):
        review.read_sheet(mixed)


def test_errors_across_sheets(tmp_path: Path) -> None:
    one = sheet(tmp_path / "one.csv", [graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Ade")])
    two = sheet(tmp_path / "two.csv", [graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Ade")])
    with pytest.raises(
        review.GradeError,
        match=re.escape("R-1 graded twice by reviewer 'Ade' (one.csv row 2, two.csv row 2)"),
    ):
        review.score([review.read_sheet(one), review.read_sheet(two)])
    other = sheet(tmp_path / "other.csv", [graded("R-1", "yo", "", "", "", "")])
    with pytest.raises(
        review.GradeError, match=re.escape("other.csv: R-1 is 'yo' here but 'ha' in another sheet")
    ):
        review.score([review.read_sheet(one), review.read_sheet(other)])


# --- CLI ------------------------------------------------------------------------------------------------------


@pytest.fixture
def review_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "kg.db"
    monkeypatch.setenv("KUDIGUARD_HASH_SECRET", SECRET)
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(db))
    monkeypatch.setenv("KUDIGUARD_EMBEDDINGS", "false")  # BM25 over the real packs: offline and fast
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)

    async def seed() -> None:
        store = await SQLiteStore.open(db, hasher=HASHER)
        when = datetime(2026, 10, 9, tzinfo=UTC)
        for n, (language, verdict) in enumerate(
            [
                ("ha", Verdict.LIKELY_SCAM),
                ("yo", Verdict.NOT_A_CHECK),
                ("ig", Verdict.CAUTION),
                ("en", Verdict.LIKELY_SCAM),
            ]
        ):
            await store.add_interaction(
                Interaction(channel="whatsapp", message_id=f"m{n}", user_hash=HASHER.user_id(f"+23480300000{n}"),
                            language=language, kind=MessageKind.TEXT, outcome=Outcome.ANSWERED, verdict=verdict,
                            flags=(Flag.ASKS_CREDENTIALS,), transcript="Turo min OTP 08031234567",
                            english="Send me your OTP", reply="-Kada ka bayar", created_at=when)
            )  # fmt: skip
        await store.close()

    asyncio.run(seed())
    return tmp_path


def test_cli_review_export_score_and_check(review_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    team = review_env / "team.txt"
    team.write_text("+234803000002\n", encoding="utf-8")
    args = [
        "review",
        "export",
        "--exclude-users",
        str(team),
        "--since",
        "2026-10-01",
        "--languages",
        "ha,yo,ha,ig",
    ]
    assert cli.main(args) == 0
    out = capsys.readouterr()
    assert (
        "sampled 2 of 2 eligible replies" in out.out and "only 2 eligible replies (asked for 50)" in out.err
    )
    assert "keyword search only" in out.out and "not ignored" not in out.err  # outside the repository
    paths = sorted((review_env / "review").glob("review-*.csv"))  # next to the database by default
    assert [p.name[-6:] for p in paths] == ["ha.csv", "yo.csv"]
    text = "".join(p.read_text(encoding="utf-8-sig") for p in paths)
    assert "08031234567" not in text and HASHER.user_id("+234803000000") not in text
    rows = [r for p in paths for r in read_cells(p)]
    assert [r["language"] for r in rows] == ["ha", "yo"] and rows[0]["reply"] == "'-Kada ka bayar"

    for r in rows:
        r.update(accuracy="4", fluency="5", faithful="yes", harmful_or_misleading="no", reviewer="Ade")
    graded_path = sheet(review_env / "graded.csv", rows)
    assert cli.main(["review", "score", str(graded_path), "--markdown"]) == 0
    assert "| **All** | 2 | 4.0 | 5.0 | 100% | 0% |" in capsys.readouterr().out
    assert cli.main(["review", "score", str(graded_path)]) == 0
    assert "Reviewers: Ade" in capsys.readouterr().out

    working = sheet(  # a half-graded working copy: check reads only the review ids
        review_env / "ha-ade.csv", [{**rows[0], "accuracy": "4", "fluency": "", "faithful": "maybe"}]
    )
    assert cli.main(["review", "check", *map(str, paths), str(working)]) == 0
    assert "all 2 review ids are still in the database" in capsys.readouterr().out

    async def erase() -> None:  # the Hausa user sends DELETE during the review
        store = await SQLiteStore.open(review_env / "kg.db", hasher=HASHER)
        await store.delete_user(HASHER.user_id("+234803000000"))
        await store.close()

    asyncio.run(erase())
    assert cli.main(["review", "check", *map(str, paths), str(working)]) == 1
    assert capsys.readouterr().out.splitlines()[1:] == [rows[0]["review_id"]]


def test_cli_review_export_reports_semantic_search(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from kudiguard.api import services

    async def with_embedder(settings: Any, client: httpx.AsyncClient) -> tuple[KnowledgeBase, object]:
        return knowledge_base_from_packs(ROOT / "data" / "packs", client=client), object()

    monkeypatch.setattr(services, "build_knowledge", with_embedder)
    assert cli.main(["review", "export", "--n", "1", "--out-dir", str(review_env / "s")]) == 0
    assert "semantic + keyword search" in capsys.readouterr().out


def test_cli_review_warnings_and_errors(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "ROOT", review_env)
    monkeypatch.setattr(cli, "IGNORED_DIRS", (review_env / "var",))
    assert cli.main(["review", "export", "--n", "1", "--out-dir", str(review_env / "docs")]) == 0
    out = capsys.readouterr()
    assert "sampled 1 of 3" in out.out and "not ignored" in out.err and "only" not in out.err
    assert cli.main(["review", "export", "--n", "1", "--out-dir", str(review_env / "var" / "r")]) == 0
    assert "not ignored" not in capsys.readouterr().err
    for bad, message in (
        (["--since", "yesterday"], "Invalid isoformat"),
        (["--n", "0"], "--n must be at least 1"),
        (["--languages", " , "], "--languages needs at least one"),
    ):
        assert cli.main(["review", "export", *bad]) == 2
        assert message in capsys.readouterr().err
    bad_sheet = sheet(review_env / "bad.csv", [graded("R-1", "ha", "9", "4", "yes", "no")])
    assert cli.main(["review", "score", str(bad_sheet)]) == 2
    assert "bad.csv row 2: accuracy" in capsys.readouterr().err


def test_read_review_ids_only(tmp_path: Path) -> None:
    path = write(tmp_path / "w.csv", "notes,review_id\nx,R-1\nonly-a-note\n,'R-2\n,\n")
    assert review.read_review_ids(path) == {"R-1", "R-2"}
    with pytest.raises(review.GradeError, match="no review_id column"):
        review.read_review_ids(write(tmp_path / "x.csv", "id\n1\n"))
    with pytest.raises(review.GradeError, match="no review_id column"):
        review.read_review_ids(write(tmp_path / "e.csv", ""))


def test_markdown_escapes_pipes_from_sheets(tmp_path: Path) -> None:
    path = sheet(tmp_path / "p.csv", [graded("R-1", "h|a", "4", "4", "yes", "no")])
    assert "| h\\|a | 1 |" in review.markdown_scorecard(review.score([review.read_sheet(path)]))


def test_cli_output_survives_a_cp1252_console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")  # Windows, stdout redirected to a file
    monkeypatch.setattr("sys.stdout", console)
    monkeypatch.setattr("sys.stderr", io.StringIO())  # not reconfigurable: left alone
    path = sheet(tmp_path / "s.csv", [graded("R-1", "yo", "4", "4", "yes", "no", reviewer="Ọlá Ɓala")])
    assert cli.main(["review", "score", str(path), "--markdown"]) == 0
    assert cli.main(["review", "score", str(path)]) == 0
    console.flush()
    assert "Reviewers: Ọlá Ɓala" in console.buffer.getvalue().decode("utf-8")


def test_cli_review_io_errors_are_clean(review_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    blocker = review_env / "file"
    blocker.write_text("not a folder", encoding="utf-8")
    assert cli.main(["review", "export", "--n", "1", "--out-dir", str(blocker / "sub")]) == 2
    assert "error: cannot write review sheets to" in capsys.readouterr().err
    assert cli.main(["review", "check", str(blocker)]) == 2  # not a sheet
    assert "no review_id column" in capsys.readouterr().err


def test_cli_review_check_needs_the_database(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(review_env / "missing.db"))
    assert cli.main(["review", "check", str(review_env / "any.csv")]) == 2
    assert "no evidence database" in capsys.readouterr().err
    assert not (review_env / "missing.db").exists()


# --- found by mutation testing (docs/SPIKES.md, G) ----------------------------------------------------------


def golden_card() -> review.Scorecard:
    g = review.Grade
    grades = (
        g("R-1", "ha", "Ade", 5, 4, True, False, "s row 2"),
        g("R-1", "ha", "Bola", 2, 4, False, True, "s row 3"),
        g("R-2", "ha", "Ade", 4, 3, True, False, "s row 4"),
    )
    return review.score([review.Sheet("s.csv", grades, {"R-1": "ha", "R-2": "ha", "R-3": "yo"})])


def test_scorecard_text_in_full() -> None:
    agreement = (
        "  agreement on 1 replies graded by 2+ reviewers: faithful 0% (AC1 -0.60, 1 split), harmful 0% (AC1 -0.60, "
        "1 split); accuracy 3.0, fluency 0.0 points apart on average"
    )
    assert review.format_scorecard(golden_card()).splitlines() == [
        "Reviewers: Ade, Bola",
        "ha: graded 2 (ungraded 0)  accuracy 3.8  fluency 3.5  faithful 50%  harmful 50%",
        agreement,
        "yo: graded 0 (ungraded 1)  accuracy n/a  fluency n/a  faithful n/a  harmful n/a",
        "all: graded 2 (ungraded 1)  accuracy 3.8  fluency 3.5  faithful 50%  harmful 50%",
        agreement,
        "Harmful or misleading (follow up): R-1",
        "Not faithful to the facts (follow up): R-1",
    ]
    empty = review.format_scorecard(review.score([review.Sheet("e.csv", (), {"R-9": "ig"})])).splitlines()
    assert empty[0] == "Reviewers: none"
    assert empty[-2:] == [
        "Harmful or misleading (follow up): none",
        "Not faithful to the facts (follow up): none",
    ]


def test_follow_up_lists_name_every_reply() -> None:
    g = review.Grade
    grades = (g("R-1", "ha", "Ade", 4, 4, False, True), g("R-2", "ha", "Ade", 4, 4, False, True))
    text = review.format_scorecard(review.score([review.Sheet("s.csv", grades, {"R-1": "ha", "R-2": "ha"})]))
    assert text.splitlines()[-2:] == [
        "Harmful or misleading (follow up): R-1, R-2",
        "Not faithful to the facts (follow up): R-1, R-2",
    ]


def test_markdown_scorecard_in_full() -> None:
    lines = review.markdown_scorecard(golden_card()).splitlines()
    assert lines[:11] == [
        "| Language | Replies graded | Accuracy (1-5) | Fluency (1-5) | Faithful to facts | Harmful or misleading |",
        "|---|---|---|---|---|---|",
        "| ha | 2 | 3.8 | 3.5 | 50% | 50% |",
        "| yo | 0 | n/a | n/a | n/a | n/a |",
        "| **All** | 2 | 3.8 | 3.5 | 50% | 50% |",
        "",
        "| Language | Graded by 2+ | Faithful: agreement (AC1, splits) | Harmful: agreement (AC1, splits) "
        "| Accuracy gap | Fluency gap |",
        "|---|---|---|---|---|---|",
        "| ha | 1 | 0% (AC1 -0.60, 1 split) | 0% (AC1 -0.60, 1 split) | 3.0 | 0.0 |",
        "| yo | 0 | n/a | n/a | n/a | n/a |",
        "| **All** | 1 | 0% (AC1 -0.60, 1 split) | 0% (AC1 -0.60, 1 split) | 3.0 | 0.0 |",
    ]
    assert lines[11] == "" and lines[12].startswith("2 native-speaker reviewers.")
    af = review.score([review.Sheet("af.csv", (), {"R-1": "af"})])  # a code sorting before "all"
    assert "| af | 0 |" in review.markdown_scorecard(af)


def test_ac1_with_three_reviewers_split_one_to_two() -> None:
    assert review.yes_no_agreement([[True, False, False]]) == review.YesNoAgreement(
        pytest.approx(1 / 3),
        pytest.approx(-0.2),
        1,  # pi = 1/3, p_e = 4/9
    )


def test_language_conflicts_are_found_in_either_order(tmp_path: Path) -> None:
    for first, second in (("yo", "ha"), ("ha", "yo")):
        rows = [graded("R-1", first, "", "", "", ""), graded("R-1", second, "", "", "", "")]
        with pytest.raises(review.GradeError, match="here but"):
            review.read_sheet(sheet(tmp_path / f"{first}.csv", rows))
        one = sheet(tmp_path / f"one-{first}.csv", rows[:1])
        two = sheet(tmp_path / f"two-{first}.csv", rows[1:])
        with pytest.raises(review.GradeError, match="in another sheet"):
            review.score([review.read_sheet(one), review.read_sheet(two)])


def test_two_reviewers_in_any_name_order_are_not_duplicates(tmp_path: Path) -> None:
    shared = sheet(
        tmp_path / "shared.csv",
        [
            graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Zainab"),
            graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Ade"),
        ],
    )
    assert review.score([review.read_sheet(shared)]).overall.agreement.items == 1


def test_every_grade_value_and_reviewer_name_is_read(tmp_path: Path) -> None:
    path = sheet(
        tmp_path / "s.csv",
        [
            graded("R-1", "ha", "1", "5", "y", "n", reviewer="Xena"),  # the whole scale, short answers
            graded("R-2", "ha", "5", "1", "n", "y", reviewer="Xena"),
        ],
    )
    first, second = review.read_sheet(path).grades
    assert (first.accuracy, first.fluency, first.faithful, first.harmful) == (1, 5, True, False)
    assert (second.accuracy, second.fluency, second.faithful, second.harmful) == (5, 1, False, True)
    assert first.reviewer == "Xena"  # only the spreadsheet text prefix (') is stripped


def test_short_and_blank_rows(tmp_path: Path) -> None:
    header = "language,review_id,accuracy,fluency,faithful,harmful_or_misleading,reviewer,notes"
    path = write(tmp_path / "trimmed.csv", f"{header}\r\nha,R-1,4,4,yes,no\r\n\r\nyo,R-2\r\n")
    parsed = review.read_sheet(path)
    assert parsed.items == {"R-1": "ha", "R-2": "yo"}  # a blank line is no item
    assert [g.reviewer for g in parsed.grades] == ["trimmed"]  # trailing empty cells left out: the file name
    assert review.read_review_ids(path) == {"R-1", "R-2"}
    short = "review_id,accuracy,fluency,faithful,harmful_or_misleading,language\r\nR-1\r\n"
    assert review.read_sheet(write(tmp_path / "short.csv", short)).items == {"R-1": ""}


def test_semicolon_sheet_with_commas_in_the_notes(tmp_path: Path) -> None:
    cells = graded(
        "R-1", "ha", "4", "4", "yes", "no", notes="wrong, " * 40
    )  # more commas than semicolons in the whole file
    text = ";".join(HEADER) + "\r\n" + ";".join(cells.get(c, "") for c in HEADER)
    assert review.read_sheet(write(tmp_path / "semi.csv", text)).items == {"R-1": "ha"}


def test_utf16_big_endian_sheet(tmp_path: Path) -> None:
    text = "review_id\tlanguage\taccuracy\tfluency\tfaithful\tharmful_or_misleading\r\nR-1\tyo\t4\t4\tyes\tno\r\n"
    path = write(tmp_path / "be.csv", b"\xfe\xff" + text.encode("utf-16-be"))
    assert review.read_sheet(path).items == {"R-1": "yo"}


def test_rewriting_a_sheet_replaces_a_longer_old_file(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("x" * 10_000, encoding="utf-8")
    review.write_sheet(path, [])
    assert path.read_bytes() == ("﻿" + ",".join(f'"{c}"' for c in review.COLUMNS) + "\r\n").encode("utf-8")


async def test_sheet_columns_and_context_cells(knowledge: KnowledgeBase, tmp_path: Path) -> None:
    rows = await review.review_rows([row(1, transcript=None)], hasher=HASHER, knowledge=knowledge)
    [path] = review.write_sheets(tmp_path, rows, stem="sample").values()
    [cells] = read_cells(path)
    assert list(cells) == [
        "review_id", "language", "channel", "verdict", "user_message", "english_pivot", "reply", "facts_rederived",
        "accuracy", "fluency", "faithful", "harmful_or_misleading", "reviewer", "notes",
    ]  # fmt: skip
    assert (cells["channel"], cells["verdict"], cells["user_message"]) == ("whatsapp", "likely_scam", "")
    assert cells["english_pivot"] == "Send me your OTP code"


def test_verdict_groups_and_language_order() -> None:
    assert review.verdict_group("not_a_check") == review.QUESTION
    assert {review.verdict_group(v) for v in ("likely_scam", "caution", "no_flags")} == {review.CHECK}
    assert sorted(["ar", "en", "ig", "zu", "ha"], key=review.language_key) == ["ha", "ig", "en", "ar", "zu"]


def test_since_keeps_that_day_and_drops_undated_rows() -> None:
    since = date(2026, 10, 9)
    undated = {k: v for k, v in row(3).items() if k != "created_at"}
    assert [r["id"] for r in review.eligible([row(1), undated], since=since)] == [1]


def test_large_samples() -> None:
    rows = [
        row(i, language=("ha", "yo")[i % 2], verdict=("caution", "not_a_check")[i // 2 % 2])
        for i in range(400)
    ]
    assert len(review.stratified_sample(rows, 300, seed=1, languages=("ha", "yo"))) == 300
