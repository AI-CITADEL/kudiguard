"""Native-speaker review kit: sampling, the reviewer sheet, grade validation, scoring and the CLI."""

import asyncio
import csv
import io
import re
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime
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


# --- export ---------------------------------------------------------------------------------------------------


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


def test_small_strata_leave_their_share_to_others() -> None:
    rows = [row(1, "ha", "not_a_check")] + [row(n, "yo") for n in range(2, 30)]
    sample = review.stratified_sample(rows, 10, seed=0)
    assert len(sample) == 10 and sum(r["language"] == "ha" for r in sample) == 1
    assert len(review.stratified_sample(rows[:3], 10, seed=0)) == 3  # fewer than asked: all of them


def test_spreadsheet_cells_cannot_run_formulas() -> None:
    assert spreadsheet_cell('=HYPERLINK("http://x","Click")') == '\'=HYPERLINK("http://x","Click")'
    assert [spreadsheet_cell(v) for v in ("+234", "-x", "@a", "\tz")] == ["'+234", "'-x", "'@a", "'\tz"]
    assert spreadsheet_cell("Ẹ kú àárọ̀") == "Ẹ kú àárọ̀"


def test_review_ids_are_opaque_and_stable() -> None:
    first = HASHER.review_id(7)
    assert first == HASHER.review_id(7) != HASHER.review_id(8)
    assert first.startswith("R-") and len(first) == 12
    assert UserHasher(b"t" * MIN_SECRET_BYTES).review_id(7) != first


def test_facts_format() -> None:
    assert review.format_facts([], []) == "(no facts found)"
    assert review.format_facts(["A.", "B."], ["Do x."]) == "1. A.\n2. B.\nSteps:\n- Do x."


@pytest.fixture
async def knowledge() -> Any:
    async with httpx.AsyncClient() as client:
        yield knowledge_base_from_packs(ROOT / "data" / "packs", client=client)


async def test_sheet_rows_carry_no_ids_and_rederive_the_facts(
    knowledge: KnowledgeBase, tmp_path: Path
) -> None:
    sample = [row(5, "yo", transcript="=cmd|' /C calc'!A0"), row(3, "ha", "not_a_check", flags=[])]
    rows = await review.review_rows(sample, hasher=HASHER, knowledge=knowledge)
    assert [r["language"] for r in rows] == ["ha", "yo"]
    otp = rows[1]
    assert otp["review_id"] == HASHER.review_id(5) and "OTP" in otp["facts_rederived"]
    assert otp["facts_rederived"].startswith("1. ")

    path = tmp_path / "out" / "sheet.csv"
    review.write_sheet(path, rows)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw  # BOM for Excel
    text = raw.decode("utf-8-sig")
    assert "user5" not in text and ',"5",' not in text
    parsed = list(csv.DictReader(io.StringIO(text)))
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


# --- score ----------------------------------------------------------------------------------------------------

HEADER = list(review.COLUMNS)


def sheet(
    path: Path, rows: list[Mapping[str, str]], *, delimiter: str = ",", encoding: str = "utf-8-sig"
) -> Path:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEADER, delimiter=delimiter)
    writer.writeheader()
    writer.writerows({c: r.get(c, "") for c in HEADER} for r in rows)
    path.write_bytes(buffer.getvalue().encode(encoding))
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
            {"review_id": "", "language": ""},  # blank line in a spreadsheet
        ],
    )
    card = review.score([review.read_sheet(path)])
    yo, ha, overall = card.languages
    assert (yo.language, yo.graded, yo.ungraded, yo.accuracy, yo.fluency) == ("yo", 2, 0, 4.0, 3.0)
    assert (yo.faithful_rate, yo.harmful_rate) == (0.5, 0.5)
    assert (ha.graded, ha.ungraded, ha.accuracy, ha.faithful_rate) == (0, 1, None, None)
    assert (overall.language, overall.graded, overall.ungraded) == ("all", 2, 1)
    assert card.reviewers == ("ade",) and card.harmful == ("R-2",) and card.unfaithful == ("R-2",)
    assert card.agreement == review.Agreement(0, None, None, None, None)
    text = review.format_scorecard(card)
    assert "yo: graded 2 (ungraded 0)  accuracy 4.0" in text and "(follow up): R-2" in text
    assert "ha: graded 0 (ungraded 1)  accuracy n/a" in text and "on 0 items" in text


def test_score_several_reviewers_with_agreement(tmp_path: Path) -> None:
    a = sheet(
        tmp_path / "a.csv",
        [
            graded("R-1", "ig", "5", "5", "yes", "no", reviewer="Chi"),
            graded("R-2", "ig", "4", "4", "yes", "no"),
        ],
    )
    b = sheet(
        tmp_path / "b.csv",
        [
            graded("R-1", "ig", "3", "4", "no", "no", reviewer="Obi"),
            graded("R-2", "ig", "4", "2", "yes", "yes"),
        ],
        delimiter=";",  # Excel in a comma-decimal locale
        encoding="cp1252",  # Excel's plain "CSV" save
    )
    (tmp_path / "b.csv").write_bytes((tmp_path / "b.csv").read_bytes().replace(b"R-2", b"R-2\x81", 0))
    card = review.score([review.read_sheet(a), review.read_sheet(b)])
    ig = card.languages[0]
    assert ig.graded == 2 and ig.accuracy == 4.0 and ig.fluency == 3.75
    assert ig.faithful_rate == 0.5 and ig.harmful_rate == 0.5  # any reviewer's flag counts
    assert card.reviewers == ("Chi", "Obi", "a", "b")
    assert card.agreement == review.Agreement(2, 0.5, 0.5, 1.0, 1.5)

    same = sheet(tmp_path / "c.csv", [graded("R-1", "ig", "3", "4", "no", "no")])
    other = sheet(tmp_path / "d.csv", [graded("R-1", "ig", "5", "5", "yes", "no")])
    card = review.score([review.read_sheet(same), review.read_sheet(other)])
    assert card.agreement == review.Agreement(1, 0.0, 1.0, 2.0, 1.0)
    md = review.markdown_scorecard(card)
    assert md.splitlines()[2] == "| ig | 1 | 4.0 | 4.5 | 0% | 0% |"
    assert "| **All** | 1 |" in md and "2 native-speaker reviewers" in md and "on 1 double-graded" in md


def test_cp1252_sheets_are_read(tmp_path: Path) -> None:
    path = tmp_path / "excel.csv"
    path.write_bytes("review_id,language,accuracy,fluency,faithful,harmful_or_misleading,notes\r\n"
                     "R-1,ha,4,4,yes,no,caf\xe9 \x81\r\n".encode("latin-1"))  # fmt: skip
    assert review.read_sheet(path).grades[0].accuracy == 4


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


def test_sheet_level_errors(tmp_path: Path) -> None:
    dup = sheet(
        tmp_path / "dup.csv", [graded("R-1", "ha", "", "", "", ""), graded("R-1", "ha", "", "", "", "")]
    )
    with pytest.raises(review.GradeError, match=re.escape("dup.csv row 3: R-1 appears twice")):
        review.read_sheet(dup)
    other = tmp_path / "other.csv"
    other.write_text("id,text\n1,x\n", encoding="utf-8")
    with pytest.raises(review.GradeError, match="not a review sheet"):
        review.read_sheet(other)
    with pytest.raises(review.GradeError, match="cannot read"):
        review.read_sheet(tmp_path / "missing.csv")
    one = sheet(tmp_path / "one.csv", [graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Ade")])
    two = sheet(tmp_path / "two.csv", [graded("R-1", "ha", "4", "4", "yes", "no", reviewer="Ade")])
    with pytest.raises(review.GradeError, match="R-1 graded twice by reviewer 'Ade'"):
        review.score([review.read_sheet(one), review.read_sheet(two)])


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


def test_cli_review_export_then_score(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "REVIEW_DIR", review_env / "var" / "review")
    team = review_env / "team.txt"
    team.write_text("+234803000002\n", encoding="utf-8")
    assert (
        cli.main(["review", "export", "--exclude-users", str(team), "--since", "2026-10-01", "--seed", "3"])
        == 0
    )
    out = capsys.readouterr()
    assert "wrote 2 of 2 eligible replies" in out.out and "only 2 eligible replies (asked for 50)" in out.err
    assert "not ignored" not in out.err  # outside the repository
    [path] = (review_env / "var" / "review").glob("review-*.csv")
    text = path.read_text(encoding="utf-8-sig")
    assert "08031234567" not in text and HASHER.user_id("+234803000000") not in text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["language"] for r in rows] == ["ha", "yo"] and rows[0]["reply"] == "'-Kada ka bayar"

    for r in rows:
        r.update(accuracy="4", fluency="5", faithful="yes", harmful_or_misleading="no", reviewer="Ade")
    graded_path = sheet(review_env / "graded.csv", rows)
    assert cli.main(["review", "score", str(graded_path), "--markdown"]) == 0
    assert "| **All** | 2 | 4.0 | 5.0 | 100% | 0% |" in capsys.readouterr().out
    assert cli.main(["review", "score", str(graded_path)]) == 0
    assert "Reviewers: Ade" in capsys.readouterr().out


def test_cli_review_warnings_and_errors(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "ROOT", review_env)
    monkeypatch.setattr(cli, "IGNORED_DIRS", (review_env / "var",))
    assert cli.main(["review", "export", "--n", "1", "--out", str(review_env / "docs" / "r.csv")]) == 0
    out = capsys.readouterr()
    assert "wrote 1 of 3" in out.out and "not ignored" in out.err and "only" not in out.err
    assert cli.main(["review", "export", "--n", "1", "--out", str(review_env / "var" / "r.csv")]) == 0
    assert "not ignored" not in capsys.readouterr().err
    assert cli.main(["review", "export", "--since", "yesterday"]) == 2
    assert "error:" in capsys.readouterr().err
    bad = sheet(review_env / "bad.csv", [graded("R-1", "ha", "9", "4", "yes", "no")])
    assert cli.main(["review", "score", str(bad)]) == 2
    assert "bad.csv row 2: accuracy" in capsys.readouterr().err
