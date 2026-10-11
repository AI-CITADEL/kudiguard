"""Every path that stores or exports user text masks numbers however they are split, including rows written before a
redaction fix (they are redacted again on the way out): evidence store, review sheets, scam patterns, voice export."""

import csv
import json
import logging
import sqlite3
import time
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from kudiguard import review, voice_cli
from kudiguard.domain import Flag, Interaction, MessageKind, Outcome, Verdict
from kudiguard.donation import DONATION_VERSION, Donation, Sealer, VoiceVault
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.patterns import ScamPattern, deidentify
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage import legacy
from kudiguard.storage import sqlite as sqlite_module
from kudiguard.storage.base import Store, VoiceSample
from kudiguard.storage.export import export_row, protect, spreadsheet_cell
from kudiguard.storage.legacy import reredact, v1_deidentify
from kudiguard.storage.memory import MemoryStore
from kudiguard.storage.sqlite import SQLiteStore
from kudiguard.text import REDACTION_VERSION, RedactingFilter

ROOT = Path(__file__).resolve().parents[2]
HASHER = UserHasher(b"r" * MIN_SECRET_BYTES)
# A phone split after one digit, the same with the country code, dotted, read out, and a 20-digit card.
LEAKY = (
    "call 8 031 234 567 or +234 8 031 234 567 or 0803.123.4567, read 0 8 0 3 123 45 67, "
    "card 5399 8312 3456 7890 1234"
)


def no_digits(text: str | None) -> bool:
    return text is not None and not any(ch.isdigit() for ch in text) and "[number]" in text


@pytest.fixture(params=["memory", "sqlite"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Store]:
    impl: Store = (
        MemoryStore(HASHER)
        if request.param == "memory"
        else await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    )
    yield impl
    await impl.close()


def interaction(**kw: object) -> Interaction:
    return Interaction(
        channel="whatsapp", message_id="m1", user_hash="u1", language="en", kind=MessageKind.TEXT,
        outcome=Outcome.ANSWERED, intent="scam_check", verdict=Verdict.LIKELY_SCAM, **kw,
    )  # type: ignore[arg-type]  # fmt: skip


def test_deidentify_masks_every_spelling() -> None:
    assert no_digits(deidentify(LEAKY))


async def test_stores_mask_interactions_samples_and_patterns(store: Store) -> None:
    interaction_id = await store.add_interaction(interaction(transcript=LEAKY, english=LEAKY, reply=LEAKY))
    await store.set_outcome(interaction_id, Outcome.ERROR, error=LEAKY)
    loaded = await store.get_interaction(interaction_id)
    assert loaded is not None
    assert all(no_digits(t) for t in (loaded.transcript, loaded.english, loaded.reply, loaded.error))
    [row] = await store.export_interactions()
    assert all(no_digits(row[k]) for k in ("transcript", "english", "reply", "error"))
    sample = VoiceSample(
        id="a" * 32, user_hash="u1", week=date(2026, 10, 12), language="en", channel="whatsapp",
        consent_version=DONATION_VERSION, hypothesis=LEAKY,
    )  # fmt: skip
    await store.add_voice_sample(sample)
    [kept] = await store.voice_samples()
    assert no_digits(kept.hypothesis)
    pattern = ScamPattern.from_check(
        language="en", channel="whatsapp", verdict=Verdict.LIKELY_SCAM, flags=(Flag.ASKS_CREDENTIALS,),
        text=LEAKY, english=LEAKY, day=date(2026, 10, 12),
    )  # fmt: skip
    await store.add_pattern(pattern)
    [exported] = await store.export_patterns()
    assert no_digits(exported["text"]) and no_digits(exported["english"])


async def test_rows_stored_before_the_fix_are_masked_on_export(tmp_path: Path) -> None:
    db = tmp_path / "kg.db"
    store = await SQLiteStore.open(db, hasher=HASHER)
    await store.add_interaction(interaction(transcript="x", english="x", reply="x"))
    await store.add_pattern(
        ScamPattern.from_check(
            language="en", channel="whatsapp", verdict=Verdict.LIKELY_SCAM, flags=(), text="x", english="x",
            day=date(2026, 10, 12),
        )
    )  # fmt: skip
    await store.close()
    with sqlite3.connect(db) as legacy:  # what an older release (or redaction) may have left in the database
        legacy.execute(
            "UPDATE interactions SET transcript = ?, english = ?, reply = ?", (LEAKY, LEAKY, LEAKY)
        )
        legacy.execute("UPDATE scam_patterns SET text = ?, english = ?", (LEAKY, LEAKY))
    legacy.close()
    store = await SQLiteStore.open(db, hasher=HASHER)
    try:
        rows = await store.export_interactions()
        assert all(no_digits(rows[0][k]) for k in ("transcript", "english", "reply"))
        [pattern] = await store.export_patterns()
        assert no_digits(pattern["text"]) and no_digits(pattern["english"])
        rows[0]["created_at"] = datetime(2026, 10, 9, tzinfo=UTC).isoformat()
        async with httpx.AsyncClient() as client:
            knowledge = knowledge_base_from_packs(ROOT / "data" / "packs", client=client)
            [sheet_row] = await review.review_rows(rows, hasher=HASHER, knowledge=knowledge)
        assert no_digits(sheet_row["user_message"]) and no_digits(sheet_row["english_pivot"])
        assert no_digits(sheet_row["reply"])
    finally:
        await store.close()


async def test_a_database_from_an_older_release_is_redacted_again_on_open(tmp_path: Path) -> None:
    """Legacy rows are rewritten once (``redaction_version``): masked at rest, and DELETE still finds the examples."""
    db = tmp_path / "kg.db"
    leaky = "they said call 8 031 234 567 now"  # what the old patterns stored unmasked
    store = await SQLiteStore.open(db, hasher=HASHER)
    await store.add_interaction(interaction(transcript="x", english="x", reply="x"))
    await store.add_interaction(
        Interaction(
            channel="whatsapp", message_id="m2", user_hash="u1", language="en", kind=MessageKind.TEXT,
            outcome=Outcome.ANSWERED, intent="question", transcript="x", english="x", reply="x",
        )
    )  # fmt: skip  # not a scam check: no example was taken from it
    await store.add_pattern(
        ScamPattern.from_check(
            language="en", channel="whatsapp", verdict=Verdict.LIKELY_SCAM, flags=(), text="x", english="x",
            day=datetime.now(UTC).date(),
        )
    )  # fmt: skip
    await store.add_voice_sample(
        VoiceSample(
            id="b" * 32, user_hash="u1", week=date(2026, 10, 12), language="en", channel="whatsapp",
            consent_version=DONATION_VERSION, hypothesis="x",
        )
    )  # fmt: skip
    await store.close()
    with sqlite3.connect(db) as legacy:
        legacy.execute(
            "UPDATE interactions SET transcript = ?, english = ?, reply = ?, error = ?",
            (leaky, leaky, leaky, leaky),
        )
        legacy.execute("UPDATE scam_patterns SET text = ?, english = ?", (leaky, leaky))
        legacy.execute("UPDATE voice_samples SET hypothesis = ?", (leaky,))
        legacy.execute("DELETE FROM meta WHERE key = 'redaction_version'")  # written before the fix
    legacy.close()
    store = await SQLiteStore.open(db, hasher=HASHER)
    try:
        with sqlite3.connect(db) as raw:  # at rest, not only on export
            rows = raw.execute("SELECT transcript, english, reply, error FROM interactions").fetchall()
            assert all(no_digits(value) for row in rows for value in row)
            assert all(
                no_digits(t) for row in raw.execute("SELECT text, english FROM scam_patterns") for t in row
            )
            assert no_digits(raw.execute("SELECT hypothesis FROM voice_samples").fetchone()[0])
            version = raw.execute("SELECT value FROM meta WHERE key = 'redaction_version'").fetchone()[0]
        raw.close()
        assert int(version) == REDACTION_VERSION
        assert await store.delete_user("u1") >= 3  # the example taken from the user's message goes too
        assert await store.export_patterns() == []
    finally:
        await store.close()
    assert not files_containing(tmp_path, b"031 234 567")  # nor in free pages or the WAL file


def all_bytes(folder: Path) -> bytes:
    """The database, WAL and shared-memory files, raw."""
    return b"".join(f.read_bytes() for f in sorted(folder.glob("kg.db*")))


def files_containing(folder: Path, needle: bytes) -> list[Path]:
    return [f for f in folder.glob("kg.db*") if needle in f.read_bytes()]


async def test_voice_export_masks_hypotheses_stored_before_the_fix(tmp_path: Path) -> None:
    store = MemoryStore(HASHER)
    vault = VoiceVault(tmp_path / "voice", store, Sealer(b"v" * MIN_SECRET_BYTES))
    donation = Donation(
        user_hash="u1", channel="whatsapp", language="en", audio=b"OggS" + bytes(64), hypothesis="x",
        consent_version=DONATION_VERSION,
    )  # fmt: skip
    sample_id = await vault.donate(donation)
    assert sample_id
    [stored] = await store.voice_samples()
    legacy = VoiceSample(
        id=sample_id, user_hash="u1", week=stored.week, language="en", channel="whatsapp",
        consent_version=DONATION_VERSION, hypothesis=LEAKY,
    )  # fmt: skip
    assert await voice_cli.export_samples(vault, [legacy], tmp_path / "out") == (1, 0)
    with (tmp_path / "out" / voice_cli.MANIFEST).open(encoding="utf-8-sig", newline="") as handle:
        [row] = list(csv.DictReader(handle))
    assert no_digits(row["asr_hypothesis"])


async def test_clean_rows_are_left_alone_and_a_failed_rewrite_closes_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kg.db"
    store = await SQLiteStore.open(db, hasher=HASHER)
    await store.add_interaction(interaction(transcript="no numbers", english=None, reply="fine"))
    await store.add_pattern(
        ScamPattern.from_check(
            language="en", channel="whatsapp", verdict=Verdict.LIKELY_SCAM, flags=(), text="send the code",
            english=None, day=date(2026, 10, 12),
        )
    )  # fmt: skip
    await store.add_voice_sample(
        VoiceSample(
            id="c" * 32, user_hash="u1", week=date(2026, 10, 12), language="en", channel="whatsapp",
            consent_version=DONATION_VERSION, hypothesis="no numbers",
        )
    )  # fmt: skip
    await store.close()
    with sqlite3.connect(db) as legacy:
        legacy.execute("DELETE FROM meta WHERE key = 'redaction_version'")
    legacy.close()
    store = await SQLiteStore.open(db, hasher=HASHER)
    try:
        [row] = await store.export_interactions()
        assert (row["transcript"], row["english"], row["reply"]) == ("no numbers", None, "fine")
        [pattern] = await store.export_patterns()
        assert (pattern["text"], pattern["english"]) == ("send the code", None)
    finally:
        await store.close()

    def broken(connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sqlite_module, "_reredact", broken)
    with pytest.raises(sqlite3.OperationalError):
        await SQLiteStore.open(db, hasher=HASHER)
    opened: list[sqlite3.Connection] = []

    def failing(connection: sqlite3.Connection) -> None:  # any exception, not only a database error
        opened.append(connection)
        raise ValueError("not a database error")

    monkeypatch.setattr(sqlite_module, "_reredact", failing)
    with pytest.raises(ValueError, match="not a database error"):
        await SQLiteStore.open(db, hasher=HASHER)
    with pytest.raises(sqlite3.ProgrammingError):  # closed, not left open (and locked) behind the failure
        opened[0].execute("SELECT 1")


def test_a_busy_checkpoint_is_retried_then_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Busy:
        calls = 0

        def execute(self, sql: str) -> "Busy":
            self.calls += 1
            return self

        def fetchone(self) -> tuple[int, int, int]:
            return (1, 10, 0)  # another process is reading: the WAL cannot be truncated yet

    class Done(Busy):
        def fetchone(self) -> tuple[int, int, int]:
            return (0, 10, 10)  # not busy: ten frames copied, the WAL truncated

    pauses: list[float] = []
    monkeypatch.setattr(sqlite_module.time, "sleep", pauses.append)
    busy, done = Busy(), Done()
    assert not sqlite_module._checkpoint(busy)  # type: ignore[arg-type]
    assert busy.calls == sqlite_module.CHECKPOINT_ATTEMPTS == 5 and "WAL checkpoint busy" in caplog.text
    assert pauses == pytest.approx([0.05, 0.1, 0.15, 0.2, 0.25])  # growing pauses, 0.75 s at most
    assert sqlite_module._checkpoint(done) is True  # type: ignore[arg-type]
    assert done.calls == 1 and len(pauses) == 5


@pytest.mark.parametrize("verdict", [Verdict.LIKELY_SCAM, Verdict.CAUTION])
async def test_an_example_is_derived_again_from_its_own_check(tmp_path: Path, verdict: Verdict) -> None:
    """b4f96f8 reduced "0803.123.4567.com" to a link: from the example alone the number cannot be told apart, from the
    check it was taken from it can, and DELETE derives it the same way."""
    db = tmp_path / "kg.db"
    raw_text = "visit 0803.123.4567.com"  # what b4f96f8 stored for the check (its patterns missed the number)
    store = await SQLiteStore.open(db, hasher=HASHER)
    await store.add_interaction(
        Interaction(
            channel="whatsapp", message_id="m1", user_hash="u1", language="en", kind=MessageKind.TEXT,
            outcome=Outcome.ANSWERED, intent="scam_check", verdict=verdict, transcript="x",
            created_at=datetime(2026, 10, 7, 12, tzinfo=UTC),
        )
    )  # fmt: skip
    await store.add_pattern(
        ScamPattern.from_check(
            language="en", channel="whatsapp", verdict=verdict, flags=(), text="x", english=None,
            day=date(2026, 10, 7),
        )
    )  # fmt: skip
    await store.close()
    with sqlite3.connect(db) as legacy_db:
        legacy_db.execute("UPDATE interactions SET transcript = ?", (raw_text,))
        legacy_db.execute("UPDATE scam_patterns SET text = ?", (v1_deidentify(raw_text),))
        legacy_db.execute("DELETE FROM meta WHERE key = 'redaction_version'")
    legacy_db.close()
    assert v1_deidentify(raw_text) == "visit [link: 0803.123.4567.com]"
    store = await SQLiteStore.open(db, hasher=HASHER)
    try:
        [pattern] = await store.export_patterns()
        assert pattern["text"] == deidentify(reredact(raw_text)) == "visit [number].com"  # not "visit [link]"
        await store.delete_user("u1")
        assert await store.export_patterns() == []
    finally:
        await store.close()


LEGACY_DB = ROOT / "tests" / "fixtures" / "legacy_b4f96f8.db"
# Identifiers in the legacy fixture (written by the b4f96f8 release itself, the last one before redaction_version):
# none may be left at rest after the open, whatever the old patterns missed.
LEGACY_SECRETS = (
    "0803", "8031234567", "0123", "1234", "5399", "482913", "2231", "john", "@", "gmail", "[link: pay",
    chr(0x0660 + 8), chr(0xFF10 + 8),
)  # fmt: skip


LEGACY_COLUMNS = (
    ("interactions", "transcript, english, reply, error"),
    ("voice_samples", "hypothesis"),
    ("scam_patterns", "text, english"),
)


def stored_values(db: Path) -> list[str]:
    with sqlite3.connect(db) as raw:
        values = [
            value
            for table, columns in LEGACY_COLUMNS
            for row in raw.execute(f"SELECT {columns} FROM {table}")
            for value in row
            if value
        ]
    raw.close()
    return values


async def test_a_database_written_by_the_b4f96f8_release_is_migrated(tmp_path: Path) -> None:
    """Built with b4f96f8's own code (two users, 22 checks with every spelling the old patterns missed): right after
    the open no old value that was rewritten is left anywhere in the files (database, WAL, shared memory: free pages
    and unallocated space included), nothing identifying is at rest, and DELETE removes every example taken from
    each user's messages."""
    db = tmp_path / "kg.db"
    db.write_bytes(LEGACY_DB.read_bytes())
    old = stored_values(db)
    store = await SQLiteStore.open(db, hasher=UserHasher(b"k" * 32))
    try:
        texts = stored_values(db)
        rewritten = sorted({v for v in old if v not in texts and v not in "".join(texts)})
        assert len(rewritten) >= 40
        raw_bytes = all_bytes(tmp_path)
        assert [v for v in rewritten if v.encode("utf-8") in raw_bytes] == []
        identifiers = [
            s for s in LEGACY_SECRETS if len(s) > 1
        ]  # a lone "@" or digit also occurs in page headers
        assert not [s for s in identifiers if s.encode("utf-8") in raw_bytes]
        with sqlite3.connect(db) as raw:
            patterns = raw.execute("SELECT count(*) FROM scam_patterns").fetchone()[0]
        raw.close()
        assert patterns == 22 and len(texts) == 140
        assert not [t for t in texts if any(secret in t for secret in LEGACY_SECRETS)]
        assert await store.delete_user("userA") > 0 and await store.delete_user("userB") > 0
        assert await store.export_patterns() == []  # 22 of 22 found again
    finally:
        await store.close()
    assert not files_containing(tmp_path, b"0803") and not files_containing(tmp_path, b"john")


async def legacy_database(tmp_path: Path, leaky: str, version: int | None) -> Path:
    db = tmp_path / "kg.db"
    store = await SQLiteStore.open(db, hasher=HASHER)
    await store.add_interaction(interaction(transcript="x", english="x", reply="x"))
    await store.add_pattern(
        ScamPattern.from_check(
            language="en", channel="whatsapp", verdict=Verdict.LIKELY_SCAM, flags=(), text="x", english="x",
            day=date(2026, 10, 12),
        )
    )  # fmt: skip
    await store.add_voice_sample(
        VoiceSample(
            id="d" * 32, user_hash="u1", week=date(2026, 10, 12), language="en", channel="whatsapp",
            consent_version=DONATION_VERSION, hypothesis="x",
        )
    )  # fmt: skip
    await store.close()
    with sqlite3.connect(db) as legacy:
        legacy.execute("UPDATE interactions SET transcript = ?", (leaky,))
        legacy.execute("UPDATE scam_patterns SET text = ?", (leaky,))
        legacy.execute("UPDATE voice_samples SET hypothesis = ?", (leaky,))
        if version is None:
            legacy.execute("DELETE FROM meta WHERE key = 'redaction_version'")
        else:
            legacy.execute("UPDATE meta SET value = ? WHERE key = 'redaction_version'", (str(version),))
    legacy.close()
    return db


def stored_texts(db: Path) -> list[str]:
    with sqlite3.connect(db) as raw:
        texts = [
            raw.execute("SELECT transcript FROM interactions").fetchone()[0],
            raw.execute("SELECT text FROM scam_patterns").fetchone()[0],
            raw.execute("SELECT hypothesis FROM voice_samples").fetchone()[0],
        ]
    raw.close()
    return texts


async def test_text_that_redacts_to_something_sorting_lower_is_rewritten(tmp_path: Path) -> None:
    db = await legacy_database(
        tmp_path, "mail john(at)gmail.com", version=None
    )  # "[email]" sorts before "john"
    await (await SQLiteStore.open(db, hasher=HASHER)).close()
    assert stored_texts(db) == ["mail [email]"] * 3


@pytest.mark.parametrize("offset", [0, 1])
async def test_a_database_at_or_after_this_version_is_not_rewritten(tmp_path: Path, offset: int) -> None:
    """Opening stays cheap: the rewrite runs once per version (and never downgrades a newer database)."""
    leaky = "call 8 031 234 567"  # only possible here because the test writes it behind the store's back
    db = await legacy_database(tmp_path, leaky, version=REDACTION_VERSION + offset)
    await (await SQLiteStore.open(db, hasher=HASHER)).close()
    assert stored_texts(db) == [leaky] * 3


def test_evidence_export_columns() -> None:
    row = export_row(interaction(transcript="t", english="e", reply="r"), None)
    assert list(row) == [
        "id", "created_at", "user", "channel", "language", "kind", "outcome", "intent", "verdict", "flags",
        "transcript", "english", "reply", "reply_to", "decode_ms", "asr_ms", "llm_ms", "total_ms", "error", "helpful",
    ]  # fmt: skip


NBSP, IDEO = chr(0x00A0), chr(0x3000)  # no-break and ideographic spaces


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("X=1", "X=1"),
        (IDEO + "=1", "'" + IDEO + "=1"),
        (NBSP + "@SUM(1)", "'" + NBSP + "@SUM(1)"),
        (" \t\r\n-1", "' \t\r\n-1"),
    ],
)
def test_only_blanks_are_skipped_before_a_formula(value: str, expected: str) -> None:
    assert spreadsheet_cell(value) == expected


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("card [number] 1234", "card [number]"),  # digits the old patterns left next to a mask go with it
        ("card [number] 1234.", "card [number]."),
        ("call 0803 [number] now", "call [number] now"),
        ("a [number] 12 34 [number] b", "a [number] [number] b"),
        (
            "[number] 10:30 and N5,000 [number]",
            "[number] 10:30 and N5,000 [number]",
        ),  # times and amounts stay
        ("[number] 1.5 kg", "[number] 1.5 kg"),
        ("call 8 031 234 567", "call [number]"),  # and everything today's rules mask
        ("no numbers", "no numbers"),
        ("12 apples and [number]", "12 apples and [number]"),  # only digits touching a mask
        ("[number] now and 1234", "[number] now and 1234"),
        ("1234 [number]", "[number]"),
        ("[number] 1234", "[number]"),
    ],
)
def test_legacy_text_is_redacted_again(stored: str, expected: str) -> None:
    assert reredact(stored) == expected
    assert reredact(expected) == expected


def test_the_frozen_v1_functions_match_the_b4f96f8_release() -> None:
    """Recorded with b4f96f8's own code (corpus, review probes, 1,000 joined cases): the match must be exact."""
    golden = json.loads((ROOT / "tests" / "fixtures" / "legacy_v1_golden.json").read_text(encoding="utf-8"))
    assert len(golden) > 1500
    assert [(text, legacy._v1_redact(text), v1_deidentify(text)) for text, _, _ in golden] == [
        tuple(g) for g in golden
    ]


def test_the_frozen_v1_deidentify_reproduces_the_old_examples() -> None:
    """b4f96f8's output for its own stored transcripts: what DELETE matched (kept only to find the source again)."""
    assert v1_deidentify("send to [number] now") == "send to [number] now"
    assert v1_deidentify("call 8 031 234 567 or 0803 123 4567") == "call 8 031 234 567 or [number]"
    assert v1_deidentify("card 5399 8312 3456 7890 and 2 2 3 1 5 6 7 8 9 0") == "card [number] and [number]"
    assert v1_deidentify("acct 1234567 pay N500000") == "acct [number] pay N500000"
    assert v1_deidentify("mail j@x.com, see https://u:p@x.com/a www.Bank.ng/x pay1234567.ng") == (
        "mail [email], see [link] [link: bank.ng] [link: pay#.ng]"
    )
    assert v1_deidentify("0803 12 34") == "0803 12 34"  # too short for a number


def test_every_redaction_path_takes_a_long_run_of_dates() -> None:
    """The verifier's crash: 64,000 characters of dates (RecursionError, then minutes) through store, patterns and
    logs. Now linear, and the pipeline cuts inbound text to 8,000 characters before any of them."""
    dates = "10/10/2026 " * 5_800 + "0803 123 4567"
    started = time.perf_counter()
    stored = protect(interaction(transcript=dates, english=dates, reply=dates, error=dates), HASHER)
    assert stored.transcript is not None and stored.transcript.endswith("[number]")
    assert deidentify(dates).startswith("10/10/2026")
    record = logging.LogRecord("kudiguard", logging.ERROR, __file__, 1, "failed: %s", (dates,), None)
    assert RedactingFilter().filter(record) and record.getMessage().endswith("[number]")
    assert time.perf_counter() - started < 3.0
