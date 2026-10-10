"""Every path that stores or exports user text masks numbers however they are split, including rows written before a
redaction fix (they are redacted again on the way out): evidence store, review sheets, scam patterns, voice export."""

import csv
import sqlite3
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
from kudiguard.storage.base import Store, VoiceSample
from kudiguard.storage.memory import MemoryStore
from kudiguard.storage.sqlite import SQLiteStore

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
