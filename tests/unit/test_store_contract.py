"""One behavioural contract, run against every Store implementation (SQLite for production, memory for tests)."""

import asyncio
from collections.abc import AsyncIterator
from datetime import date, timedelta
from pathlib import Path

import pytest

from kudiguard.domain import Flag, Interaction, MessageKind, Outcome, Verdict
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.base import Store, UserState, VoiceSample, utcnow
from kudiguard.storage.memory import MemoryStore
from kudiguard.storage.sqlite import SQLiteStore


@pytest.fixture(params=["memory", "sqlite"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Store]:
    hasher = UserHasher(b"s" * MIN_SECRET_BYTES)
    if request.param == "memory":
        impl: Store = MemoryStore(hasher)
    else:
        impl = await SQLiteStore.open(tmp_path / "kg.db", hasher=hasher)
    yield impl
    await impl.close()


def turn(user: str = "u1", message_id: str = "m1", **kw: object) -> Interaction:
    return Interaction(
        channel="whatsapp", message_id=message_id, user_hash=user, language="ha", kind=MessageKind.AUDIO, **kw
    )  # type: ignore[arg-type]


async def test_user_state_round_trip(store: Store) -> None:
    assert await store.get_user("u1") is None
    state = UserState("u1", language="ha", consented_at=utcnow())
    await store.save_user(state)
    loaded = await store.get_user("u1")
    assert loaded is not None and loaded.language == "ha" and loaded.consented_at is not None
    loaded.stopped, loaded.language, loaded.last_interaction_id = True, "yo", 7
    await store.save_user(loaded)
    again = await store.get_user("u1")
    assert again is not None and again.stopped and again.language == "yo" and again.last_interaction_id == 7


async def test_claim_message_is_idempotent_and_concurrency_safe(store: Store) -> None:
    results = await asyncio.gather(*(store.claim_message("whatsapp", "wamid.1") for _ in range(10)))
    assert results.count(True) == 1
    assert await store.claim_message("web", "wamid.1")  # ids are per channel


async def test_interaction_round_trip_preserves_every_field(store: Store) -> None:
    original = turn(
        outcome=Outcome.ANSWERED,
        intent="scam_check",
        transcript="an toshe BVN dina",
        english="my BVN was blocked",
        verdict=Verdict.LIKELY_SCAM,
        flags=(Flag.ASKS_CREDENTIALS, Flag.ACCOUNT_THREAT),
        reply="🔴 Wannan damfara ce",
        decode_ms=90,
        asr_ms=1800,
        llm_ms=2900,
        total_ms=5000,
    )
    interaction_id = await store.add_interaction(original)
    loaded = await store.get_interaction(interaction_id)
    assert loaded is not None and loaded.id == interaction_id
    for name in (
        "outcome",
        "intent",
        "transcript",
        "english",
        "verdict",
        "flags",
        "reply",
        "asr_ms",
        "total_ms",
    ):
        assert getattr(loaded, name) == getattr(original, name), name
    assert loaded.kind is MessageKind.AUDIO and loaded.created_at.tzinfo is not None
    assert await store.get_interaction(999_999) is None


async def test_feedback_latest_vote_wins_and_appears_in_export(store: Store) -> None:
    interaction_id = await store.add_interaction(turn())
    await store.add_feedback("u1", interaction_id, helpful=False)
    await store.add_feedback("u1", interaction_id, helpful=True)
    rows = await store.export_interactions()
    assert len(rows) == 1 and rows[0]["helpful"] is True


async def test_feedback_only_for_own_interactions(store: Store) -> None:
    interaction_id = await store.add_interaction(turn(user="u1"))
    await store.add_feedback("intruder", interaction_id, helpful=False)
    assert (await store.export_interactions())[0]["helpful"] is None


async def test_rate_limit_counting(store: Store) -> None:
    for n in range(3):
        await store.add_interaction(turn(message_id=f"m{n}"))
    await store.add_interaction(turn(user="u2", message_id="other"))
    assert await store.count_interactions_since("u1", utcnow() - timedelta(hours=1)) == 3
    assert await store.count_interactions_since("u1", utcnow() + timedelta(seconds=1)) == 0


async def test_delete_user_erases_everything(store: Store) -> None:
    await store.save_user(UserState("u1", language="ha"))
    interaction_id = await store.add_interaction(turn())
    await store.add_feedback("u1", interaction_id, helpful=True)
    await store.add_interaction(turn(user="u2", message_id="keep"))
    removed = await store.delete_user("u1")
    assert removed >= 3
    assert await store.get_user("u1") is None
    assert await store.get_interaction(interaction_id) is None
    assert [r["user"] for r in await store.export_interactions()] == ["u2"]


async def test_retention_purge(store: Store) -> None:
    old = turn(message_id="old")
    old.created_at = utcnow() - timedelta(days=100)
    old_id = await store.add_interaction(old)
    await store.add_feedback("u1", old_id, helpful=True)
    new_id = await store.add_interaction(turn(message_id="new"))
    stale = UserState("stale", last_seen_at=utcnow() - timedelta(days=100))
    await store.save_user(stale)
    await store.save_user(UserState("u1"))
    removed = await store.purge_before(utcnow() - timedelta(days=90))
    assert removed >= 3
    assert await store.get_interaction(old_id) is None and await store.get_interaction(new_id) is not None
    assert await store.get_user("stale") is None and await store.get_user("u1") is not None


async def test_export_is_privacy_safe_and_complete(store: Store) -> None:
    await store.add_interaction(turn(verdict=Verdict.CAUTION, flags=(Flag.URGENCY,), total_ms=4200))
    (row,) = await store.export_interactions()
    assert row["user"] == "u1" and row["verdict"] == "caution" and row["flags"] == ["urgency"]
    assert row["total_ms"] == 4200 and row["language"] == "ha" and row["kind"] == "audio"
    assert "phone" not in row and "message_id" not in row  # provider ids can be linked back to a number


WAMID = "wamid.HBgNMjM0ODAzMTIzNDU2NxUCABIYFjNFQjBGNEY="  # base64 protobuf embedding 2348031234567


async def test_provider_message_ids_are_never_stored_raw(store: Store, tmp_path: Path) -> None:
    assert await store.claim_message("whatsapp", WAMID)
    assert not await store.claim_message("whatsapp", WAMID)  # de-dup still works on the hash
    interaction_id = await store.add_interaction(turn(message_id=WAMID))
    stored = await store.get_interaction(interaction_id)
    assert stored is not None and stored.message_id != WAMID and "HBgN" not in stored.message_id

    def raw_bytes_on_disk() -> bool:  # SQLite: main file and WAL
        return any(b"HBgNMjM0ODAz" in f.read_bytes() for f in tmp_path.glob("kg.db*"))

    assert not await asyncio.to_thread(raw_bytes_on_disk)


async def test_store_redacts_free_text(store: Store) -> None:
    interaction_id = await store.add_interaction(
        turn(
            transcript="my account 0123456789",
            english="call 08031234567",
            reply="ok",
            error="x@y.example failed",
        )
    )
    stored = await store.get_interaction(interaction_id)
    assert stored is not None
    assert "0123456789" not in (stored.transcript or "") and "08031234567" not in (stored.english or "")
    assert "x@y.example" not in (stored.error or "")


async def test_count_active_users(store: Store) -> None:
    for user, n in (("u1", 1), ("u1", 2), ("u2", 3)):
        await store.add_interaction(turn(user=user, message_id=f"m{n}"))
    old = turn(user="u3", message_id="old")
    old.created_at = utcnow() - timedelta(days=40)
    await store.add_interaction(old)
    assert await store.count_active_users_since(utcnow() - timedelta(days=30)) == 2


async def test_naive_datetimes_rejected(store: Store) -> None:
    from datetime import datetime

    with pytest.raises(ValueError, match="naive"):
        await store.count_interactions_since("u1", datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="naive"):
        await store.save_user(UserState("u1", created_at=datetime(2026, 1, 1)))


async def test_consent_version_and_reply_to_round_trip(store: Store) -> None:
    await store.save_user(UserState("u1", consent_version="2026-10-07"))
    user = await store.get_user("u1")
    assert user is not None and user.consent_version == "2026-10-07"
    first = await store.add_interaction(turn(message_id="a"))
    follow_up = await store.add_interaction(turn(message_id="b", reply_to=first))
    stored = await store.get_interaction(follow_up)
    assert stored is not None and stored.reply_to == first


async def test_closed_store_refuses_use(store: Store) -> None:
    from kudiguard.storage.base import StoreClosedError

    await store.close()
    await store.close()  # idempotent
    with pytest.raises(StoreClosedError):
        await store.get_user("u1")
    with pytest.raises(StoreClosedError):
        await store.claim_message("whatsapp", "m1")


async def test_set_outcome_marks_undelivered_answers(store: Store) -> None:
    interaction_id = await store.add_interaction(turn(message_id="a"))
    await store.set_outcome(interaction_id, Outcome.ERROR, "undelivered: call 08031234567")
    stored = await store.get_interaction(interaction_id)
    assert stored is not None and stored.outcome is Outcome.ERROR
    assert stored.error is not None and "08031234567" not in stored.error
    await store.set_outcome(999, Outcome.ERROR)  # unknown id: no-op


async def test_retention_purge_forgets_old_message_claims(store: Store) -> None:
    assert await store.claim_message("whatsapp", "wamid.1")
    assert await store.purge_before(utcnow() - timedelta(days=1)) == 0  # a fresh claim is kept...
    assert not await store.claim_message("whatsapp", "wamid.1")  # ...so a redelivery is still a duplicate
    assert (
        await store.purge_before(utcnow() + timedelta(minutes=1)) == 1
    )  # the claim itself counts as removed
    assert await store.claim_message("whatsapp", "wamid.1")  # nothing about the old message id remains


async def test_count_active_users_by_channel(store: Store) -> None:
    for n, (channel, user) in enumerate(
        (("whatsapp", "u1"), ("web", "w1"), ("web", "w2"), ("whatsapp", "u1"))
    ):
        item = turn(user=user, message_id=f"c{n}")
        item.channel = channel
        await store.add_interaction(item)
    since = utcnow() - timedelta(days=30)
    assert await store.count_active_users_since(since) == 3
    assert await store.count_active_users_since(since, channels={"whatsapp"}) == 1
    assert await store.count_active_users_since(since, channels={"web", "whatsapp"}) == 3
    assert await store.count_active_users_since(since, channels=()) == 0


# --- voice-note donation ------------------------------------------------------------------------------------------


def sample(sample_id: str, user: str = "u1", day: date = date(2026, 10, 15), **kw: object) -> VoiceSample:
    values: dict[str, object] = {
        "id": sample_id, "user_hash": user, "week": day, "language": "yo", "channel": "whatsapp",
        "consent_version": "2026-10-10", "hypothesis": "pe 08031234567", "duration_s": 3.5,
    }  # fmt: skip
    values.update(kw)
    return VoiceSample(**values)  # type: ignore[arg-type]


async def test_donation_consent_round_trips(store: Store) -> None:
    given, offered = utcnow(), utcnow() - timedelta(days=1)
    await store.save_user(
        UserState("u1", donation_version="2026-10-10", donation_at=given, donation_offered_at=offered)
    )
    loaded = await store.get_user("u1")
    assert loaded is not None and loaded.donation_version == "2026-10-10"
    assert loaded.donation_at == given and loaded.donation_offered_at == offered
    loaded.donation_version, loaded.donation_at = None, None  # withdrawal
    await store.save_user(loaded)
    again = await store.get_user("u1")
    assert again is not None and again.donation_version is None and again.donation_at is None
    assert again.donation_offered_at == offered  # offered once, remembered


async def test_voice_samples_are_minimised_and_ordered_by_week(store: Store) -> None:
    await store.add_voice_sample(sample("b" * 32, day=date(2026, 10, 15)))  # Thursday
    await store.add_voice_sample(sample("a" * 32, user="u2", day=date(2026, 10, 19), duration_s=None))
    await store.add_voice_sample(sample("c" * 32, day=date(2026, 10, 12)))
    rows = await store.voice_samples()
    assert [(r.id, r.week) for r in rows] == [
        ("b" * 32, date(2026, 10, 12)), ("c" * 32, date(2026, 10, 12)), ("a" * 32, date(2026, 10, 19)),
    ]  # fmt: skip
    assert all("08031234567" not in r.hypothesis for r in rows)  # redacted by the store
    assert rows[0].duration_s == 3.5 and rows[2].duration_s is None
    assert rows[0].consent_version == "2026-10-10" and rows[0].channel == "whatsapp"
    assert [r.id for r in await store.voice_samples("u2")] == ["a" * 32]
    assert await store.voice_samples("nobody") == []


async def test_withdrawal_deletes_only_that_donors_samples(store: Store) -> None:
    for n, user in enumerate(("u1", "u1", "u2")):
        await store.add_voice_sample(sample(f"{n}" * 32, user=user))
    assert await store.delete_voice_samples("u1") == ["0" * 32, "1" * 32]
    assert [s.id for s in await store.voice_samples()] == ["2" * 32]
    assert await store.delete_voice_samples("u1") == []


async def test_welcome_and_withdrawal_round_trip(store: Store) -> None:
    withdrawn = utcnow()
    await store.save_user(UserState("u1", welcome_version="2026-10-10", donation_withdrawn_at=withdrawn))
    loaded = await store.get_user("u1")
    assert (
        loaded is not None
        and loaded.welcome_version == "2026-10-10"
        and loaded.donation_withdrawn_at == withdrawn
    )
    loaded.welcome_version, loaded.donation_withdrawn_at = None, None
    await store.save_user(loaded)
    again = await store.get_user("u1")
    assert again is not None and again.welcome_version is None and again.donation_withdrawn_at is None


async def test_delete_leaves_only_a_coded_marker_that_expires(store: Store, tmp_path: Path) -> None:
    assert await store.deleted_at("u1") is None
    await store.save_user(UserState("u1"))
    before = utcnow()
    assert await store.delete_user("u1") == 1  # the marker is not counted as data
    marked = await store.deleted_at("u1")
    assert marked is not None and marked >= before and await store.deleted_at("u2") is None
    await store.delete_user("u1")  # a second DELETE moves the time on
    later = await store.deleted_at("u1")
    assert later is not None and later >= marked
    if isinstance(store, SQLiteStore):  # only a keyed hash of the user hash is stored, never the hash itself
        import sqlite3

        db = sqlite3.connect(tmp_path / "kg.db")
        assert db.execute("SELECT marker FROM deletion_markers").fetchall() == [
            (UserHasher(b"s" * MIN_SECRET_BYTES).deletion_marker("u1"),)
        ]
        db.close()
    assert await store.purge_deletion_markers(utcnow() - timedelta(days=1)) == 0
    assert await store.purge_deletion_markers(utcnow() + timedelta(seconds=1)) == 1
    assert await store.deleted_at("u1") is None


async def test_sample_ids_are_unique(store: Store) -> None:
    await store.add_voice_sample(sample("1" * 32))
    with pytest.raises(ValueError, match="duplicate"):
        await store.add_voice_sample(sample("1" * 32, user="u2"))
    assert [s.user_hash for s in await store.voice_samples()] == ["u1"]


async def test_the_voice_vault_id_is_stable_per_database(store: Store) -> None:
    first = await store.voice_vault_id()
    assert len(first) == 32 and await store.voice_vault_id() == first


async def test_a_single_flagged_sample_can_be_deleted(store: Store) -> None:
    await store.add_voice_sample(sample("1" * 32))
    await store.add_voice_sample(sample("2" * 32))
    assert await store.delete_voice_sample("1" * 32)
    assert not await store.delete_voice_sample("1" * 32)
    assert [s.id for s in await store.voice_samples()] == ["2" * 32]


async def test_voice_retention_purges_whole_weeks_on_or_before_the_cutoff(store: Store) -> None:
    await store.add_voice_sample(sample("1" * 32, day=date(2025, 10, 6)))  # week of Monday 2025-10-06
    await store.add_voice_sample(sample("2" * 32, day=date(2025, 10, 13)))
    assert await store.purge_voice_samples(date(2025, 10, 5)) == []
    assert await store.purge_voice_samples(date(2025, 10, 6)) == ["1" * 32]
    assert [s.id for s in await store.voice_samples()] == ["2" * 32]


async def test_delete_user_also_removes_donated_sample_rows(store: Store) -> None:
    await store.save_user(UserState("u1", donation_version="2026-10-10", donation_at=utcnow()))
    await store.add_voice_sample(sample("1" * 32))
    await store.add_voice_sample(sample("2" * 32, user="u2"))
    assert await store.delete_user("u1") == 2  # the user row and one sample
    assert [s.id for s in await store.voice_samples()] == ["2" * 32]


async def test_a_donors_consent_record_outlives_inactivity_while_samples_exist(store: Store) -> None:
    long_ago = utcnow() - timedelta(days=200)
    for user in ("donor", "idle"):
        await store.save_user(UserState(user, created_at=long_ago, last_seen_at=long_ago))
    await store.add_voice_sample(sample("1" * 32, user="donor"))
    assert await store.purge_before(utcnow() - timedelta(days=90)) == 1  # only the idle non-donor
    assert await store.get_user("donor") is not None and await store.get_user("idle") is None
    await store.delete_voice_samples("donor")
    assert await store.purge_before(utcnow() - timedelta(days=90)) == 1  # now it ages out as usual
