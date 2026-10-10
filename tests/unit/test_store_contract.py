"""One behavioural contract, run against every Store implementation (SQLite for production, memory for tests)."""

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from kudiguard.domain import Flag, Interaction, MessageKind, Outcome, Verdict
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.base import Store, UserState, utcnow
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
    with pytest.raises(StoreClosedError):
        await store.interaction_ids()


async def test_interaction_ids_follow_erasure(store: Store) -> None:
    assert await store.interaction_ids() == frozenset()
    gone = await store.add_interaction(turn())
    kept = await store.add_interaction(turn(user="u2", message_id="keep"))
    assert await store.interaction_ids() == {gone, kept}
    await store.delete_user("u1")
    assert await store.interaction_ids() == {kept}
    assert {r["id"] for r in await store.export_interactions()} == await store.interaction_ids()


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
