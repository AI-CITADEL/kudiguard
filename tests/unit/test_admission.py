"""The one licence-cap gate shared by WhatsApp, the website and the partner API (``kudiguard.admission``)."""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from kudiguard.admission import Admission, AtCapacity
from kudiguard.domain import Interaction, MessageKind
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.base import utcnow
from kudiguard.storage.memory import MemoryStore
from kudiguard.storage.sqlite import SQLiteStore

HASHER = UserHasher(b"a" * MIN_SECRET_BYTES)


def row(user_hash: str, channel: str = "whatsapp", days_ago: float = 0) -> Interaction:
    return Interaction(
        channel=channel, message_id=f"m-{user_hash}-{days_ago}", user_hash=user_hash, language="en",
        kind=MessageKind.TEXT, created_at=utcnow() - timedelta(days=days_ago),
    )  # fmt: skip


async def test_no_cap_and_no_checks_admits_everyone_without_a_slot() -> None:
    gate = Admission(MemoryStore(HASHER), None)
    reservation = await gate.reserve("u1", "web")
    assert not reservation.holds_slot and gate.in_flight == 0


async def test_new_users_hold_a_slot_until_released_and_returning_users_never_need_one() -> None:
    store = MemoryStore(HASHER)
    gate = Admission(store, 2)
    await store.add_interaction(row("active"))
    assert not (await gate.reserve("active", "whatsapp")).holds_slot  # active in the window: always served
    first = await gate.reserve("new-1", "api")
    assert first.holds_slot and gate.in_flight == 1
    assert not (await gate.reserve("new-1", "api")).holds_slot  # its first turn already holds the slot
    with pytest.raises(AtCapacity, match="licence cap"):
        await gate.reserve("new-2", "web")  # 1 active + 1 in flight = cap 2
    first.release()
    first.release()  # idempotent
    assert gate.in_flight == 0
    async with gate.admitted("new-2", "web") as second:
        assert second.holds_slot and gate.in_flight == 1
    assert gate.in_flight == 0


async def test_users_outside_the_window_count_as_new() -> None:
    store = MemoryStore(HASHER)
    gate = Admission(store, 1)
    await store.add_interaction(row("dormant", days_ago=31))
    async with gate.admitted("dormant", "whatsapp") as reservation:
        assert reservation.holds_slot


async def test_a_channel_share_limits_only_that_channel() -> None:
    store = MemoryStore(HASHER)
    gate = Admission(store, 10, shares={"web": 1})
    await store.add_interaction(row("w1", channel="web"))
    with pytest.raises(AtCapacity, match="web share"):
        await gate.reserve("w2", "web")
    held = await gate.reserve("a1", "api")
    assert held.holds_slot
    held.release()


async def test_a_delete_keeps_its_slot_for_the_rest_of_the_window() -> None:
    store = MemoryStore(HASHER)
    gate = Admission(store, 1)
    await store.add_interaction(row("leaver"))
    await store.delete_user("leaver")
    assert await store.count_active_users_since(utcnow() - timedelta(days=30)) == 0
    with pytest.raises(AtCapacity):
        await gate.reserve("newcomer", "whatsapp")


async def test_new_user_checks_plug_in_and_run_only_for_new_users() -> None:
    """Where a new-user rate gate (workstream F) plugs in: under the same lock, before the cap."""
    store = MemoryStore(HASHER)
    await store.add_interaction(row("known"))
    seen: list[tuple[str, str]] = []

    class TooManyNewUsers(AtCapacity):
        pass

    async def gate_check(user_hash: str, channel: str) -> None:
        seen.append((user_hash, channel))
        if user_hash == "blocked":
            raise TooManyNewUsers("new-user rate")

    gate = Admission(store, None)
    gate.add_check(gate_check)
    assert not (await gate.reserve("known", "web")).holds_slot
    assert (await gate.reserve("fresh", "api")).holds_slot
    with pytest.raises(TooManyNewUsers):
        await gate.reserve("blocked", "whatsapp")
    assert seen == [("fresh", "api"), ("blocked", "whatsapp")] and gate.in_flight == 1


async def test_concurrent_new_users_on_sqlite_never_overshoot(tmp_path: Path) -> None:
    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    gate = Admission(store, 5, shares={"web": 2})

    async def arrive(n: int) -> bool:
        channel = ("whatsapp", "web", "api")[n % 3]
        try:
            async with gate.admitted(f"user-{n}", channel):
                await asyncio.sleep(0.01)  # the first turn runs, then logs
                await store.add_interaction(row(f"user-{n}", channel=channel))
        except AtCapacity:
            return False
        return True

    admitted = await asyncio.gather(*(arrive(n) for n in range(30)))
    since = utcnow() - timedelta(days=30)
    # never more than the cap; a slot seen twice for an instant (logged, not yet released) may refuse one more
    assert 3 <= sum(admitted) <= 5 and await store.count_active_users_since(since) == sum(admitted)
    assert await store.count_active_users_since(since, channels={"web"}) <= 2
    assert await store.count_deletions_since(since) == 0
    await store.delete_user("user-0")
    assert await store.count_deletions_since(since) == 1
    assert await store.count_deletions_since(utcnow() + timedelta(seconds=1)) == 0
    await store.close()
