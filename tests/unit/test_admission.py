"""The one licence-cap gate shared by WhatsApp, the website and the partner API (``kudiguard.admission``)."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kudiguard.admission import Admission, AtCapacity, NewUsersBusy
from kudiguard.domain import Interaction, MessageKind
from kudiguard.i18n import Msg
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


async def test_no_cap_and_no_gates_admits_everyone_without_a_slot() -> None:
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
    with pytest.raises(AtCapacity, match="licence cap"):
        await gate.reserve("new-2", "web")  # 1 active + 1 in flight = cap 2
    first.release()
    first.release()  # idempotent
    assert gate.in_flight == 0
    async with gate.admitted("new-2", "web") as second:
        assert second.holds_slot and gate.in_flight == 1
    assert gate.in_flight == 0


async def test_concurrent_requests_of_one_new_user_each_hold_the_slot() -> None:
    """The twin attack: a first request that ends without a log (a 422) must not free the slot its twin is using."""
    store = MemoryStore(HASHER)
    gate = Admission(store, 1)
    first = await gate.reserve("twin", "web")
    second = await gate.reserve("twin", "web")  # concurrent: counted, not free
    assert first.holds_slot and second.holds_slot and gate.in_flight == 1
    first.release()  # the slow, empty request 422s
    with pytest.raises(AtCapacity):
        await gate.reserve("other", "web")  # the twin is still being served
    second.release()
    assert gate.in_flight == 0
    (await gate.reserve("other", "web")).release()


async def test_a_failed_log_keeps_the_slot_for_a_window() -> None:
    now = [datetime(2026, 10, 11, 12, tzinfo=utcnow().tzinfo)]
    store = MemoryStore(HASHER)
    gate = Admission(store, 1, clock=lambda: now[0])
    served = await gate.reserve("served", "api")
    served.hold()  # its interaction could not be written: served, but invisible to the store
    served.hold()  # idempotent once released
    assert gate.in_flight == 1
    with pytest.raises(AtCapacity):
        await gate.reserve("next", "api")
    now[0] += timedelta(days=30)
    (await gate.reserve("next", "api")).release()  # the hold expired with the window
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


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_a_delete_keeps_its_slot_only_for_users_who_were_active(kind: str, tmp_path: Path) -> None:
    store = (
        MemoryStore(HASHER) if kind == "memory" else await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    )
    gate = Admission(store, 2)
    since = utcnow() - timedelta(days=30)
    for stranger in range(5):  # numbers that never used KudiGuard: no slot taken (and no way to block others)
        await store.delete_user(f"stranger-{stranger}")
    await store.add_interaction(row("old", days_ago=40))
    await store.delete_user("old")  # active only before the window
    assert await store.count_deletions_since(since) == 0
    await store.add_interaction(row("leaver"))
    await store.delete_user("leaver")
    assert await store.count_active_users_since(since) == 0 and await store.count_deletions_since(since) == 1
    assert await store.deleted_at("stranger-0") is not None  # the sign-up import marker is unchanged
    (await gate.reserve("newcomer", "whatsapp")).release()  # 1 erased-but-active + 0 in flight < 2
    await store.add_interaction(row("active"))
    with pytest.raises(AtCapacity):
        await gate.reserve("another", "whatsapp")  # 1 active + 1 erased = cap
    if kind == "sqlite":
        await store.close()  # type: ignore[attr-defined]


async def test_pure_checks_and_on_admit_hooks_for_a_new_user_gate() -> None:
    """Where a new-user rate gate (workstream F) plugs in: a pure check, and a hook that takes its token only once
    the cap has passed, so a user refused at the cap never uses one."""
    store = MemoryStore(HASHER)
    await store.add_interaction(row("known"))
    checked: list[str] = []
    taken: list[tuple[str, str]] = []

    async def allows(user_hash: str, channel: str) -> None:
        checked.append(user_hash)
        if user_hash == "blocked":
            raise NewUsersBusy("new-user rate")

    async def take_token(user_hash: str, channel: str) -> None:
        taken.append((user_hash, channel))

    gate = Admission(store, 2)
    gate.add_check(allows)
    gate.add_on_admit(take_token)
    assert not (await gate.reserve("known", "web")).holds_slot
    assert (await gate.reserve("fresh", "api")).holds_slot  # 1 known + 1 in flight: the cap is now full
    with pytest.raises(NewUsersBusy):
        await gate.reserve("blocked", "whatsapp")
    with pytest.raises(AtCapacity):
        await gate.reserve("late", "whatsapp")  # refused at the cap: no token taken
    assert checked == ["fresh", "blocked", "late"] and taken == [("fresh", "api")]
    hooks_only = Admission(store, None, on_admit=(take_token,))
    assert (await hooks_only.reserve("solo", "web")).holds_slot and taken[-1] == ("solo", "web")


def test_refusals_say_what_to_tell_the_user() -> None:
    assert (AtCapacity.message, AtCapacity.problem, AtCapacity.retry_after_s) == (
        Msg.AT_CAPACITY,
        "at_capacity",
        None,
    )
    busy = NewUsersBusy("x")
    assert (busy.message, busy.problem, busy.retry_after_s) == (Msg.NEW_USERS_BUSY, "busy_new_users", 3600)


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

    admitted = await asyncio.gather(
        *(arrive(n) for n in range(30))
    )  # all arrive before any logs: deterministic
    since = utcnow() - timedelta(days=30)
    assert sum(admitted) == 5 and await store.count_active_users_since(since) == 5
    assert await store.count_active_users_since(since, channels={"web"}) == 2
    await store.close()


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_licence_cap_markers_are_purged_at_the_window_start_not_earlier(
    kind: str, tmp_path: Path
) -> None:
    """The review's capmarker probe: a retention pass must not free the slot of a user erased 29 days 20 hours after
    their last message (the DELETE markers for the sign-up import go ~7 h early; the cap markers must not)."""
    from kudiguard.api.services import PURGE_INTERVAL_S, _purge_once

    store = (
        MemoryStore(HASHER) if kind == "memory" else await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    )
    now = utcnow()
    for n, hours in enumerate((30 * 24 - 4, 10)):  # active 29 d 20 h ago (then erased), and 10 h ago
        await store.add_interaction(row(f"u{n}", days_ago=hours / 24))
    await store.delete_user("u0")  # its cap marker is dated with the last message: 29 d 20 h ago
    await store.add_interaction(row("u2"))
    since = now - timedelta(days=30)
    await _purge_once(store, now, 90, 730, PURGE_INTERVAL_S, None, 365)
    assert await store.count_active_users_since(since) + await store.count_deletions_since(since) == 3
    with pytest.raises(AtCapacity):
        await Admission(store, 3).reserve("newcomer", "whatsapp")
    assert (
        await store.purge_cap_markers(now - timedelta(days=29)) == 1
    )  # due once its message leaves the window
    assert await store.deleted_at("u0") is not None  # the import marker is separate
    if kind == "sqlite":
        await store.close()  # type: ignore[attr-defined]
