"""N-ATLaS licence cap: one admission point for every channel (WhatsApp, website, partner API).

The licence allows ``cap`` active end-users per rolling 30 days, project-wide. A user with an interaction logged in
the window is "returning" and always served. A user new to the window is admitted only while

    users with an interaction in the window
    + erased users whose last interaction is in the window (licence-cap markers, purged exactly at the window start)
    + new users in flight (or held after a failed log)                                                    <  cap

and, for a channel with its own share (the anonymous website), the same inside that share.

What is guaranteed, and under which assumptions:
* **One server process.** The lock and the in-flight set live in memory; several instances need a shared gate, and
  a restart forgets in-flight and held slots (a held slot is a served user whose log write failed). With
  ``cap=None`` and a gate installed, held slots are not bounded by any cap (they still expire after the window).
* **Every served request holds a reservation until its interaction is written.** Check and reservation happen under
  one lock. Reservations are counted per arrival (a user's concurrent requests each hold one), so the slot stays
  taken until the last of them ends, whatever order they end in. A turn that ends without a log (a 422, a welcome)
  frees it; a turn whose log *write fails* keeps it for a whole window (served but not in the store).
* **Only admitted turns are logged** (callers' duty: ``pipeline`` and ``api.app`` log after ``reserve``), so a row
  always means "admitted"; the evidence log is kept for at least the window (``config``: retention >= 31 days, and the
  purge never cuts inside the window).
* In-flight users are read before the store, so a user who is logged-then-released during the check is counted once
  or twice, never zero times. Double counting is the safe direction and also happens in steadier states: a logged
  user whose twin request is still running, and a held user who later logs successfully (counted twice until the
  hold expires, 30 days). So a burst can be refused early, never admitted past the cap.

Pluggable for new users only, under the same lock: ``checks`` are pure tests (raise ``AtCapacity`` or a subclass such
as ``NewUsersBusy`` to refuse; they must not consume anything), and ``on_admit`` hooks run only once the cap has
passed and the slot is being reserved (e.g. a new-user rate gate taking its token there, so a user refused at the cap
never uses one).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from kudiguard.i18n import Msg
from kudiguard.storage.base import ACTIVE_USER_WINDOW_DAYS, Store, utcnow

NewUserHook = Callable[[str, str], Awaitable[None]]  # (user_hash, channel)


class AtCapacity(Exception):
    """A new user cannot be admitted now. Carries what to tell them (``message``) and the API's ``problem`` code."""

    message: Msg = Msg.AT_CAPACITY
    problem: str = "at_capacity"
    retry_after_s: int | None = None


class NewUsersBusy(AtCapacity):
    """Too many new users at once (a new-user rate gate): try again in about an hour."""

    message = Msg.NEW_USERS_BUSY
    problem = "busy_new_users"
    retry_after_s = 3600


@dataclass
class _Slot:
    channel: str
    holders: int = 0  # reservations not yet released
    held_until: datetime | None = None  # kept after a failed log: served, but the store does not count them


class Reservation:
    """One arrival's hold on a new user's slot; ``release`` once the interaction is logged (idempotent), or ``hold``
    if the log could not be written. Returning users get an empty reservation."""

    def __init__(self, admission: Admission | None = None, user_hash: str = "") -> None:
        self._admission = admission
        self._user_hash = user_hash

    @property
    def holds_slot(self) -> bool:
        return self._admission is not None

    def release(self) -> None:
        if self._admission is not None:
            self._admission._release(self._user_hash)
            self._admission = None

    def hold(self) -> None:
        """The user was served but their interaction is not in the store: keep the slot for a whole window."""
        if self._admission is not None:
            self._admission._hold(self._user_hash)
            self.release()


class Admission:
    def __init__(
        self,
        store: Store,
        cap: int | None,
        *,
        shares: Mapping[str, int] | None = None,
        checks: tuple[NewUserHook, ...] = (),
        on_admit: tuple[NewUserHook, ...] = (),
        clock: Callable[[], datetime] = utcnow,
        window_days: int = ACTIVE_USER_WINDOW_DAYS,
    ) -> None:
        self.store = store
        self.cap = cap
        self.shares = dict(shares or {})
        self.checks = list(checks)
        self.on_admit = list(on_admit)
        self._clock = clock
        self._window = timedelta(days=window_days)
        self._lock = asyncio.Lock()
        self._pending: dict[str, _Slot] = {}  # new users in flight (or held after a failed log)

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    def add_check(self, check: NewUserHook) -> None:
        """A pure test for new users (after the returning-user test, before the cap); raises to refuse."""
        self.checks.append(check)

    def add_on_admit(self, hook: NewUserHook) -> None:
        """Runs when a new user is admitted (the cap has passed), under the lock, before the slot is reserved."""
        self.on_admit.append(hook)

    async def reserve(self, user_hash: str, channel: str) -> Reservation:
        """Admit ``user_hash`` arriving on ``channel`` or raise ``AtCapacity``."""
        if self.cap is None and not self.checks and not self.on_admit:
            return Reservation()
        async with self._lock:  # check + reserve atomically (store counts yield to other turns)
            now = self._clock()
            self._expire(now)
            if (slot := self._pending.get(user_hash)) is not None:
                slot.holders += (
                    1  # a concurrent request of a user in flight: counted, so the slot outlives both
                )
                return Reservation(self, user_hash)
            since = now - self._window
            if await self.store.count_interactions_since(user_hash, since):
                return Reservation()
            for check in self.checks:
                await check(user_hash, channel)
            if self.cap is not None:
                await self._check_cap(channel, since)
            for hook in self.on_admit:
                await hook(user_hash, channel)
            self._pending[user_hash] = _Slot(channel, holders=1)
            return Reservation(self, user_hash)

    async def _check_cap(self, channel: str, since: datetime) -> None:
        assert self.cap is not None
        # Snapshot the in-flight users BEFORE reading the store: a turn logs its interaction and only then releases
        # its slot (without the lock), so every user is then seen at least once (twice at worst: the safe side).
        in_flight = len(self._pending)
        in_channel = sum(slot.channel == channel for slot in self._pending.values())
        share = self.shares.get(channel)
        if (
            share is not None
            and await self.store.count_active_users_since(since, channels={channel}) + in_channel >= share
        ):
            raise AtCapacity(f"{channel} share of the licence cap is full")
        active = await self.store.count_active_users_since(since)
        deleted = await self.store.count_deletions_since(since)
        if active + deleted + in_flight >= self.cap:
            raise AtCapacity("licence cap of active end-users reached")

    def _release(self, user_hash: str) -> None:
        slot = self._pending[user_hash]  # a live reservation's slot is never expired (holders > 0)
        slot.holders -= 1
        if slot.holders <= 0 and slot.held_until is None:
            del self._pending[user_hash]

    def _hold(self, user_hash: str) -> None:
        self._pending[user_hash].held_until = self._clock() + self._window

    def _expire(self, now: datetime) -> None:
        for user_hash, slot in list(self._pending.items()):
            if slot.holders <= 0 and slot.held_until is not None and slot.held_until <= now:
                del self._pending[user_hash]

    @asynccontextmanager
    async def admitted(self, user_hash: str, channel: str) -> AsyncIterator[Reservation]:
        """``reserve`` for the duration of a block (released at its end, or earlier by the caller)."""
        reservation = await self.reserve(user_hash, channel)
        try:
            yield reservation
        finally:
            reservation.release()
