"""N-ATLaS licence cap: one admission point for every channel (WhatsApp, website, partner API).

The licence allows ``cap`` active end-users per rolling 30 days, project-wide. A user active in the window (an
interaction logged since then) is always served. A user new to the window is admitted only while

    active users in the store + DELETEs in the window + new users whose first turn is still running  <  cap

and, for a channel with its own share (the anonymous website), the same inside that share. The check and the
reservation happen under one lock, and the reservation is held until the user's interaction is logged (then the store
counts them) or the turn ends without a log, so a burst across channels can never overshoot. A DELETE in the window
keeps its slot (the coded deletion marker is counted): over-counting a returning user is the safe direction.

``checks`` are further gates for new users only (e.g. a new-user rate limit), run under the same lock before the
slot is reserved; a gate refuses by raising (``AtCapacity`` or its own subclass).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from kudiguard.storage.base import ACTIVE_USER_WINDOW_DAYS, Store, utcnow

NewUserCheck = Callable[[str, str], Awaitable[None]]  # (user_hash, channel); raises to refuse


class AtCapacity(Exception):
    """A new user cannot be admitted now (licence cap, a channel's share, or a new-user gate)."""


class Reservation:
    """A new user's slot while their first turn runs; ``release`` once the interaction is logged (idempotent).
    Returning users get an empty reservation."""

    def __init__(self, admission: Admission | None = None, user_hash: str = "") -> None:
        self._admission = admission
        self._user_hash = user_hash

    @property
    def holds_slot(self) -> bool:
        return self._admission is not None

    def release(self) -> None:
        if self._admission is not None:
            self._admission.release_slot(self._user_hash)
            self._admission = None


class Admission:
    def __init__(
        self,
        store: Store,
        cap: int | None,
        *,
        shares: Mapping[str, int] | None = None,
        checks: tuple[NewUserCheck, ...] = (),
        clock: Callable[[], datetime] = utcnow,
        window_days: int = ACTIVE_USER_WINDOW_DAYS,
    ) -> None:
        self.store = store
        self.cap = cap
        self.shares = dict(shares or {})
        self.checks = list(checks)
        self._clock = clock
        self._window = timedelta(days=window_days)
        self._lock = asyncio.Lock()
        self._pending: dict[str, str] = {}  # user_hash -> channel, new users whose first turn is running

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    def release_slot(self, user_hash: str) -> None:
        self._pending.pop(user_hash, None)

    def add_check(self, check: NewUserCheck) -> None:
        """Plug in another gate for new users (runs after the returning-user test, before the cap)."""
        self.checks.append(check)

    async def reserve(self, user_hash: str, channel: str) -> Reservation:
        """Admit ``user_hash`` arriving on ``channel`` or raise ``AtCapacity``."""
        if self.cap is None and not self.checks:
            return Reservation()
        async with self._lock:  # check + reserve atomically (store counts yield to other turns)
            if user_hash in self._pending:
                return Reservation()  # its first turn holds the slot
            since = self._clock() - self._window
            if await self.store.count_interactions_since(user_hash, since):
                return Reservation()
            for check in self.checks:
                await check(user_hash, channel)
            if self.cap is not None:
                await self._check_cap(channel, since)
            self._pending[user_hash] = channel
            return Reservation(self, user_hash)

    async def _check_cap(self, channel: str, since: datetime) -> None:
        assert self.cap is not None
        # Snapshot the in-flight users BEFORE reading the store: a turn logs its interaction and only then releases
        # its slot (without the lock), so every user is then seen at least once (twice at worst: the safe side).
        in_flight = len(self._pending)
        in_channel = sum(c == channel for c in self._pending.values())
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

    @asynccontextmanager
    async def admitted(self, user_hash: str, channel: str) -> AsyncIterator[Reservation]:
        """``reserve`` for the duration of a block (released at its end, or earlier by the caller)."""
        reservation = await self.reserve(user_hash, channel)
        try:
            yield reservation
        finally:
            reservation.release()
