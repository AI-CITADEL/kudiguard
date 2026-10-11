"""In-process abuse limits (docs/ABUSE.md, threats 3 and 5). One process per store (docs/ARCHITECTURE.md), so these
counters are per process and reset on restart; each is bounded in memory.

* ``NewUserGate``: a cap on users new to the licence window per hour and per day, so throwaway numbers or minted web
  sessions cannot burn the N-ATLAS 1,000-user cap in one go. A user is counted once (repeat messages while
  onboarding are not new again); returning users are never asked.
* ``FloodGuard``: per-user queue cap (a flooder cannot pile up unbounded work or memory), at most one "rate
  limited" notice per interval (a flood does not become a flood of our replies), and an hourly voice-note budget
  (speech recognition is the scarcest CPU resource and is shared with every user).
* ``LearningGate``: what may enter the scam pattern library and the voice donations: per-user daily cap,
  near-duplicate cap per week, nothing from a message the abuse screen flagged, no hallucinated transcripts.

Every limit has a mode: ``off``, ``shadow`` (counted and logged, never refused) or ``enforce``.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from kudiguard.admission import Admission, NewUsersBusy, NewUsersBusyToday
from kudiguard.patterns import week_of
from kudiguard.safety.abuse import ScreenMode
from kudiguard.text import normalize

log = logging.getLogger(__name__)

HOUR_S = 3600.0
DAY_S = 24 * HOUR_S
MAX_KEYS = 10_000  # per-user windows kept before the idle ones are swept


def _allowed(mode: ScreenMode, over: bool, what: str) -> bool:
    if over:
        log.warning("limit reached: %s (mode=%s)", what, mode)
    return not over or mode is not ScreenMode.ENFORCE


class NewUserGate:
    def __init__(
        self,
        *,
        per_hour: int | None = 100,  # = the 1,000-user licence cap / 10: a demo-day burst still gets in
        per_day: int | None = 300,
        mode: ScreenMode = ScreenMode.ENFORCE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (per_hour is not None and per_hour < 1) or (per_day is not None and per_day < 1):
            raise ValueError("new-user caps must be positive (or None for no cap)")
        self.per_hour, self.per_day, self.mode = per_hour, per_day, mode
        self._clock = clock
        self._seen: OrderedDict[str, float] = OrderedDict()  # user key -> first admitted (oldest first)

    def admit(self, user_key: str) -> bool:
        """``refusal`` then ``reserve``: whether a user new to the licence window may start now (counted if so)."""
        if self.refusal(user_key) is not None:
            return False
        self.reserve(user_key)
        return True

    def allows(self, user_key: str) -> bool:
        return self.refusal(user_key) is None

    def refusal(self, user_key: str) -> str | None:
        """Which limit refuses this user now: "hour", "day" or None (admitted; shadow mode only logs). A pure
        check: meant to run inside a shared check-and-reserve admission step (one lock for the licence cap and this
        gate), followed by ``reserve`` only once every check has passed."""
        if self.mode is ScreenMode.OFF:
            return None
        now = self._clock()
        while self._seen and now - next(iter(self._seen.values())) >= DAY_S:
            self._seen.popitem(last=False)
        if user_key in self._seen:
            return None
        last_hour = sum(1 for t in self._seen.values() if now - t < HOUR_S)
        limit = (
            "day" if self.per_day is not None and len(self._seen) >= self.per_day
            else "hour" if self.per_hour is not None and last_hour >= self.per_hour
            else None
        )  # fmt: skip
        return None if _allowed(self.mode, limit is not None, f"new users per {limit}") else limit

    def attach(self, admission: Admission) -> None:
        """Plug into the shared licence-cap admission (every channel): a pure check before the cap, the count only
        once the cap has passed (a user refused at the cap uses no quota). Applies even without a licence cap: it is
        an anti-flood guard (docs/ABUSE.md)."""
        admission.add_check(self._check)
        admission.add_on_admit(self._on_admit)

    async def _check(self, user_key: str, channel: str) -> None:
        limit = self.refusal(user_key)
        if limit == "hour":
            raise NewUsersBusy("new users per hour")
        if limit == "day":
            raise NewUsersBusyToday("new users per day")

    async def _on_admit(self, user_key: str, channel: str) -> None:
        self.reserve(user_key)

    def reserve(self, user_key: str) -> None:
        """Count this user as admitted (idempotent: onboarding messages are not new users again)."""
        if self.mode is not ScreenMode.OFF:
            self._seen.setdefault(user_key, self._clock())


class _Window:
    """Timestamps per key within ``span_s`` (bounded: keys idle for a whole span are dropped)."""

    def __init__(self, span_s: float, clock: Callable[[], float]) -> None:
        self._span, self._clock = span_s, clock
        self._events: dict[str, deque[float]] = {}

    def count(self, key: str) -> int:
        events = self._events.get(key)
        if events is None:
            return 0
        now = self._clock()
        while events and now - events[0] >= self._span:
            events.popleft()
        if not events:
            del self._events[key]
        return len(events)

    def add(self, key: str) -> None:
        self._events.setdefault(key, deque()).append(self._clock())
        if len(self._events) > MAX_KEYS:  # many keys: drop the idle ones
            for stale in list(self._events):
                self.count(stale)


class FloodGuard:
    def __init__(
        self,
        *,
        max_queued: int = 10,
        notice_interval_s: float = 600.0,
        voice_per_hour: int | None = 30,
        mode: ScreenMode = ScreenMode.ENFORCE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_queued < 1 or notice_interval_s < 0 or (voice_per_hour is not None and voice_per_hour < 1):
            raise ValueError("flood limits must be positive")
        self.max_queued, self.mode = max_queued, mode
        self._notices = _Window(notice_interval_s, clock)
        self._voice = _Window(HOUR_S, clock)
        self._voice_per_hour = voice_per_hour

    def queue_full(self, queued: int) -> bool:
        """A user already has ``queued`` messages waiting: refuse more (enforce)."""
        if self.mode is ScreenMode.OFF:
            return False
        return not _allowed(self.mode, queued >= self.max_queued, "per-user queue")

    def notify_rate_limit(self, user_key: str) -> bool:
        """Whether to send the "rate limited" notice now (once per interval per user in enforce)."""
        if self.mode is ScreenMode.OFF:
            return True
        repeated = self._notices.count(user_key) > 0
        self._notices.add(user_key)
        return _allowed(self.mode, repeated, "rate-limit notices")

    def voice_allowed(self, user_key: str) -> bool:
        """Count one voice note; False when the user's hourly budget is spent (enforce)."""
        if self.mode is ScreenMode.OFF or self._voice_per_hour is None:
            return True
        over = self._voice.count(user_key) >= self._voice_per_hour
        if not over:
            self._voice.add(user_key)
        return _allowed(self.mode, over, "voice notes per hour")


# --- what KudiGuard learns from ------------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")
SHINGLE = 3
SIMILAR = 0.8  # Jaccard similarity of word 3-shingles: near-duplicates ("Ref 1" / "Ref 2", one word changed)
MAX_CLUSTERS = 5000  # per week (bounded memory)


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    words = _DIGITS.sub("#", normalize(text)).split()
    if len(words) < SHINGLE:
        return frozenset({tuple(words)})
    return frozenset(tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1))


def _similar(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> bool:
    return len(a & b) / max(len(a | b), 1) >= SIMILAR


def _crisis(screened: Iterable[str]) -> bool:
    return any(c in ("crisis", "crisis_other") for c in screened)


def _blocking(screened: Iterable[str]) -> tuple[str, ...]:
    """Categories that keep a message out of what KudiGuard learns from. `injection_suspect` is not one: its broad
    phrases match genuine messages too (docs/ABUSE.md)."""
    return tuple(c for c in screened if c != "injection_suspect")


@dataclass(slots=True)
class _Cluster:
    shingles: frozenset[tuple[str, ...]]
    count: int = 0
    users: set[str] = field(default_factory=set)


class LearningGate:
    def __init__(
        self,
        *,
        per_user_per_day: int = 5,
        similar_per_week: int = 20,
        mode: ScreenMode = ScreenMode.ENFORCE,
        today: Callable[[], date] = lambda: datetime.now(UTC).date(),
    ) -> None:
        if per_user_per_day < 1 or similar_per_week < 1:
            raise ValueError("learning limits must be positive")
        self.mode = mode
        self._per_user, self._similar = per_user_per_day, similar_per_week
        self._today = today
        self._day: tuple[date, Counter[str]] = (date.min, Counter())
        self._week: tuple[date, list[_Cluster]] = (date.min, [])
        self.dropped: Counter[str] = Counter()

    def allow_pattern(self, user_key: str, text: str, screened: Iterable[str] = ()) -> bool:
        """May this 🔴/🟠 check become a scam example? Counted only when it is kept. Never a crisis message,
        whatever the mode (sensitive personal data)."""
        if _crisis(screened):
            return False
        if self.mode is ScreenMode.OFF:
            return True
        reason = self._pattern_problem(user_key, text, _blocking(screened))
        return self._decide(reason, "pattern")

    def release(self, user_key: str, text: str) -> None:
        """Undo the count of an allowed pattern that could not be stored (so a failed write uses no quota)."""
        mark = shingles(text)
        clusters = self._week[1]
        cluster = next((c for c in clusters if user_key in c.users and _similar(c.shingles, mark)), None)
        if cluster is None:  # e.g. the week rolled over in between: nothing to undo
            return
        cluster.count -= 1
        cluster.users.discard(user_key)
        if cluster.count <= 0:
            clusters.remove(cluster)  # its slot is free again
        if self._day[1][user_key] > 0:  # the day may have rolled over in between
            self._day[1][user_key] -= 1

    def allow_donation(self, screened: Iterable[str] = (), *, filtered: bool = False) -> bool:
        """May this voice note be donated? Not when the message was flagged or its transcript was hallucinated.
        Never a crisis voice note, whatever the mode."""
        if _crisis(screened):
            self.dropped["donation:crisis"] += 1
            return False
        if self.mode is ScreenMode.OFF:
            return True
        screened = _blocking(screened)
        reason = f"screened:{screened[0]}" if screened else "hallucinated" if filtered else None
        return self._decide(reason, "donation")

    def _decide(self, reason: str | None, what: str) -> bool:
        if reason is None:
            return True
        self.dropped[f"{what}:{reason}"] += 1
        log.info("learning_gate mode=%s %s not kept: %s", self.mode, what, reason)
        return self.mode is ScreenMode.SHADOW

    def _pattern_problem(self, user_key: str, text: str, screened: tuple[str, ...]) -> str | None:
        if screened:
            return f"screened:{screened[0]}"
        today = self._today()
        if self._day[0] != today:
            self._day = (today, Counter())
        week = week_of(today)
        if self._week[0] != week:
            self._week = (week, [])
        if self._day[1][user_key] >= self._per_user:
            return "per_user_cap"
        mark = shingles(text)
        clusters = self._week[1]
        cluster = next((c for c in clusters if _similar(c.shingles, mark)), None)
        if cluster is not None and user_key in cluster.users:
            return "duplicate_from_user"
        if cluster is not None and cluster.count >= self._similar:
            return "similar_cap"
        if cluster is None:
            if len(clusters) >= MAX_CLUSTERS:
                return "week_full"
            cluster = _Cluster(mark)
            clusters.append(cluster)
        cluster.count += 1
        cluster.users.add(user_key)
        self._day[1][user_key] += 1
        return None
