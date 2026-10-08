"""In-process token-bucket rate limiter keyed by API key or client IP (bounded memory, LRU eviction).

Single-instance by design (like the conversation's per-user ordering); a multi-instance deployment would move this to
the shared store or the edge proxy.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        per_minute: int,
        *,
        burst: int | None = None,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be positive")
        self._rate = per_minute / 60.0
        self._capacity = float(burst or per_minute)
        self._max_keys = max_keys
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # key -> (tokens, updated_at)

    def allow(self, key: str) -> bool:
        """Take one token for ``key``; False when the bucket is empty."""
        now = self._clock()
        tokens, updated = self._buckets.pop(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - updated) * self._rate)
        allowed = tokens >= 1.0
        self._buckets[key] = (tokens - 1.0 if allowed else tokens, now)
        if len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)  # least recently seen
        return allowed

    def retry_after(self, key: str) -> int:
        """Whole seconds until ``key`` has a token again (for the Retry-After header)."""
        tokens, updated = self._buckets.get(key, (self._capacity, self._clock()))
        tokens = min(self._capacity, tokens + (self._clock() - updated) * self._rate)
        return max(1, math.ceil((1.0 - tokens) / self._rate))
