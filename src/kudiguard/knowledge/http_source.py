"""Partner knowledge served over HTTP.

Contract (v1), JSON both ways, payload = ``KnowledgePayload`` ({"entries": [...], "contacts": [...]}):

* sync mode: ``GET  {base_url}/export`` -> full payload, indexed locally and re-synced every ``refresh_s`` in the
  background. Once a copy exists, requests never wait on the partner; the last good copy serves through outages.
* live mode: ``POST {base_url}/search`` with ``{"query", "flags", "language", "limit"}`` -> ranked payload,
  cached for ``cache_ttl_s``. The query is redacted (numbers, emails) before it leaves; no user identifiers are sent.

Every call has a hard timeout, a response-size cap, schema validation and a circuit breaker (after
``FAILURE_THRESHOLD`` consecutive failures the source is skipped for ``COOLDOWN_S``, then one probe is let through).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kudiguard.knowledge.base import SourceUnavailable
from kudiguard.knowledge.embeddings import Embedder
from kudiguard.knowledge.file_pack import KnowledgeConfigError
from kudiguard.knowledge.index import EntryIndex
from kudiguard.knowledge.models import (
    Contact,
    Entry,
    Hit,
    HttpSourceSpec,
    KnowledgePayload,
    PackManifest,
    Query,
    SourceResult,
)
from kudiguard.text import redact

log = logging.getLogger(__name__)

FAILURE_THRESHOLD = 3
COOLDOWN_S = 60.0
RETRY_SYNC_S = 60.0
CACHE_SIZE = 512
LIVE_CANDIDATES = 3  # partner candidates re-scored per requested hit

_CacheKey = tuple[str, tuple[str, ...], str, int]


class _TooLarge(Exception):
    pass


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: list[Any] = Field(default_factory=list, max_length=2000)
    contacts: list[Any] = Field(default_factory=list, max_length=200)


_M = TypeVar("_M", bound=BaseModel)


def _valid(model: type[_M], item: Any) -> _M | None:
    try:
        return model.model_validate(item)
    except ValidationError:
        return None


def _consume_result(task: asyncio.Task[None]) -> None:
    """Mark a background task's exception as retrieved (it is already logged)."""
    if not task.cancelled():
        task.exception()


class HttpSource:
    def __init__(
        self,
        manifest: PackManifest,
        client: httpx.AsyncClient,
        *,
        env: Mapping[str, str] = os.environ,
        clock: Callable[[], float] = time.monotonic,
        embedder: Embedder | None = None,
    ) -> None:
        if not isinstance(manifest.source, HttpSourceSpec):
            raise KnowledgeConfigError(f"{manifest.id}: source type must be 'http'")
        self.manifest = manifest
        self._spec: HttpSourceSpec = manifest.source
        self._client = client
        self._clock = clock
        self._embedder = embedder
        self._headers = {"Accept": "application/json", "User-Agent": "KudiGuard/0.1"}
        if auth := self._spec.auth:
            token = (env.get(auth.env) or "").strip()
            if not token:
                raise KnowledgeConfigError(f"{manifest.id}: environment variable {auth.env} is not set")
            self._headers[auth.header] = f"{auth.scheme} {token}" if auth.scheme else token

        self._index: EntryIndex | None = None
        self._next_sync = 0.0
        self._sync_task: asyncio.Task[None] | None = None
        self._cache: OrderedDict[_CacheKey, tuple[float, SourceResult]] = OrderedDict()
        self._failures = 0
        self._open_until = 0.0
        self._probing = False

    # --- public API -----------------------------------------------------------------------------------

    async def refresh(self) -> None:
        """Sync now (sync mode); concurrent callers share one in-flight sync. Raises ``SourceUnavailable``."""
        if self._spec.mode == "sync":
            await asyncio.shield(self._start_sync())

    async def search(self, query: Query) -> SourceResult:
        return await (self._search_sync(query) if self._spec.mode == "sync" else self._search_live(query))

    # --- sync mode ------------------------------------------------------------------------------------

    def _start_sync(self) -> asyncio.Task[None]:
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.create_task(self._sync(), name=f"kb-sync-{self.manifest.id}")
            self._sync_task.add_done_callback(_consume_result)
        return self._sync_task

    async def _sync(self) -> None:
        try:
            payload = await self._call("GET", "/export")
            # Embedding a large export is CPU-heavy: keep it off the event loop.
            self._index = await asyncio.to_thread(
                EntryIndex, payload.entries, payload.contacts, embedder=self._embedder
            )
        except SourceUnavailable as exc:
            self._next_sync = self._clock() + RETRY_SYNC_S
            log.warning("knowledge source %s: sync failed (%s)", self.manifest.id, exc)
            raise
        self._next_sync = self._clock() + self._spec.refresh_s

    async def _search_sync(self, query: Query) -> SourceResult:
        if self._index is None:
            await asyncio.shield(self._start_sync())  # first use: must wait (bounded by the call timeout)
        elif self._clock() >= self._next_sync:
            self._start_sync()  # stale: refresh in the background, answer from the current copy now
        assert self._index is not None
        return self._index.search(query)

    # --- live mode ------------------------------------------------------------------------------------

    async def _search_live(self, query: Query) -> SourceResult:
        text = redact(query.text)
        key: _CacheKey = (text, tuple(query.flags), query.language, query.limit)
        now = self._clock()
        if (cached := self._cache.get(key)) and now - cached[0] < self._spec.cache_ttl_s:
            self._cache.move_to_end(key)
            return cached[1]

        body = {"query": text, "flags": list(query.flags), "language": query.language, "limit": query.limit}
        payload = await self._call("POST", "/search", body)
        wanted = set(query.flags)
        others = [e for e in payload.entries if not wanted.intersection(e.flags)][
            : query.limit * LIVE_CANDIDATES
        ]
        result = SourceResult(
            flag_entries=tuple(e for e in payload.entries if wanted.intersection(e.flags)),
            # Re-score the partner's candidates locally so they compete on the same scale as every other source.
            hits=await asyncio.to_thread(self._score, others, query) if others else (),
            contacts=tuple(c for c in payload.contacts if wanted.intersection(c.flags))
            + tuple(c for c in payload.contacts if not c.flags),
        )
        if self._spec.cache_ttl_s:
            self._cache[key] = (now, result)
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return result

    def _score(self, entries: list[Entry], query: Query) -> tuple[Hit, ...]:
        return (
            EntryIndex(entries, (), embedder=self._embedder).search(dataclasses.replace(query, flags=())).hits
        )

    # --- transport ------------------------------------------------------------------------------------

    def _admit(self) -> None:
        """Circuit breaker: closed -> open after repeated failures -> half-open lets exactly one probe through."""
        if self._failures < FAILURE_THRESHOL***REMOVED***
            return
        if self._clock() < self._open_until or self._probing:
            raise SourceUnavailable(f"{self.manifest.id}: circuit open")
        self._probing = True

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= FAILURE_THRESHOL***REMOVED***
            self._open_until = self._clock() + COOLDOWN_S

    async def _call(self, method: str, path: str, body: Any = None) -> KnowledgePayload:
        self._admit()
        try:
            async with asyncio.timeout(self._spec.timeout_s):
                raw = await self._fetch(method, path, body)
            payload = self._parse(raw)
        except (httpx.HTTPError, ValidationError, _TooLarge, TimeoutError) as exc:
            self._record_failure()
            raise SourceUnavailable(f"{self.manifest.id}: {type(exc).__name__}: {exc}") from exc
        except asyncio.CancelledError:
            # The caller's deadline expired first: the partner was too slow, which is a failure too.
            self._record_failure()
            raise
        finally:
            self._probing = False
        self._failures = 0
        return payload

    def _parse(self, raw: bytes) -> KnowledgePayload:
        """Validate the envelope strictly, items leniently: one bad record must not discard a partner's knowledge."""
        envelope = _Envelope.model_validate_json(raw)
        entries = [e for e in (_valid(Entry, item) for item in envelope.entries) if e is not None]
        contacts = [c for c in (_valid(Contact, item) for item in envelope.contacts) if c is not None]
        if dropped := len(envelope.entries) - len(entries) + len(envelope.contacts) - len(contacts):
            log.warning("knowledge source %s: dropped %d invalid records", self.manifest.id, dropped)
        return KnowledgePayload(entries=tuple(entries), contacts=tuple(contacts))

    async def _fetch(self, method: str, path: str, body: Any) -> bytes:
        url = self._spec.base_url.rstrip("/") + path
        async with self._client.stream(
            method, url, json=body, headers=self._headers, timeout=self._spec.timeout_s
        ) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self._spec.max_bytes:
                raise _TooLarge(f"declared size {declared} exceeds {self._spec.max_bytes} bytes")
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data += chunk
                if len(data) > self._spec.max_bytes:
                    raise _TooLarge(f"response exceeds {self._spec.max_bytes} bytes")
        return bytes(data)
