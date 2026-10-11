"""KnowledgeBase: queries every in-scope source concurrently and merges the answers.

Merge policy (safety first, then relevance):
* red-flag entries: core pack first, then partners by priority; round-robin so each source is represented
* search hits: by relevance across all sources (cosine similarity when embeddings are on), source priority breaks
  ties; weak partner hits are dropped so a partner can never crowd out a better core answer
* contacts: pinned first, then flag-relevant, then general (each by source priority); de-duplicated
* a character budget keeps the result within the LLM's context window
A failing or slow source is skipped and reported; it never blocks or breaks the reply.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, zip_longest
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray

from kudiguard.knowledge.base import ContentScreen, KnowledgeSource, SourceUnavailable
from kudiguard.knowledge.embeddings import Embedder
from kudiguard.knowledge.file_pack import FilePackSource, KnowledgeConfigError, load_manifest
from kudiguard.knowledge.http_source import HttpSource
from kudiguard.knowledge.index import DENSE_MIN
from kudiguard.knowledge.models import (
    Contact,
    Entry,
    HttpSourceSpec,
    KnowledgeResult,
    PackKind,
    Query,
    Relevance,
    SourcedEntry,
    SourceResult,
)
from kudiguard.text import normalize

log = logging.getLogger(__name__)

HTTP_GRACE_S = 0.25  # let an HTTP source's own timeout fire (and count toward its breaker) before ours
PARTNER_MIN_BM25_RELEVANCE = 0.3  # squashed BM25 floor for partner hits when embeddings are off
QUERY_CACHE_SIZE = 256
MIN_FACT_CHARS = 120  # a trimmed safety entry always keeps at least this much of its first fact
ELLIPSIS = "\u2026"
PARTNER_MARGIN = 0.03  # a partner hit must beat the core hit by this much to be shown first


@dataclass(frozen=True, slots=True)
class Limits:
    flag_entries: int = 2
    hits: int = 3
    contacts: int = 4
    chars: int = 3000  # title + facts + actions across entries; ~750 tokens of a 4K context


DEFAULT_LIMITS = Limits()


def _contact_key(contact: Contact) -> str:
    return "".join(contact.value.lower().split())


def _entry_chars(item: SourcedEntry) -> int:
    return sum(map(len, (item.entry.title, *item.entry.facts, *item.entry.actions)))


def _size(title: str, facts: list[str], actions: list[str]) -> int:
    return len(title) + sum(map(len, facts)) + sum(map(len, actions))


def _trim(item: SourcedEntry, budget: int) -> SourcedEntry:
    """Drop trailing actions, then facts, until the entry fits; the first fact is kept (truncated if needed)."""
    entry = item.entry
    facts, actions = list(entry.facts), list(entry.actions)
    while actions and _size(entry.title, facts, actions) > budget:
        actions.pop()
    while len(facts) > 1 and _size(entry.title, facts, actions) > budget:
        facts.pop()
    room = max(budget - len(entry.title), MIN_FACT_CHARS)
    if len(facts[0]) > room:
        facts[0] = facts[0][: room - 1].rstrip() + ELLIPSIS
    if facts == list(entry.facts) and actions == list(entry.actions):
        return item
    trimmed = entry.model_copy(update={"facts": tuple(facts), "actions": tuple(actions)})
    return dataclasses.replace(item, entry=trimmed)


_KEYWORD_SPLIT = re.compile(r"[^a-z0-9]+")


def _cued(entry: Entry, text: str) -> bool:
    """One of the entry's keywords appears in the query (``text``: normalised words, space-padded)."""
    return any(
        f" {phrase} " in text
        for keyword in entry.keywords
        if (phrase := " ".join(_KEYWORD_SPLIT.split(normalize(keyword))).strip())
    )


def _sourced(source: KnowledgeSource, entry: Entry, relevance: float | None = None) -> SourcedEntry:
    m = source.manifest
    return SourcedEntry(entry, m.id, m.name, m.sample, relevance, m.kind, m.cite)


class KnowledgeBase:
    def __init__(
        self,
        sources: Sequence[KnowledgeSource],
        *,
        embedder: Embedder | None = None,
        deadline_s: float = 1.5,
        limits: Limits = DEFAULT_LIMITS,
        min_relevance: Relevance | None = None,
    ) -> None:
        """``min_relevance``: search hits below it are dropped (None = keep every hit the index returns). Red-flag
        entries are never dropped: they match the message's flags, not its wording."""
        ids = [s.manifest.id for s in sources]
        if len(ids) != len(set(ids)):
            raise KnowledgeConfigError(f"duplicate knowledge source ids in {ids}")
        if not any(s.manifest.kind is PackKind.CORE for s in sources):
            raise KnowledgeConfigError("a core knowledge pack is required (safety guidance)")
        self.sources = tuple(sources)
        self._embedder = embedder
        self._deadline_s = deadline_s
        self._limits = limits
        self._min_relevance = min_relevance
        self._partner_floor = DENSE_MIN if embedder is not None else PARTNER_MIN_BM25_RELEVANCE
        self._query_cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._safety_order = sorted(
            sources, key=lambda s: (s.manifest.kind is not PackKind.CORE, -s.manifest.priority, s.manifest.id)
        )
        by_priority = sorted(sources, key=lambda s: (-s.manifest.priority, s.manifest.id))
        self._rank_of = {s.manifest.id: r for r, s in enumerate(by_priority)}
        self._core_ids = frozenset(s.manifest.id for s in sources if s.manifest.kind is PackKind.CORE)

    async def refresh_all(self) -> dict[str, str]:
        """Refresh every source; returns ``{source_id: error}`` for the ones that failed."""
        outcomes = await asyncio.gather(*(s.refresh() for s in self.sources), return_exceptions=True)
        return {
            s.manifest.id: str(o)
            for s, o in zip(self.sources, outcomes, strict=True)
            if isinstance(o, BaseException)
        }

    async def search(self, query: Query) -> KnowledgeResult:
        active = [
            s
            for s in self.sources
            if s.manifest.kind is PackKind.CORE or query.packs is None or s.manifest.id in query.packs
        ]
        if self._embedder is not None and query.vector is None and query.text.strip():
            query = dataclasses.replace(query, vector=await self._embed(query.text))  # None -> BM25 only

        answers = await asyncio.gather(*(self._ask(s, query) for s in active))
        results = {s.manifest.id: a for s, a in zip(active, answers, strict=True) if a is not None}
        answered = [s for s in self._safety_order if s.manifest.id in results]
        dense = query.vector is not None
        hits = self._best_hits(answered, results, query)
        if (floor := self._min_relevance) is not None:
            hits = [h for h in hits if floor.met(h.relevance, dense=dense)]

        return KnowledgeResult(
            entries=self._within_budget(self._flag_entries(answered, results), hits),
            contacts=self._contacts(answered, results, query),
            failed_sources=tuple(s.manifest.id for s in active if s.manifest.id not in results),
            dense=dense,
        )

    # --- merging --------------------------------------------------------------------------------------

    def _flag_entries(
        self, answered: Sequence[KnowledgeSource], results: Mapping[str, SourceResult]
    ) -> list[SourcedEntry]:
        per_source = [[_sourced(s, e) for e in results[s.manifest.id].flag_entries] for s in answered]
        round_robin = [item for group in zip_longest(*per_source) for item in group if item is not None]
        return round_robin[: self._limits.flag_entries]

    def _best_hits(
        self, answered: Sequence[KnowledgeSource], results: Mapping[str, SourceResult], query: Query
    ) -> list[SourcedEntry]:
        # Each source's own order (BM25 + dense fusion) is kept; relevance only decides which source's next hit
        # comes first, so fusion quality is never thrown away by re-sorting on raw cosine.
        queues: list[list[SourcedEntry]] = []
        weight: dict[str, float] = {}  # entry id -> uncued weight (situational packs only)
        text = f" {' '.join(_KEYWORD_SPLIT.split(normalize(query.text)))} "
        for source in sorted(answered, key=lambda s: self._rank_of[s.manifest.id]):
            is_core = source.manifest.kind is PackKind.CORE
            queues.append(
                [
                    _sourced(source, hit.entry, hit.relevance)
                    for hit in results[source.manifest.id].hits
                    if is_core or (hit.relevance is not None and hit.relevance >= self._partner_floor)
                ]
            )
            if (w := source.manifest.uncued_weight) < 1 and not query.flags:
                for hit in results[source.manifest.id].hits:
                    if not _cued(hit.entry, text):
                        weight[hit.entry.id] = w

        def head_score(queue: list[SourcedEntry]) -> float:
            head = queue[0]
            bonus = PARTNER_MARGIN if head.source_id in self._core_ids else 0.0
            relevance = (head.relevance if head.relevance is not None else 0.0) * weight.get(
                head.entry.id, 1.0
            )
            return relevance + bonus

        merged: list[SourcedEntry] = []
        while len(merged) < self._limits.hits and (live := [q for q in queues if q]):
            merged.append(max(live, key=head_score).pop(0))
        return merged

    def _within_budget(
        self, flagged: list[SourcedEntry], hits: list[SourcedEntry]
    ) -> tuple[SourcedEntry, ...]:
        """Safety (red-flag) entries are trimmed to fit, never dropped for a smaller hit; hits are skipped."""
        kept: list[SourcedEntry] = []
        remaining = self._limits.chars
        for item in flagged:
            item = _trim(item, remaining)
            kept.append(item)
            remaining -= _entry_chars(item)
        for item in hits:
            if (size := _entry_chars(item)) <= remaining:
                kept.append(item)
                remaining -= size
        return tuple(kept)

    def _contacts(
        self, answered: Sequence[KnowledgeSource], results: Mapping[str, SourceResult], query: Query
    ) -> tuple[Contact, ...]:
        by_priority = sorted(answered, key=lambda s: self._rank_of[s.manifest.id])
        everything = [
            (c, c.pinned and s.manifest.id in self._core_ids)  # only the core pack may pin contacts
            for s in by_priority
            for c in results[s.manifest.id].contacts
        ]
        wanted = set(query.flags)
        pinned = (c for c, pin in everything if pin)
        relevant = (c for c, pin in everything if not pin and wanted.intersection(c.flags))
        general = (c for c, pin in everything if not pin and not c.flags)
        unique: dict[str, Contact] = {}
        for contact in chain(pinned, relevant, general):
            unique.setdefault(_contact_key(contact), contact)
        return tuple(unique.values())[: self._limits.contacts]

    # --- I/O ------------------------------------------------------------------------------------------

    async def _embed(self, text: str) -> NDArray[np.float32] | None:
        """Embed once per request, off the event loop; repeated questions hit a small LRU cache.

        Any embedding failure degrades search to BM25 instead of failing the reply.
        """
        if (cached := self._query_cache.get(text)) is not None:
            self._query_cache.move_to_end(text)
            return cached
        assert self._embedder is not None
        try:
            vector = await asyncio.to_thread(self._embedder.embed_query, text)
        except Exception:
            log.exception("query embedding failed; falling back to keyword search")
            return None
        self._query_cache[text] = vector
        if len(self._query_cache) > QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        return vector

    async def _ask(self, source: KnowledgeSource, query: Query) -> SourceResult | None:
        spec = source.manifest.source
        timeout = (
            min(self._deadline_s, spec.timeout_s) + HTTP_GRACE_S
            if isinstance(spec, HttpSourceSpec)
            else self._deadline_s
        )
        try:
            return await asyncio.wait_for(source.search(query), timeout=timeout)
        except (SourceUnavailable, TimeoutError) as exc:
            log.warning("knowledge source %s skipped: %r", source.manifest.id, exc)
        except Exception:
            log.exception("knowledge source %s failed unexpectedly", source.manifest.id)
        return None


def knowledge_base_from_packs(
    packs_dir: Path,
    *,
    client: httpx.AsyncClient,
    include: Iterable[str] | None = None,
    embedder: Embedder | None = None,
    allow_samples: bool = False,
    allow_live: bool = False,
    deadline_s: float = 1.5,
    limits: Limits = DEFAULT_LIMITS,
    screen: ContentScreen | None = None,
    min_relevance: Relevance | None = None,
) -> KnowledgeBase:
    """The supported way to wire a KnowledgeBase: one embedder shared by every index and the query path."""
    sources = build_sources(
        packs_dir, client=client, include=include, embedder=embedder, allow_samples=allow_samples,
        allow_live=allow_live, screen=screen,
    )  # fmt: skip
    return KnowledgeBase(
        sources, embedder=embedder, deadline_s=deadline_s, limits=limits, min_relevance=min_relevance
    )


def build_sources(
    packs_dir: Path,
    *,
    client: httpx.AsyncClient,
    include: Iterable[str] | None = None,
    embedder: Embedder | None = None,
    allow_samples: bool = False,
    allow_live: bool = False,
    env: Mapping[str, str] = os.environ,
    clock: Callable[[], float] = time.monotonic,
    screen: ContentScreen | None = None,
) -> list[KnowledgeSource]:
    """Instantiate sources from ``packs_dir``: every enabled pack, or exactly the ``include`` list.

    Fictional sample packs (``sample: true``) are refused unless ``allow_samples`` (demos and tests only). Partner
    packs in ``mode: live`` send the English text of each user message to the partner, so they are refused unless
    ``allow_live`` (``KUDIGUARD_ALLOW_LIVE_PACKS``, which also discloses it on /privacy).
    """
    available = {d.name: d for d in sorted(packs_dir.iterdir()) if (d / "pack.yaml").is_file()}
    manifests = {name: load_manifest(d) for name, d in available.items()}
    for name, manifest in manifests.items():
        if manifest.id != name:
            raise KnowledgeConfigError(f"{available[name]}: folder name must match pack id {manifest.id!r}")

    if include is None:
        selected = [name for name, m in manifests.items() if m.enabled]
    else:
        selected = list(include)
        if unknown := [name for name in selected if name not in manifests]:
            raise KnowledgeConfigError(f"unknown knowledge packs {unknown} in {packs_dir}")
    if not allow_samples and (samples := [n for n in selected if manifests[n].sample]):
        raise KnowledgeConfigError(f"sample packs {samples} are fictional; pass allow_samples=True for demos")
    live = [n for n in selected if isinstance(s := manifests[n].source, HttpSourceSpec) and s.mode == "live"]
    if live and not allow_live:
        raise KnowledgeConfigError(
            f"packs {live} use mode: live, which sends users' messages (in English) to the partner; use mode: sync, "
            "or set KUDIGUARD_ALLOW_LIVE_PACKS=true (disclosed on /privacy)"
        )

    sources: list[KnowledgeSource] = []
    for name in selected:
        manifest = manifests[name]
        if isinstance(manifest.source, HttpSourceSpec):
            sources.append(
                HttpSource(manifest, client, env=env, clock=clock, embedder=embedder, screen=screen)
            )
        else:
            sources.append(FilePackSource.load(available[name], embedder=embedder, screen=screen))
    return sources
