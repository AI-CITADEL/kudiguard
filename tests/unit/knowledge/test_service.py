import asyncio
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np
import pytest

from kudiguard.domain import Flag
from kudiguard.knowledge.base import SourceUnavailable
from kudiguard.knowledge.file_pack import FilePackSource, KnowledgeConfigError
from kudiguard.knowledge.http_source import HttpSource
from kudiguard.knowledge.models import (
    Contact,
    ContactKind,
    Entry,
    Hit,
    PackKind,
    PackManifest,
    Query,
    SourceResult,
)
from kudiguard.knowledge.service import KnowledgeBase, Limits, build_sources

PACKS = Path(__file__).resolve().parents[3] / "data" / "packs"


def e(id_: str, facts: str = "fact") -> Entry:
    return Entry(id=id_, title=id_, facts=(facts,))


def phone(number: str, *flags: Flag, pinned: bool = False) -> Contact:
    return Contact(label=f"line {number}", kind=ContactKind.PHONE, value=number, flags=flags, pinned=pinned)


class FakeSource:
    def __init__(
        self,
        id_: str,
        *,
        kind: PackKind = PackKind.PARTNER,
        priority: int = 0,
        result: SourceResult | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
        sample: bool = False,
    ) -> None:
        self.manifest = PackManifest(
            id=id_, name=id_.title(), owner="x", kind=kind, priority=priority, sample=sample
        )
        self._result = result or SourceResult()
        self._error = error
        self._delay = delay
        self.refreshed = False
        self.queries: list[Query] = []

    async def search(self, query: Query) -> SourceResult:
        self.queries.append(query)
        await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return self._result

    async def refresh(self) -> None:
        if self._error:
            raise self._error
        self.refreshed = True


def core() -> FakeSource:
    result = SourceResult(
        flag_entries=(e("core-f1"), e("core-f2")),
        hits=(Hit(e("core-h1"), 0.80), Hit(e("core-h2"), 0.70)),
        contacts=(phone("111", pinned=True), phone("622", Flag.ASKS_CREDENTIALS), phone("999")),
    )
    return FakeSource("core", kind=PackKind.CORE, priority=10, result=result)


def bank() -> FakeSource:
    result = SourceResult(
        flag_entries=(e("bank-f1"),),
        hits=(Hit(e("bank-h1"), 0.90), Hit(e("bank-weak"), 0.50), Hit(e("bank-h2"), 0.75)),
        contacts=(phone("0700", Flag.ASKS_CREDENTIALS), phone("0800"), phone("6 2 2", Flag.ASKS_CREDENTIALS)),
    )
    return FakeSource("bank", priority=50, result=result)


QUERY = Query(text="q", flags=(Flag.ASKS_CREDENTIALS,))


# --- merge policy -------------------------------------------------------------------------------------


async def test_flag_entries_core_first_round_robin() -> None:
    result = await KnowledgeBase([bank(), core()], limits=Limits(flag_entries=2, hits=0)).search(QUERY)
    assert [(s.source_id, s.entry.id) for s in result.entries] == [("core", "core-f1"), ("bank", "bank-f1")]


async def test_hits_interleaved_by_head_relevance_keeping_each_source_order() -> None:
    kb = KnowledgeBase([bank(), core()], limits=Limits(flag_entries=0, hits=4))
    result = await kb.search(QUERY)
    # bank's own order (h1, weak, h2) is preserved; relevance only picks which source goes next
    assert [s.entry.id for s in result.entries] == ["bank-h1", "core-h1", "core-h2", "bank-weak"]
    assert all(s.relevance is not None for s in result.entries)


async def test_partner_floor_with_embeddings_drops_weak_partner_hits() -> None:
    kb = KnowledgeBase([bank(), core()], embedder=CountingEmbedder(), limits=Limits(flag_entries=0, hits=5))
    ids = [s.entry.id for s in (await kb.search(QUERY)).entries]
    assert "bank-weak" not in ids  # 0.50 < cosine floor 0.62
    assert ids[0] == "bank-h1"


async def test_contacts_pinned_then_relevant_then_general_deduplicated() -> None:
    result = await KnowledgeBase([bank(), core()], limits=Limits(contacts=5)).search(QUERY)
    # "6 2 2" and "622" are the same number; the bank's (higher priority) copy wins
    assert [c.value for c in result.contacts] == ["111", "0700", "6 2 2", "0800", "999"]


async def test_character_budget_keeps_first_entry_and_skips_overflow() -> None:
    big = FakeSource(
        "core",
        kind=PackKind.CORE,
        result=SourceResult(
            flag_entries=(e("huge", "x" * 900),),
            hits=(Hit(e("big", "y" * 600), 0.9), Hit(e("small", "z" * 50), 0.8)),
        ),
    )
    result = await KnowledgeBase([big], limits=Limits(chars=1000)).search(QUERY)
    assert [s.entry.id for s in result.entries] == ["huge", "small"]


# --- scope, failures, refresh ------------------------------------------------------------------------


async def test_pack_scope_keeps_core_and_selected_partners() -> None:
    other = FakeSource("other", priority=60, result=SourceResult(hits=(Hit(e("other-h"), 0.99),)))
    result = await KnowledgeBase([core(), bank(), other]).search(Query(text="q", packs=frozenset({"bank"})))
    assert {s.source_id for s in result.entries} == {"core", "bank"}
    assert not other.queries


async def test_failing_and_slow_sources_are_skipped_within_deadline() -> None:
    broken = FakeSource("broken", error=SourceUnavailable("down"))
    buggy = FakeSource("buggy", error=KeyError("bug"))
    slow = FakeSource("slow", delay=5)
    kb = KnowledgeBase([core(), broken, buggy, slow], deadline_s=0.05)
    started = asyncio.get_running_loop().time()
    result = await kb.search(QUERY)
    assert asyncio.get_running_loop().time() - started < 1
    assert set(result.failed_sources) == {"broken", "buggy", "slow"}
    assert result.entries[0].entry.id == "core-f1"


async def test_sample_flag_is_carried_for_attribution() -> None:
    demo = FakeSource("demo", sample=True, result=SourceResult(hits=(Hit(e("demo-h"), 0.99),)))
    result = await KnowledgeBase([core(), demo]).search(Query(text="q"))
    assert next(s for s in result.entries if s.source_id == "demo").sample is True


async def test_refresh_all_reports_errors_without_raising() -> None:
    ok, bad = FakeSource("ok"), FakeSource("bad", error=SourceUnavailable("down"))
    errors = await KnowledgeBase([core(), ok, bad]).refresh_all()
    assert ok.refreshed
    assert errors == {"bad": "down"}


def test_core_source_is_mandatory() -> None:
    with pytest.raises(KnowledgeConfigError, match="core"):
        KnowledgeBase([bank()])


def test_duplicate_source_ids_rejected() -> None:
    with pytest.raises(KnowledgeConfigError, match="duplicate"):
        KnowledgeBase([core(), core()])


# --- embedding once per request -----------------------------------------------------------------------


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 2), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        self.calls += 1
        return np.array([1.0, 0.0], np.float32)


async def test_query_embedded_once_shared_and_cached() -> None:
    embedder = CountingEmbedder()
    sources = [core(), bank()]
    kb = KnowledgeBase(sources, embedder=embedder)
    await kb.search(Query(text="same question"))
    await kb.search(Query(text="same question"))
    assert embedder.calls == 1
    assert all(q.vector is not None for s in sources for q in s.queries)
    await kb.search(Query(text="   "))
    assert embedder.calls == 1


# --- building from the shipped packs directory --------------------------------------------------------


async def test_build_sources_default_is_every_enabled_real_pack() -> None:
    async with httpx.AsyncClient() as client:
        sources = build_sources(PACKS, client=client)
    assert [s.manifest.id for s in sources] == [
        "genuine-notices",
        "kudiguard-core",
        "money-skills",
    ]  # fictional samples stay off


async def test_sample_packs_refused_unless_allowed() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(KnowledgeConfigError, match="fictional"):
            build_sources(PACKS, client=client, include=["kudiguard-core", "demo-bank"])
        sources = build_sources(
            PACKS, client=client, include=["kudiguard-core", "demo-bank"], allow_samples=True
        )
    assert [type(s) for s in sources] == [FilePackSource, FilePackSource]


async def test_build_sources_include_creates_http_source() -> None:
    async with httpx.AsyncClient() as client:
        sources = build_sources(
            PACKS,
            client=client,
            include=["kudiguard-core", "demo-bank-api"],
            allow_samples=True,
            env={"DEMO_BANK_TOKEN": "t"},
        )
    assert [type(s) for s in sources] == [FilePackSource, HttpSource]


async def test_live_partner_packs_are_refused_unless_allowed(tmp_path: Path) -> None:
    """mode: live sends each user's message (in English) to the partner: off unless the operator allows and discloses."""
    for name, mode in (("live-bank", "live"), ("sync-bank", "sync")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pack.yaml").write_text(
            f"id: {name}\nname: B\nowner: B\nkind: partner\nsource:\n  type: http\n"
            f"  base_url: https://partner.example\n  mode: {mode}\n",
            encoding="utf-8",
        )
    async with httpx.AsyncClient() as client:
        assert len(build_sources(tmp_path, client=client, include=["sync-bank"])) == 1
        with pytest.raises(KnowledgeConfigError, match=r"live-bank.*mode: live"):
            build_sources(tmp_path, client=client)
        assert len(build_sources(tmp_path, client=client, allow_live=True)) == 2
        from kudiguard.knowledge.service import knowledge_base_from_packs

        with pytest.raises(KnowledgeConfigError, match="KUDIGUARD_ALLOW_LIVE_PACKS"):
            knowledge_base_from_packs(tmp_path, client=client)
        with pytest.raises(KnowledgeConfigError, match="core knowledge pack"):  # got past the live check
            knowledge_base_from_packs(tmp_path, client=client, allow_live=True)


async def test_build_sources_unknown_pack() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(KnowledgeConfigError, match="nope"):
            build_sources(PACKS, client=client, include=["kudiguard-core", "nope"])


async def test_folder_name_must_match_pack_id(tmp_path: Path) -> None:
    (tmp_path / "wrong").mkdir()
    (tmp_path / "wrong" / "pack.yaml").write_text(
        "id: right\nname: R\nowner: R\nkind: core\n", encoding="utf-8"
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(KnowledgeConfigError, match="folder name"):
            build_sources(tmp_path, client=client)


async def test_end_to_end_with_core_and_sample_bank() -> None:
    async with httpx.AsyncClient() as client:
        sources = build_sources(
            PACKS, client=client, include=["kudiguard-core", "demo-bank"], allow_samples=True
        )
        result = await KnowledgeBase(sources).search(
            Query(text="someone asked for my otp", flags=(Flag.ASKS_CREDENTIALS,))
        )
    assert [s.entry.id for s in result.entries][:2] == ["flag-asks-credentials", "demo-never-ask"]
    assert result.contacts[0].label == "Your own bank"  # pinned
    assert result.contacts[1].label.startswith("Demo Bank fraud desk")


class BrokenEmbedder(CountingEmbedder):
    def embed_query(self, text: str) -> np.ndarray:
        raise RuntimeError("onnx exploded")


async def test_embedding_failure_degrades_to_keyword_search() -> None:
    source = core()
    result = await KnowledgeBase([source], embedder=BrokenEmbedder()).search(Query(text="q"))
    assert result.entries  # still answered
    assert source.queries[0].vector is None


async def test_partners_cannot_pin_contacts() -> None:
    pushy = FakeSource(
        "pushy", priority=99, result=SourceResult(contacts=(phone("0900", pinned=True), phone("0901")))
    )
    result = await KnowledgeBase([pushy, core()]).search(Query(text="q"))
    assert result.contacts[0].value == "111"  # core's pinned contact stays first
    assert [c.value for c in result.contacts][1:3] == ["0900", "0901"]


async def test_partner_needs_margin_to_outrank_core() -> None:
    near = FakeSource("near", priority=99, result=SourceResult(hits=(Hit(e("near-h"), 0.82),)))
    far = FakeSource("far", priority=99, result=SourceResult(hits=(Hit(e("far-h"), 0.84),)))
    hits_only = Limits(flag_entries=0)

    async def first(partner: FakeSource) -> str:
        kb = KnowledgeBase([core(), partner], limits=hits_only)
        return (await kb.search(Query(text="q"))).entries[0].entry.id

    assert await first(near) == "core-h1"  # 0.80 + 0.03 margin > 0.82
    assert await first(far) == "far-h"  # 0.84 > 0.83


async def test_unscored_partner_hits_are_dropped() -> None:
    unscored = FakeSource("unscored", priority=99, result=SourceResult(hits=(Hit(e("u")),)))
    result = await KnowledgeBase([core(), unscored]).search(Query(text="q"))
    assert "u" not in [s.entry.id for s in result.entries]


async def test_safety_entries_are_trimmed_not_dropped() -> None:
    long_flag = Entry(id="f2", title="T", facts=("a" * 300, "b" * 300), actions=("c" * 300,))
    src = FakeSource(
        "core",
        kind=PackKind.CORE,
        result=SourceResult(flag_entries=(e("f1", "x" * 500), long_flag), hits=(Hit(e("h", "z" * 50), 0.9),)),
    )
    result = await KnowledgeBase([src], limits=Limits(chars=900)).search(QUERY)
    ids = [s.entry.id for s in result.entries]
    assert ids[:2] == ["f1", "f2"]  # second safety entry kept (trimmed) rather than skipped for a hit
    trimmed = result.entries[1].entry
    assert trimmed.actions == () and len(trimmed.facts) == 1


async def test_factory_shares_one_embedder() -> None:
    from kudiguard.knowledge.service import knowledge_base_from_packs

    embedder = CountingEmbedder()
    async with httpx.AsyncClient() as client:
        kb = knowledge_base_from_packs(PACKS, client=client, embedder=embedder)
        await kb.search(Query(text="how do i save money"))
    assert embedder.calls == 1


async def test_overlong_first_safety_fact_is_truncated_with_ellipsis() -> None:
    from kudiguard.knowledge.service import ELLIPSIS, MIN_FACT_CHARS

    src = FakeSource(
        "core",
        kind=PackKind.CORE,
        result=SourceResult(flag_entries=(e("f1", "a" * 500), e("f2", "b " * 300))),
    )
    result = await KnowledgeBase([src], limits=Limits(chars=200, hits=0)).search(QUERY)
    first, second = (s.entry for s in result.entries)
    assert first.facts[0] == "a" * 197 + ELLIPSIS  # room = budget - len(title "f1")
    assert len(first.facts[0]) == 198
    # The budget is spent, yet the second safety entry still keeps MIN_FACT_CHARS of its first fact.
    assert second.facts[0].endswith(ELLIPSIS) and len(second.facts[0]) <= MIN_FACT_CHARS
    assert not second.facts[0][:-1].endswith(" ")  # trailing whitespace stripped before the ellipsis


async def test_query_cache_evicts_least_recently_used(monkeypatch: pytest.MonkeyPatch) -> None:
    import kudiguard.knowledge.service as service

    monkeypatch.setattr(service, "QUERY_CACHE_SIZE", 2)
    embedder = CountingEmbedder()
    kb = KnowledgeBase([core()], embedder=embedder)
    for text in (
        "one",
        "two",
        "one",
        "three",
    ):  # "one" refreshed, so "two" is the oldest when "three" arrives
        await kb.search(Query(text=text))
    assert embedder.calls == 3
    await kb.search(Query(text="one"))
    assert embedder.calls == 3  # still cached
    await kb.search(Query(text="two"))
    assert embedder.calls == 4  # evicted, embedded again
