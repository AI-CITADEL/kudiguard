import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from kudiguard.domain import Flag
from kudiguard.knowledge.base import SourceUnavailable
from kudiguard.knowledge.file_pack import KnowledgeConfigError
from kudiguard.knowledge.http_source import HttpSource
from kudiguard.knowledge.models import PackManifest, Query

BASE = "https://partner.example/kb"
EXPORT = {
    "entries": [
        {
            "id": "otp",
            "title": "We never ask for OTP",
            "facts": ["Never share OTP."],
            "flags": ["asks_credentials"],
        },
        {
            "id": "limit",
            "title": "Daily transfer limit",
            "facts": ["Limit is N200,000."],
            "keywords": ["limit"],
        },
    ],
    "contacts": [
        {"label": "Fraud desk", "kind": "phone", "value": "000", "flags": ["asks_credentials"]},
        {"label": "Website", "kind": "web", "value": "https://partner.example"},
    ],
}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def manifest(mode: str = "sync", **source: object) -> PackManifest:
    return PackManifest.model_validate(
        {
            "id": "partner",
            "name": "Partner",
            "owner": "Partner Ltd",
            "source": {
                "type": "http",
                "base_url": BASE,
                "mode": mode,
                "auth": {"env": "PARTNER_TOKEN"},
                **source,
            },
        }
    )


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as c:
        yield c


@pytest.fixture
def clock() -> Clock:
    return Clock()


def source(client: httpx.AsyncClient, clock: Clock, mode: str = "sync", **spec: object) -> HttpSource:
    return HttpSource(manifest(mode, **spec), client, env={"PARTNER_TOKEN": "s3cret"}, clock=clock)


# --- sync mode ----------------------------------------------------------------------------------------


@respx.mock
async def test_sync_refresh_then_search_locally(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.get(f"{BASE}/export").respond(json=EXPORT)
    src = source(client, clock)
    await src.refresh()
    result = await src.search(Query(text="what is the daily transfer limit", flags=(Flag.ASKS_CREDENTIALS,)))
    await src.search(Query(text="limit"))
    assert route.call_count == 1
    assert route.calls[0].request.headers["Authorization"] == "Bearer s3cret"
    assert [e.id for e in result.flag_entries] == ["otp"]
    assert [h.entry.id for h in result.hits] == ["limit"]
    assert [c.value for c in result.contacts] == ["000", "https://partner.example"]


@respx.mock
async def test_sync_stale_copy_served_immediately_while_resyncing(
    client: httpx.AsyncClient, clock: Clock
) -> None:
    route = respx.get(f"{BASE}/export").respond(json=EXPORT)
    src = source(client, clock, refresh_s=600)
    await src.search(Query(text="limit"))  # first use waits for the initial sync
    clock.now += 599
    await src.search(Query(text="limit"))
    assert route.call_count == 1
    clock.now += 2
    result = await src.search(Query(text="limit"))  # stale: answered from the old copy, sync in background
    assert result.hits[0].entry.id == "limit"
    await src.refresh()  # joins the in-flight background sync instead of starting another
    assert route.call_count == 2


@respx.mock
async def test_concurrent_first_syncs_share_one_call(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.get(f"{BASE}/export").respond(json=EXPORT)
    src = source(client, clock)
    await asyncio.gather(*(src.search(Query(text="limit")) for _ in range(5)))
    assert route.call_count == 1


@respx.mock
async def test_sync_keeps_last_good_index_when_refresh_fails(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.get(f"{BASE}/export")
    route.side_effect = [httpx.Response(200, json=EXPORT), httpx.Response(503)]
    src = source(client, clock, refresh_s=60)
    await src.refresh()
    clock.now += 61
    assert (await src.search(Query(text="limit"))).hits[0].entry.id == "limit"


@respx.mock
async def test_sync_without_any_index_raises(client: httpx.AsyncClient, clock: Clock) -> None:
    respx.get(f"{BASE}/export").respond(503)
    with pytest.raises(SourceUnavailable):
        await source(client, clock).search(Query(text="limit"))


# --- live mode ----------------------------------------------------------------------------------------


@respx.mock
async def test_live_posts_query_and_splits_results(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.post(f"{BASE}/search").respond(json=EXPORT)
    src = source(client, clock, "live")
    result = await src.search(Query(text="otp limit", flags=(Flag.ASKS_CREDENTIALS,), language="ha", limit=2))
    body = json.loads(route.calls[0].request.content)
    assert body == {"query": "otp limit", "flags": ["asks_credentials"], "language": "ha", "limit": 2}
    assert [e.id for e in result.flag_entries] == ["otp"]
    assert [h.entry.id for h in result.hits] == ["limit"]  # partner's own ranking is kept
    assert [c.value for c in result.contacts] == ["000", "https://partner.example"]


@respx.mock
async def test_live_results_are_cached_until_ttl(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.post(f"{BASE}/search").respond(json=EXPORT)
    src = source(client, clock, "live", cache_ttl_s=300)
    query = Query(text="limit")
    await src.search(query)
    await src.search(query)
    assert route.call_count == 1
    await src.search(Query(text="other question"))
    assert route.call_count == 2
    clock.now += 301
    await src.search(query)
    assert route.call_count == 3


@respx.mock
async def test_refresh_is_noop_in_live_mode(client: httpx.AsyncClient, clock: Clock) -> None:
    await source(client, clock, "live").refresh()
    assert not respx.calls


# --- failure handling ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"entries": [], "surprise": 1}),  # unknown field
        httpx.Response(200, content=b'{"entries": []}' + b" " * 2_000),  # over max_bytes
    ],
)
@respx.mock
async def test_bad_responses_raise_source_unavailable(
    client: httpx.AsyncClient, clock: Clock, response: httpx.Response
) -> None:
    respx.post(f"{BASE}/search").mock(return_value=response)
    with pytest.raises(SourceUnavailable, match="partner"):
        await source(client, clock, "live", max_bytes=1_000).search(Query(text="limit"))


@respx.mock
async def test_timeout_raises_source_unavailable(client: httpx.AsyncClient, clock: Clock) -> None:
    respx.post(f"{BASE}/search").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(SourceUnavailable):
        await source(client, clock, "live").search(Query(text="limit"))


@respx.mock
async def test_circuit_breaker_opens_and_recovers(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.post(f"{BASE}/search").respond(500)
    src = source(client, clock, "live", cache_ttl_s=0)
    for _ in range(3):
        with pytest.raises(SourceUnavailable):
            await src.search(Query(text="limit"))
    with pytest.raises(SourceUnavailable, match="circuit open"):
        await src.search(Query(text="limit"))
    assert route.call_count == 3

    clock.now += 61
    route.respond(json=EXPORT)
    assert (await src.search(Query(text="limit"))).hits
    assert route.call_count == 4


# --- configuration ------------------------------------------------------------------------------------


async def test_missing_token_env_is_a_config_error(client: httpx.AsyncClient, clock: Clock) -> None:
    with pytest.raises(KnowledgeConfigError, match="PARTNER_TOKEN"):
        HttpSource(manifest(), client, env={}, clock=clock)


@respx.mock
async def test_no_auth_and_raw_scheme(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.get(f"{BASE}/export").respond(json=EXPORT)
    plain = PackManifest.model_validate(
        {"id": "p", "name": "P", "owner": "P", "source": {"type": "http", "base_url": BASE}}
    )
    await HttpSource(plain, client, env={}, clock=clock).refresh()
    assert "Authorization" not in route.calls[0].request.headers

    raw = manifest(auth={"env": "PARTNER_TOKEN", "header": "X-Api-Key", "scheme": None})
    await HttpSource(raw, client, env={"PARTNER_TOKEN": "k"}, clock=clock).refresh()
    assert route.calls[1].request.headers["X-Api-Key"] == "k"


def test_files_manifest_rejected(client: httpx.AsyncClient, clock: Clock) -> None:
    files = PackManifest.model_validate({"id": "p", "name": "P", "owner": "P"})
    with pytest.raises(KnowledgeConfigError, match="http"):
        HttpSource(files, client, env={}, clock=clock)


@respx.mock
async def test_live_cache_evicts_oldest(
    client: httpx.AsyncClient, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kudiguard.knowledge.http_source.CACHE_SIZE", 2)
    route = respx.post(f"{BASE}/search").respond(json=EXPORT)
    src = source(client, clock, "live")
    for text in ("a", "b", "c", "a"):
        await src.search(Query(text=text))
    assert route.call_count == 4  # "a" was evicted when "c" arrived


class Hang:
    """respx side effect that never answers in time; counts attempts (respx only records completed calls)."""

    def __init__(self) -> None:
        self.attempts = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        await asyncio.sleep(10)
        return httpx.Response(200, json=EXPORT)


@respx.mock
async def test_hanging_partner_times_out_fast_and_trips_breaker(
    client: httpx.AsyncClient, clock: Clock
) -> None:
    hang = Hang()
    respx.post(f"{BASE}/search").mock(side_effect=hang)
    src = source(client, clock, "live", timeout_s=0.05, cache_ttl_s=0)
    started = time.perf_counter()
    for _ in range(3):
        with pytest.raises(SourceUnavailable, match="Timeout"):
            await src.search(Query(text="limit"))
    with pytest.raises(SourceUnavailable, match="circuit open"):
        await src.search(Query(text="limit"))
    assert time.perf_counter() - started < 1
    assert hang.attempts == 3


@respx.mock
async def test_half_open_lets_exactly_one_probe_through(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.post(f"{BASE}/search").respond(500)
    src = source(client, clock, "live", cache_ttl_s=0, timeout_s=0.2)
    for _ in range(3):
        with pytest.raises(SourceUnavailable):
            await src.search(Query(text="limit"))
    clock.now += 61
    hang = Hang()
    route.mock(side_effect=hang)
    outcomes = await asyncio.gather(
        *(src.search(Query(text="limit")) for _ in range(3)), return_exceptions=True
    )
    assert sum("circuit open" in str(o) for o in outcomes) == 2
    assert hang.attempts == 1  # exactly one probe


@respx.mock
async def test_live_query_is_redacted_before_sending(client: httpx.AsyncClient, clock: Clock) -> None:
    route = respx.post(f"{BASE}/search").respond(json=EXPORT)
    await source(client, clock, "live").search(Query(text="my account 0123456789 and ada@mail.example.com"))
    assert json.loads(route.calls[0].request.content)["query"] == "my account [number] and [email]"


@respx.mock
async def test_declared_oversize_rejected_before_reading(client: httpx.AsyncClient, clock: Clock) -> None:
    respx.post(f"{BASE}/search").respond(200, headers={"content-length": "99999999"}, content=b"{}")
    with pytest.raises(SourceUnavailable, match="declared size"):
        await source(client, clock, "live").search(Query(text="limit"))


@pytest.mark.parametrize(
    ("url", "extra", "ok"),
    [
        ("https://kb.partner.example", {}, True),
        ("http://127.0.0.1:8900", {}, True),
        ("http://localhost:8900", {}, True),
        ("http://kb.partner.example", {}, False),
        ("http://kb.partner.example", {"allow_insecure": True}, True),
        ("https://169.254.169.254/latest", {}, False),
    ],
)
def test_base_url_policy(url: str, extra: dict[str, object], ok: bool) -> None:
    spec = {"id": "p", "name": "P", "owner": "P", "source": {"type": "http", "base_url": url, **extra}}
    if ok:
        PackManifest.model_validate(spec)
    else:
        with pytest.raises(ValueError, match=r"https|link-local"):
            PackManifest.model_validate(spec)


@respx.mock
async def test_cancellation_by_caller_deadline_counts_toward_breaker(
    client: httpx.AsyncClient, clock: Clock
) -> None:
    hang = Hang()
    respx.post(f"{BASE}/search").mock(side_effect=hang)
    src = source(client, clock, "live", timeout_s=3, cache_ttl_s=0)  # own timeout longer than caller's
    for _ in range(3):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(src.search(Query(text="limit")), timeout=0.02)
    with pytest.raises(SourceUnavailable, match="circuit open"):
        await src.search(Query(text="limit"))
    assert hang.attempts == 3


@respx.mock
async def test_one_bad_record_does_not_discard_the_partner(client: httpx.AsyncClient, clock: Clock) -> None:
    payload = {
        "entries": [*EXPORT["entries"], {"id": "BAD ID", "title": "x", "facts": ["y"]}],
        "contacts": [*EXPORT["contacts"], {"label": "Phish", "kind": "web", "value": "http://evil.example"}],
    }
    respx.get(f"{BASE}/export").respond(json=payload)
    result = await source(client, clock).search(Query(text="limit", flags=(Flag.ASKS_CREDENTIALS,)))
    assert [e.id for e in result.flag_entries] == ["otp"]
    assert "http://evil.example" not in [c.value for c in result.contacts]


@respx.mock
async def test_live_hits_are_rescored_locally(client: httpx.AsyncClient, clock: Clock) -> None:
    noisy = {"entries": [{"id": "junk", "title": "Weather today", "facts": ["Sunny."]}, *EXPORT["entries"]]}
    respx.post(f"{BASE}/search").respond(json=noisy)
    hits = (await source(client, clock, "live").search(Query(text="daily transfer limit"))).hits
    assert [h.entry.id for h in hits] == ["limit"]  # partner's irrelevant first candidate dropped
    assert hits[0].relevance is not None


@respx.mock
async def test_non_object_items_are_dropped_not_fatal(client: httpx.AsyncClient, clock: Clock) -> None:
    respx.get(f"{BASE}/export").respond(
        json={"entries": [*EXPORT["entries"], "junk", 42], "contacts": [None]}
    )
    result = await source(client, clock).search(Query(text="daily transfer limit"))
    assert result.hits[0].entry.id == "limit"
    assert result.contacts == ()
