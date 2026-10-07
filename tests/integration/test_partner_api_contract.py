"""HttpSource <-> reference partner server, end to end over ASGI (no sockets)."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from kudiguard.domain import Flag
from kudiguard.knowledge.base import SourceUnavailable
from kudiguard.knowledge.http_source import HttpSource
from kudiguard.knowledge.models import PackManifest, Query
from kudiguard.knowledge.reference_server import create_app

PACK = Path(__file__).resolve().parents[2] / "data" / "packs" / "demo-bank"
BASE = "http://partner.test"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(PACK, token="dev-token"))
    async with httpx.AsyncClient(transport=transport) as c:
        yield c


def http_source(client: httpx.AsyncClient, mode: str, token: str = "dev-token") -> HttpSource:
    manifest = PackManifest.model_validate(
        {
            "id": "demo-bank-api",
            "name": "Demo Bank API",
            "owner": "x",
            "sample": True,
            "source": {
                "type": "http",
                "base_url": BASE,
                "mode": mode,
                "allow_insecure": True,
                "auth": {"env": "DEMO_BANK_TOKEN"},
            },
        }
    )
    return HttpSource(manifest, client, env={"DEMO_BANK_TOKEN": token})


@pytest.mark.parametrize("mode", ["sync", "live"])
async def test_contract_round_trip(client: httpx.AsyncClient, mode: str) -> None:
    result = await http_source(client, mode).search(
        Query(text="what is my daily transfer limit", flags=(Flag.ASKS_CREDENTIALS,))
    )
    assert [e.id for e in result.flag_entries] == ["demo-never-ask"]
    assert result.hits[0].entry.id == "transfer-limit"
    assert result.contacts[0].label.startswith("Demo Bank fraud desk")


@pytest.mark.parametrize("mode", ["sync", "live"])
async def test_contract_rejects_bad_token(client: httpx.AsyncClient, mode: str) -> None:
    with pytest.raises(SourceUnavailable, match="401"):
        await http_source(client, mode, token="wrong").search(Query(text="limit"))


async def test_search_request_validation(client: httpx.AsyncClient) -> None:
    response = await client.post(
        f"{BASE}/search",
        json={"query": "x", "flags": ["not_a_flag"]},
        headers={"Authorization": "Bearer dev-token"},
    )
    assert response.status_code == 422
