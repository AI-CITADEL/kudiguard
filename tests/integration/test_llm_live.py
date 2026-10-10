"""End-to-end against the real N-ATLAS server (``pytest -m live``; skipped when the server is not running)."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from kudiguard.domain import Verdict
from kudiguard.knowledge.models import Query
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.openai_compat import OpenAICompatLLM
from kudiguard.prompts import analyse_request, clean_reply, compose_request, parse_analysis
from kudiguard.safety.decision import decide, resolve_analysis
from kudiguard.safety.output_guard import OutputGuard

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.live


@pytest.fixture
async def llm() -> AsyncIterator[OpenAICompatLLM]:
    async with httpx.AsyncClient() as client:
        model = OpenAICompatLLM("http://127.0.0.1:8081", client=client)
        if not await model.ready():
            pytest.skip("llama-server not running on :8081")
        yield model


async def test_hausa_scam_end_to_end(llm: OpenAICompatLLM, registry: LanguageRegistry) -> None:
    text = "Wani ya kira ni ya ce an toshe BVN dina, ya ce in tura masa lambar OTP da aka turo min."
    analysis = parse_analysis(
        (await llm.complete(analyse_request(text, "Hausa"))).text, text, source_language="ha"
    )
    assert analysis is not None
    analysis = resolve_analysis(analysis, text, "ha")
    result = decide(text, analysis, registry.lexicon_for("ha"))
    assert result.verdict is Verdict.LIKELY_SCAM

    async with httpx.AsyncClient() as client:
        knowledge = await knowledge_base_from_packs(ROOT / "data" / "packs", client=client).search(
            Query(text=analysis.english, flags=result.flags)
        )
    completion = await llm.complete(
        compose_request(
            language_name="Hausa",
            english=analysis.english,
            verdict=result.verdict,
            flags=result.flags,
            knowledge=knowledge,
        )
    )
    reply = clean_reply(completion.text, truncated=completion.truncated)
    assert reply
    assert OutputGuard(registry).check(reply, "ha") == ()
    assert completion.latency_ms < 15_000


async def test_same_request_twice_gives_the_same_text() -> None:
    """Greedy decoding with the prompt cache off is bit-exact (with it on, near-tied tokens can flip)."""
    request = analyse_request(
        "Someone called and asked me to send the OTP they just sent to my phone.", "English"
    )
    async with httpx.AsyncClient() as client:
        llm = OpenAICompatLLM("http://127.0.0.1:8081", client=client, cache_prompt=False)
        if not await llm.ready():
            pytest.skip("llama-server not running on :8081")
        first, second = (await llm.complete(request)).text, (await llm.complete(request)).text
    assert first == second
