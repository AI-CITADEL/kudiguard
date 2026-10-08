import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from kudiguard.llm.base import ChatRequest, LLMError, Purpose
from kudiguard.llm.openai_compat import OpenAICompatLLM, ServerKind

BASE = "http://llm.test"
URL = f"{BASE}/v1/chat/completions"


def ok(content: str = "Hello", finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )


def request(purpose: Purpose = Purpose.COMPOSE) -> ChatRequest:
    return ChatRequest(messages=({"role": "user", "content": "hi"},), max_tokens=50, purpose=purpose)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as c:
        yield c


@respx.mock
async def test_completion_payload_and_parsing(client: httpx.AsyncClient) -> None:
    route = respx.post(URL).mock(return_value=ok("Do not share it.", "length"))
    llm = OpenAICompatLLM(BASE, client=client, slots={Purpose.ANALYSE: 0, Purpose.COMPOSE: 1})
    completion = await llm.complete(request(Purpose.COMPOSE))
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "n-atlas"
    assert body["temperature"] == pytest.approx(0.1)
    assert body["cache_prompt"] is True and body["id_slot"] == 1
    assert completion.text == "Do not share it."
    assert completion.truncated
    assert (completion.prompt_tokens, completion.completion_tokens) == (12, 3)
    assert completion.latency_ms >= 0


@respx.mock
async def test_unmapped_purpose_gets_no_slot_and_generic_server_no_extras(client: httpx.AsyncClient) -> None:
    route = respx.post(URL).mock(return_value=ok())
    await OpenAICompatLLM(BASE, client=client, slots={Purpose.ANALYSE: 0}).complete(request(Purpose.WARN))
    assert "id_slot" not in json.loads(route.calls[0].request.content)
    generic = OpenAICompatLLM(BASE, client=client, server=ServerKind.OPENAI, model="m", api_key="k")
    await generic.complete(request())
    body = json.loads(route.calls[1].request.content)
    assert "cache_prompt" not in body and "repeat_penalty" not in body
    assert route.calls[1].request.headers["Authorization"] == "Bearer k"


@respx.mock
async def test_requests_are_serialised_on_the_gpu(client: httpx.AsyncClient) -> None:
    in_flight = 0
    peak = 0

    async def slow(_: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return ok()

    respx.post(URL).mock(side_effect=slow)
    llm = OpenAICompatLLM(BASE, client=client, max_concurrency=1)
    await asyncio.gather(*(llm.complete(request()) for _ in range(4)))
    assert peak == 1


@respx.mock
async def test_retries_once_on_connection_error_or_loading(client: httpx.AsyncClient) -> None:
    route = respx.post(URL)
    route.side_effect = [
        httpx.ConnectError("refused"),
        httpx.Response(503, json={"error": "loading"}),
        ok("Hi"),
    ]
    llm = OpenAICompatLLM(BASE, client=client, retries=2, retry_delay_s=0)
    assert (await llm.complete(request())).text == "Hi"
    assert route.call_count == 3


@respx.mock
async def test_gives_up_after_retries(client: httpx.AsyncClient) -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(LLMError, match="unreachable"):
        await OpenAICompatLLM(BASE, client=client, retries=1, retry_delay_s=0).complete(request())


@respx.mock
async def test_timeout_is_not_retried(client: httpx.AsyncClient) -> None:
    route = respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError, match=r"budget 20s, compose"):
        await OpenAICompatLLM(BASE, client=client, retries=3, retry_delay_s=0).complete(request())
    assert route.call_count == 1


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, text="boom"), "HTTP 500"),
        (httpx.Response(400, json={"error": {"message": "context too long"}}), "context too long"),
        (httpx.Response(200, text="not json"), "malformed"),
        (httpx.Response(200, json={"choices": []}), "malformed"),
        (httpx.Response(200, json={"choices": [{"message": {"content": 5}}]}), "malformed"),
    ],
)
@respx.mock
async def test_errors_become_llm_error(
    client: httpx.AsyncClient, response: httpx.Response, message: str
) -> None:
    respx.post(URL).mock(return_value=response)
    with pytest.raises(LLMError, match=message):
        await OpenAICompatLLM(BASE, client=client, retries=0).complete(request())


@respx.mock
async def test_null_content_is_empty_text(client: httpx.AsyncClient) -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
        )
    )
    assert (await OpenAICompatLLM(BASE, client=client).complete(request())).text == ""


@respx.mock
async def test_ready(client: httpx.AsyncClient) -> None:
    respx.get(f"{BASE}/health").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    assert await OpenAICompatLLM(BASE, client=client).ready()
    respx.get(f"{BASE}/health").mock(return_value=httpx.Response(503, json={"status": "loading model"}))
    assert not await OpenAICompatLLM(BASE, client=client).ready()
    respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("down"))
    assert not await OpenAICompatLLM(BASE, client=client).ready()


@respx.mock
async def test_ready_for_generic_server_uses_models_endpoint(client: httpx.AsyncClient) -> None:
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    assert await OpenAICompatLLM(BASE, client=client, server=ServerKind.OPENAI).ready()


async def test_base_url_normalisation_and_custom_chat_path(client: httpx.AsyncClient) -> None:
    assert OpenAICompatLLM("http://x/v1/", client=client).base_url == "http://x"
    assert OpenAICompatLLM("http://x/V1", client=client).base_url == "http://x"
    llm = OpenAICompatLLM("https://api.example/v1beta/openai", client=client, chat_path="/chat/completions")
    assert llm.base_url == "https://api.example/v1beta/openai"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(400, json=["bad", "request"]), "HTTP 400"),
        (httpx.Response(500, json="kaboom"), "kaboom"),
        (httpx.Response(400, text="<html>bad</html>"), "HTTP 400"),
        (httpx.Response(200, json=["not", "an", "object"]), "malformed"),
        (
            httpx.Response(200, json={"choices": [{"message": None, "finish_reason": "stop"}], "usage": "x"}),
            None,
        ),
    ],
)
@respx.mock
async def test_odd_bodies_never_escape_as_other_exceptions(
    client: httpx.AsyncClient, response: httpx.Response, message: str | None
) -> None:
    respx.post(URL).mock(return_value=response)
    llm = OpenAICompatLLM(BASE, client=client, retries=0)
    if message is None:
        completion = await llm.complete(request())
        assert completion.text == "" and completion.prompt_tokens is None
    else:
        with pytest.raises(LLMError, match=message):
            await llm.complete(request())


@respx.mock
async def test_error_text_is_capped(client: httpx.AsyncClient) -> None:
    respx.post(URL).mock(return_value=httpx.Response(400, json={"error": {"message": "x" * 5000}}))
    with pytest.raises(LLMError) as info:
        await OpenAICompatLLM(BASE, client=client, retries=0).complete(request())
    assert len(str(info.value)) < 300


@respx.mock
async def test_503_on_every_attempt_is_mapped(client: httpx.AsyncClient) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(503, json={"error": "loading model"}))
    with pytest.raises(LLMError, match="HTTP 503"):
        await OpenAICompatLLM(BASE, client=client, retries=1, retry_delay_s=0).complete(request())
    assert route.call_count == 2


@respx.mock
async def test_total_deadline_covers_queueing(client: httpx.AsyncClient) -> None:
    async def slow(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return ok()

    respx.post(URL).mock(side_effect=slow)
    llm = OpenAICompatLLM(BASE, client=client)
    first = asyncio.create_task(llm.complete(request()))
    await asyncio.sleep(0.01)
    quick = ChatRequest(messages=({"role": "user", "content": "x"},), max_tokens=5, timeout_s=0.1)
    with pytest.raises(LLMError, match=r"timed out after 0\.1s"):
        await llm.complete(quick)  # spends its whole budget waiting for the GPU slot
    assert (await first).queued_ms < 50


@respx.mock
async def test_cancelled_waiter_releases_nothing_it_did_not_take(client: httpx.AsyncClient) -> None:
    async def slow(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return ok()

    respx.post(URL).mock(side_effect=slow)
    llm = OpenAICompatLLM(BASE, client=client)
    running = asyncio.create_task(llm.complete(request()))
    waiting = asyncio.create_task(llm.complete(request()))
    await asyncio.sleep(0.01)
    waiting.cancel()
    await running
    completion = await asyncio.wait_for(llm.complete(request()), timeout=1)  # the gate is free again
    assert completion.text == "Hello" and completion.attempts == 1


@respx.mock
async def test_retry_skipped_when_budget_cannot_fit_one(client: httpx.AsyncClient) -> None:
    route = respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    short = ChatRequest(messages=({"role": "user", "content": "x"},), max_tokens=5, timeout_s=0.5)
    with pytest.raises(LLMError, match="unreachable"):
        await OpenAICompatLLM(BASE, client=client, retries=3, retry_delay_s=0.2).complete(short)
    assert route.call_count == 1


@respx.mock
async def test_metrics_report_attempts(client: httpx.AsyncClient) -> None:
    respx.post(URL).mock(side_effect=[httpx.Response(503), ok()])
    completion = await OpenAICompatLLM(BASE, client=client, retries=1, retry_delay_s=0).complete(request())
    assert completion.attempts == 2 and completion.queued_ms >= 0


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_non_positive_timeout_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        ChatRequest(messages=(), max_tokens=1, timeout_s=bad)


@respx.mock
async def test_empty_error_body_uses_reason_phrase(client: httpx.AsyncClient) -> None:
    respx.post(URL).mock(return_value=httpx.Response(503, content=b""))
    with pytest.raises(LLMError, match="HTTP 503: Service Unavailable"):
        await OpenAICompatLLM(BASE, client=client, retries=0).complete(request())
