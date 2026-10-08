"""Adapter for OpenAI-compatible chat servers: llama.cpp (default, local N-ATLAS GGUF), vLLM, Ollama, hosted APIs.

llama.cpp extras (``ServerKind.LLAMACPP``): ``repeat_penalty``, ``cache_prompt`` and optional per-purpose ``id_slot``.
Generic servers get a plain OpenAI body.

Every call has a **total** deadline (queueing + all attempts), per purpose by default. llama.cpp stops generating
when the client disconnects, so enforcing the deadline client-side is safe (M4 review). The concurrency gate is held
across retries on purpose: with one GPU this keeps requests first-in-first-out.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import httpx

from kudiguard.llm.base import ChatRequest, Completion, LLMError, Purpose

RETRYABLE_STATUS = frozenset({502, 503})  # 503 = llama.cpp still loading the model
MAX_ERROR_CHARS = 200  # provider errors can echo user input (PII): keep them short
DEFAULT_TIMEOUTS: Mapping[Purpose, float] = {
    Purpose.ANALYSE: 10.0,
    Purpose.TRANSLATE: 10.0,
    Purpose.COMPOSE: 20.0,
    Purpose.WARN: 10.0,
    Purpose.GENERAL: 30.0,
}
MIN_ATTEMPT_S = 1.0  # do not start a retry with less budget than this


class ServerKind(StrEnum):
    LLAMACPP = "llamacpp"
    OPENAI = "openai"


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.AsyncClient,
        server: ServerKind = ServerKind.LLAMACPP,
        model: str = "n-atlas",
        api_key: str | None = None,
        chat_path: str = "/v1/chat/completions",
        timeouts: Mapping[Purpose, float] = DEFAULT_TIMEOUTS,
        max_concurrency: int = 1,  # one GPU: queue rather than thrash the KV cache
        slots: Mapping[Purpose, int] | None = None,
        retries: int = 1,
        retry_delay_s: float = 0.5,
    ) -> None:
        url = base_url.rstrip("/")
        self.base_url = url[:-3] if url.lower().endswith("/v1") else url
        self._client = client
        self._server = server
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._chat_url = f"{self.base_url}{chat_path}"
        self._timeouts = dict(timeouts)
        self._gate = asyncio.Semaphore(max_concurrency)
        self._slots = dict(slots or {})
        self._retries = retries
        self._retry_delay_s = retry_delay_s

    def build_body(self, request: ChatRequest) -> dict[str, Any]:
        """The exact JSON body sent for ``request`` (public for benchmarks and debugging)."""
        body = request.payload()
        body["model"] = self._model
        if self._server is ServerKind.LLAMACPP:
            body["cache_prompt"] = True
            if (slot := self._slots.get(request.purpose)) is not None:
                body["id_slot"] = slot
        else:
            body.pop("repeat_penalty", None)
        return body

    def _budget(self, request: ChatRequest) -> float:
        if request.timeout_s is not None:
            return request.timeout_s
        return self._timeouts.get(request.purpose, DEFAULT_TIMEOUTS[Purpose.GENERAL])

    async def complete(self, request: ChatRequest) -> Completion:
        body = self.build_body(request)
        budget = self._budget(request)
        deadline = asyncio.get_running_loop().time() + budget
        try:
            async with asyncio.timeout_at(deadline):
                queued_at = time.perf_counter()
                async with self._gate:
                    started = time.perf_counter()
                    response, attempts = await self._post_with_retries(body, deadline)
                    finished = time.perf_counter()
        except TimeoutError as exc:
            elapsed = asyncio.get_running_loop().time() - (deadline - budget)
            raise LLMError(
                f"LLM timed out after {elapsed:.1f}s (budget {budget:g}s, {request.purpose.value})"
            ) from exc
        return dataclasses.replace(
            _parse(response, int((finished - started) * 1000)),
            queued_ms=int((started - queued_at) * 1000),
            attempts=attempts,
        )

    async def _post_with_retries(self, body: dict[str, Any], deadline: float) -> tuple[httpx.Response, int]:
        loop = asyncio.get_running_loop()
        for attempt in range(1, self._retries + 2):
            remaining = deadline - loop.time()
            last = attempt > self._retries or remaining < self._retry_delay_s + MIN_ATTEMPT_S
            try:
                response = await self._client.post(
                    self._chat_url, json=body, headers=self._headers, timeout=max(remaining, 0.001)
                )
            except httpx.TimeoutException as exc:  # never retried: it would double the user's wait
                raise TimeoutError from exc
            except httpx.TransportError as exc:
                if last:
                    raise LLMError(f"LLM unreachable at {self.base_url}: {type(exc).__name__}") from exc
            else:
                if response.status_code not in RETRYABLE_STATUS or last:
                    return response, attempt
            await asyncio.sleep(self._retry_delay_s)
        raise AssertionError("unreachable")  # pragma: no cover

    async def ready(self) -> bool:
        path = "/health" if self._server is ServerKind.LLAMACPP else "/v1/models"
        try:
            response = await self._client.get(f"{self.base_url}{path}", headers=self._headers, timeout=5.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200


def _parse(response: httpx.Response, latency_ms: int) -> Completion:
    if response.status_code >= 400:
        raise LLMError(f"LLM HTTP {response.status_code}: {_error_text(response)}")
    try:
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise TypeError("content is not a string")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError(f"malformed LLM response: {type(exc).__name__}") from exc
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return Completion(
        text=content or "",
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        latency_ms=latency_ms,
    )


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else body
    if isinstance(error, dict):
        error = error.get("message", error)
    text = str(error) if error else response.text
    return text.strip()[:MAX_ERROR_CHARS] or response.reason_phrase
