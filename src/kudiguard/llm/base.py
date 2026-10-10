"""LLM port: the request/response types and the protocol every model backend implements.

This package imports nothing else from ``kudiguard`` so it can be lifted unchanged into the N-ATLAS gateway.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

DEFAULT_TEMPERATURE = 0.1  # N-ATLAS model card
DEFAULT_REPEAT_PENALTY = 1.12  # N-ATLAS model card (llama.cpp name)


class Purpose(StrEnum):
    """What a call is for: routes it to a cache slot and labels metrics."""

    ANALYSE = "analyse"
    TRANSLATE = "translate"
    COMPOSE = "compose"
    WARN = "warn"
    STEPS = "steps"
    GENERAL = "general"


# Per-purpose total budget (seconds, incl. queueing and retries) when a request sets no ``timeout_s``.
DEFAULT_TIMEOUTS: Mapping[Purpose, float] = {
    Purpose.ANALYSE: 10.0,
    Purpose.TRANSLATE: 10.0,
    Purpose.COMPOSE: 20.0,
    Purpose.WARN: 10.0,
    Purpose.STEPS: 10.0,
    Purpose.GENERAL: 30.0,
}


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[Mapping[str, str], ...]
    max_tokens: int
    temperature: float = DEFAULT_TEMPERATURE
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY
    json_schema: Mapping[str, Any] | None = None
    stop: tuple[str, ...] = ()
    purpose: Purpose = Purpose.GENERAL
    # Total budget incl. queueing and retries; None = the backend's per-purpose default.
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")

    def payload(self) -> dict[str, Any]:
        """OpenAI-compatible chat-completions body (llama.cpp also accepts ``repeat_penalty``)."""
        body: dict[str, Any] = {
            "messages": [dict(m) for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.stop:
            body["stop"] = list(self.stop)
        if self.json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": self.purpose.value, "schema": dict(self.json_schema)},
            }
        return body


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0  # request time incl. retries, excluding queueing
    queued_ms: int = 0  # time waiting for the backend's concurrency slot
    attempts: int = 1

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMError(RuntimeError):
    """The model could not produce a completion (unreachable, timeout, HTTP error, malformed response)."""


class LLM(Protocol):
    async def complete(self, request: ChatRequest) -> Completion:
        """Run one chat completion. Raises ``LLMError``."""
        ...

    async def ready(self) -> bool:
        """True when the backend is loaded and accepting requests."""
        ...


async def complete_within(
    llm: LLM, request: ChatRequest, deadline: float | None, minimum_s: float
) -> Completion:
    """One call capped to the time left before ``deadline`` (``time.monotonic()`` seconds).

    Raises ``LLMError`` without calling when less than ``minimum_s`` is left: a turn's fixed text must still go out.
    """
    if deadline is not None:
        left = deadline - time.monotonic()
        if left < minimum_s:
            raise LLMError(f"{request.purpose}: skipped, {left:.1f}s left in the turn")
        budget = min(request.timeout_s or DEFAULT_TIMEOUTS[request.purpose], left)
        request = dataclasses.replace(request, timeout_s=budget)
    return await llm.complete(request)
