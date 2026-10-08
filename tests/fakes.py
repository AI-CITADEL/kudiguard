"""Test doubles for the pipeline's ports (LLM, ASR, Messenger). Deterministic and inspectable."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from kudiguard.domain import InboundMessage
from kudiguard.llm.base import ChatRequest, Completion, LLMError, Purpose
from kudiguard.pipeline import Outbound
from kudiguard.speech.base import Audio, Transcript

Reply = str | Exception | Callable[[ChatRequest], str]


def analysis(english: str, intent: str = "scam_check", flags: Sequence[str] = ()) -> str:
    return json.dumps({"english": english, "intent": intent, "flags": list(flags)})


@dataclass
class ConcurrencyTracker:
    peak: int = 0


class FakeLLM:
    """Scripted per purpose: each call pops the next reply for its purpose (the last one repeats)."""

    def __init__(self, **script: Reply | list[Reply]) -> None:
        self.script: dict[Purpose, list[Reply]] = {
            Purpose(k): (v if isinstance(v, list) else [v]) for k, v in script.items()
        }
        self.requests: list[ChatRequest] = []
        self.delay_s = 0.0
        self.latency_ms = 100
        self._active = 0
        self._tracker: ConcurrencyTracker | None = None

    def track_concurrency(self) -> ConcurrencyTracker:
        self._tracker = ConcurrencyTracker()
        return self._tracker

    def calls(self, purpose: Purpose) -> list[ChatRequest]:
        return [r for r in self.requests if r.purpose is purpose]

    async def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        self._active += 1
        if self._tracker:
            self._tracker.peak = max(self._tracker.peak, self._active)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
        finally:
            self._active -= 1
        replies = self.script.get(request.purpose)
        if not replies:
            raise LLMError(f"no scripted reply for {request.purpose}")
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        text = reply(request) if callable(reply) else reply
        return Completion(text=text, finish_reason="stop", latency_ms=self.latency_ms)

    async def ready(self) -> bool:
        return True


class FakeASR:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.delay_s = 0.0
        self.languages: list[str] = []

    async def transcribe(self, audio: Audio, language: str) -> Transcript:
        self.languages.append(language)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error:
            raise self.error
        return Transcript(self.text, language, audio.duration_s, latency_ms=50)

    def supports(self, language: str) -> bool:
        return True


@dataclass
class FakeMessenger:
    sent: list[tuple[str, Outbound]] = field(default_factory=list)
    acknowledged: list[str] = field(default_factory=list)
    fail_sends: int = 0  # the next N sends raise
    fail_acknowledge: bool = False
    fail_on: set[str] = field(default_factory=set)  # sends of these texts raise
    hang: bool = False  # every send blocks forever (a stuck channel)
    media: dict[str, bytes] = field(default_factory=dict)  # media_id -> bytes served by fetch_media

    async def acknowledge(self, message: InboundMessage) -> None:
        if self.fail_acknowledge:
            raise RuntimeError("receipt failed")
        self.acknowledged.append(message.message_id)

    async def send(self, user_id: str, message: Outbound) -> None:
        if self.hang:
            await asyncio.Event().wait()
        if message.text in self.fail_on:
            raise RuntimeError("send failed")
        if self.fail_sends:
            self.fail_sends -= 1
            raise RuntimeError("send failed")
        self.sent.append((user_id, message))

    @property
    def texts(self) -> list[str]:
        return [m.text for _, m in self.sent]

    async def fetch_media(self, message: InboundMessage) -> bytes:
        if message.media_id not in self.media:
            raise RuntimeError(f"unknown media {message.media_id}")
        return self.media[message.media_id]

    def clear(self) -> None:
        self.sent.clear()
        self.acknowledged.clear()
