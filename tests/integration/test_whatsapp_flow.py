"""End to end over the WhatsApp channel: signed webhook → pipeline → Graph API (mocked), with the real store,
rules, knowledge base and speech decoding; only the model and the speech recogniser are fakes."""

import asyncio
import hashlib
import hmac
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI

from fakes import FakeASR, FakeLLM, analysis
from kudiguard.channels.whatsapp import WhatsAppConfig, WhatsAppMessenger
from kudiguard.channels.whatsapp_routes import whatsapp_router
from kudiguard.engine import Engine
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.pipeline import Conversation
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.speech.audio import find_ffmpeg
from kudiguard.storage.sqlite import SQLiteStore

ROOT = Path(__file__).resolve().parents[2]
CONFIG = WhatsAppConfig(phone_number_id="PNID123", access_token="tok", app_secret="sec", verify_token="v")
GRAPH = f"https://graph.facebook.com/{CONFIG.api_version}"
USER = "2348000000001"
OTP_SCAM = "Someone called and asked me to send the OTP sent to my phone. Is this a scam?"


@dataclass
class Stack:
    web: httpx.AsyncClient
    conversation: Conversation
    store: SQLiteStore
    sent: respx.Route
    media_lookup: respx.Route
    counter: int = 0

    async def deliver(self, kind: str, body: dict[str, Any]) -> None:
        self.counter += 1
        message = {"from": USER, "id": f"wamid.E2E{self.counter}", "timestamp": str(int(time.time())), "type": kind,
                   kind: body}  # fmt: skip
        payload = {"object": "whatsapp_business_account", "entry": [{"id": "W", "changes": [{"field": "messages",
                   "value": {"metadata": {"phone_number_id": "PNID123"}, "messages": [message]}}]}]}  # fmt: skip
        raw = json.dumps(payload).encode()
        signature = "sha256=" + hmac.new(b"sec", raw, hashlib.sha256).hexdigest()
        response = await self.web.post(
            "/webhooks/whatsapp", content=raw, headers={"x-hub-signature-256": signature}
        )
        assert response.status_code == 200
        await self.conversation.wait_idle()

    def outbound(self) -> list[dict[str, Any]]:
        return [json.loads(c.request.content) for c in self.sent.calls]


@pytest.fixture
async def stack(registry: LanguageRegistry, tmp_path: Path) -> AsyncIterator[Stack]:
    hasher = UserHasher(b"e" * MIN_SECRET_BYTES)
    with respx.mock(assert_all_called=False) as graph:
        sent = graph.post(f"{GRAPH}/PNID123/messages").mock(return_value=httpx.Response(200, json={}))
        media_lookup = graph.get(f"{GRAPH}/MEDIA1").mock(
            return_value=httpx.Response(
                200, json={"url": "https://lookaside.fbsbx.com/media1", "file_size": 20000}
            )
        )
        note = (ROOT / "tests" / "fixtures" / "voice_note_en_otp.ogg").read_bytes()
        graph.get("https://lookaside.fbsbx.com/media1").mock(return_value=httpx.Response(200, content=note))
        graph.route(host="test").pass_through()
        async with httpx.AsyncClient() as http:
            knowledge = knowledge_base_from_packs(
                ROOT / "data" / "packs", client=http, include=["kudiguard-core"]
            )
            llm = FakeLLM(analyse=analysis("x", "scam_check"), compose="1. Do not share the code.")
            engine = Engine(
                registry=registry, llm=llm, knowledge=knowledge, asr=FakeASR(OTP_SCAM), ffmpeg=find_ffmpeg()
            )
            store = await SQLiteStore.open(tmp_path / "kg.db", hasher=hasher)
            conversation = Conversation(
                engine=engine,
                store=store,
                messenger=WhatsAppMessenger(CONFIG, client=http, backoff_s=0),
                hasher=hasher,
            )
            app = FastAPI()
            app.include_router(whatsapp_router(CONFIG, conversation))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as web:
                yield Stack(web, conversation, store, sent, media_lookup)
            await conversation.drain()
            await store.close()


async def test_voice_note_end_to_end(stack: Stack) -> None:
    await stack.deliver("text", {"body": "Hello"})  # first contact: welcome + language menu
    await stack.deliver(
        "interactive", {"type": "list_reply", "list_reply": {"id": "lang:en", "title": "English"}}
    )
    before = len(stack.outbound())
    await stack.deliver("audio", {"id": "MEDIA1", "mime_type": "audio/ogg; codecs=opus", "voice": True})

    out = stack.outbound()[before:]
    receipt = next(p for p in out if p.get("status") == "read")
    assert receipt["typing_indicator"] == {"type": "text"}
    texts = [p for p in out if p.get("type") == "text"]
    interactive = [p for p in out if p.get("type") == "interactive"]
    assert texts[0]["text"]["body"] == "🎧 Listening…"
    reply = next(p for p in out if "SCAM" in json.dumps(p, ensure_ascii=False))
    body = reply["text"]["body"] if reply["type"] == "text" else reply["interactive"]["body"]["text"]
    assert "I heard" in body and "Where to report" in body
    button_ids = [
        b["reply"]["id"] for p in interactive for b in p["interactive"]["action"].get("buttons", [])
    ]
    assert any(i.startswith("steps:") for i in button_ids) and any(
        i.startswith("helpful:") for i in button_ids
    )
    assert all(p["to"] == USER for p in out if "to" in p)

    assert stack.media_lookup.call_count == 1  # fetched once, after consent
    rows = await stack.store.export_interactions()
    assert rows[-1]["verdict"] == "likely_scam" and rows[-1]["kind"] == "audio"
    assert USER not in json.dumps(rows) and "wamid.E2E" not in json.dumps(rows)


async def test_media_is_not_downloaded_before_consent(stack: Stack) -> None:
    await stack.deliver("audio", {"id": "MEDIA1", "mime_type": "audio/ogg; codecs=opus", "voice": True})
    assert stack.media_lookup.call_count == 0  # welcome + menu only; the voice note was never fetched
    assert any(p.get("type") == "interactive" for p in stack.outbound())


async def test_duplicate_webhook_delivery_is_answered_once(stack: Stack) -> None:
    await stack.deliver("text", {"body": "Hello"})
    before = len(stack.outbound())
    stack.counter -= 1  # same wamid again (Meta retry)
    await stack.deliver("text", {"body": "Hello"})
    await asyncio.sleep(0)
    assert len(stack.outbound()) == before
