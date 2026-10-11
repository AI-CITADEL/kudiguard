import hashlib
import hmac
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from kudiguard.channels.whatsapp import WhatsAppConfig
from kudiguard.channels.whatsapp_routes import whatsapp_router
from kudiguard.domain import InboundMessage

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "whatsapp"
CONFIG = WhatsAppConfig(
    phone_number_id="PNID123", access_token="t", app_secret="app-secret", verify_token="verify-me"
)
PATH = "/webhooks/whatsapp"


class Recorder:
    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.messages: list[InboundMessage] = []

    def submit(self, message: InboundMessage) -> bool:
        if self.accept:
            self.messages.append(message)
        return self.accept


def client_for(recorder: Recorder, **kwargs: int) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(whatsapp_router(CONFIG, recorder, **kwargs))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def signed(body: bytes) -> dict[str, str]:
    digest = hmac.new(b"app-secret", body, hashlib.sha256).hexdigest()
    return {"x-hub-signature-256": f"sha256={digest}", "content-type": "application/json"}


def fresh_payload() -> bytes:
    """The recorded fixture with current timestamps (old deliveries are skipped by design)."""
    payload = json.loads((FIXTURES / "inbound_mixed.json").read_text(encoding="utf-8"))
    for message in payload["entry"][0]["changes"][0]["value"]["messages"]:
        message["timestamp"] = str(int(time.time()))
    return json.dumps(payload).encode()


async def test_subscription_handshake() -> None:
    async with client_for(Recorder()) as client:
        ok = await client.get(
            PATH, params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "42"}
        )
        bad = await client.get(
            PATH, params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "42"}
        )
    assert ok.status_code == 200 and ok.text == "42"
    assert bad.status_code == 403


async def test_signed_webhook_is_accepted_and_submitted() -> None:
    recorder = Recorder()
    body = fresh_payload()
    async with client_for(recorder) as client:
        response = await client.post(PATH, content=body, headers=signed(body))
    assert response.status_code == 200 and response.json() == {"ok": True}
    assert (
        len(recorder.messages) == 7 and recorder.messages[0].message_id == "wamid.TEXT1"
    )  # equal timestamps: payload order


async def test_bad_signature_is_rejected_before_parsing() -> None:
    recorder = Recorder()
    body = fresh_payload()
    async with client_for(recorder) as client:
        unsigned = await client.post(PATH, content=body)
        forged = await client.post(PATH, content=body + b" ", headers=signed(body))
    assert unsigned.status_code == 401 and forged.status_code == 401 and recorder.messages == []


async def test_invalid_json_and_oversized_bodies() -> None:
    async with client_for(Recorder(), max_body_bytes=100) as client:
        garbage = b"{not json"
        assert (await client.post(PATH, content=garbage, headers=signed(garbage))).status_code == 400
        big = b"x" * 101
        assert (await client.post(PATH, content=big, headers=signed(big))).status_code == 413


async def test_oversized_body_without_content_length_header() -> None:
    async def stream():  # type: ignore[no-untyped-def]
        yield b"x" * 60
        yield b"x" * 60

    async with client_for(Recorder(), max_body_bytes=100) as client:
        response = await client.post(PATH, content=stream())
    assert response.status_code == 413


async def test_shutting_down_asks_meta_to_retry() -> None:
    body = fresh_payload()
    async with client_for(Recorder(accept=False)) as client:
        response = await client.post(PATH, content=body, headers=signed(body))
    assert response.status_code == 503


async def test_hostile_signed_bodies_never_cause_a_500() -> None:
    deep = b"[" * 100_000 + b"]" * 100_000
    odd = json.dumps({"object": "whatsapp_business_account", "entry": [{"changes": 5}]}).encode()
    async with client_for(Recorder(), max_body_bytes=1_000_000) as client:
        assert (await client.post(PATH, content=deep, headers=signed(deep))).status_code == 400
        assert (await client.post(PATH, content=odd, headers=signed(odd))).status_code == 200


async def test_failed_delivery_status_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    # A real wamid is base64 of a protobuf that embeds the recipient's number (privacy.py).
    wamid = "wamid.HBgNMjM0ODAzMTIzNDU2NxUCABEYEjAzQkE5RTQ2"
    payload = json.loads((FIXTURES / "statuses_only.json").read_text(encoding="utf-8"))
    payload["entry"][0]["changes"][0]["value"]["statuses"].append(
        {
            "id": wamid,
            "status": "failed",
            "recipient_id": "2348031234567",
            "errors": [{"code": 131047, "title": "Re-engagement message"}],
        }
    )
    body = json.dumps(payload).encode()
    async with client_for(Recorder()) as client:
        assert (await client.post(PATH, content=body, headers=signed(body))).status_code == 200
    # The title identifies the failure; the 6-digit code itself may be masked by log redaction (records are
    # redacted at creation once any app has started, kudiguard.text.install_log_redaction).
    assert any(
        "could not deliver" in r.getMessage() and "Re-engagement message" in r.getMessage()
        for r in caplog.records
    )
    logged = caplog.text + " ".join(str(r.args) for r in caplog.records)
    for secret in (wamid, "HBgNMjM0ODAz", "2348031234567", "8031234567"):
        assert secret not in logged
