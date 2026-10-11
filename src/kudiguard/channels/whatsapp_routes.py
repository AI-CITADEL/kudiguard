"""FastAPI routes for the WhatsApp webhook (mounted by the app factory in ``kudiguard.api``).

GET: Meta's subscription handshake. POST: verify the signature over the *raw* body, parse, hand each message to the
conversation in the background and answer 200 at once (Meta retries slow or failed webhooks, and duplicates are
dropped by ``Store.claim_message``). While shutting down, answer 503 so Meta redelivers to a running instance.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from kudiguard.channels.whatsapp import (
    WhatsAppConfig,
    parse_failed_deliveries,
    parse_webhook,
    verify_signature,
    verify_subscription,
)
from kudiguard.domain import InboundMessage

log = logging.getLogger(__name__)

MAX_WEBHOOK_BYTES = 1_000_000  # real payloads are a few KB; anything far larger is abuse


class Submitter(Protocol):
    def submit(self, message: InboundMessage) -> bool: ...


def whatsapp_router(
    config: WhatsAppConfig,
    conversation: Submitter,
    *,
    path: str = "/webhooks/whatsapp",
    max_body_bytes: int = MAX_WEBHOOK_BYTES,
) -> APIRouter:
    router = APIRouter(tags=["whatsapp"])

    @router.get(path, include_in_schema=False)
    async def subscribe(request: Request) -> Response:
        challenge = verify_subscription(dict(request.query_params), config.verify_token)
        if challenge is None:
            return Response(status_code=403)
        return PlainTextResponse(challenge)

    @router.post(path, include_in_schema=False)
    async def receive(request: Request) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_body_bytes:
            return Response(status_code=413)
        body = bytearray()
        # Bounded read: an unauthenticated sender cannot make us buffer more than the cap (chunked bodies too).
        async for chunk in request.stream():
            body += chunk
            if len(body) > max_body_bytes:
                return Response(status_code=413)
        raw = bytes(body)
        if not verify_signature(raw, request.headers.get("x-hub-signature-256"), config.app_secret):
            log.warning("rejected WhatsApp webhook with a bad signature")
            return Response(status_code=401)
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return Response(status_code=400)
        for failure in parse_failed_deliveries(payload, phone_number_id=config.phone_number_id):
            # Never the wamid (it encodes the recipient's number, see privacy.py) nor the recipient: code and title only.
            log.warning("WhatsApp could not deliver a reply: code %s %s", failure.code, failure.title)
        for message in parse_webhook(payload, phone_number_id=config.phone_number_id):
            if not conversation.submit(message):
                return Response(status_code=503)  # shutting down: Meta retries the delivery later
        return JSONResponse({"ok": True})

    return router
