"""WhatsApp Cloud API channel: webhook verification and parsing, sending, read receipts and lazy media download.

Webhook flow (``kudiguard.channels.whatsapp_routes``): verify the ``X-Hub-Signature-256`` HMAC of the raw body →
``parse_webhook`` → ``Conversation.submit`` per message → HTTP 200 immediately. Reactions and system notices are
ignored; media types we cannot use become ``UNSUPPORTED`` (the user gets a friendly reply); failed-delivery statuses
are reported by ``parse_failed_deliveries`` (Meta may report send errors only there).

Sending respects WhatsApp's limits, counted in UTF-16 units (conservative: emoji count 2): text ≤ 4096 (split at
paragraph, then line, then word breaks), reply-button body ≤ 1024 and list body ≤ 4096 (a longer reply goes out as
text followed by a short interactive message), ≤ 3 unique button titles ≤ 20, list rows ≤ 10 (title ≤ 24,
description ≤ 72).

Retries follow Meta's error *codes* (https://developers.facebook.com/docs/whatsapp/cloud-api/support/error-codes),
honour ``Retry-After``, and never risk a duplicate message: a POST is retried only when the request provably did not
reach Meta (connection errors) or Meta rejected it as throttled/temporarily unavailable. Idempotent GETs (media) are
retried on any transient failure. The access token is sent only to Meta hosts and never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import math
import os
import re
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from kudiguard.domain import InboundMessage, MessageKind
from kudiguard.pipeline import Outbound
from kudiguard.speech.audio import MAX_AUDIO_BYTES
from kudiguard.speech.base import AudioError, AudioProblem

log = logging.getLogger(__name__)

CHANNEL = "whatsapp"
# Graph API v25 (Feb 2026), supported for two years; newer versions work via WHATSAPP_API_VERSION.
DEFAULT_API_VERSION = "v25.0"
_API_VERSION = re.compile(r"v\d{1,3}\.\d{1,3}")
MAX_TEXT = 4096
MAX_BUTTON_BODY = 1024
MAX_LIST_BODY = 4096
MAX_BUTTONS = 3
MAX_BUTTON_TITLE = 20
MAX_ROWS = 10
MAX_ROW_TITLE = 24
MAX_ROW_DESCRIPTION = 72
MAX_MESSAGE_AGE_S = (
    24 * 3600
)  # older deliveries (replays after an outage) are skipped, not answered days late
MAX_RETRY_AFTER_S = 10.0
BUTTONS_ONLY_BODY = "👇"  # body of the interactive message when the reply itself is too long (or empty)
# Meta error codes worth retrying: throttling / temporary unavailability (the request was not processed).
RETRYABLE_CODES = frozenset({4, 80007, 130429, 131016, 131056, 133004})
# "Something went wrong": Meta may have processed it, so only idempotent requests are retried.
RETRYABLE_IF_IDEMPOTENT_CODES = frozenset({131000})
TOKEN_CODES = frozenset({190})  # expired/invalid access token: an operator must act
# Media download URLs are served by Meta's CDNs; never send the bearer token anywhere else.
MEDIA_HOST_SUFFIXES = (".fbsbx.com", ".facebook.com", ".fbcdn.net", ".whatsapp.net")
UNSUPPORTED_TYPES = frozenset(
    {"image", "video", "document", "sticker", "location", "contacts", "order", "unsupported", "unknown"}
)
IGNORED_TYPES = frozenset({"reaction", "system", "ephemeral", "request_welcome"})


class WhatsAppError(RuntimeError):
    def __init__(self, status: int | None, detail: str, code: int | None = None) -> None:
        super().__init__(f"WhatsApp API error {status} (code {code}): {detail}")
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class WhatsAppConfig:
    phone_number_id: str
    access_token: str
    app_secret: str
    verify_token: str
    api_version: str = DEFAULT_API_VERSION
    graph_url: str = "https://graph.facebook.com"

    def __post_init__(self) -> None:
        required = ("phone_number_id", "access_token", "app_secret", "verify_token")
        if missing := [name for name in required if not getattr(self, name)]:
            raise ValueError(f"WhatsApp config is missing {missing}")
        if not _API_VERSION.fullmatch(self.api_version):  # it becomes part of every Graph API URL
            raise ValueError(f"WHATSAPP_API_VERSION must look like v25.0, not {self.api_version!r}")

    def __repr__(self) -> str:  # never print secrets
        return f"WhatsAppConfig(phone_number_id={self.phone_number_id!r}, api_version={self.api_version!r})"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> WhatsAppConfig:
        return cls(
            phone_number_id=env.get("WHATSAPP_PHONE_NUMBER_ID", ""),
            access_token=env.get("WHATSAPP_ACCESS_TOKEN", ""),
            app_secret=env.get("WHATSAPP_APP_SECRET", ""),
            verify_token=env.get("WHATSAPP_VERIFY_TOKEN", ""),
            api_version=env.get("WHATSAPP_API_VERSION") or DEFAULT_API_VERSION,
        )

    @property
    def base(self) -> str:
        return f"{self.graph_url.rstrip('/')}/{self.api_version}"


# --- webhook verification --------------------------------------------------------------------------------


def verify_signature(body: bytes, header: str | None, app_secret: str) -> bool:
    """``X-Hub-Signature-256: sha256=<hex HMAC-SHA256 of the raw body with the app secret>`` (constant time)."""
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256=").strip().lower())


def verify_subscription(params: Mapping[str, str], verify_token: str) -> str | None:
    """Meta's GET handshake: return ``hub.challenge`` when the mode and token match, else None (→ HTTP 403)."""
    token = params.get("hub.verify_token", "")
    if params.get("hub.mode") == "subscribe" and hmac.compare_digest(token.encode(), verify_token.encode()):
        return params.get("hub.challenge", "")
    return None


# --- webhook parsing -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeliveryFailure:
    """A ``statuses[].status == "failed"`` callback: one of our messages was not delivered."""

    message_id: str  # our outbound wamid
    code: int | None
    title: str


def parse_webhook(
    payload: Any,
    *,
    phone_number_id: str | None = None,
    now: float | None = None,
    max_age_s: float = MAX_MESSAGE_AGE_S,
) -> list[InboundMessage]:
    """Messages addressed to our number, oldest first. Never raises: malformed items are skipped and logged."""
    current = time.time() if now is None else now
    found: list[tuple[int, InboundMessage]] = []
    try:
        for value in _values(payload, phone_number_id):
            for raw in _items(value.get("messages")):
                parsed = _parse_message(raw, current, max_age_s)
                if parsed is not None:
                    found.append(parsed)
    except Exception:  # a hostile-but-signed body must not turn into a 500 that Meta redelivers for days
        log.exception("malformed WhatsApp webhook payload; ignored")
        return []
    found.sort(key=lambda item: item[0])
    return [message for _, message in found]


def parse_failed_deliveries(payload: Any, *, phone_number_id: str | None = None) -> list[DeliveryFailure]:
    """Failed-delivery statuses (e.g. 131047 outside the 24-hour window, 131026 undeliverable). Never raises."""
    failures: list[DeliveryFailure] = []
    try:
        for value in _values(payload, phone_number_id):
            for status in _items(value.get("statuses")):
                if not isinstance(status, dict) or status.get("status") != "failed":
                    continue
                error = next((e for e in _items(status.get("errors")) if isinstance(e, dict)), {})
                code = error.get("code") if isinstance(error.get("code"), int) else None
                failures.append(
                    DeliveryFailure(str(status.get("id", "")), code, str(error.get("title", ""))[:200])
                )
    except Exception:
        log.exception("malformed WhatsApp status payload; ignored")
        return []
    return failures


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _values(payload: Any, phone_number_id: str | None) -> Iterator[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("object") != "whatsapp_business_account":
        return
    for entry in _items(payload.get("entry")):
        if not isinstance(entry, dict):
            continue
        for change in _items(entry.get("changes")):
            if not (isinstance(change, dict) and change.get("field") == "messages"):
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                continue
            if phone_number_id:
                metadata = value.get("metadata")
                if not isinstance(metadata, dict) or metadata.get("phone_number_id") != phone_number_id:
                    continue  # another number on the same app, or no way to tell
            yield value


def _timestamp(raw: Any, now: float) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return int(now)
    return value if 0 < value < now + 86400 else int(now)


def _forwarded(raw: dict[str, Any]) -> bool:
    """``context.forwarded`` / ``context.frequently_forwarded``. Meta documents these for text, image and video,
    not audio, so this is a safety net: donation also requires ``audio.voice`` (recorded in the app)."""
    context = raw.get("context")
    return isinstance(context, dict) and (
        context.get("forwarded") is True or context.get("frequently_forwarded") is True
    )


def _parse_message(raw: Any, now: float, max_age_s: float) -> tuple[int, InboundMessage] | None:
    if not isinstance(raw, dict):
        return None
    message_id, sender, kind = raw.get("id"), raw.get("from"), raw.get("type")
    if not (isinstance(message_id, str) and message_id and isinstance(sender, str) and sender):
        log.warning("skipping malformed WhatsApp message (missing id/from)")
        return None
    if not isinstance(kind, str):
        log.warning("skipping malformed WhatsApp message (missing type)")
        return None
    sent_at = _timestamp(raw.get("timestamp", now), now)
    if now - sent_at > max_age_s:
        log.info("skipping stale WhatsApp message (%.0f h old)", (now - sent_at) / 3600)
        return None
    if kind in IGNORED_TYPES:
        return None

    def make(kind_: MessageKind, **fields: Any) -> tuple[int, InboundMessage]:
        return sent_at, InboundMessage(CHANNEL, message_id, sender, kind_, sent_at=float(sent_at), **fields)

    section = raw.get(kind)
    body: dict[str, Any] = section if isinstance(section, dict) else {}
    if kind == "text" and isinstance(body.get("body"), str):
        return make(MessageKind.TEXT, text=body["body"])
    if kind == "audio" and isinstance(body.get("id"), str):
        mime = body.get("mime_type") if isinstance(body.get("mime_type"), str) else None
        voice_note, forwarded = body.get("voice") is True, _forwarded(raw)
        log.info("voice message: recorded=%s forwarded=%s", voice_note, forwarded)  # flags only, no content
        return make(
            MessageKind.AUDIO,
            media_id=body["id"],
            audio_mime=mime,
            voice_note=voice_note,
            forwarded=forwarded,
        )
    if kind == "interactive":
        reply = body.get("button_reply") or body.get("list_reply")
        if isinstance(reply, dict) and isinstance(reply.get("id"), str):
            return make(MessageKind.BUTTON, button_id=reply["id"])
    if kind == "button" and isinstance(body.get("payload"), str):  # template quick-reply button
        return make(MessageKind.BUTTON, button_id=body["payload"])
    if kind not in UNSUPPORTED_TYPES:
        log.info("unknown WhatsApp message type %r treated as unsupported", kind)
    return make(MessageKind.UNSUPPORTED)


# --- outbound formatting ---------------------------------------------------------------------------------


def units(text: str) -> int:
    """Length in UTF-16 code units (emoji count 2): never under-counts against WhatsApp's character limits."""
    return len(text.encode("utf-16-le")) // 2


def _prefix_within(text: str, limit: int) -> int:
    """Number of code points of ``text`` whose UTF-16 length fits ``limit``."""
    total = 0
    for index, char in enumerate(text):
        total += 2 if ord(char) > 0xFFFF else 1
        if total > limit:
            return index
    return len(text)


def split_text(text: str, limit: int = MAX_TEXT) -> list[str]:
    """Split so each part fits ``limit`` UTF-16 units, preferring paragraph, then line, then word breaks."""
    if limit < 2:  # an emoji needs 2 units: a smaller limit could never make progress
        raise ValueError("limit must be at least 2 UTF-16 units")
    parts: list[str] = []
    rest = text.strip()
    while units(rest) > limit:
        window = rest[: _prefix_within(rest, limit)]
        cut = 0
        for separator in ("\n\n", "\n", " "):
            position = window.rfind(separator)
            if position > len(window) // 2:
                cut = position
                break
        cut = cut or len(window)
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts


def _clip(text: str, limit: int) -> str:
    return text if units(text) <= limit else text[: _prefix_within(text, limit - 1)].rstrip() + "…"


def _unique_titles(titles: list[str], limit: int) -> list[str]:
    """Clip titles to ``limit``; Meta rejects duplicate button titles, so numbered suffixes break ties."""
    seen: set[str] = set()
    result: list[str] = []
    for n, title in enumerate(titles, start=1):
        clipped = _clip(title, limit)
        if clipped in seen:
            suffix = f" {n}"
            clipped = _clip(title, limit - len(suffix)) + suffix
        seen.add(clipped)
        result.append(clipped)
    return result


def _text_payloads(envelope: dict[str, str], text: str) -> list[dict[str, Any]]:
    return [
        {**envelope, "type": "text", "text": {"body": part, "preview_url": False}}
        for part in split_text(text)
    ]


def build_payloads(to: str, message: Outbound) -> list[dict[str, Any]]:
    """Graph API message bodies for one ``Outbound`` (pure; limits enforced here)."""
    envelope = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to}
    text = message.text.strip()
    interactive: dict[str, Any] | None = None
    body_limit = MAX_BUTTON_BODY
    if message.buttons:
        buttons = message.buttons[:MAX_BUTTONS]
        titles = _unique_titles([b.title for b in buttons], MAX_BUTTON_TITLE)
        interactive = {
            "type": "button",
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": b.id[:256], "title": title}}
                    for b, title in zip(buttons, titles, strict=True)
                ]
            },
        }
    elif message.options:
        body_limit = MAX_LIST_BODY
        rows = []
        for option in message.options[:MAX_ROWS]:
            row = {"id": option.id[:200], "title": _clip(option.title, MAX_ROW_TITLE)}
            if option.description:
                row["description"] = _clip(option.description, MAX_ROW_DESCRIPTION)
            rows.append(row)
        interactive = {
            "type": "list",
            "action": {
                "button": _clip(message.menu_button or "Menu", MAX_BUTTON_TITLE),
                "sections": [{"rows": rows}],
            },
        }
    if interactive is None:
        return _text_payloads(envelope, text)
    if text and units(text) <= body_limit:
        return [{**envelope, "type": "interactive", "interactive": {**interactive, "body": {"text": text}}}]
    leading = _text_payloads(envelope, text) if text else []
    closing = {
        **envelope,
        "type": "interactive",
        "interactive": {**interactive, "body": {"text": BUTTONS_ONLY_BODY}},
    }
    return [*leading, closing]


# --- the Messenger -------------------------------------------------------------------------------------------


def _retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    try:
        seconds = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None  # absent, or an HTTP-date (not used by Meta): fall back to our backoff
    return max(0.0, min(seconds, MAX_RETRY_AFTER_S)) if math.isfinite(seconds) else None


class WhatsAppMessenger:
    """``pipeline.Messenger`` over the WhatsApp Cloud API."""

    def __init__(
        self,
        config: WhatsAppConfig,
        *,
        client: httpx.AsyncClient,
        timeout_s: float = 10.0,
        retries: int = 2,
        backoff_s: float = 0.5,
        max_media_bytes: int = MAX_AUDIO_BYTES,
    ) -> None:
        self._config = config
        self._client = client
        self._timeout = httpx.Timeout(timeout_s)
        self._retries = retries
        self._backoff_s = backoff_s
        self._max_media_bytes = max_media_bytes
        self._auth = {"Authorization": f"Bearer {config.access_token}"}

    async def acknowledge(self, message: InboundMessage) -> None:
        """Blue ticks + typing indicator (shown until our reply, or up to 25 s)."""
        await self._post(
            {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message.message_id,
                "typing_indicator": {"type": "text"},
            }
        )

    async def send(self, user_id: str, message: Outbound) -> None:
        for payload in build_payloads(user_id, message):
            await self._post(payload)

    async def fetch_media(self, message: InboundMessage) -> bytes:
        """Look up the media URL and download it. A URL that expired between lookup and download is looked up once
        more (Meta's guidance); media that is gone or unusable is an ``AudioError`` (user is asked to resend)."""
        if not message.media_id:
            raise AudioError(AudioProblem.EMPTY, "no media id")
        for attempt in range(2):
            url = await self._media_url(message.media_id)
            try:
                return await self._download(url)
            except WhatsAppError as exc:
                if exc.status in (403, 404, 410) and attempt == 0:
                    log.info("media URL expired; looking it up again")
                    continue
                if exc.status in (400, 403, 404, 410):
                    raise AudioError(AudioProblem.UNDECODABLE, f"media unavailable ({exc.status})") from exc
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _media_url(self, media_id: str) -> str:
        try:
            meta = await self._request(
                "GET",
                f"{self._config.base}/{media_id}",
                params={"phone_number_id": self._config.phone_number_id},
            )
        except WhatsAppError as exc:
            if exc.status in (400, 404):  # unknown or deleted media id
                raise AudioError(AudioProblem.UNDECODABLE, f"media lookup failed ({exc.status})") from exc
            raise
        url, size = meta.get("url"), meta.get("file_size")
        if not isinstance(url, str):
            raise WhatsAppError(None, "media lookup returned no url")
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.scheme != "https" or not host.endswith(MEDIA_HOST_SUFFIXES):
            raise WhatsAppError(None, f"refusing media URL on unexpected host {host!r}")
        if isinstance(size, int) and size > self._max_media_bytes:
            raise AudioError(AudioProblem.TOO_LARGE, f"{size} bytes")
        return url

    # --- HTTP ---------------------------------------------------------------------------------------------

    async def _post(self, payload: dict[str, Any]) -> None:
        await self._request(
            "POST", f"{self._config.base}/{self._config.phone_number_id}/messages", json=payload
        )

    async def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        idempotent = method == "GET"
        for attempt in range(self._retries + 1):
            response: httpx.Response | None = None
            try:
                response = await self._client.request(
                    method, url, headers=self._auth, timeout=self._timeout, **kwargs
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                if attempt == self._retries:  # never reached Meta: safe to retry even a send
                    raise WhatsAppError(None, f"network: {type(exc).__name__}") from exc
            except httpx.TransportError as exc:
                # The request may have reached Meta: retrying a send could deliver the message twice.
                if not idempotent or attempt == self._retries:
                    raise WhatsAppError(None, f"network: {type(exc).__name__}") from exc
            else:
                if response.status_code < 300:
                    try:
                        data = response.json()
                    except ValueError:
                        return {}
                    return data if isinstance(data, dict) else {}
                error = _error_of(response)
                if error.code in TOKEN_CODES:
                    log.error(
                        "WhatsApp access token rejected (code %s): renew WHATSAPP_ACCESS_TOKEN", error.code
                    )
                if not self._retryable(error, idempotent) or attempt == self._retries:
                    raise error
                log.warning("WhatsApp API %s (code %s), retrying", error.status, error.code)
            await asyncio.sleep(_retry_after(response) or self._backoff_s * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _retryable(error: WhatsAppError, idempotent: bool) -> bool:
        if error.code is not None:
            return error.code in RETRYABLE_CODES or (
                idempotent and error.code in RETRYABLE_IF_IDEMPOTENT_CODES
            )
        if error.status in (429, 503):  # throttled / unavailable: Meta did not process it
            return True
        return idempotent and error.status in (500, 502, 504)

    async def _download(self, url: str) -> bytes:
        for attempt in range(self._retries + 1):
            response_status: int | None = None
            try:
                async with self._client.stream(
                    "GET", url, headers=self._auth, timeout=self._timeout
                ) as response:
                    response_status = response.status_code
                    if response.status_code >= 300:
                        error = _error_of(response, body=await response.aread())
                        if response.status_code not in (429, 500, 502, 503, 504) or attempt == self._retries:
                            raise error
                    else:
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > self._max_media_bytes:
                                raise AudioError(AudioProblem.TOO_LARGE, f"> {self._max_media_bytes} bytes")
                            chunks.append(chunk)
                        return b"".join(chunks)
            except httpx.TransportError as exc:
                if attempt == self._retries:
                    raise WhatsAppError(response_status, f"media download: {type(exc).__name__}") from exc
            await asyncio.sleep(self._backoff_s * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover


def _error_of(response: httpx.Response, *, body: bytes | None = None) -> WhatsAppError:
    """Meta's ``{"error": {"message", "code"}}`` → WhatsAppError (message only; never request headers)."""
    code: int | None = None
    detail = response.reason_phrase
    try:
        data = response.json() if body is None else httpx.Response(200, content=body).json()
        error = data.get("error", {}) if isinstance(data, dict) else {}
        if isinstance(error, dict):
            detail = str(error.get("message", detail))[:200]
            code = error.get("code") if isinstance(error.get("code"), int) else None
    except ValueError:
        pass
    return WhatsAppError(response.status_code, detail, code)
