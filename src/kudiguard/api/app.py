"""FastAPI application: public REST API (``/v1``), web voice page, WhatsApp webhook, health checks.

    kudiguard serve            (or: uvicorn "kudiguard.api.app:create_app" --factory)

The REST API is stateless for callers (no phone numbers needed) but every check is logged as redacted, pseudonymous
evidence (channel ``api`` / ``web``). The web page is a public client of the same engine, rate-limited per client IP;
the API requires an ``X-API-Key`` and is rate-limited per key.

Partner API and the N-ATLaS licence: ``/v1`` is mounted only when ``KUDIGUARD_PARTNER_API`` is on (off in the pilot).
Not mounting it (rather than answering 404 from live routes) keeps it out of ``/docs`` and leaves nothing to probe.
When on, every check names its end user with a pseudonymous ``X-KudiGuard-End-User`` id (never a phone number); the id
is hashed in its own domain (``api:<key name>:<id>``) and counts toward the licence cap of active end-users per
rolling 30 days, shared with WhatsApp and the website (one admission lock, so a burst cannot overshoot).

Request hygiene (review M9): bodies over ``MAX_REQUEST_BYTES`` are refused before any parsing; authentication,
consent and rate limits are checked *before* a multipart form is read; validation errors never echo the input back.
"""

from __future__ import annotations

import asyncio
import hmac
import html
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from kudiguard.admission import AtCapacity, Reservation
from kudiguard.api.ratelimit import RateLimiter
from kudiguard.api.services import Services, build_services
from kudiguard.config import ROOT, Settings
from kudiguard.domain import InboundMessage, Interaction, MessageKind, Outcome
from kudiguard.donation import DONATION_VERSION, Donation
from kudiguard.engine import Answer, AnswerKind, crisis_message
from kudiguard.envfile import load_env_file
from kudiguard.hosting import StorageLocation
from kudiguard.i18n import VERDICT_HEADLINE, Msg
from kudiguard.pipeline import CONSENT_VERSION, record_pattern
from kudiguard.render import plain_text
from kudiguard.safety.abuse import CRISES
from kudiguard.speech.audio import MAX_AUDIO_BYTES, MAX_SECONDS
from kudiguard.speech.base import ASRError, AudioError, AudioProblem
from kudiguard.storage.base import DELETION_MARKER_DAYS, utcnow

log = logging.getLogger(__name__)

STATIC = Path(__file__).with_name("static")
API_DEADLINE_S = 45.0  # model work per check; the engine skips model steps that no longer fit
REQUEST_TIMEOUT_S = 75.0  # whole check incl. speech: beyond this the caller gets 504
QUEUE_TIMEOUT_S = 15.0  # waiting for a free check slot: beyond this the caller gets 503 (retry later)
MAX_TEXT_CHARS = 4000
MAX_CONCURRENT_API_CHECKS = 8
MAX_REQUEST_BYTES = MAX_AUDIO_BYTES + 1024 * 1024  # one voice note plus form fields
SMALL_REQUEST_BYTES = 64 * 1024  # JSON endpoints: a 4000-char message is ~16 KB even in UTF-8
# Paths that legitimately carry larger bodies (the WhatsApp webhook enforces its own 1 MB cap).
LARGE_BODY_PATHS = {"/v1/check/audio": MAX_REQUEST_BYTES, "/web/check": MAX_REQUEST_BYTES,
                    "/webhooks/whatsapp": 2 * 1024 * 1024}  # fmt: skip
SESSION_PATTERN = r"^[A-Za-z0-9_-]{16,64}$"
# The licence-cap reservation of the request being served (set by ``_admitted``): kept if its log cannot be written.
_RESERVATION: ContextVar[Reservation] = ContextVar("kudiguard_reservation")
END_USER_HEADER = "X-KudiGuard-End-User"
_END_USER = re.compile(r"[A-Za-z0-9._:-]{8,137}")  # 137 = a short tag + a SHA-512 hex digest
# A guardrail against mistakes, not proof of pseudonymity: ids that look like a phone number or an email address are
# refused. Hex digests (32/40/64/128 hex characters) and UUIDs, optionally behind a short letter tag ("c-<sha256>"), are
# accepted as they are: what partners should send. Anything else may not hold a run of 10+ digits once the separators
# are removed (catches "user-0803-123-4567", "x8031234567", "cust-447911123456"), nor be digits only.
_HASHED_ID = re.compile(
    r"(?i)(?:[a-z]{1,8}[._:-]?)?(?:[0-9a-f]{128}|[0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{32}"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_ONLY_DIGITS = re.compile(r"[\d._:-]+")
_SEPARATORS = re.compile(r"[._:-]")
_LONG_DIGITS = re.compile(r"\d{10,}")
_EMAIL_LIKE = re.compile(
    r"(?i)(?:^|[._:-])(?:gmail|googlemail|yahoo|ymail|hotmail|outlook|icloud|aol|proton|protonmail)(?:[._:-]|$)"
    r"|[._-](?:com|net|org|ng|edu|gov|io|co)$"
)


def contact_shaped(value: str) -> bool:
    """True for an end-user id that looks like a phone number or an email address (see the guardrail above)."""
    if _HASHED_ID.fullmatch(value):
        return False
    if _ONLY_DIGITS.fullmatch(value):
        return True
    return bool(_LONG_DIGITS.search(_SEPARATORS.sub("", value)) or _EMAIL_LIKE.search(value))


# --- schemas --------------------------------------------------------------------------------------------------


class CheckRequest(BaseModel):
    text: str = Field(
        min_length=1, max_length=MAX_TEXT_CHARS, description="The message to check, or a question"
    )
    language: str = Field("en", description="ISO code from GET /v1/languages (ha, yo, ig, en)")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    language: str = "en"


class ReportChannel(BaseModel):
    label: str
    kind: str
    value: str


class CheckResponse(BaseModel):
    verdict: Literal["likely_scam", "caution", "no_flags", "not_a_check"]
    headline: str | None = Field(description="Fixed, reviewed verdict text in the requested language")
    explanation: str = Field(description="Grounded explanation and next steps in the requested language")
    flags: list[str]
    actions: list[str] = Field(description="Next steps from the knowledge base (English source text)")
    report_channels: list[ReportChannel]
    transcript: str | None = None
    english: str | None = None
    language: str
    degraded: bool = Field(description="True when the model was unavailable and fixed text was used")
    disclaimer: str
    latency_ms: int


class AskResponse(BaseModel):
    answer: str
    verdict: str
    language: str
    sources: list[str]
    degraded: bool
    disclaimer: str
    latency_ms: int


AUDIO_FORM = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["audio"],
                    "properties": {
                        "audio": {"type": "string", "format": "binary",
                                  "description": "Voice note: ogg/opus, m4a, mp3, wav, webm; ≤ 16 MB, ≤ 90 s of speech"},
                        "language": {"type": "string", "default": "en"},
                    },
                }
            }
        },
    }
}  # fmt: skip


# --- request size guard -------------------------------------------------------------------------------------------


class BodySizeLimit:
    """ASGI middleware: refuse bodies over the path's limit before any parsing (declared or streamed/chunked).

    Small by default (JSON endpoints are read before the API key is checked, by FastAPI's design); larger only on
    the upload and webhook paths, which check credentials before reading the body.
    """

    def __init__(self, app: ASGIApp, limit: int, large_paths: dict[str, int] | None = None) -> None:
        self.app = app
        self.default = limit
        self.large_paths = large_paths or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.large_paths.get(scope.get("path", ""), self.default)
        declared = dict(scope.get("headers") or ()).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await _too_large(send)
            return
        received = 0
        started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await _too_large(send)


class _BodyTooLarge(BaseException):
    """BaseException on purpose: FastAPI turns ordinary exceptions raised while reading a JSON body into a 400;
    this must reach the middleware so the client gets 413."""


async def _too_large(send: Send) -> None:
    body = b'{"detail":"request body too large"}'
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})  # fmt: skip


# --- application ----------------------------------------------------------------------------------------------


class _ConversationProxy:
    """Lets the WhatsApp router be mounted at creation time while services start in the lifespan."""

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def submit(self, message: InboundMessage) -> bool:
        services: Services | None = getattr(self._app.state, "services", None)
        if services is None or services.conversation is None:
            return False  # not ready: 503, Meta retries
        return services.conversation.submit(message)


def create_app(
    settings: Settings | None = None,
    *,
    services_factory: Callable[[Settings], Awaitable[Services]] = build_services,
) -> FastAPI:
    if settings is None:  # production entry point (also `uvicorn --factory`): WHATSAPP_* etc. live in .env
        load_env_file(ROOT / ".env")
        settings = Settings()
    config = settings
    api_keys = config.api_key_map()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = await services_factory(config)
        app.state.services = services
        try:
            yield
        finally:
            app.state.services = None
            await services.aclose()

    app = FastAPI(
        title="KudiGuard API",
        version="1.0.0",
        summary="Scam checks and money-safety answers in Hausa, Yoruba, Igbo and English, powered by N-ATLAS.",
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimit, limit=SMALL_REQUEST_BYTES, large_paths=LARGE_BODY_PATHS)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.state.services = None
    app.state.settings = config
    app.state.api_limiter = RateLimiter(config.api_rate_per_minute)
    app.state.web_limiter = RateLimiter(config.web_rate_per_minute)
    app.state.check_slots = asyncio.Semaphore(MAX_CONCURRENT_API_CHECKS)
    app.state.api_keys = api_keys
    if config.partner_api:
        if not api_keys:
            log.warning(
                "KUDIGUARD_PARTNER_API is on but KUDIGUARD_API_KEYS is empty: every /v1 call is refused"
            )
        app.include_router(_v1_router())
    elif api_keys:
        log.warning(
            "KUDIGUARD_API_KEYS is set but the partner API is off (KUDIGUARD_PARTNER_API): /v1 not mounted"
        )
    app.include_router(_web_router())
    app.include_router(_health_router())

    from kudiguard.channels.whatsapp import WhatsAppConfig
    from kudiguard.channels.whatsapp_routes import whatsapp_router

    if not config.privacy_contact.strip():
        log.warning(
            "KUDIGUARD_PRIVACY_CONTACT is empty: /privacy offers no way to request access or deletion"
        )
    if config.whatsapp_enabled:
        try:
            app.include_router(whatsapp_router(WhatsAppConfig.from_env(), _ConversationProxy(app)))
        except ValueError:
            log.warning("WhatsApp webhook not mounted (WHATSAPP_* not configured)")
    return app


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    """422 without echoing the caller's input (it may be a long, sensitive message)."""
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    detail = [{k: v for k, v in error.items() if k in ("type", "loc", "msg")} for error in errors]
    return JSONResponse({"detail": detail}, status_code=422)


def _services(request: Request) -> Services:
    services: Services | None = request.app.state.services
    if services is None:
        raise HTTPException(503, "starting up")
    return services


ServicesDep = Annotated[Services, Depends(_services)]


def _api_client(request: Request) -> str:
    """The caller's name for a valid ``X-API-Key`` (constant-time compare), rate-limited per client."""
    key = request.headers.get("x-api-key", "")
    for known, name in request.app.state.api_keys.items():
        if hmac.compare_digest(key.encode(), known.encode()):
            _too_many(request.app.state.api_limiter, name)
            return str(name)
    raise HTTPException(401, "missing or invalid X-API-Key", headers={"WWW-Authenticate": "ApiKey"})


Client = Annotated[str, Depends(_api_client)]


def _end_user(request: Request, client: Client) -> str:
    """The licence-cap identity of a partner call: ``api:<key name>:<end-user id>`` (hashed by ``UserHasher``).

    The partner must name its end user with a stable pseudonymous id (its own random or hashed id), so each person
    counts toward the shared cap of active end-users; a phone number (or anything shaped like one) is refused."""
    values = request.headers.getlist(END_USER_HEADER)
    if len(values) > 1:
        raise HTTPException(400, f"send {END_USER_HEADER} once")
    value = values[0].strip() if values else ""
    if not value:
        raise HTTPException(
            400,
            f"missing {END_USER_HEADER}: send a stable pseudonymous id for the end user (8-137 characters), "
            "never a phone number; each end user counts toward the N-ATLaS licence cap",
        )
    if not _END_USER.fullmatch(value):
        raise HTTPException(
            400, f"{END_USER_HEADER} must be 8-137 characters: letters, digits, '.', '_', ':', '-'"
        )
    if contact_shaped(value):
        raise HTTPException(
            400,
            f"{END_USER_HEADER} looks like a phone number or an email address: send a pseudonymous id, e.g. a "
            "SHA-256 hex digest of your own customer id, or a UUID",
        )
    return f"api:{client}:{value}"


EndUser = Annotated[str, Depends(_end_user)]


def client_ip(request: Request, trust_proxy: bool) -> str:
    """The caller's IP for rate limiting. Behind our tunnel (``trust_proxy``) the client cannot choose it:
    Cloudflare's ``CF-Connecting-IP`` is set by the edge, and otherwise the *rightmost* ``X-Forwarded-For`` entry is
    the one our own proxy appended (entries to its left come from the client)."""
    if trust_proxy:
        if cf := request.headers.get("cf-connecting-ip"):
            return cf.strip()
        if forwarded := request.headers.get("x-forwarded-for"):
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


# --- shared check logic --------------------------------------------------------------------------------------


def _audio_problem(services: Services, language: str, problem: AudioProblem) -> HTTPException:
    """A user-facing, localised reason the audio could not be used (422)."""
    key = (
        Msg.AUDIO_TOO_LONG
        if problem in (AudioProblem.TOO_LONG, AudioProblem.TOO_LARGE)
        else Msg.NOT_UNDERSTOOD
    )
    message = plain_text(services.engine.messages.text(language, key, seconds=int(MAX_SECONDS)))
    return HTTPException(422, {"problem": problem.value, "message": message})


async def _run_check(
    request: Request,
    services: Services,
    *,
    channel: str,
    user_key: str,
    language: str,
    text: str | None,
    audio: bytes | None,
) -> tuple[Answer, str | None, int]:
    if language not in {lang.code for lang in services.registry.enabled}:
        raise HTTPException(422, f"unsupported language {language!r}; see GET /v1/languages")
    slots: asyncio.Semaphore = request.app.state.check_slots
    try:
        async with asyncio.timeout(QUEUE_TIMEOUT_S):
            await slots.acquire()
    except TimeoutError as exc:
        raise HTTPException(503, "busy, please retry", headers={"Retry-After": "5"}) from exc
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            return await _check_in_slot(services, channel, user_key, language, text, audio)
    except TimeoutError as exc:
        raise HTTPException(504, "the check took too long, please retry") from exc
    finally:
        slots.release()


async def _check_in_slot(
    services: Services, channel: str, user_key: str, language: str, text: str | None, audio: bytes | None
) -> tuple[Answer, str | None, int]:
    started = time.perf_counter()
    deadline = time.monotonic() + API_DEADLINE_S
    transcript: str | None = None
    decode_ms = asr_ms = None
    if audio is not None:
        if not services.engine.has_speech:
            raise HTTPException(501, "voice is not enabled on this server")
        try:
            speech = await services.engine.transcribe(audio, language)
        except AudioError as exc:
            raise _audio_problem(services, language, exc.problem) from exc
        except ASRError as exc:
            log.exception("speech recognition failed")
            raise HTTPException(503, "speech recognition unavailable") from exc
        transcript = speech.transcript.text.strip()
        decode_ms, asr_ms = speech.decode_ms, speech.asr_ms
        text = transcript
    message_id = uuid.uuid4().hex
    turn_id = services.hasher.message_key(channel, message_id)  # same key scheme as WhatsApp turns
    answer = await services.engine.check(text or "", language, deadline=deadline, turn_id=turn_id)
    created = utcnow()
    total_ms = int((time.perf_counter() - started) * 1000)
    private = answer.screen_action in CRISES
    try:  # evidence must never break the answer
        await services.store.add_interaction(
            Interaction(
                channel=channel,
                message_id=message_id,
                user_hash=services.hasher.user_id(user_key),
                language=language,
                kind=MessageKind.AUDIO if audio is not None else MessageKind.TEXT,
                outcome=answer.outcome,
                intent=answer.intent.value if answer.intent else None,
                transcript=None if private else text,  # a crisis message is sensitive: not kept
                english=None if private else answer.english or None,
                verdict=answer.verdict if answer.kind is AnswerKind.CHECK else None,
                flags=answer.flags,
                reply=services.engine.screen.scrub_crisis(answer.body) if private else answer.body,
                decode_ms=decode_ms,
                asr_ms=asr_ms,
                llm_ms=answer.llm_ms,
                total_ms=total_ms,
                created_at=created,
            )
        )
    except Exception:
        log.exception("could not log %s check", channel)
        # served but not in the store: the user keeps their licence-cap slot for the window
        _RESERVATION.get(Reservation()).hold()
    await record_pattern(services.store, channel, answer, text or "", created, gate=services.learning,
                         user_key=services.hasher.user_id(user_key))  # fmt: skip
    return answer, transcript, total_ms


def _check_response(
    services: Services, answer: Answer, transcript: str | None, total_ms: int
) -> CheckResponse:
    messages = services.engine.messages
    language = answer.language
    if answer.kind is AnswerKind.GREETING:
        explanation = messages.text(language, Msg.GREETING_REPLY)
    elif answer.kind is AnswerKind.CONTENT_FREE:
        explanation = messages.text(language, Msg.NOT_UNDERSTOOD)
    else:  # a crisis message that is also a scam: the crisis reply first (docs/ABUSE.md)
        explanation = "\n\n".join(p for p in (answer.preface, answer.body) if p)
    headline_key = VERDICT_HEADLINE.get(answer.verdict) if answer.kind is AnswerKind.CHECK else None
    return CheckResponse(
        verdict=answer.verdict.value if answer.kind is AnswerKind.CHECK else "not_a_check",
        headline=plain_text(messages.text(language, headline_key)) if headline_key else None,
        explanation=plain_text(explanation),
        flags=[f.value for f in answer.flags],
        actions=list(answer.actions),
        report_channels=[
            ReportChannel(label=c.label_for(language), kind=c.kind.value, value=c.value_for(language))
            for c in answer.contacts
        ],
        transcript=transcript,
        english=answer.english or None,
        language=language,
        degraded=answer.outcome is Outcome.FALLBACK,
        disclaimer=plain_text(messages.text(language, Msg.DISCLAIMER)),
        latency_ms=total_ms,
    )


async def _form(request: Request) -> tuple[dict[str, str], bytes | None]:
    """Parse a multipart/urlencoded form *after* auth/consent/rate checks: one file, few small fields."""
    try:
        form = await request.form(max_files=1, max_fields=8, max_part_size=MAX_TEXT_CHARS * 4 + 1024)
    except _BodyTooLarge:
        raise  # the size guard answers 413
    except Exception as exc:  # malformed multipart, too many parts, oversized field
        raise HTTPException(422, "malformed form data") from exc
    fields: dict[str, str] = {}
    audio: bytes | None = None
    for name, value in form.multi_items():
        if isinstance(value, UploadFile):
            if name == "audio":
                audio = await _read_upload(value)
        else:
            fields[name] = value
    return fields, audio


async def _read_upload(upload: UploadFile) -> bytes:
    data = bytearray()
    while chunk := await upload.read(64 * 1024):
        data += chunk
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, f"audio larger than {MAX_AUDIO_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(422, "empty audio")
    return bytes(data)


def _too_many(limiter: RateLimiter, key: str) -> None:
    if not limiter.allow(key):
        raise HTTPException(
            429, "rate limit exceeded", headers={"Retry-After": str(limiter.retry_after(key))}
        )


# --- /v1: public API -----------------------------------------------------------------------------------------


def _v1_router() -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1"])

    @router.post("/check", response_model=CheckResponse, summary="Check a message (JSON text)")
    async def check(
        body: CheckRequest, request: Request, services: ServicesDep, user: EndUser
    ) -> CheckResponse:
        async with _admitted(request, services, user):
            answer, transcript, ms = await _run_check(
                request,
                services,
                channel="api",
                user_key=user,
                language=body.language,
                text=body.text,
                audio=None,
            )
        return _check_response(services, answer, transcript, ms)

    @router.post(
        "/check/audio",
        response_model=CheckResponse,
        summary="Check a voice note (multipart upload)",
        openapi_extra=AUDIO_FORM,
    )
    async def check_audio(request: Request, services: ServicesDep, user: EndUser) -> CheckResponse:
        async with _admitted(request, services, user):
            # only after the key, end user, rate limit and licence cap were checked
            fields, audio = await _form(request)
            if audio is None:
                raise HTTPException(422, "missing 'audio' file")
            answer, transcript, ms = await _run_check(
                request, services, channel="api", user_key=user,
                language=fields.get("language", "en"), text=None, audio=audio,
            )  # fmt: skip
        return _check_response(services, answer, transcript, ms)

    @router.post("/ask", response_model=AskResponse, summary="Ask a money-safety question")
    async def ask(body: AskRequest, request: Request, services: ServicesDep, user: EndUser) -> AskResponse:
        async with _admitted(request, services, user):
            answer, _, ms = await _run_check(
                request, services, channel="api", user_key=user, language=body.language,
                text=body.question, audio=None,
            )  # fmt: skip
        result = _check_response(services, answer, None, ms)
        return AskResponse(
            answer=result.explanation, verdict=result.verdict, language=result.language,
            sources=list(answer.sources), degraded=result.degraded, disclaimer=result.disclaimer, latency_ms=ms,
        )  # fmt: skip

    @router.get("/languages", summary="Supported languages")
    async def languages(services: ServicesDep, client: Client) -> list[dict[str, Any]]:
        voice = services.engine.has_speech
        return [
            {
                "code": lang.code,
                "name": lang.name,
                "native_name": lang.native_name,
                "status": lang.status.value,
                "voice": voice,
            }
            for lang in services.registry.enabled
        ]

    @router.get("/topics", summary="Knowledge topics the answers are grounded on")
    async def topics(services: ServicesDep, client: Client) -> list[dict[str, Any]]:
        return services.engine.topics()

    return router


# --- web voice page --------------------------------------------------------------------------------------------

# /privacy, rendered from settings: what reaches partners only when the operator switches it on.
LIVE_PACKS_NOTICE = (
    "<li><strong>Partner knowledge sources:</strong> to answer some questions, the English translation of your message "
    "(most numbers masked; names are not removed), the warning signs we found and your language are sent to a partner "
    "organisation's server, which searches its own help content. The partner follows its own privacy policy.</li>"
)
PARTNER_API_OFF = "An API for partner organisations exists but is switched off during this pilot."
PARTNER_API_COLLECT_ON = (
    "<li><strong>Partner organisations' customers</strong> (through our API): the message or voice note the partner "
    "sends, our reply, the channel (API), and a coded form of the partner's own id for that customer (never a phone "
    "number), kept and deleted like the messages above. The partner is responsible for telling its customers and "
    "getting their agreement.</li>"
)
PARTNER_API_ON = (
    "Partner organisations can also use it through an API for their own customers; each partner names its users only "
    "by a coded id, never a phone number."
)


def _days(days: int) -> str:
    """ "1 day", "90 days", "730 days (2 years)"."""
    text = f"{days} day{'s' if days != 1 else ''}"
    years, rest = divmod(days, 365)
    return f"{text} ({years} year{'s' if years > 1 else ''})" if years and not rest else text


def _fill(page: str, values: dict[str, str]) -> str:
    """Replace ``{{NAME}}`` placeholders in one pass: a value can never introduce another placeholder."""
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values.get(m.group(1), m.group(0)), page)


@asynccontextmanager
async def _admitted(request: Request, services: Services, user_key: str) -> AsyncIterator[None]:
    """N-ATLaS licence cap for the website and the partner API (``user_key`` is ``web:<session>`` or
    ``api:<key name>:<end user>``), through the admission gate shared with WhatsApp (``Services.admission``): a user
    new to the 30-day window is refused with 503 ``at_capacity`` when the cap (or the web share) is full. The slot is
    held for the check, which logs the interaction before it ends."""
    channel = user_key.partition(":")[0]
    try:
        reservation = await services.admission.reserve(services.hasher.user_id(user_key), channel)
    except AtCapacity as exc:  # the cap, the web share, or a new-user gate (e.g. NewUsersBusy)
        settings: Settings = request.app.state.settings
        message = plain_text(services.engine.messages.text(settings.default_language, exc.message))
        headers = {"Retry-After": str(exc.retry_after_s)} if exc.retry_after_s else None
        raise HTTPException(503, {"problem": exc.problem, "message": message}, headers=headers) from exc
    token = _RESERVATION.set(reservation)
    try:
        yield
    finally:
        _RESERVATION.reset(token)
        reservation.release()


def _web_router() -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.get("/")
    async def page(request: Request) -> HTMLResponse:
        """The web page, told whether voice donation is on, where data is stored and the configured retention
        periods (its consent texts render them)."""
        settings: Settings = request.app.state.settings
        config = {
            "donation": settings.voice_donation,
            "storageInNigeria": settings.storage_location is StorageLocation.NIGERIA,
            "retentionDays": settings.retention_days,
            "patternRetentionDays": settings.pattern_retention_days,
        }
        values = {
            "CONFIG": json.dumps(config),
            "RETENTION_DAYS": str(settings.retention_days),
            "PATTERN_RETENTION_DAYS": str(settings.pattern_retention_days),
        }
        page = (STATIC / "voice.html").read_text(encoding="utf-8")
        return HTMLResponse(_fill(page, values))

    @router.get("/privacy")
    async def privacy(request: Request) -> HTMLResponse:
        """Privacy notice (required by Meta for Live mode; NDPA). Operator and contact come from settings."""
        settings: Settings = request.app.state.settings
        contact = settings.privacy_contact.strip()

        values = {
            "OPERATOR": html.escape(settings.operator_name),
            "CONTACT": (
                f'email <a href="mailto:{html.escape(contact)}">{html.escape(contact)}</a>'
                if contact
                else "contact the operator"
            ),
            "RETENTION_DAYS": str(settings.retention_days),
            "PATTERN_RETENTION": _days(settings.pattern_retention_days),
            "STORAGE_LOCATION": html.escape(settings.storage_location.english),
            "STORAGE_RISK": html.escape(settings.storage_location.english_risk),
            "DELETION_MARKER_DAYS": str(DELETION_MARKER_DAYS),
            "DONATION_STATUS": (
                "Available now."
                if settings.voice_donation
                else "Not switched on yet: this describes how it will work once it is."
            ),
            "VOICE_RETENTION": (
                "12 months" if settings.voice_retention_days == 365 else _days(settings.voice_retention_days)
            ),
            "LIVE_PACKS": LIVE_PACKS_NOTICE if settings.allow_live_packs else "",
            "PARTNER_API": PARTNER_API_ON if settings.partner_api else PARTNER_API_OFF,
            "PARTNER_API_COLLECT": PARTNER_API_COLLECT_ON if settings.partner_api else "",
        }
        page = (STATIC / "privacy.html").read_text(encoding="utf-8")
        return HTMLResponse(_fill(page, values))

    @router.post("/web/check")
    async def web_check(
        request: Request, services: ServicesDep, background: BackgroundTasks
    ) -> CheckResponse:
        """A typed crisis message gets the crisis reply whatever happens next (docs/ABUSE.md): no consent header,
        rate limit, licence cap or busy server stands between the user and help, and nothing is stored then."""
        crisis = await _typed_crisis(request, services)
        try:
            return await _gated_web_check(request, services, background)
        except HTTPException:
            if crisis is None:
                raise
            return crisis

    async def _gated_web_check(
        request: Request, services: Services, background: BackgroundTasks
    ) -> CheckResponse:
        # Consent, session and rate limit come in headers so they are checked before any upload is read.
        if request.headers.get("x-kudiguard-consent") != CONSENT_VERSION:  # consent to the current text only
            raise HTTPException(403, "consent is required")
        # Voice donation: a separate, optional box with its own version; absent = no donation.
        donating = request.headers.get("x-kudiguard-donation", "")
        settings: Settings = request.app.state.settings
        if donating and not settings.voice_donation:
            raise HTTPException(403, "voice donation is not enabled on this server")
        if donating and donating != DONATION_VERSION:
            raise HTTPException(403, "voice donation consent is out of date: reload the page")
        consented_at = utcnow()  # a withdrawal after this moment wins over the background write
        session = request.headers.get("x-kudiguard-session", "")
        if not re.fullmatch(SESSION_PATTERN, session):
            raise HTTPException(422, "invalid session id")
        _too_many(request.app.state.web_limiter, client_ip(request, settings.trust_proxy_headers))
        async with _admitted(request, services, f"web:{session}"):
            fields, audio = await _form(request)
            text = fields.get("text", "")[:MAX_TEXT_CHARS]
            if audio is None and not text.strip():
                raise HTTPException(422, "send text or audio")
            answer, transcript, ms = await _run_check(
                request, services, channel="web", user_key=f"web:{session}", language=fields.get("language", "en"),
                text=text if audio is None else None, audio=audio,
            )  # fmt: skip
            if (  # after the response is sent; never a flagged message (docs/ABUSE.md)
                donating and audio is not None and services.donations is not None
                and services.learning.allow_donation(answer.screened)
            ):  # fmt: skip
                background.add_task(
                    services.donations.donate,
                    Donation(
                        user_hash=services.hasher.user_id(f"web:{session}"), channel="web", language=answer.language,
                        audio=audio, hypothesis=transcript or "", consent_version=donating,
                        consented_at=consented_at,
                    ),
                )  # fmt: skip
            return _check_response(services, answer, transcript, ms)

    @router.post("/web/donation/withdraw")
    async def withdraw_donation(request: Request, services: ServicesDep) -> dict[str, int]:
        """Unticking the donation box: stop and delete this browser session's donated voice notes (NDPA s.35:
        as easy as giving it). Needs only the session id the page keeps; never rate-limited or consent-gated."""
        session = request.headers.get("x-kudiguard-session", "")
        if not re.fullmatch(SESSION_PATTERN, session):
            raise HTTPException(422, "invalid session id")
        user_hash = services.hasher.user_id(f"web:{session}")
        if services.donations is not None:
            return {"deleted": await services.donations.erase(user_hash)}
        return {"deleted": len(await services.store.delete_voice_samples(user_hash))}

    return router


async def _typed_crisis(request: Request, services: Services) -> CheckResponse | None:
    """The crisis reply for a small typed web message, read before any gate (a voice note is only known to be a
    crisis after transcription, behind the gates). None: not a crisis, or not a small text form."""
    declared = request.headers.get("content-length", "")
    kind = request.headers.get("content-type", "")
    if not declared.isdigit() or int(declared) > SMALL_REQUEST_BYTES or "form" not in kind:
        return None
    try:  # parsed once; the gated path reuses it (a file part is left unread)
        form = await request.form(max_files=1, max_fields=8, max_part_size=MAX_TEXT_CHARS * 4 + 1024)
    except Exception:
        return None  # malformed: the gated path answers it
    fields = {k: v for k, v in form.multi_items() if isinstance(v, str)}
    if len(fields) < len(form.multi_items()):
        return None  # a voice note: known to be a crisis only after transcription
    text = fields.get("text", "")[:MAX_TEXT_CHARS]
    category = services.engine.screen.crisis(text)
    if category is None:
        return None
    codes = {lang.code for lang in services.registry.enabled}
    asked = fields.get("language", "")
    language = asked if asked in codes else services.engine.screen.crisis_language(text) or "en"
    messages = services.engine.messages
    return CheckResponse(
        verdict="not_a_check", headline=None, flags=[], actions=[], report_channels=[], language=language,
        explanation=plain_text(messages.text(language, crisis_message(category))), degraded=False,
        disclaimer=plain_text(messages.text(language, Msg.DISCLAIMER)), latency_ms=0,
    )  # fmt: skip


# --- health -------------------------------------------------------------------------------------------------------


def _health_router() -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        services: Services | None = request.app.state.services
        if services is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        try:
            model = await asyncio.wait_for(services.llm.ready(), timeout=3)
        except Exception:
            model = False
        body = {
            "status": "ok" if model else "degraded",
            "model": model,  # rules-only answers still work when False
            "whatsapp": services.conversation is not None,
            "voice": services.engine.has_speech,
        }
        return JSONResponse(body, status_code=200)

    return router
