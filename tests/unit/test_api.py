"""The FastAPI app with real engine/knowledge/store and fake model + speech recogniser."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from fakes import FakeASR, FakeLLM, analysis
from kudiguard.api import app as app_module
from kudiguard.api.app import create_app
from kudiguard.api.services import Services
from kudiguard.config import Settings
from kudiguard.engine import Engine
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import LLMError, Purpose
from kudiguard.pipeline import CONSENT_VERSION
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.speech.audio import find_ffmpeg
from kudiguard.speech.base import ASRError
from kudiguard.storage.memory import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
VOICE_NOTE = ROOT / "tests" / "fixtures" / "voice_note_en_otp.ogg"
KEY = "k" * 32
OTP_SCAM = "Someone called and asked me to send the OTP sent to my phone. Is this a scam?"
SESSION = "abcdef0123456789abcdef"
END_USER = "partner-user-0001"  # the partner's pseudonymous id for its customer (licence cap)
API = {"x-api-key": KEY, "x-kudiguard-end-user": END_USER}


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "api_keys": (f"tester:{KEY}",),
        "partner_api": True,
        "whatsapp_enabled": False,
        "hash_secret": "s" * MIN_SECRET_BYTES,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@dataclass
class Api:
    client: httpx.AsyncClient
    app: FastAPI
    llm: FakeLLM
    asr: FakeASR
    store: MemoryStore

    async def check(
        self, text: str = OTP_SCAM, language: str = "en", key: str = KEY, user: str = END_USER
    ) -> httpx.Response:
        return await self.client.post(
            "/v1/check",
            json={"text": text, "language": language},
            headers={"x-api-key": key, "x-kudiguard-end-user": user},
        )


async def _start(registry: LanguageRegistry, config: Settings, *, speech: bool = True) -> AsyncIterator[Api]:
    llm = FakeLLM(
        analyse=analysis("x", "scam_check"),
        compose="1. Do not share the code.",
        translate="x",
        reply_translate="1. Kada ka bayar da lambar.",
    )
    asr = FakeASR(OTP_SCAM)
    hasher = UserHasher(b"s" * MIN_SECRET_BYTES)
    store = MemoryStore(hasher)
    async with httpx.AsyncClient() as http:
        knowledge = knowledge_base_from_packs(
            ROOT / "data" / "packs", client=http, include=["kudiguard-core"]
        )
        engine = Engine(
            registry=registry, llm=llm, knowledge=knowledge,
            asr=asr if speech else None, ffmpeg=find_ffmpeg() if speech else None,
        )  # fmt: skip

        async def factory(_: Settings) -> Services:
            return Services(config, registry, engine, store, hasher, llm)

        app = create_app(config, services_factory=factory)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield Api(client, app, llm, asr, store)


@pytest.fixture
async def api(registry: LanguageRegistry) -> AsyncIterator[Api]:
    async for running in _start(registry, settings()):
        yield running


ApiFactory = Callable[..., AsyncIterator[Api]]


# --- health ------------------------------------------------------------------------------------------------------


async def test_health_and_readiness(api: Api) -> None:
    assert (await api.client.get("/healthz")).json() == {"status": "ok"}
    ready = (await api.client.get("/readyz")).json()
    assert ready == {"status": "ok", "model": True, "whatsapp": False, "voice": True}


async def test_not_ready_before_startup() -> None:
    app = create_app(settings())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/readyz")).status_code == 503
        assert (await client.post("/v1/check", json={"text": "x"}, headers=API)).status_code == 503


# --- /v1/check -------------------------------------------------------------------------------------------------


async def test_check_requires_a_valid_key(api: Api) -> None:
    assert (await api.client.post("/v1/check", json={"text": "x"})).status_code == 401
    assert (await api.check(key="wrong" * 8)).status_code == 401


async def test_check_scam_in_hausa(api: Api) -> None:
    response = await api.check(language="ha")
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "likely_scam" and "asks_credentials" in body["flags"]
    assert body["headline"] and "*" not in body["headline"]  # WhatsApp markup stripped for API clients
    assert body["language"] == "ha" and body["degraded"] is False
    assert body["report_channels"] and body["report_channels"][0]["label"] == "Bankinku"  # localised label
    assert body["actions"] and body["disclaimer"] and body["latency_ms"] >= 0
    rows = await api.store.export_interactions()
    assert rows[-1]["channel"] == "api" and rows[-1]["verdict"] == "likely_scam"


@pytest.mark.parametrize(
    ("payload", "status"),
    [({"text": "x", "language": "fr"}, 422), ({"text": ""}, 422), ({"text": "x" * 4001}, 422), ({}, 422)],
)
async def test_check_validation(api: Api, payload: dict[str, object], status: int) -> None:
    response = await api.client.post("/v1/check", json=payload, headers=API)
    assert response.status_code == status


async def test_degraded_when_model_is_down(api: Api) -> None:
    api.llm.script[Purpose.ANALYSE] = [LLMError("down")]
    body = (await api.check()).json()
    assert body["verdict"] == "likely_scam" and body["degraded"] is True and body["explanation"]


async def test_greeting_and_content_free(api: Api) -> None:
    hello = (await api.check(text="Hello!")).json()
    assert hello["verdict"] == "not_a_check" and hello["headline"] is None and "Hello" in hello["explanation"]
    emoji = (await api.check(text="👍")).json()
    assert emoji["verdict"] == "not_a_check" and emoji["explanation"]


async def test_per_key_rate_limit(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(api_rate_per_minute=2)):
        assert (await api.check()).status_code == 200
        assert (await api.check()).status_code == 200
        limited = await api.check()
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1


async def test_evidence_failure_never_breaks_the_answer(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(*args: object) -> int:
        raise RuntimeError("disk full")

    monkeypatch.setattr(api.store, "add_interaction", broken)
    assert (await api.check()).status_code == 200


# --- /v1/check/audio -------------------------------------------------------------------------------------------


async def test_check_audio(api: Api) -> None:
    files = {"audio": ("note.ogg", VOICE_NOTE.read_bytes(), "audio/ogg")}
    response = await api.client.post("/v1/check/audio", files=files, data={"language": "en"}, headers=API)
    body = response.json()
    assert response.status_code == 200 and body["transcript"] == OTP_SCAM and body["verdict"] == "likely_scam"
    assert (await api.store.export_interactions())[-1]["kind"] == "audio"


@pytest.mark.parametrize(("content", "status"), [(b"", 422), (b"not audio at all", 422)])
async def test_bad_audio(api: Api, content: bytes, status: int) -> None:
    files = {"audio": ("x.ogg", content, "audio/ogg")}
    response = await api.client.post("/v1/check/audio", files=files, headers=API)
    assert response.status_code == status


async def test_audio_too_large(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "MAX_AUDIO_BYTES", 10)
    files = {"audio": ("x.ogg", b"x" * 11, "audio/ogg")}
    response = await api.client.post("/v1/check/audio", files=files, headers=API)
    assert response.status_code == 413


async def test_speech_failure_is_503(api: Api) -> None:
    api.asr.error = ASRError("model missing")
    files = {"audio": ("note.ogg", VOICE_NOTE.read_bytes(), "audio/ogg")}
    response = await api.client.post("/v1/check/audio", files=files, headers=API)
    assert response.status_code == 503


async def test_voice_disabled_is_501(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(), speech=False):
        files = {"audio": ("note.ogg", VOICE_NOTE.read_bytes(), "audio/ogg")}
        response = await api.client.post("/v1/check/audio", files=files, headers=API)
        assert response.status_code == 501
        assert all(
            lang["voice"] is False for lang in (await api.client.get("/v1/languages", headers=API)).json()
        )


# --- /v1/ask, discovery ----------------------------------------------------------------------------------------


async def test_ask(api: Api) -> None:
    api.llm.script[Purpose.ANALYSE] = [analysis("x", "question")]
    api.llm.script[Purpose.COMPOSE] = ["Open an account with your BVN and a valid ID."]
    response = await api.client.post(
        "/v1/ask",
        json={"question": "How do I open a savings account?", "language": "en"},
        headers=API,
    )
    body = response.json()
    assert response.status_code == 200 and body["verdict"] == "not_a_check" and "BVN" in body["answer"]
    assert isinstance(body["sources"], list)


async def test_languages_and_topics(api: Api) -> None:
    languages = (await api.client.get("/v1/languages", headers=API)).json()
    assert [lang["code"] for lang in languages] == ["ha", "yo", "ig", "en"]
    topics = (await api.client.get("/v1/topics", headers=API)).json()
    assert any(t["id"] == "flag-asks-credentials" and t["pack"] == "kudiguard-core" for t in topics)


async def test_openapi_documents_the_public_api(api: Api) -> None:
    paths = (await api.client.get("/openapi.json")).json()["paths"]
    assert {"/v1/check", "/v1/check/audio", "/v1/ask", "/v1/languages", "/v1/topics"} <= set(paths)
    assert "/web/check" not in paths and "/webhooks/whatsapp" not in paths


# --- web voice page --------------------------------------------------------------------------------------------

WEB = {"x-kudiguard-session": SESSION, "x-kudiguard-consent": CONSENT_VERSION}


async def test_web_page_is_served(api: Api) -> None:
    page = await api.client.get("/")
    assert (
        page.status_code == 200 and "KudiGuard" in page.text and "text/html" in page.headers["content-type"]
    )


async def test_web_check_text(api: Api) -> None:
    response = await api.client.post("/web/check", data={"language": "yo", "text": OTP_SCAM}, headers=WEB)
    body = response.json()
    assert body["verdict"] == "likely_scam" and body["language"] == "yo"
    rows = await api.store.export_interactions()
    assert rows[-1]["channel"] == "web" and rows[-1]["user"] != SESSION  # pseudonymous session hash


async def test_web_check_voice(api: Api) -> None:
    files = {"audio": ("voice-note", VOICE_NOTE.read_bytes(), "audio/ogg")}
    body = (await api.client.post("/web/check", data={"language": "en"}, files=files, headers=WEB)).json()
    assert body["transcript"] == OTP_SCAM


@pytest.mark.parametrize(
    ("headers", "data", "status"),
    [
        ({"x-kudiguard-session": SESSION}, {"text": "x"}, 403),  # no consent
        ({**WEB, "x-kudiguard-consent": "no"}, {"text": "x"}, 403),
        ({**WEB, "x-kudiguard-session": "short"}, {"text": "x"}, 422),
        ({**WEB, "x-kudiguard-session": "bad session id with spaces!"}, {"text": "x"}, 422),
        (WEB, {}, 422),
        (WEB, {"text": "   "}, 422),
    ],
)
async def test_web_check_validation(
    api: Api, headers: dict[str, str], data: dict[str, str], status: int
) -> None:
    assert (await api.client.post("/web/check", data=data, headers=headers)).status_code == status


async def test_web_audio_problem_has_a_localised_message(api: Api) -> None:
    files = {"audio": ("voice-note", b"not audio at all", "audio/ogg")}
    response = await api.client.post("/web/check", data={"language": "ha"}, files=files, headers=WEB)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["problem"] == "undecodable" and detail["message"] and "*" not in detail["message"]


async def test_web_rate_limit_per_ip_cannot_be_spoofed(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(web_rate_per_minute=1, trust_proxy_headers=True)):

        async def post(forwarded: str, api: Api = api) -> int:
            headers = {**WEB, "x-forwarded-for": forwarded}
            return (await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=headers)).status_code

        # Our proxy appends the real address on the right; the client controls the entries to its left.
        assert await post("1.1.1.1, 203.0.113.9") == 200
        assert await post("9.9.9.9, 203.0.113.9") == 429  # same real client, spoofed left entry
        assert await post("203.0.113.10") == 200  # a different client


def test_client_ip_rules() -> None:
    from kudiguard.api.app import client_ip

    class Fake:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers
            self.client = None

    spoofable = Fake({"x-forwarded-for": "6.6.6.6, 5.5.5.5"})
    assert client_ip(spoofable, trust_proxy=False) == "unknown"  # type: ignore[arg-type]
    assert client_ip(spoofable, trust_proxy=True) == "5.5.5.5"  # type: ignore[arg-type]
    cloudflare = Fake({"cf-connecting-ip": "7.7.7.7", "x-forwarded-for": "6.6.6.6"})
    assert client_ip(cloudflare, trust_proxy=True) == "7.7.7.7"  # type: ignore[arg-type]


# --- request hygiene (review M9) -------------------------------------------------------------------------------


async def test_oversized_bodies_are_refused_before_parsing(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    from kudiguard.api.app import MAX_REQUEST_BYTES

    big = b"x" * (MAX_REQUEST_BYTES + 1)
    declared = await api.client.post("/v1/check/audio", content=big, headers=API)
    assert declared.status_code == 413

    async def chunks():  # type: ignore[no-untyped-def]
        for _ in range(18):
            yield b"x" * (1024 * 1024)

    async def huge_file_part():  # type: ignore[no-untyped-def]
        yield b'--x\r\nContent-Disposition: form-data; name="audio"; filename="a.ogg"\r\n\r\n'
        for _ in range(18):
            yield b"x" * (1024 * 1024)

    multipart = {**WEB, "content-type": "multipart/form-data; boundary=x"}  # no Content-Length: chunked
    streamed = await api.client.post("/web/check", content=huge_file_part(), headers=multipart)
    assert streamed.status_code == 413
    not_a_form = await api.client.post("/web/check", content=chunks(), headers=WEB)
    assert not_a_form.status_code == 422  # never read at all: nothing was buffered


async def test_auth_and_consent_are_checked_before_the_form_is_read(api: Api) -> None:
    garbage = {"content-type": "multipart/form-data; boundary=x"}
    assert (await api.client.post("/v1/check/audio", content=b"garbage", headers=garbage)).status_code == 401
    assert (await api.client.post("/web/check", content=b"garbage", headers=garbage)).status_code == 403
    malformed = await api.client.post("/v1/check/audio", content=b"garbage", headers={**garbage, **API})
    assert malformed.status_code == 422


async def test_audio_form_without_file(api: Api) -> None:
    response = await api.client.post("/v1/check/audio", data={"language": "en"}, headers=API)
    assert response.status_code == 422 and "audio" in response.json()["detail"]


async def test_validation_errors_do_not_echo_input(api: Api) -> None:
    secret = "my pin is 1234 " * 400
    response = await api.client.post("/v1/check", json={"text": secret}, headers=API)
    assert response.status_code == 422 and "my pin" not in response.text


async def test_busy_and_timeout(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    monkeypatch.setattr(app_module, "QUEUE_TIMEOUT_S", 0.05)
    api.app.state.check_slots = asyncio.Semaphore(0)  # every slot taken
    busy = await api.check()
    assert busy.status_code == 503 and busy.headers["retry-after"] == "5"
    api.app.state.check_slots = asyncio.Semaphore(1)
    monkeypatch.setattr(app_module, "REQUEST_TIMEOUT_S", 0.05)
    api.llm.delay_s = 1.0
    assert (await api.check()).status_code == 504


async def test_discovery_requires_a_key(api: Api) -> None:
    assert (await api.client.get("/v1/languages")).status_code == 401
    assert (await api.client.get("/v1/topics")).status_code == 401


# --- WhatsApp mounting -----------------------------------------------------------------------------------------


async def test_whatsapp_webhook_mounted_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {"WHATSAPP_PHONE_NUMBER_ID": "P", "WHATSAPP_ACCESS_TOKEN": "t",
                        "WHATSAPP_APP_SECRET": "s", "WHATSAPP_VERIFY_TOKEN": "v"}.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    app = create_app(settings(whatsapp_enabled=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        params = {"hub.mode": "subscribe", "hub.verify_token": "v", "hub.challenge": "7"}
        assert (await client.get("/webhooks/whatsapp", params=params)).text == "7"
        from kudiguard.api.app import _ConversationProxy

        assert _ConversationProxy(app).submit(None) is False  # type: ignore[arg-type]  # not started: Meta retries


def test_weak_api_keys_fail_fast() -> None:
    with pytest.raises(ValueError, match="at least 24"):
        create_app(settings(api_keys=("bank:short",)))


async def test_proxy_submits_to_a_running_conversation(api: Api) -> None:
    from kudiguard.api.app import _ConversationProxy

    submitted: list[object] = []

    class Conversation:
        def submit(self, message: object) -> bool:
            submitted.append(message)
            return True

        async def drain(self) -> None:
            return None

    api.app.state.services.conversation = Conversation()
    assert _ConversationProxy(api.app).submit("msg") is True and submitted == ["msg"]  # type: ignore[arg-type]
    api.app.state.services.conversation = None


async def test_webhook_not_mounted_without_whatsapp_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
    app = create_app(settings(whatsapp_enabled=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/webhooks/whatsapp")).status_code == 404


async def test_readiness_reports_a_failing_model_as_degraded(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken() -> bool:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(api.llm, "ready", broken)
    ready = await api.client.get("/readyz")
    assert (
        ready.status_code == 200 and ready.json()["status"] == "degraded" and ready.json()["model"] is False
    )


async def test_body_size_limit_middleware_edges() -> None:
    from kudiguard.api.app import BodySizeLimit

    seen: list[str] = []

    async def passthrough(scope, receive, send):  # type: ignore[no-untyped-def]
        seen.append(scope["type"])

    await BodySizeLimit(passthrough, limit=1)({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    assert seen == ["lifespan"]

    async def responds_then_reads(scope, receive, send):  # type: ignore[no-untyped-def]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await receive()  # a disconnect message is not a body
        await receive()  # the body overflows after the response already started

    sent: list[dict[str, object]] = []
    messages = iter(
        [{"type": "http.disconnect"}, {"type": "http.request", "body": b"xx", "more_body": False}]
    )

    async def receive():  # type: ignore[no-untyped-def]
        return next(messages)

    async def send(message):  # type: ignore[no-untyped-def]
        sent.append(message)

    await BodySizeLimit(responds_then_reads, limit=1)({"type": "http", "headers": []}, receive, send)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200]  # no second response


async def test_extra_file_fields_are_ignored(api: Api) -> None:
    files = [("audio", ("note.ogg", VOICE_NOTE.read_bytes(), "audio/ogg"))]
    response = await api.client.post("/web/check", data={"language": "en"}, files=files, headers=WEB)
    assert response.status_code == 200
    other = [("photo", ("x.jpg", b"jpeg", "image/jpeg"))]
    response = await api.client.post("/web/check", data={"text": OTP_SCAM}, files=other, headers=WEB)
    assert response.status_code == 200 and response.json()["transcript"] is None  # the photo is ignored


def test_create_app_without_settings_loads_the_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[Path] = []
    monkeypatch.setattr(app_module, "load_env_file", lambda path: loaded.append(path))
    monkeypatch.setenv("KUDIGUARD_WHATSAPP_ENABLED", "false")
    monkeypatch.setenv("KUDIGUARD_API_KEYS", "")
    app = create_app()
    assert loaded and loaded[0].name == ".env" and app.state.settings.whatsapp_enabled is False


def test_client_ip_falls_back_to_the_socket(api: Api) -> None:
    from kudiguard.api.app import client_ip

    class Fake:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.client = type("C", (), {"host": "10.0.0.5"})()

    assert client_ip(Fake(), trust_proxy=True) == "10.0.0.5"  # type: ignore[arg-type]


async def test_json_endpoints_have_a_small_body_cap(api: Api) -> None:
    from kudiguard.api.app import SMALL_REQUEST_BYTES

    big = b'{"text": "' + b"x" * SMALL_REQUEST_BYTES + b'"}'
    declared = await api.client.post("/v1/check", content=big, headers={"content-type": "application/json"})
    assert declared.status_code == 413  # refused before reading, even without a key

    async def chunked():  # type: ignore[no-untyped-def]
        yield b'{"text": "'
        for _ in range(4):
            yield b"x" * SMALL_REQUEST_BYTES

    streamed = await api.client.post(
        "/v1/check", content=chunked(), headers={"content-type": "application/json", **API}
    )
    assert streamed.status_code == 413  # not FastAPI's generic 400


async def test_privacy_page(registry: LanguageRegistry) -> None:
    config = settings(operator_name="Veldt <Labs>", privacy_contact="privacy@example.org")
    async for api in _start(registry, config):
        page = await api.client.get("/privacy")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "90 days" in page.text and "DELETE" in page.text and "ndpc.gov.ng" in page.text
        assert "outside" in page.text and "names or addresses" in page.text
        assert "Veldt &lt;Labs&gt;" in page.text and "mailto:privacy@example.org" in page.text  # escaped
        assert "{{" not in page.text


async def test_privacy_page_without_contact(api: Api) -> None:
    page = await api.client.get("/privacy")
    assert "contact the operator" in page.text and "the KudiGuard team" in page.text
    assert "90 days" in page.text and "2 years" in page.text
    assert "/privacy" not in (await api.client.get("/openapi.json")).text
    assert 'href="/privacy"' in (await api.client.get("/")).text


async def test_privacy_page_reflects_configured_retention(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(retention_days=45, pattern_retention_days=400,
                                               operator_name="{{CONTACT}} Ltd")):  # fmt: skip
        page = (await api.client.get("/privacy")).text
        assert "45 days" in page and "400 days" in page and "{{CONTACT}} Ltd" in page  # one-pass substitution


async def test_web_scam_checks_feed_the_library_but_api_checks_do_not(api: Api) -> None:
    await api.check()  # B2B API: its end users did not consent to us
    assert await api.store.export_patterns() == []
    await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)
    [row] = await api.store.export_patterns()
    assert row["channel"] == "web" and SESSION not in str(row)


async def test_web_sessions_have_their_own_share_of_the_licence_cap(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(web_session_cap=1, active_user_cap=100)):
        assert (await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)).status_code == 200
        other = {**WEB, "x-kudiguard-session": "zzzzzzzzzzzzzzzzzzzz"}
        full = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=other)
        assert full.status_code == 503 and full.json()["detail"]["problem"] == "at_capacity"
        again = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)
        assert again.status_code == 200  # a returning session is always served


async def test_web_respects_the_project_wide_cap(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(web_session_cap=50, active_user_cap=1)):
        await api.check()  # one API client already active
        response = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)
        assert response.status_code == 503


async def test_no_cap_configured(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(active_user_cap=None)):
        assert (await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)).status_code == 200


async def test_a_burst_of_new_web_sessions_cannot_overshoot_the_share(registry: LanguageRegistry) -> None:
    import asyncio

    async for api in _start(
        registry, settings(web_session_cap=2, active_user_cap=100, web_rate_per_minute=100)
    ):
        api.llm.delay_s = 0.05  # first checks overlap

        async def new_session(n: int, api: Api = api) -> int:
            headers = {**WEB, "x-kudiguard-session": f"burst-session-{n:04d}-xxxx"}
            return (await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=headers)).status_code

        codes = await asyncio.gather(*(new_session(n) for n in range(5)))
        assert sorted(codes) == [200, 200, 503, 503, 503]
        assert api.app.state.services.admission.in_flight == 0


async def test_outdated_web_consent_is_refused(api: Api) -> None:
    old = {**WEB, "x-kudiguard-consent": "2020-01-01"}
    assert (await api.client.post("/web/check", data={"text": "x"}, headers=old)).status_code == 403


async def test_burst_cannot_overshoot_the_share_with_sqlite(
    registry: LanguageRegistry, tmp_path: Path
) -> None:
    """SQLite counts run in a thread (they yield), so admission must be atomic."""
    import asyncio

    from kudiguard.storage.sqlite import SQLiteStore

    config = settings(web_session_cap=2, active_user_cap=100, web_rate_per_minute=100)
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose="1. Do not share the code.")
    hasher = UserHasher(b"s" * MIN_SECRET_BYTES)
    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=hasher)
    async with httpx.AsyncClient() as http:
        knowledge = knowledge_base_from_packs(
            ROOT / "data" / "packs", client=http, include=["kudiguard-core"]
        )
        engine = Engine(registry=registry, llm=llm, knowledge=knowledge)

        async def factory(_: Settings) -> Services:
            return Services(config, registry, engine, store, hasher, llm, closers=[store.close])

        app = create_app(config, services_factory=factory)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

                async def new_session(n: int) -> int:
                    headers = {**WEB, "x-kudiguard-session": f"sqlite-burst-{n:04d}-xxxxxx"}
                    return (
                        await client.post("/web/check", data={"text": OTP_SCAM}, headers=headers)
                    ).status_code

                codes = await asyncio.gather(*(new_session(n) for n in range(8)))
    assert codes.count(200) == 2 and codes.count(503) == 6


@pytest.mark.parametrize("trial", range(3))
async def test_a_mixed_whatsapp_web_and_api_burst_never_exceeds_the_cap(
    registry: LanguageRegistry, tmp_path: Path, trial: int
) -> None:
    """One admission gate (Services.admission) for every channel: 90 new users at once on SQLite, cap 20, web share
    10. Before it, each channel saw only its own in-flight users and 36 got in."""
    import asyncio
    from datetime import UTC, datetime

    from fakes import FakeMessenger
    from kudiguard.domain import InboundMessage, MessageKind
    from kudiguard.pipeline import Conversation, Limits
    from kudiguard.storage.base import UserState
    from kudiguard.storage.sqlite import SQLiteStore

    cap = 20
    config = settings(
        active_user_cap=cap, web_session_cap=10, web_rate_per_minute=1000, api_rate_per_minute=1000
    )
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose="1. Do not share the code.")
    llm.delay_s = 0.02  # first checks overlap
    hasher = UserHasher(b"s" * MIN_SECRET_BYTES)
    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=hasher)
    phones = [f"+23480{trial}{n:07d}" for n in range(30)]
    long_ago = datetime(2026, 1, 1, tzinfo=UTC)
    for phone in phones:  # consented WhatsApp users with no interaction in the window: new to the cap
        await store.save_user(
            UserState(
                hasher.user_id(phone), language="en", consented_at=long_ago, consent_version=CONSENT_VERSION,
                welcome_version=CONSENT_VERSION, welcome_at=long_ago, created_at=long_ago, last_seen_at=long_ago,
            )
        )  # fmt: skip
    messenger = FakeMessenger()
    async with httpx.AsyncClient() as http:
        knowledge = knowledge_base_from_packs(
            ROOT / "data" / "packs", client=http, include=["kudiguard-core"]
        )
        engine = Engine(registry=registry, llm=llm, knowledge=knowledge)

        async def factory(_: Settings) -> Services:
            services = Services(config, registry, engine, store, hasher, llm, closers=[store.close])
            services.conversation = Conversation(
                engine=engine, store=store, messenger=messenger, hasher=hasher,
                limits=Limits(active_user_cap=cap), admission=services.admission,
            )  # fmt: skip
            return services

        app = create_app(config, services_factory=factory)
        async with app.router.lifespan_context(app):
            services = app.state.services
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

                async def api_user(n: int) -> int:
                    headers = {**API, "x-kudiguard-end-user": f"burst-user-{trial}-{n:04d}"}
                    return (
                        await client.post("/v1/check", json={"text": OTP_SCAM}, headers=headers)
                    ).status_code

                async def web_user(n: int) -> int:
                    headers = {**WEB, "x-kudiguard-session": f"websess-{trial}-{n:04d}-xxxxxx"}
                    return (
                        await client.post("/web/check", data={"text": OTP_SCAM}, headers=headers)
                    ).status_code

                for n, phone in enumerate(phones):
                    services.conversation.submit(
                        InboundMessage(channel="whatsapp", message_id=f"wamid.{trial}.{n}", user_id=phone,
                                       kind=MessageKind.TEXT, text=OTP_SCAM)
                    )  # fmt: skip
                codes = await asyncio.gather(
                    *(api_user(n) for n in range(30)),
                    *(web_user(n) for n in range(30)),
                    services.conversation.wait_idle(),
                )
                await services.conversation.drain()
                assert services.admission.in_flight == 0
                since = datetime(2000, 1, 1, tzinfo=UTC)
                total = await store.count_active_users_since(since)
                web = await store.count_active_users_since(since, channels={"web"})
                whatsapp = await store.count_active_users_since(since, channels={"whatsapp"})
    served = codes[:60].count(200)
    assert cap - 3 <= total <= cap and web <= 10  # never one more (a transient double count may waste a slot)
    assert served + whatsapp == total and set(codes[:60]) <= {200, 503}


async def test_check_passes_a_hashed_turn_id(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    services = api.app.state.services
    seen: list[object] = []
    original = services.engine.check

    async def spy(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("turn_id"))
        return await original(*args, **kwargs)

    monkeypatch.setattr(services.engine, "check", spy)
    assert (await api.check()).status_code == 200
    (turn_id,) = seen
    # the stored row keeps the same hashed key (storage.protect), so logs and evidence can be joined later
    stored = list(api.store._interactions.values())[-1]  # type: ignore[attr-defined]
    assert isinstance(turn_id, str) and len(turn_id) == 32 and turn_id == stored.message_id


# --- partner API: off by default, end users named and counted (N-ATLaS licence) ------------------------------


async def test_partner_api_is_off_by_default(
    registry: LanguageRegistry, caplog: pytest.LogCaptureFixture
) -> None:
    async for api in _start(registry, settings(partner_api=False)):
        for path in ("/v1/check", "/v1/ask", "/v1/check/audio"):
            assert (await api.client.post(path, json={"text": "x"}, headers=API)).status_code == 404
        assert (await api.client.get("/v1/languages", headers=API)).status_code == 404
        assert not any(p.startswith("/v1") for p in (await api.client.get("/openapi.json")).json()["paths"])
        assert (await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)).status_code == 200
    assert "partner API is off" in caplog.text  # keys configured but unused: say so
    assert Settings(_env_file=None).partner_api is False  # type: ignore[call-arg]


def test_partner_api_on_without_keys_warns(caplog: pytest.LogCaptureFixture) -> None:
    create_app(settings(api_keys=()))
    assert "every /v1 call is refused" in caplog.text


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "missing"),
        ("", "missing"),
        ("short", "8-137"),
        ("x" * 138, "8-137"),
        ("has space inside", "8-137"),
        ("+2348031234567", "8-137"),  # '+' is not allowed at all
        ("08031234567", "phone"),
        ("2348031234567", "phone"),
        ("0803-123-4567", "phone"),
        ("447911123456", "phone"),  # digits only: any country
        ("user_08031234567", "phone"),
        ("tel:2348031234567", "phone"),
        # the review's bypasses of the first version: separators, bare 10 digits, other countries, emails
        ("user-0803-123-4567", "phone"),
        ("ab.0803.123.4567", "phone"),
        ("0701-234-5678-ab", "phone"),
        ("x8031234567", "phone"),
        ("x2340803123456", "phone"),
        ("cust-447911123456", "phone"),
        ("u15551234567", "phone"),
        ("john.doe_gmail.com", "email"),
        ("jd.at.yahoo-mail", "email"),
    ],
)
async def test_every_partner_call_names_a_pseudonymous_end_user(
    api: Api, value: str | None, reason: str
) -> None:
    headers = {"x-api-key": KEY} | ({} if value is None else {"x-kudiguard-end-user": value})
    for path, body in (("/v1/check", {"text": OTP_SCAM}), ("/v1/ask", {"question": "What is BVN?"})):
        response = await api.client.post(path, json=body, headers=headers)
        assert response.status_code == 400 and reason in response.json()["detail"]
    audio = await api.client.post("/v1/check/audio", content=b"garbage", headers=headers)
    assert audio.status_code == 400  # before the form is read
    assert await api.store.export_interactions() == []


@pytest.mark.parametrize(
    "value",
    [
        "550e8400-e29b-41d4-a716-446655440000",
        "a3f9c2d17be04c55a1e2f3b4c5d6e7f8",
        "cust.000123:web",
        "ab12" * 32,  # SHA-512 hex
        "c-" + "ab12" * 32,  # tagged SHA-512 (130 characters)
        "cust-" + "0f" * 64,  # the longest tag: 137 characters
    ],
)
async def test_pseudonymous_ids_are_accepted(api: Api, value: str) -> None:
    assert (await api.check(user=value)).status_code == 200


async def test_a_missing_key_is_401_before_the_end_user_is_checked(api: Api) -> None:
    response = await api.client.post(
        "/v1/check", json={"text": "x"}, headers={"x-kudiguard-end-user": "08031234567"}
    )
    assert response.status_code == 401


async def test_each_end_user_is_hashed_in_the_partner_domain(api: Api) -> None:
    await api.check(user="customer-aaaa")
    await api.check(user="customer-bbbb")
    await api.check(user="customer-aaaa")
    rows = list(api.store._interactions.values())  # type: ignore[attr-defined]
    hasher = api.app.state.services.hasher
    hashes = {r.user_hash for r in rows}
    assert {r.channel for r in rows} == {"api"}
    assert len(hashes) == 2  # one per end user, not one per partner key
    assert hasher.user_id("api:tester:customer-aaaa") in hashes
    assert hasher.user_id("web:customer-aaaa") != hasher.user_id("api:tester:customer-aaaa")  # own domain
    assert "customer-aaaa" not in str(rows)


async def test_partner_end_users_count_toward_the_shared_licence_cap(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(active_user_cap=2, web_session_cap=50)):
        assert (await api.check(user="customer-aaaa")).status_code == 200
        assert (await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)).status_code == 200
        full = await api.check(user="customer-bbbb")  # a new end user: the project-wide cap is reached
        assert full.status_code == 503 and full.json()["detail"]["problem"] == "at_capacity"
        assert full.json()["detail"]["message"]
        bbbb = {**API, "x-kudiguard-end-user": "customer-bbbb"}
        assert (await api.client.post("/v1/ask", json={"question": "x"}, headers=bbbb)).status_code == 503
        audio = await api.client.post("/v1/check/audio", content=b"garbage", headers=bbbb)
        assert audio.status_code == 503  # refused before the upload is read
        assert (await api.check(user="customer-aaaa")).status_code == 200  # a returning end user is served
        other = {**WEB, "x-kudiguard-session": "zzzzzzzzzzzzzzzzzzzz"}
        assert (
            await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=other)
        ).status_code == 503
        assert api.app.state.services.admission.in_flight == 0


async def test_whatsapp_users_fill_the_cap_for_partners_too(registry: LanguageRegistry) -> None:
    from kudiguard.domain import Interaction, MessageKind

    async for api in _start(registry, settings(active_user_cap=1)):
        await api.store.add_interaction(
            Interaction(
                channel="whatsapp", message_id="m1", user_hash="h" * 32, language="en", kind=MessageKind.TEXT
            )
        )
        assert (await api.check()).status_code == 503


async def test_the_web_share_does_not_limit_partner_users(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(active_user_cap=10, web_session_cap=1)):
        assert (await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)).status_code == 200
        assert (
            await api.check(user="customer-aaaa")
        ).status_code == 200  # the web share is full, the cap is not


async def test_a_burst_of_new_partner_users_cannot_overshoot_the_cap(registry: LanguageRegistry) -> None:
    import asyncio

    async for api in _start(registry, settings(active_user_cap=2, api_rate_per_minute=100)):
        api.llm.delay_s = 0.05  # first checks overlap

        async def new_user(n: int, api: Api = api) -> int:
            return (await api.check(user=f"burst-user-{n:04d}")).status_code

        codes = await asyncio.gather(*(new_user(n) for n in range(6)))
        assert sorted(codes) == [200, 200, 503, 503, 503, 503]
        assert api.app.state.services.admission.in_flight == 0
        assert len({r.user_hash for r in api.store._interactions.values()}) == 2  # type: ignore[attr-defined]


async def test_privacy_page_says_whether_the_partner_api_and_live_packs_are_on(
    registry: LanguageRegistry,
) -> None:
    from kudiguard.api.app import LIVE_PACKS_NOTICE, PARTNER_API_COLLECT_ON, PARTNER_API_OFF, PARTNER_API_ON

    async for api in _start(registry, settings(partner_api=False)):
        page = (await api.client.get("/privacy")).text
        assert PARTNER_API_OFF in page and PARTNER_API_ON not in page and LIVE_PACKS_NOTICE not in page
        assert "training" not in page and "{{" not in page
        assert "the deletion check runs every 6 hours" in page
    async for api in _start(registry, settings(allow_live_packs=True)):
        page = (await api.client.get("/privacy")).text
        assert PARTNER_API_ON in page and LIVE_PACKS_NOTICE in page and "{{" not in page
        assert PARTNER_API_COLLECT_ON in page  # what is kept for partners' customers, only when the API is on


async def test_web_page_renders_the_configured_retention(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(retention_days=45, pattern_retention_days=400)):
        page = (await api.client.get("/")).text
        assert "for up to 45 days" in page and "for up to 400 days" in page  # the no-JavaScript fallback
        assert '"retentionDays": 45, "patternRetentionDays": 400' in page and "{{" not in page
        assert "90 days" not in page and "2 years" not in page
        assert page.count("{days}") == page.count("{patternDays}") == 5  # 4 languages + the renderer


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retention_days", 91),
        ("pattern_retention_days", 731),
        ("retention_days", 30),
        ("pattern_retention_days", 29),
    ],
)
def test_retention_cannot_exceed_the_published_promise(field: str, value: int) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        settings(**{field: value})


async def test_the_end_user_header_must_be_sent_once(api: Api) -> None:
    headers = [
        ("x-api-key", KEY),
        ("x-kudiguard-end-user", "goodid-1234"),
        ("x-kudiguard-end-user", "08031234567"),
    ]
    response = await api.client.post("/v1/check", json={"text": OTP_SCAM}, headers=headers)
    assert response.status_code == 400 and "once" in response.json()["detail"]


def test_hashed_and_random_ids_are_never_mistaken_for_contacts() -> None:
    """The guardrail's false positives on what partners are told to send (seeded; measured 0 in 200,000 each)."""
    import hashlib
    import random
    import uuid

    from kudiguard.api.app import contact_shaped

    rng = random.Random(7)
    for _ in range(5000):
        digest = hashlib.sha256(rng.randbytes(16)).hexdigest()
        ident = uuid.UUID(int=rng.getrandbits(128), version=4)
        for value in (digest, "c-" + digest, digest[:40], digest[:32], str(ident), ident.hex):
            assert not contact_shaped(value), value


def test_day_counts_read_naturally() -> None:
    from kudiguard.api.app import _days

    assert (_days(1), _days(90), _days(730), _days(400)) == (
        "1 day",
        "90 days",
        "730 days (2 years)",
        "400 days",
    )
    assert _days(365) == "365 days (1 year)"


@pytest.mark.parametrize("channel", ["web", "api"])
async def test_a_twin_request_cannot_ride_on_a_slot_its_failing_sibling_frees(
    registry: LanguageRegistry, channel: str
) -> None:
    """The review's twin attack: per user, a request whose (empty) body trickles in holds the slot, its valid twin
    arrives 20 ms later; the slow one then fails (422) and must not free the slot the twin is still using."""
    import asyncio

    async for api in _start(registry, settings(active_user_cap=20, web_session_cap=3, web_rate_per_minute=10_000,
                                               api_rate_per_minute=10_000)):  # fmt: skip
        api.llm.delay_s = 0.3

        async def slow_empty(headers: dict[str, str], path: str, content_type: str, api: Api = api) -> int:
            async def trickle():  # type: ignore[no-untyped-def]
                yield b"text=" if content_type.endswith("urlencoded") else b"--x\r\n"
                await asyncio.sleep(0.15)
                yield b"+" if content_type.endswith("urlencoded") else b"--x--\r\n"

            response = await api.client.post(
                path, content=trickle(), headers={**headers, "content-type": content_type}
            )
            return response.status_code

        async def pair(k: int, api: Api = api) -> tuple[int, int]:
            await asyncio.sleep(k * 0.03)
            if channel == "web":
                headers = {**WEB, "x-kudiguard-session": f"twin-{k:04d}-xxxxxxxxxx"}
                bad = asyncio.create_task(
                    slow_empty(headers, "/web/check", "application/x-www-form-urlencoded")
                )
                await asyncio.sleep(0.02)
                good = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=headers)
            else:
                headers = {**API, "x-kudiguard-end-user": f"twin-user-{k:04d}"}
                bad = asyncio.create_task(
                    slow_empty(headers, "/v1/check/audio", "multipart/form-data; boundary=x")
                )
                await asyncio.sleep(0.02)
                good = await api.client.post("/v1/check", json={"text": OTP_SCAM}, headers=headers)
            return await bad, good.status_code

        results = await asyncio.gather(*(pair(k) for k in range(30)))
        rows = list(api.store._interactions.values())  # type: ignore[attr-defined]
        users = {r.user_hash for r in rows}
        limit = 3 if channel == "web" else 20
        assert len(users) <= limit and {bad for bad, _ in results} <= {422, 503}  # 503: cap reached
        assert api.app.state.services.admission.in_flight == 0


async def test_a_new_user_gate_refusal_is_its_own_problem(api: Api) -> None:
    from kudiguard.admission import NewUsersBusy

    async def busy(user_hash: str, channel: str) -> None:
        raise NewUsersBusy("too many new users")

    api.app.state.services.admission.add_check(busy)
    response = await api.check()
    assert response.status_code == 503 and response.headers["retry-after"] == "3600"
    assert (
        response.json()["detail"]["problem"] == "busy_new_users"
        and "hour" in response.json()["detail"]["message"]
    )
    web = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)
    assert web.status_code == 503 and web.json()["detail"]["problem"] == "busy_new_users"


async def test_a_failed_evidence_write_keeps_the_partner_users_slot(
    registry: LanguageRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    async for api in _start(registry, settings(active_user_cap=1)):

        async def broken(*args: object) -> int:
            raise RuntimeError("disk full")

        monkeypatch.setattr(api.store, "add_interaction", broken)
        assert (await api.check(user="customer-aaaa")).status_code == 200  # served, but not in the store
        monkeypatch.undo()
        assert api.app.state.services.admission.in_flight == 1
        assert (await api.check(user="customer-bbbb")).status_code == 503
