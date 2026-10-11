"""Review R: crisis-first on the partner API, the web fallback on any failure, the shared voice budget, the bounded
pattern write, log redaction under every entry point and the WhatsApp API version format."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
from test_api import API, KEY, OTP_SCAM, VOICE_NOTE, WEB, Api, _start, settings

from kudiguard import cli
from kudiguard.api import app as app_module
from kudiguard.channels.whatsapp import WhatsAppConfig
from kudiguard.domain import Flag, Verdict
from kudiguard.i18n import Messages, Msg
from kudiguard.languages import LanguageRegistry
from kudiguard.patterns import ScamPattern
from kudiguard.render import plain_text
from kudiguard.safety.limits import FloodGuard
from kudiguard.text import RedactingFilter, install_log_redaction

CRISIS = "They took all my savings. I want to kill myself."
PHONE_NUMBER = "08031234567"


@pytest.fixture
async def api(registry: LanguageRegistry) -> AsyncIterator[Api]:
    async for running in _start(registry, settings()):
        yield running


def crisis_text(registry: LanguageRegistry, language: str = "en") -> str:
    return plain_text(Messages(registry).text(language, Msg.CRISIS_REPLY))


# --- partner API: crisis before the gates (the key still first) ----------------------------------------------


async def test_partner_crisis_is_answered_before_the_end_user_and_rate_gates(
    registry: LanguageRegistry,
) -> None:
    async for api in _start(registry, settings(api_rate_per_minute=1)):
        no_user = await api.client.post("/v1/check", json={"text": CRISIS, "language": "ha"},
                                        headers={"x-api-key": KEY})  # fmt: skip
        assert no_user.status_code == 200 and no_user.json()["explanation"] == crisis_text(registry, "ha")
        assert (await api.check()).status_code == 200  # spends the one call per minute
        limited = await api.check(CRISIS)
        assert limited.status_code == 200 and limited.json()["verdict"] == "not_a_check"
        assert (await api.check()).status_code == 429  # an ordinary message still meets the limit
        asked = await api.client.post("/v1/ask", json={"question": CRISIS}, headers={"x-api-key": KEY})
        assert asked.status_code == 200 and asked.json()["answer"] == crisis_text(registry)
        assert asked.json()["sources"] == [] and asked.json()["latency_ms"] == 0
        rows = await api.store.export_interactions()
        assert len(rows) == 1 and rows[0]["transcript"] == OTP_SCAM


async def test_partner_crisis_without_a_key_is_still_401(api: Api) -> None:
    for path, body in (("/v1/check", {"text": CRISIS}), ("/v1/ask", {"question": CRISIS})):
        assert (await api.client.post(path, json=body)).status_code == 401
    assert (await api.client.post("/v1/check/audio", data={"text": CRISIS})).status_code == 401


async def test_partner_audio_form_with_typed_crisis_text(api: Api, registry: LanguageRegistry) -> None:
    response = await api.client.post("/v1/check/audio", data={"text": CRISIS}, headers={"x-api-key": KEY})
    assert response.status_code == 200 and response.json()["explanation"] == crisis_text(registry)
    calm = await api.client.post("/v1/check/audio", data={"text": "hello"}, headers={"x-api-key": KEY})
    assert calm.status_code == 400  # not a crisis: the end-user gate answers
    assert await api.store.export_interactions() == []


async def test_partner_crisis_passing_every_gate_is_checked_and_kept_private(api: Api) -> None:
    response = await api.check(CRISIS)
    assert response.status_code == 200
    [row] = await api.store.export_interactions()
    assert row["transcript"] is None


async def test_ask_still_answers_without_a_crisis(api: Api) -> None:
    response = await api.client.post("/v1/ask", json={"question": "how do I spot a fake alert?"}, headers=API)
    assert response.status_code == 200 and response.json()["answer"]


# --- web: any failure behind a typed crisis still gets the crisis reply ----------------------------------------


async def test_web_crisis_survives_an_unexpected_failure(
    api: Api, registry: LanguageRegistry, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError(f"store down while reading {PHONE_NUMBER}")

    monkeypatch.setattr(app_module, "_run_check", broken)
    with caplog.at_level(logging.ERROR):
        response = await api.client.post("/web/check", data={"text": CRISIS}, headers=WEB)
    assert response.status_code == 200 and response.json()["explanation"] == crisis_text(registry)
    assert PHONE_NUMBER not in caplog.text and "crisis reply served" in caplog.text
    with pytest.raises(RuntimeError):  # without a crisis the failure is not hidden
        await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)


# --- the voice budget shared with WhatsApp ------------------------------------------------------------------


async def test_web_and_api_voice_notes_share_the_whatsapp_budget(
    api: Api, registry: LanguageRegistry
) -> None:
    api.app.state.services.flood = FloodGuard(voice_per_hour=1)
    files = {"audio": ("voice-note", VOICE_NOTE.read_bytes(), "audio/ogg")}
    assert (
        await api.client.post("/web/check", data={"language": "ha"}, files=files, headers=WEB)
    ).status_code == 200
    over = await api.client.post("/web/check", data={"language": "ha"}, files=files, headers=WEB)
    assert over.status_code == 429
    expected = plain_text(Messages(registry).text("ha", Msg.VOICE_LIMIT))
    assert over.json()["detail"] == {"problem": "voice_limit", "message": expected}
    assert (await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)).status_code == 200
    assert (await api.client.post("/v1/check/audio", files=files, headers=API)).status_code == 200
    partner_over = await api.client.post("/v1/check/audio", files=files, headers=API)
    assert partner_over.status_code == 429 and partner_over.json()["detail"]["problem"] == "voice_limit"


# --- a slow pattern write never costs the caller the answer ------------------------------------------------


async def test_a_slow_pattern_write_is_dropped(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(*_: Any, **__: Any) -> None:
        await asyncio.sleep(5)

    monkeypatch.setattr(app_module, "record_pattern", slow)
    monkeypatch.setattr(app_module, "PATTERN_WRITE_S", 0.01)
    response = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)
    assert response.status_code == 200 and response.json()["verdict"] == "likely_scam"
    assert len(await api.store.export_interactions()) == 1


def test_scam_patterns_are_redacted_whatever_the_caller_passes() -> None:
    from datetime import date

    pattern = ScamPattern.from_check(
        language="en", channel="web", verdict=Verdict.LIKELY_SCAM, flags=(Flag.MONEY_REQUEST,),
        text=f"send the OTP to {PHONE_NUMBER}", english=f"call {PHONE_NUMBER}", day=date(2026, 10, 11),
    )  # fmt: skip
    assert PHONE_NUMBER not in pattern.text and PHONE_NUMBER not in (pattern.english or "")


# --- log redaction for every entry point ---------------------------------------------------------------------


def test_records_are_redacted_for_handlers_added_later(api: Api) -> None:
    """The app's lifespan installed redaction; a handler attached afterwards (as `uvicorn --factory` does) and
    without any filter still never sees the number."""
    seen: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(self.format(record))

    handler = Collect()
    logger = logging.getLogger("uvicorn.error")
    logger.addHandler(handler)
    try:
        try:
            raise ValueError(f"bad input {PHONE_NUMBER}")
        except ValueError:
            logger.exception("failed for %s", PHONE_NUMBER)
    finally:
        logger.removeHandler(handler)
    assert seen and all(PHONE_NUMBER not in line for line in seen)


def test_install_log_redaction_is_idempotent() -> None:
    install_log_redaction()
    factory = logging.getLogRecordFactory()
    root = logging.getLogger()
    handler = logging.NullHandler()
    root.addHandler(handler)
    try:
        install_log_redaction()
        install_log_redaction()
        assert logging.getLogRecordFactory() is factory
        assert sum(isinstance(f, RedactingFilter) for f in handler.filters) == 1
    finally:
        root.removeHandler(handler)


def test_cli_installs_redaction_for_every_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "install_log_redaction", lambda: calls.append("installed"))
    assert cli.main(["purge", "--days", "0"]) == 2
    assert calls == ["installed"]


# --- WhatsApp API version --------------------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["25", "v25", "v25.0/../x", "latest"])
def test_whatsapp_api_version_is_format_checked(version: str) -> None:
    env = {"WHATSAPP_PHONE_NUMBER_ID": "1", "WHATSAPP_ACCESS_TOKEN": "t", "WHATSAPP_APP_SECRET": "s",
           "WHATSAPP_VERIFY_TOKEN": "v", "WHATSAPP_API_VERSION": version}  # fmt: skip
    with pytest.raises(ValueError, match="WHATSAPP_API_VERSION"):
        WhatsAppConfig.from_env(env)
    assert WhatsAppConfig.from_env({**env, "WHATSAPP_API_VERSION": "v26.0"}).api_version == "v26.0"
