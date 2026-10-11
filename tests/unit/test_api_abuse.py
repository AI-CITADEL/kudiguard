"""Malicious use on the web page and in the wiring (docs/ABUSE.md): settings, services, new web sessions,
crisis privacy, the learning gate on web checks and donations."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_api import OTP_SCAM, SESSION, VOICE_NOTE, WEB, Api, _start, settings
from test_m9_support import base_settings

from kudiguard.api.services import build_abuse_screens, build_limits, build_pack_screen, build_services
from kudiguard.config import Settings
from kudiguard.donation import DONATION_VERSION, Sealer, VoiceVault
from kudiguard.i18n import Messages, Msg
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import Purpose
from kudiguard.privacy import MIN_SECRET_BYTES
from kudiguard.render import plain_text
from kudiguard.safety.abuse import ScreenMode
from kudiguard.safety.limits import NewUserGate

CRISIS = "They took all my savings. I want to kill myself."


@pytest.fixture
async def api(registry: LanguageRegistry) -> AsyncIterator[Api]:
    async for running in _start(registry, settings()):
        yield running


def use_gate(api: Api, gate: NewUserGate) -> None:
    """Replace the new-user gate on the shared admission (Services attaches the default one)."""
    admission = api.app.state.services.admission
    admission.checks.clear()
    admission.on_admit.clear()
    gate.attach(admission)


def session(n: int) -> dict[str, str]:
    return {**WEB, "x-kudiguard-session": f"abuse-session-{n:04d}-xxxxx"}


async def test_minted_web_sessions_meet_the_new_user_gate(api: Api, registry: LanguageRegistry) -> None:
    use_gate(api, NewUserGate(per_hour=1))
    assert (
        await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=session(1))
    ).status_code == 200
    busy = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=session(2))
    assert busy.status_code == 503 and busy.headers["retry-after"] == "3600"
    detail = busy.json()["detail"]
    assert detail == {"problem": "busy_new_users",
                      "message": plain_text(Messages(registry).text("en", Msg.NEW_USERS_BUSY))}  # fmt: skip
    again = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=session(1))
    assert again.status_code == 200  # a returning session is always served
    assert api.app.state.services.admission.in_flight == 0  # no leaked holder


async def test_web_crisis_reply_keeps_no_text(api: Api, registry: LanguageRegistry) -> None:
    response = await api.client.post("/web/check", data={"text": CRISIS}, headers=WEB)
    body = response.json()
    assert body["verdict"] == "not_a_check" and body["headline"] is None
    assert body["explanation"] == plain_text(Messages(registry).text("en", Msg.CRISIS_REPLY))
    [row] = await api.store.export_interactions()
    assert row["transcript"] is None and not api.llm.calls(Purpose.COMPOSE)


async def test_web_pattern_flood_keeps_one_example(api: Api) -> None:
    for _ in range(4):
        await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=WEB)
    assert len(await api.store.export_patterns()) == 1


async def test_a_flagged_web_voice_note_is_not_donated(registry: LanguageRegistry, tmp_path: Path) -> None:
    async for api in _start(registry, settings(voice_donation=True)):
        api.app.state.services.donations = VoiceVault(
            tmp_path / "voice", api.store, Sealer(b"s" * MIN_SECRET_BYTES)
        )
        files = {"audio": ("voice-note", VOICE_NOTE.read_bytes(), "audio/ogg")}
        headers = {**WEB, "x-kudiguard-donation": DONATION_VERSION}
        api.asr.text = "you stupid useless bot"
        response = await api.client.post("/web/check", data={"language": "en"}, files=files, headers=headers)
        assert response.status_code == 200 and await api.store.voice_samples() == []
        assert SESSION not in str(await api.store.export_interactions())


# --- settings and wiring -----------------------------------------------------------------------------------------


def test_settings_defaults_and_validation() -> None:
    config = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (config.abuse_crisis_mode, config.abuse_injection_mode, config.reply_screen_mode) == (
        "enforce", "enforce", "enforce"
    )  # fmt: skip
    assert config.pack_screen_core_mode == "shadow" and config.new_users_per_hour == 100
    unset = Settings(_env_file=None, new_users_per_day="", voice_notes_per_user_per_hour=" ")  # type: ignore[call-arg, arg-type]
    assert unset.new_users_per_day is None and unset.voice_notes_per_user_per_hour is None
    with pytest.raises(ValidationError, match="cannot exceed"):
        Settings(_env_file=None, new_users_per_hour=50, new_users_per_day=10)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, abuse_crisis_mode="on")  # type: ignore[call-arg, arg-type]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, reply_echo_words=2)  # type: ignore[call-arg]


def test_builders_follow_settings(registry: LanguageRegistry) -> None:
    config = Settings(_env_file=None, abuse_injection_mode="shadow", abuse_crisis_mode="off",  # type: ignore[call-arg]
                      reply_screen_mode="shadow", reply_echo_words=9, asr_hallucination_mode="off",
                      pack_screen_mode="shadow", pack_screen_core_mode="enforce", learning_gate_mode="off",
                      new_user_mode="shadow", new_users_per_hour=5, new_users_per_day=7)  # fmt: skip
    screens = build_abuse_screens(config, registry)
    assert screens["screen"].modes.injection is ScreenMode.SHADOW
    assert screens["screen"].modes.crisis is ScreenMode.OFF
    assert screens["reply_screen"].mode is ScreenMode.SHADOW and screens["reply_screen"]._echo_words == 9
    assert screens["transcripts"].mode is ScreenMode.OFF
    pack = build_pack_screen(config, registry)
    assert (pack.partner_mode, pack.core_mode) == (ScreenMode.SHADOW, ScreenMode.ENFORCE)
    gates = build_limits(config)
    assert gates["learning"].mode is ScreenMode.OFF
    assert (gates["new_users"].per_hour, gates["new_users"].per_day) == (5, 7)
    assert gates["new_users"].mode is ScreenMode.SHADOW


async def test_build_services_wires_the_defences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {"WHATSAPP_PHONE_NUMBER_ID": "P", "WHATSAPP_ACCESS_TOKEN": "t",
                        "WHATSAPP_APP_SECRET": "s", "WHATSAPP_VERIFY_TOKEN": "v"}.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    config = base_settings(tmp_path, whatsapp_enabled=True, packs=("kudiguard-core",), flood_mode="shadow",
                           max_queued_per_user=3, abuse_scam_tool_mode="shadow")  # fmt: skip
    services = await build_services(config)
    try:
        conversation = services.conversation
        assert conversation is not None
        assert conversation._admission is services.admission and conversation._learning is services.learning
        assert len(services.admission.checks) == 1 and len(services.admission.on_admit) == 1  # the gate
        assert conversation._flood.mode is ScreenMode.SHADOW and conversation._flood.max_queued == 3
        assert services.engine.screen.modes.scam_tool is ScreenMode.SHADOW
    finally:
        await services.aclose()


async def test_web_new_users_beyond_the_daily_cap(api: Api, registry: LanguageRegistry) -> None:
    use_gate(api, NewUserGate(per_hour=5, per_day=1))
    assert (
        await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=session(1))
    ).status_code == 200
    busy = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=session(2))
    assert busy.json()["detail"]["message"] == plain_text(
        Messages(registry).text("en", Msg.NEW_USERS_BUSY_TODAY)
    )


async def test_web_crisis_that_is_also_a_scam(api: Api, registry: LanguageRegistry) -> None:
    message = "I wan kill myself. Somebody text me say make I send BVN and OTP to unlock my account."
    body = (await api.client.post("/web/check", data={"text": message}, headers=WEB)).json()
    messages = Messages(registry)
    assert body["verdict"] == "likely_scam" and body["report_channels"]
    assert body["explanation"].startswith(plain_text(messages.text("en", Msg.CRISIS_REPLY)))
    [row] = await api.store.export_interactions()
    assert row["transcript"] is None and row["english"] is None
    assert await api.store.export_patterns() == []


@pytest.mark.parametrize(
    "headers",
    [
        {"x-kudiguard-session": SESSION},  # no consent header
        {**WEB, "x-kudiguard-session": "bad"},  # invalid session
    ],
)
async def test_web_crisis_is_answered_before_every_gate(
    api: Api, registry: LanguageRegistry, headers: dict[str, str]
) -> None:
    response = await api.client.post("/web/check", data={"text": CRISIS, "language": "ha"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["explanation"] == plain_text(Messages(registry).text("ha", Msg.CRISIS_REPLY))
    assert await api.store.export_interactions() == []


async def test_web_crisis_when_rate_limited_or_full(registry: LanguageRegistry) -> None:
    async for api in _start(registry, settings(web_rate_per_minute=1, active_user_cap=1)):
        await api.check()  # the licence cap is now full for new sessions
        busy = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=session(3))
        assert busy.status_code in (429, 503)
        helped = await api.client.post(
            "/web/check", data={"text": "I want to kill myself"}, headers=session(4)
        )
        assert helped.status_code == 200 and "112" in helped.json()["explanation"]
        unknown = await api.client.post(
            "/web/check", data={"text": "Ya ce zai kashe kansa"}, headers=session(5)
        )
        assert unknown.json()["language"] == "ha"


async def test_web_crisis_first_skips_voice_and_large_or_malformed_bodies(api: Api) -> None:
    files = {"audio": ("voice-note", VOICE_NOTE.read_bytes(), "audio/ogg")}
    voice = await api.client.post(
        "/web/check", data={"text": CRISIS}, files=files, headers={"x-kudiguard-session": SESSION}
    )
    assert voice.status_code == 403  # a voice note meets the gates first
    small_voice = {"audio": ("voice-note", b"tiny", "audio/ogg")}
    tiny = await api.client.post("/web/check", data={"text": CRISIS}, files=small_voice,
                                 headers={"x-kudiguard-session": SESSION})  # fmt: skip
    assert tiny.status_code == 403
    broken = await api.client.post(
        "/web/check", content=b"--x\r\nbroken", headers={"content-type": "multipart/form-data; boundary=x"}
    )
    assert broken.status_code == 403
    plain = await api.client.post("/web/check", content=b"{}", headers={"content-type": "application/json"})
    assert plain.status_code == 403
    calm = await api.client.post("/web/check", data={"text": "how do I save money"}, headers={})
    assert calm.status_code == 403


async def test_the_new_user_gate_also_applies_without_a_licence_cap(registry: LanguageRegistry) -> None:
    """An anti-flood guard, not a licence measure (docs/ABUSE.md): it holds with KUDIGUARD_ACTIVE_USER_CAP unset."""
    async for api in _start(registry, settings(active_user_cap=None)):
        use_gate(api, NewUserGate(per_hour=1))
        assert (
            await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=session(1))
        ).status_code == 200
        busy = await api.client.post("/web/check", data={"text": OTP_SCAM}, headers=session(2))
        assert busy.status_code == 503 and busy.json()["detail"]["problem"] == "busy_new_users"
        assert api.app.state.services.admission.in_flight == 0  # no leaked holder
