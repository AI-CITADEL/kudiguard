"""Voice donation on the web page: its own versioned header, kept after the response, withdrawn by unticking."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from test_api import SESSION, VOICE_NOTE, WEB, Api, _start, settings

from kudiguard.donation import DONATION_VERSION, Sealer, VoiceVault
from kudiguard.languages import LanguageRegistry
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.base import VoiceSample, utcnow

DONATING = {**WEB, "x-kudiguard-donation": DONATION_VERSION}
FILES = {"audio": ("voice-note", VOICE_NOTE.read_bytes(), "audio/ogg")}


@pytest.fixture
async def api(registry: LanguageRegistry, tmp_path: Path) -> AsyncIterator[Api]:
    async for running in _start(registry, settings()):
        running.app.state.services.donations = VoiceVault(
            tmp_path / "voice", running.store, Sealer(b"s" * MIN_SECRET_BYTES)
        )
        yield running


def vault(api: Api) -> VoiceVault:
    found: VoiceVault = api.app.state.services.donations
    return found


async def test_ticked_box_keeps_the_recording_after_answering(api: Api) -> None:
    response = await api.client.post("/web/check", data={"language": "en"}, files=FILES, headers=DONATING)
    assert response.status_code == 200
    [kept] = await api.store.voice_samples()
    assert kept.channel == "web" and kept.language == "en" and kept.consent_version == DONATION_VERSION
    assert kept.user_hash == UserHasher(b"s" * MIN_SECRET_BYTES).user_id(f"web:{SESSION}")
    assert kept.hypothesis == response.json()["transcript"]
    assert await vault(api).read(kept.id) == VOICE_NOTE.read_bytes()


@pytest.mark.parametrize(
    ("headers", "data", "files"),
    [(WEB, {"language": "en"}, FILES),  # box not ticked
     (DONATING, {"language": "en", "text": "typed text is never donated"}, None),
     ({**WEB, "x-kudiguard-donation": ""}, {"language": "en"}, FILES)],
)  # fmt: skip
async def test_nothing_is_kept_without_the_box_or_for_text(
    api: Api, headers: dict[str, str], data: dict[str, str], files: object
) -> None:
    response = await api.client.post("/web/check", data=data, files=files, headers=headers)  # type: ignore[arg-type]
    assert response.status_code == 200 and await api.store.voice_samples() == []


@pytest.mark.parametrize("version", ["2020-01-01", "yes", "true"])
async def test_an_outdated_or_malformed_donation_header_is_refused_before_the_upload(
    api: Api, version: str
) -> None:
    headers = {**WEB, "x-kudiguard-donation": version}
    response = await api.client.post("/web/check", data={"language": "en"}, files=FILES, headers=headers)
    assert response.status_code == 403 and "reload" in response.json()["detail"]
    assert await api.store.voice_samples() == [] and await api.store.export_interactions() == []


async def test_a_failing_donation_never_affects_the_answer(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(sample: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(api.store, "add_voice_sample", broken)
    response = await api.client.post("/web/check", data={"language": "en"}, files=FILES, headers=DONATING)
    assert response.status_code == 200 and response.json()["verdict"] == "likely_scam"


async def test_unticking_deletes_this_sessions_recordings_only(api: Api) -> None:
    await api.client.post("/web/check", data={"language": "en"}, files=FILES, headers=DONATING)
    other = {**DONATING, "x-kudiguard-session": "zzzzzzzzzzzzzzzzzzzz"}
    await api.client.post("/web/check", data={"language": "en"}, files=FILES, headers=other)
    mine = {"x-kudiguard-session": SESSION}  # no consent header needed to withdraw
    response = await api.client.post("/web/donation/withdraw", headers=mine)
    assert response.status_code == 200 and response.json() == {"deleted": 1}
    [left] = await api.store.voice_samples()
    assert vault(api).path(left.id).exists() and len(list(vault(api).directory.iterdir())) == 1
    assert (await api.client.post("/web/donation/withdraw", headers=mine)).json() == {"deleted": 0}


async def test_withdrawal_needs_a_valid_session(api: Api) -> None:
    assert (await api.client.post("/web/donation/withdraw")).status_code == 422
    bad = {"x-kudiguard-session": "short"}
    assert (await api.client.post("/web/donation/withdraw", headers=bad)).status_code == 422


async def test_withdrawal_without_a_vault_still_deletes_rows(registry: LanguageRegistry) -> None:
    async for running in _start(registry, settings()):
        user_hash = UserHasher(b"s" * MIN_SECRET_BYTES).user_id(f"web:{SESSION}")
        await running.store.add_voice_sample(
            VoiceSample("a" * 32, user_hash, utcnow().date(), "en", "web", DONATION_VERSION, "x")
        )
        response = await running.client.post(
            "/web/donation/withdraw", headers={"x-kudiguard-session": SESSION}
        )
        assert response.json() == {"deleted": 1} and await running.store.voice_samples() == []
        # And without a vault, a ticked box keeps nothing.
        await running.client.post("/web/check", data={"language": "en"}, files=FILES, headers=DONATING)
        assert await running.store.voice_samples() == []


async def test_privacy_page_describes_donation_and_reviewers(api: Api) -> None:
    page = (await api.client.get("/privacy")).text
    assert 'id="donation"' in page and "12 months" in page and "18 or older" in page
    assert "STOP DONATING" in page and "AES-256-GCM" in page and "72 hours" in page
    assert "trusted reviewers" in page and "written confidentiality" in page
    assert "identify you" in page and "forward" in page and "{{" not in page


async def test_privacy_page_reflects_configured_voice_retention(registry: LanguageRegistry) -> None:
    async for running in _start(registry, settings(voice_retention_days=180)):
        page = (await running.client.get("/privacy")).text
        assert "180 days" in page and "12 months" not in page
