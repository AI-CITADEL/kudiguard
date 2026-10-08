"""Lazy media download inside the pipeline (WhatsApp voice notes arrive as a media id, not bytes)."""

import asyncio

import pytest
from test_pipeline import VOICE_NOTE, Harness, interactions
from test_pipeline import h as _harness_fixture

from kudiguard.domain import MessageKind, Outcome
from kudiguard.i18n import Msg
from kudiguard.speech.base import AudioError, AudioProblem

h = _harness_fixture  # the shared conversation-harness fixture (pytest finds fixtures by module attribute)


async def test_voice_note_is_downloaded_lazily_and_answered(h: Harness) -> None:
    await h.onboard("en")
    h.messenger.media["MEDIA1"] = VOICE_NOTE.read_bytes()
    texts = await h.say(kind=MessageKind.AUDIO, media_id="MEDIA1")
    assert texts[0] == h.t(Msg.LISTENING) and h.t(Msg.VERDICT_LIKELY_SCAM) in texts[1]


@pytest.mark.parametrize(
    ("problem", "expected"),
    [(AudioProblem.TOO_LARGE, Msg.AUDIO_TOO_LONG), (AudioProblem.UNDECODABLE, Msg.NOT_UNDERSTOOD)],
)
async def test_unusable_downloads_get_a_friendly_reply(
    h: Harness, monkeypatch: pytest.MonkeyPatch, problem: AudioProblem, expected: Msg
) -> None:
    await h.onboard("en")

    async def failing(message):  # type: ignore[no-untyped-def]
        raise AudioError(problem, "from the channel")

    monkeypatch.setattr(h.messenger, "fetch_media", failing)
    texts = await h.say(kind=MessageKind.AUDIO, media_id="GONE")
    assert texts[-1] == h.t(expected, seconds=90)
    assert (await interactions(h))[-1]["outcome"] == Outcome.AUDIO_ERROR.value


async def test_download_fault_gets_retry_message(h: Harness) -> None:
    await h.onboard("en")
    texts = await h.say(kind=MessageKind.AUDIO, media_id="UNKNOWN")  # the fake raises RuntimeError
    assert texts[-1] == h.t(Msg.ERROR_RETRY)


async def test_download_happens_outside_the_speech_slots(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    h.engine.speech_slots = asyncio.Semaphore(1)
    await h.onboard("en")
    held_during_download: list[bool] = []
    slots = h.engine.speech_slots

    async def fetch(message):  # type: ignore[no-untyped-def]
        held_during_download.append(slots.locked())
        return VOICE_NOTE.read_bytes()

    monkeypatch.setattr(h.messenger, "fetch_media", fetch)
    await h.say(kind=MessageKind.AUDIO, media_id="M")
    assert held_during_download == [False]


async def test_audio_without_bytes_or_media_reference(h: Harness) -> None:
    await h.onboard("en")
    texts = await h.say(kind=MessageKind.AUDIO)
    assert texts[-1] == h.t(Msg.NOT_UNDERSTOOD, seconds=90)


async def test_stuck_download_is_bounded_by_the_turn_deadline(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    h.configure(turn_deadline_s=0.3)
    await h.onboard("en")

    async def stuck(message):  # type: ignore[no-untyped-def]
        await asyncio.Event().wait()

    monkeypatch.setattr(h.messenger, "fetch_media", stuck)
    texts = await asyncio.wait_for(h.say(kind=MessageKind.AUDIO, media_id="SLOW"), timeout=3)
    assert texts[-1] == h.t(Msg.NOT_UNDERSTOOD, seconds=90)  # asked to resend, not a generic error
    assert (await interactions(h))[-1]["outcome"] == Outcome.AUDIO_ERROR.value


async def test_no_time_left_to_download(h: Harness) -> None:
    from kudiguard.domain import InboundMessage

    message = InboundMessage("whatsapp", "w1", "u", MessageKind.AUDIO, media_id="M")
    with pytest.raises(AudioError) as caught:
        await h.conversation._audio_of(message, deadline=0.0)  # long past
    assert caught.value.problem is AudioProblem.TIMEOUT


async def test_busy_speech_slots_are_bounded_by_the_turn_deadline(h: Harness) -> None:
    h.configure(turn_deadline_s=0.3)
    await h.onboard("en")
    h.engine.speech_slots = asyncio.Semaphore(0)  # every slot busy with web/API voice
    h.messenger.media["M"] = VOICE_NOTE.read_bytes()
    texts = await asyncio.wait_for(h.say(kind=MessageKind.AUDIO, media_id="M"), timeout=3)
    assert texts[-1] == h.t(Msg.NOT_UNDERSTOOD, seconds=90)  # "please resend", not a generic failure
    assert (await interactions(h))[-1]["outcome"] == Outcome.AUDIO_ERROR.value


async def test_no_time_left_to_transcribe(h: Harness) -> None:
    from kudiguard.pipeline import _Turn

    turn = _Turn(h.msg("x"), h.user_hash, "en", 0.0, deadline=0.0)
    with pytest.raises(AudioError) as caught:
        await h.conversation._transcribe(b"", turn)
    assert caught.value.problem is AudioProblem.TIMEOUT
