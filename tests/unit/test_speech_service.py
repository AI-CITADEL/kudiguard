from pathlib import Path

import pytest

from kudiguard.speech.audio import find_ffmpeg
from kudiguard.speech.base import Audio, AudioError, AudioProblem, Transcript
from kudiguard.speech.service import transcribe_upload

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "voice_note_en_otp.ogg"


class FakeASR:
    def __init__(self) -> None:
        self.durations: list[float] = []

    def supports(self, language: str) -> bool:
        return True

    async def transcribe(self, audio: Audio, language: str) -> Transcript:
        self.durations.append(audio.duration_s)
        return Transcript("hello", language, audio.duration_s, latency_ms=7)


async def test_upload_to_transcript_with_timings() -> None:
    asr = FakeASR()
    result = await transcribe_upload(asr, FIXTURE.read_bytes(), "en", ffmpeg=find_ffmpeg())
    assert result.transcript.text == "hello"
    assert result.asr_ms == 7 and result.decode_ms > 0 and result.total_ms >= result.decode_ms
    assert 9 < asr.durations[0] < 11.5  # silence trimmed before the model sees it


async def test_voiced_limit_enforced_before_transcription() -> None:
    asr = FakeASR()
    with pytest.raises(AudioError) as info:
        await transcribe_upload(asr, FIXTURE.read_bytes(), "en", ffmpeg=find_ffmpeg(), max_seconds=5)
    assert info.value.problem is AudioProblem.TOO_LONG
    assert asr.durations == []


async def test_too_short_voiced_speech_rejected_before_transcription() -> None:
    asr = FakeASR()
    with pytest.raises(AudioError) as info:
        await transcribe_upload(asr, FIXTURE.read_bytes(), "en", ffmpeg=find_ffmpeg(), min_voiced_seconds=20)
    assert info.value.problem is AudioProblem.NO_SPEECH
    assert asr.durations == []
