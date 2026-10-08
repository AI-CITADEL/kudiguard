"""One call for the pipeline: uploaded bytes -> transcript, with per-stage timings and typed failure reasons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from kudiguard.speech.audio import MAX_SECONDS, check_duration, decode, trim_silence
from kudiguard.speech.base import ASR, Transcript


@dataclass(frozen=True, slots=True)
class SpeechResult:
    transcript: Transcript
    decode_ms: int
    asr_ms: int
    total_ms: int


async def transcribe_upload(
    asr: ASR,
    data: bytes,
    language: str,
    *,
    ffmpeg: str,
    max_seconds: float = MAX_SECONDS,
    temp_dir: Path | None = None,
) -> SpeechResult:
    """Decode, trim, enforce the voiced-speech limit, transcribe.

    Raises ``AudioError`` (user-facing: ``.problem`` maps to a localised message) or ``ASRError`` (system fault).
    """
    started = time.perf_counter()
    audio = await decode(data, ffmpeg=ffmpeg, max_seconds=max_seconds, temp_dir=temp_dir)
    decoded = time.perf_counter()
    voiced = check_duration(trim_silence(audio), max_seconds)
    transcript = await asr.transcribe(voiced, language)
    finished = time.perf_counter()
    return SpeechResult(
        transcript=transcript,
        decode_ms=int((decoded - started) * 1000),
        asr_ms=transcript.latency_ms,
        total_ms=int((finished - started) * 1000),
    )
