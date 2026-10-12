"""One call for the pipeline: uploaded bytes -> transcript, with per-stage timings and typed failure reasons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from kudiguard.speech.audio import MAX_SECONDS, MIN_VOICED_SECONDS, check_duration, decode, trim_silence
from kudiguard.speech.base import ASR, Transcript


@dataclass(frozen=True, slots=True)
class SpeechResult:
    transcript: Transcript
    decode_ms: int
    asr_ms: int
    total_ms: int
    filtered: bool = False  # recogniser hallucinations were removed from ``transcript`` (never donated)
    # the recogniser's own words when ``asr_terms`` corrected ``transcript`` ("bifi ẹni" -> "BVN"); None = unchanged
    recognised: str | None = None

    @property
    def donated_text(self) -> str:
        """The ASR hypothesis kept with a donated voice note: the recogniser's words, never the term corrections
        (they would flatter ``kudiguard voice wer`` and anchor transcribers)."""
        return self.transcript.text if self.recognised is None else self.recognised


async def transcribe_upload(
    asr: ASR,
    data: bytes,
    language: str,
    *,
    ffmpeg: str,
    max_seconds: float = MAX_SECONDS,
    min_voiced_seconds: float = MIN_VOICED_SECONDS,
    temp_dir: Path | None = None,
) -> SpeechResult:
    """Decode, trim, enforce the voiced-speech limits (too short = NO_SPEECH), transcribe.

    Raises ``AudioError`` (user-facing: ``.problem`` maps to a localised message) or ``ASRError`` (system fault).
    """
    started = time.perf_counter()
    audio = await decode(data, ffmpeg=ffmpeg, max_seconds=max_seconds, temp_dir=temp_dir)
    decoded = time.perf_counter()
    voiced = check_duration(trim_silence(audio), max_seconds, min_voiced_seconds=min_voiced_seconds)
    transcript = await asr.transcribe(voiced, language)
    finished = time.perf_counter()
    return SpeechResult(
        transcript=transcript,
        decode_ms=int((decoded - started) * 1000),
        asr_ms=transcript.latency_ms,
        total_ms=int((finished - started) * 1000),
    )
