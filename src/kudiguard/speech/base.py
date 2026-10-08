"""Speech port: audio and transcript types, errors, and the protocol every ASR backend implements.

This package imports nothing else from ``kudiguard`` so it can be lifted unchanged into the N-ATLAS gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000  # NCAIR1 ASR models (Whisper-small) expect 16 kHz mono

Samples = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Audio:
    """Mono float32 PCM in [-1, 1] at ``SAMPLE_RATE``."""

    samples: Samples

    @property
    def duration_s(self) -> float:
        return len(self.samples) / SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str
    duration_s: float
    latency_ms: int = 0
    chunks: int = 1


class AudioProblem(StrEnum):
    """Why an upload cannot be used; the pipeline maps each to a localised message (internals are only logged)."""

    EMPTY = "empty"
    TOO_LARGE = "too_large"
    UNDECODABLE = "undecodable"
    TIMEOUT = "timeout"
    TOO_LONG = "too_long"
    NO_SPEECH = "no_speech"


class AudioError(ValueError):
    """The audio cannot be used (a user-facing condition, not a system fault). ``detail`` is for logs only."""

    def __init__(self, problem: AudioProblem, detail: str = "") -> None:
        super().__init__(f"{problem.value}: {detail}" if detail else problem.value)
        self.problem = problem
        self.detail = detail

    @property
    def too_long(self) -> bool:
        return self.problem is AudioProblem.TOO_LONG


class ASRError(RuntimeError):
    """The recogniser failed (model missing, load error, inference error)."""


class ASR(Protocol):
    async def transcribe(self, audio: Audio, language: str) -> Transcript:
        """Transcribe ``audio`` spoken in ``language``. Raises ``ASRError``."""
        ...

    def supports(self, language: str) -> bool: ...
