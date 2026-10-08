"""Speech recognition against real models (``pytest -m live``).

* Proxy: public ``openai/whisper-small`` (same architecture as the NCAIR1 fine-tunes) — runs whenever the converted
  CTranslate2 model is cached (first run converts it, ~2 min).
* NCAIR1: the official N-ATLAS models — skipped until the gated weights are accessible (HF_TOKEN + accepted terms).
"""

import os
from pathlib import Path

import pytest

from kudiguard.speech.audio import decode, find_ffmpeg
from kudiguard.speech.metrics import word_error_rate
from kudiguard.speech.natlas_asr import NatlasASR, ct2_loader

pytestmark = pytest.mark.live
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "voice_note_en_otp.ogg"
CACHE = Path(os.environ.get("KUDIGUARD_MODEL_CACHE", r"***REMOVED***\ncair\.cache"))


async def test_whatsapp_voice_note_with_proxy_model() -> None:
    pytest.importorskip("faster_whisper")
    audio = await decode(FIXTURE.read_bytes(), ffmpeg=find_ffmpeg())
    asr = NatlasASR(
        {"en": "openai/whisper-small"},
        loader=ct2_loader(cache_dir=CACHE, whisper_languages={"openai/whisper-small": "en"}),
    )
    transcript = await asr.transcribe(audio, "en")
    reference = (
        "Hello. Someone called me and said my account will be blocked. He asked me to send the OTP code that was "
        "sent to my phone. Is this a scam?"
    )
    assert word_error_rate(reference, transcript.text) <= 0.1
    assert transcript.latency_ms < 10_000


async def test_official_hausa_model_loads() -> None:
    pytest.importorskip("faster_whisper")
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("NCAIR1 models are gated: set HF_TOKEN after accepting the model terms")
    asr = NatlasASR({"ha": "NCAIR1/Hausa-ASR"}, loader=ct2_loader(cache_dir=CACHE, whisper_languages={}))
    await asr.preload(["ha"])
