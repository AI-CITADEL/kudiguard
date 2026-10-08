import asyncio
import threading
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from kudiguard.speech.base import SAMPLE_RATE, ASRError, Audio, AudioError
from kudiguard.speech.natlas_asr import NatlasASR, collapse_repeats

MODELS = {"ha": "NCAIR1/Hausa-ASR", "en": "NCAIR1/NigerianAccentedEnglish", "yo": "NCAIR1/Yoruba-ASR"}


def speech(seconds: float) -> Audio:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return Audio((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


class FakeLoader:
    def __init__(self, *, fail_on: str | None = None, delay: float = 0.0) -> None:
        self.loads: list[str] = []
        self.calls: list[tuple[str, float]] = []
        self.fail_on = fail_on
        self.delay = delay
        self.threads: set[int] = set()

    def __call__(self, repo: str):  # type: ignore[no-untyped-def]
        time.sleep(self.delay)
        if repo == self.fail_on:
            raise OSError("401 gated repo")
        self.loads.append(repo)

        def transcribe(samples: np.ndarray) -> str:
            self.threads.add(threading.get_ident())
            self.calls.append((repo, len(samples) / SAMPLE_RATE))
            return f" {repo.split('/')[1]} chunk {len(self.calls)} "

        return transcribe


async def test_routes_by_language_and_reports_metrics() -> None:
    loader = FakeLoader()
    asr = NatlasASR(MODELS, loader=loader)
    transcript = await asr.transcribe(speech(5), "ha")
    assert transcript.text == "Hausa-ASR chunk 1"
    assert transcript.language == "ha" and transcript.chunks == 1
    assert transcript.duration_s == pytest.approx(5.0, abs=0.5)
    assert loader.loads == ["NCAIR1/Hausa-ASR"]
    assert threading.get_ident() not in loader.threads  # inference runs off the event loop


async def test_long_audio_is_chunked_and_joined() -> None:
    loader = FakeLoader()
    transcript = await NatlasASR(MODELS, loader=loader).transcribe(speech(70), "en")
    assert transcript.chunks == 3
    assert transcript.text == " ".join(f"NigerianAccentedEnglish chunk {i}" for i in (1, 2, 3))
    assert all(seconds <= 30.0 for _, seconds in loader.calls)


async def test_model_loaded_once_under_concurrency() -> None:
    loader = FakeLoader(delay=0.05)
    asr = NatlasASR(MODELS, loader=loader)
    await asyncio.gather(*(asr.transcribe(speech(2), "ha") for _ in range(5)))
    assert loader.loads == ["NCAIR1/Hausa-ASR"]


async def test_least_recently_used_model_is_evicted() -> None:
    loader = FakeLoader()
    asr = NatlasASR(MODELS, loader=loader, max_loaded=2)
    for language in ("ha", "en", "ha", "yo", "en"):
        await asr.transcribe(speech(2), language)
    # ha, en loaded; ha used; yo loaded -> evicts en (LRU); en again -> reloaded, evicts ha
    assert loader.loads == [
        "NCAIR1/Hausa-ASR",
        "NCAIR1/NigerianAccentedEnglish",
        "NCAIR1/Yoruba-ASR",
        "NCAIR1/NigerianAccentedEnglish",
    ]


async def test_unsupported_language() -> None:
    asr = NatlasASR(MODELS, loader=FakeLoader())
    assert asr.supports("ha") and not asr.supports("ig")
    with pytest.raises(ASRError, match="ig"):
        await asr.transcribe(speech(2), "ig")


async def test_load_failure_fails_fast_during_cooldown_then_retries() -> None:
    now = [0.0]
    loader = FakeLoader(fail_on="NCAIR1/Hausa-ASR", delay=0.05)
    asr = NatlasASR(MODELS, loader=loader, failure_cooldown_s=300, clock=lambda: now[0])
    with pytest.raises(ASRError, match="Hausa-ASR"):
        await asr.transcribe(speech(2), "ha")
    loader.fail_on = None
    started = time.perf_counter()
    with pytest.raises(ASRError, match="unavailable"):
        await asr.transcribe(speech(2), "ha")  # no second slow load attempt on a user's request
    assert time.perf_counter() - started < 0.04 and loader.loads == []
    now[0] = 301
    assert (await asr.transcribe(speech(2), "ha")).text.startswith("Hausa-ASR chunk")


async def test_preload_reports_failures_without_raising() -> None:
    errors = await NatlasASR(MODELS, loader=FakeLoader(fail_on="NCAIR1/Yoruba-ASR")).preload(["ha", "yo"])
    assert list(errors) == ["yo"]


async def test_inference_failure_is_asr_error() -> None:
    def loader(repo: str):  # type: ignore[no-untyped-def]
        def broken(samples: np.ndarray) -> str:
            raise RuntimeError("onnx/torch exploded")

        return broken

    with pytest.raises(ASRError, match="exploded"):
        await NatlasASR(MODELS, loader=loader).transcribe(speech(2), "ha")


async def test_silence_is_rejected_before_the_model() -> None:
    loader = FakeLoader()
    with pytest.raises(AudioError, match="speech"):
        await NatlasASR(MODELS, loader=loader).transcribe(Audio(np.zeros(SAMPLE_RATE * 3, np.float32)), "ha")
    assert loader.loads == []


async def test_preload_warms_models() -> None:
    loader = FakeLoader()
    asr = NatlasASR(MODELS, loader=loader)
    await asr.preload(["ha", "en"])
    assert loader.loads == ["NCAIR1/Hausa-ASR", "NCAIR1/NigerianAccentedEnglish"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Thank you. Thank you. Thank you. Thank you.", "Thank you."),
        ("Ku tura OTP ku tura OTP ku tura OTP ku tura OTP", "Ku tura OTP"),
        ("small small small money", "small small small money"),
        ("No no no, do not pay", "No no no, do not pay"),
        ("kíákíá kíákíá kíákíá", "kíákíá kíákíá kíákíá"),
        ("yes yes yes yes yes yes yes", "yes"),
        ("Send the code. Send the code. Then call me.", "Send the code. Send the code. Then call me."),
        ("  hello   world ", "hello world"),
        ("", ""),
    ],
)
def test_collapse_whisper_repetition_loops(raw: str, expected: str) -> None:
    assert collapse_repeats(raw) == expected


# --- runtime loaders (fake libraries: no models, no network) ------------------------------------------


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperModel:
    instances: ClassVar[list["FakeWhisperModel"]] = []

    def __init__(self, path: str, *, device: str, compute_type: str, cpu_threads: int) -> None:
        self.path, self.device, self.compute_type, self.cpu_threads = path, device, compute_type, cpu_threads
        self.kwargs: dict[str, object] = {}
        FakeWhisperModel.instances.append(self)

    def transcribe(self, samples: np.ndarray, **kwargs: object) -> tuple[list[_Segment], object]:
        self.kwargs = kwargs
        return [_Segment(" Hello "), _Segment("world. ")], None


class FakeConverter:
    conversions: ClassVar[list[str]] = []

    def __init__(self, repo: str, copy_files: list[str]) -> None:
        self.repo = repo

    def convert(self, output: str, quantization: str) -> None:
        FakeConverter.conversions.append(self.repo)
        out = Path(output)
        out.mkdir(parents=True)
        (out / "model.bin").write_bytes(b"ct2")


@pytest.fixture
def fake_ct2(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    FakeWhisperModel.instances.clear()
    FakeConverter.conversions.clear()
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    converters = types.SimpleNamespace(TransformersConverter=FakeConverter)
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(converters=converters))
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", converters)


def test_ct2_loader_converts_once_then_reuses(tmp_path: Path, fake_ct2: None) -> None:
    from kudiguard.speech.natlas_asr import ct2_loader

    loader = ct2_loader(cache_dir=tmp_path, whisper_languages={"NCAIR1/Hausa-ASR": "ha"}, threads=6)
    transcribe = loader("NCAIR1/Hausa-ASR")
    assert transcribe(np.zeros(16_000, np.float32)) == "Hello world."
    model = FakeWhisperModel.instances[-1]
    assert model.path.endswith("NCAIR1--Hausa-ASR-int8") and model.cpu_threads == 6 and model.device == "cpu"
    assert model.kwargs["language"] == "ha" and model.kwargs["condition_on_previous_text"] is False
    loader("NCAIR1/Hausa-ASR")
    assert FakeConverter.conversions == ["NCAIR1/Hausa-ASR"]  # cached after the first conversion
    assert not [p for p in (tmp_path / "ct2").iterdir() if p.name.startswith("tmp")]  # staging cleaned up


def test_ct2_loader_language_none_lets_model_decide(tmp_path: Path, fake_ct2: None) -> None:
    from kudiguard.speech.natlas_asr import ct2_loader

    ct2_loader(cache_dir=tmp_path, whisper_languages={"NCAIR1/Igbo-ASR": None})("NCAIR1/Igbo-ASR")(
        np.zeros(16_000, np.float32)
    )
    assert FakeWhisperModel.instances[-1].kwargs["language"] is None


def test_failed_conversion_leaves_no_partial_model(
    tmp_path: Path, fake_ct2: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kudiguard.speech.natlas_asr import ct2_loader

    def boom(self: FakeConverter, output: str, quantization: str) -> None:
        Path(output).mkdir(parents=True)
        (Path(output) / "half.bin").write_bytes(b"x")
        raise OSError("401 gated")

    monkeypatch.setattr(FakeConverter, "convert", boom)
    with pytest.raises(OSError, match="gated"):
        ct2_loader(cache_dir=tmp_path)("NCAIR1/Hausa-ASR")
    assert not (tmp_path / "ct2" / "NCAIR1--Hausa-ASR-int8").exists()


async def test_first_call_latency_excludes_model_load() -> None:
    transcript = await NatlasASR(MODELS, loader=FakeLoader(delay=0.3)).transcribe(speech(2), "ha")
    assert transcript.latency_ms < 300
