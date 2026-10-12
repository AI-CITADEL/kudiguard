"""Official N-ATLAS speech recognition: one NCAIR1 Whisper-small fine-tune per language, on CPU.

The GPU is fully used by the N-ATLAS LLM (spike S1), so ASR runs on CPU. Models are loaded lazily, once per language
even under concurrent requests, and at most ``max_loaded`` stay in memory (least recently used is evicted). A failed
load is remembered for ``failure_cooldown_s`` so a missing/gated model fails fast instead of on every voice note.
Inference runs in a worker thread, one chunk at a time (CPU-bound; parallel chunks would just contend).

Runtimes (``Loader``): ``ct2_loader`` (default: the same NCAIR1 weights converted to CTranslate2 and run with
faster-whisper, ~4-10x faster on CPU, spike S4) and ``transformers_loader`` (reference / fallback). Both use the same
decoding settings (forced language token, temperature 0, no timestamps) so their accuracy can be compared fairly.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from kudiguard.speech.audio import split, trim_silence
from kudiguard.speech.base import SAMPLE_RATE, ASRError, Audio, AudioError, AudioProblem, Samples, Transcript

log = logging.getLogger(__name__)

Transcriber = Callable[[Samples], str]
Loader = Callable[[str], Transcriber]  # model repo id -> transcriber (slow, blocking)

DEFAULT_THREADS = 8  # measured: 8 is within 1% of 16 and leaves cores for the app; 24 is slower (M5 review)
DEFAULT_MAX_LOADED = 4  # one per enabled language (~250 MB each in int8)
FAILURE_COOLDOWN_S = 300.0
MAX_REPEAT_UNIT = 8  # words
MIN_PHRASE_REPEATS = 3  # "send the code send the code send the code"
MIN_WORD_REPEATS = 5  # single words repeat legitimately: "small small", "no no no", Yoruba reduplication
TOKENIZER_FILES = ("tokenizer.json", "preprocessor_config.json")
# A transcript that is mostly one loop ("ìdáná-ìdáná-ìdáná-…", live Yoruba voice note 2026-10-12) is not speech the
# model understood: the LLM then invents a message. Rejected when at least LOOP_MIN_UNITS words/hyphen parts shrink
# below LOOP_KEPT_SHARE once loops are collapsed.
LOOP_MIN_UNITS = 20
LOOP_KEPT_SHARE = 0.4
_HYPHEN_LOOP = re.compile(rf"([^\s\-\u2013]+)(?:[\-\u2013]\1){{{MIN_WORD_REPEATS - 1},}}")
_UNIT = re.compile(r"[^\s\-\u2013]+")


def transformers_loader(
    *,
    cache_dir: Path | None = None,
    token: str | None = None,
    whisper_languages: Mapping[str, str | None] | None = None,
    threads: int = DEFAULT_THREADS,
) -> Loader:
    """Load NCAIR1 models with Hugging Face ``transformers`` on CPU (reference runtime; extra ``kudiguard[asr]``)."""
    languages = dict(whisper_languages or {})

    def load(repo: str) -> Transcriber:
        import torch  # optional dependency
        from transformers import pipeline

        torch.set_num_threads(threads)
        asr: Any = pipeline(
            "automatic-speech-recognition",
            model=repo,
            device="cpu",
            token=token,
            model_kwargs={"cache_dir": str(cache_dir)} if cache_dir else None,
        )
        generate: dict[str, Any] = {"task": "transcribe", "temperature": 0.0}
        if language := languages.get(repo):
            generate["language"] = language

        def transcribe(samples: Samples) -> str:
            return str(asr({"raw": samples, "sampling_rate": SAMPLE_RATE}, generate_kwargs=generate)["text"])

        return transcribe

    return load


def ct2_loader(
    *,
    cache_dir: Path,
    whisper_languages: Mapping[str, str | None] | None = None,
    threads: int = DEFAULT_THREADS,
    quantization: str = "int8",
) -> Loader:
    """Load NCAIR1 models via CTranslate2/faster-whisper (default runtime; extra ``kudiguard[asr]``).

    Each Hugging Face model is converted once to CTranslate2 ``quantization`` (``int8`` default; ``float32`` keeps
    full precision) and cached under ``cache_dir/ct2``. Spike S4: an 11 s voice note in ~1.7 s vs 6-20 s with
    transformers on the same CPU. ``whisper_languages`` maps repo -> Whisper language token (None = the model's own
    generation config decides, e.g. Igbo, which Whisper has no token for). Gated repos need ``HF_TOKEN`` in the
    environment for the one-time conversion (``scripts/prepare_asr.py``).
    """
    languages = dict(whisper_languages or {})

    def load(repo: str) -> Transcriber:
        from faster_whisper import WhisperModel  # optional dependency

        target = cache_dir / "ct2" / f"{repo.replace('/', '--')}-{quantization}"
        if not (target / "model.bin").is_file():
            _convert(repo, target, quantization)
        model = WhisperModel(str(target), device="cpu", compute_type=quantization, cpu_threads=threads)
        language = languages.get(repo)

        def transcribe(samples: Samples) -> str:
            segments, _info = model.transcribe(
                samples,
                language=language,
                task="transcribe",
                beam_size=1,
                temperature=0.0,
                without_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(segment.text.strip() for segment in segments)

        return transcribe

    return load


def _convert(repo: str, target: Path, quantization: str) -> None:
    """One-time Hugging Face -> CTranslate2 conversion, written atomically (a crash never leaves a half model)."""
    from ctranslate2.converters import TransformersConverter  # optional dependency

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        output = Path(staging) / "model"
        try:
            TransformersConverter(repo, copy_files=list(TOKENIZER_FILES)).convert(
                str(output), quantization=quantization
            )
        except (OSError, ValueError) as exc:
            if "tokenizer" not in str(exc) and "preprocessor" not in str(exc):
                raise
            log.warning("converting %s without optional tokenizer files: %s", repo, exc)
            shutil.rmtree(output, ignore_errors=True)
            TransformersConverter(repo).convert(str(output), quantization=quantization)
        shutil.move(str(output), str(target))


def collapse_repeats(text: str) -> str:
    """Collapse Whisper repetition loops ("Thank you. Thank you. Thank you.") to one occurrence.

    Single words need ``MIN_WORD_REPEATS`` (legitimate: "small small", "no no no", reduplication); phrases need
    ``MIN_PHRASE_REPEATS``.
    """
    words = _HYPHEN_LOOP.sub(r"\1", text).split()  # "a-a-a-a-a" written as one hyphenated "word"
    keys = [re.sub(r"[^\w]", "", w.lower()) for w in words]  # compare ignoring case and punctuation
    out: list[str] = []
    i = 0
    while i < len(words):
        if loop := _repetition_at(keys, i):
            size, reps = loop
            out.extend(words[i : i + size])
            i += size * reps
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _repetition_at(keys: list[str], i: int) -> tuple[int, int] | None:
    """(unit size, repeat count) of the shortest looping phrase starting at ``i``, if any."""
    for size in range(1, MAX_REPEAT_UNIT + 1):
        unit = keys[i : i + size]
        if len(unit) < size:
            return None
        if not any(unit):
            continue
        reps = 1
        while keys[i + reps * size : i + (reps + 1) * size] == unit:
            reps += 1
        if reps >= (MIN_WORD_REPEATS if size == 1 else MIN_PHRASE_REPEATS):
            return size, reps
    return None


class NatlasASR:
    def __init__(
        self,
        models: Mapping[str, str],
        *,
        loader: Loader,
        max_loaded: int = DEFAULT_MAX_LOADED,
        failure_cooldown_s: float = FAILURE_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._models = dict(models)  # language code -> model repo
        self._loader = loader
        self._max_loaded = max_loaded
        self._cooldown_s = failure_cooldown_s
        self._clock = clock
        self._loaded: OrderedDict[str, Transcriber] = OrderedDict()
        self._failed: dict[str, tuple[float, str]] = {}  # language -> (retry after, reason)
        self._load_locks: dict[str, asyncio.Lock] = {}
        self._infer_lock = asyncio.Lock()

    def supports(self, language: str) -> bool:
        return language in self._models

    async def preload(self, languages: Iterable[str]) -> dict[str, str]:
        """Load models up front (startup). Returns ``{language: error}`` for the ones that failed."""
        errors = {}
        for language in languages:
            try:
                await self._model(language)
            except ASRError as exc:
                errors[language] = str(exc)
        return errors

    async def transcribe(self, audio: Audio, language: str) -> Transcript:
        if not self.supports(language):
            raise ASRError(f"no speech model configured for language {language!r}")
        voiced = trim_silence(audio)  # raises AudioError(NO_SPEECH) before any model work
        chunks = split(voiced)
        model = await self._model(language)  # a first-use load is not counted as transcription latency
        started = time.perf_counter()
        texts: list[str] = []
        async with self._infer_lock:
            for chunk in chunks:
                try:
                    texts.append(await asyncio.to_thread(model, chunk.samples))
                except Exception as exc:
                    raise ASRError(f"speech recognition failed ({language}): {exc}") from exc
        raw = " ".join(t.strip() for t in texts if t.strip())
        text = collapse_repeats(raw)
        units = len(_UNIT.findall(raw))
        if units >= LOOP_MIN_UNITS and len(_UNIT.findall(text)) < LOOP_KEPT_SHARE * units:
            raise AudioError(AudioProblem.NO_SPEECH, "recogniser repetition loop")
        return Transcript(
            text=text,
            language=language,
            duration_s=voiced.duration_s,
            latency_ms=int((time.perf_counter() - started) * 1000),
            chunks=len(chunks),
        )

    async def _model(self, language: str) -> Transcriber:
        if (model := self._loaded.get(language)) is not None:
            self._loaded.move_to_end(language)
            return model
        if (failure := self._failed.get(language)) and self._clock() < failure[0]:
            raise ASRError(f"speech model for {language!r} unavailable: {failure[1]}")
        lock = self._load_locks.setdefault(language, asyncio.Lock())
        async with lock:
            if (model := self._loaded.get(language)) is not None:  # loaded while we waited
                return model
            repo = self._models[language]
            try:
                model = await asyncio.to_thread(self._loader, repo)
            except Exception as exc:
                reason = f"cannot load {repo}: {type(exc).__name__}: {exc}"[:300]
                self._failed[language] = (self._clock() + self._cooldown_s, reason)
                raise ASRError(reason) from exc
            self._failed.pop(language, None)
            self._loaded[language] = model
            while len(self._loaded) > self._max_loaded:
                self._loaded.popitem(last=False)
            return model
