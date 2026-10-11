"""Audio handling for voice notes: decode untrusted uploads to 16 kHz mono, trim silence, split for Whisper.

Uploads are untrusted (WhatsApp media, API clients). Decoding runs ffmpeg on a private temp file (seekable, so MP4/M4A
with the index at the end decode — Android recorders, Safari's MediaRecorder) via ``subprocess.run`` in a worker
thread, which kills ffmpeg on timeout and works on every asyncio loop type. Defences: a size cap before ffmpeg starts,
an input-format allowlist (audio containers only: no playlists/concat, so no reading of other files or URLs), a
``file``-only protocol allowlist, a raw-duration cap and a wall-clock timeout.

The temp file holds the user's audio in plain form for the length of one decode (deleted in ``finally``). A crash or
power cut in between could leave it behind, so decoding uses its own folder (``kudiguard-audio`` in the system temp
dir) and ``sweep_stale_audio`` (run at startup) deletes leftover ``kg-audio-*`` files, never anything else.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from kudiguard.speech.base import SAMPLE_RATE, ASRError, Audio, AudioError, AudioProblem

log = logging.getLogger(__name__)

MAX_AUDIO_BYTES = 16 * 1024 * 1024  # WhatsApp's audio media limit
MAX_SECONDS = 90.0  # voiced speech allowed after trimming silence
RAW_SLACK_S = 30.0  # extra raw audio decoded so leading/trailing silence does not count against the limit
DECODE_TIMEOUT_S = 15.0
WHISPER_WINDOW_S = 30.0  # NCAIR1 models are Whisper-small: 30 s context per pass
INPUT_FORMATS = "ogg,matroska,webm,mov,mp4,m4a,3gp,3g2,mj2,aac,amr,wav,mp3,flac"
SILENCE_FLOOR_DB = -55.0  # never treat louder than this as silence...
SILENCE_BELOW_PEAK_DB = 35.0  # ...and otherwise: 35 dB below the loudest frame (quiet phones, noisy rooms)
FRAME_S = 0.02
PAD_S = 0.2
CUT_SEARCH_S = 3.0  # search this far either side of an ideal cut for the quietest moment
TEMP_PREFIX = "kg-audio-"
TEMP_FOLDER = "kudiguard-audio"
STALE_AUDIO_S = 300.0  # far longer than any decode (DECODE_TIMEOUT_S): such a file was left by a crash


def audio_temp_dir() -> Path | None:
    """The decode folder (``<temp>/kudiguard-audio``, private to this account), or None (= the system temp dir) when
    it cannot be created or belongs to someone else."""
    folder = Path(tempfile.gettempdir()) / TEMP_FOLDER
    try:
        folder.mkdir(mode=0o700, exist_ok=True)
        owner = folder.stat().st_uid
    except OSError:
        return None
    getuid = getattr(os, "getuid", None)  # POSIX only; on Windows the folder sits in the per-user temp dir
    if getuid is not None and owner != getuid():
        return None
    return folder


def sweep_stale_audio(folders: list[Path] | None = None, *, max_age_s: float = STALE_AUDIO_S) -> int:
    """Delete ``kg-audio-*`` files older than ``max_age_s`` (left behind by a crash mid-decode) from the decode folder
    and the system temp dir (older versions decoded there). Only that prefix, only plain files. Returns the count."""
    if folders is None:
        folders = [Path(tempfile.gettempdir()), Path(tempfile.gettempdir()) / TEMP_FOLDER]
    cutoff = time.time() - max_age_s
    removed = 0
    for folder in folders:
        for path in folder.glob(f"{TEMP_PREFIX}*"):
            try:
                if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue  # in use or already gone
    if removed:
        log.warning("deleted %d leftover temporary audio files (an earlier run stopped mid-decode)", removed)
    return removed


def find_ffmpeg(explicit: str | None = None) -> str:
    """Locate ffmpeg: explicit path, $KUDIGUARD_FFMPEG, PATH, then the project's ``tools/ffmpeg`` (development)."""
    candidates = [explicit] if explicit else [os.environ.get("KUDIGUARD_FFMPEG"), shutil.which("ffmpeg")]
    if not explicit:
        project_tools = (
            Path(__file__).resolve().parents[3] / "tools" / "ffmpeg"
        )  # source checkout (development)
        candidates += [str(p) for p in sorted(project_tools.glob("*/bin/ffmpeg*"))]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(f"ffmpeg not found ({explicit or 'set KUDIGUARD_FFMPEG or install ffmpeg'})")


async def decode(
    data: bytes,
    *,
    ffmpeg: str,
    max_seconds: float = MAX_SECONDS,
    timeout_s: float = DECODE_TIMEOUT_S,
    temp_dir: Path | None = None,
) -> Audio:
    """Decode an uploaded audio file to mono float32 at 16 kHz.

    Returns at most ``max_seconds + RAW_SLACK_S`` of audio; the voiced duration is checked after trimming
    (``check_duration``). Raises ``AudioError`` for unusable audio and ``ASRError`` if ffmpeg is unavailable.
    """
    if not data:
        raise AudioError(AudioProblem.EMPTY)
    if len(data) > MAX_AUDIO_BYTES:
        raise AudioError(AudioProblem.TOO_LARGE, f"{len(data)} bytes")
    raw_limit = max_seconds + RAW_SLACK_S
    return await asyncio.to_thread(_decode_file, data, ffmpeg, raw_limit, timeout_s, temp_dir)


def _decode_file(
    data: bytes, ffmpeg: str, raw_limit: float, timeout_s: float, temp_dir: Path | None
) -> Audio:
    fd, name = tempfile.mkstemp(prefix=TEMP_PREFIX, dir=temp_dir or audio_temp_dir())
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        args = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", "file", "-format_whitelist", INPUT_FORMATS,
            "-i", str(path),
            "-t", f"{raw_limit + 1:g}", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
        ]  # fmt: skip
        try:
            result = subprocess.run(args, capture_output=True, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AudioError(AudioProblem.TIMEOUT) from exc
        except OSError as exc:
            raise ASRError(f"ffmpeg not available: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)

    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode(errors="replace").strip()[-300:]
        log.info("ffmpeg could not decode upload: %s", detail)
        raise AudioError(AudioProblem.UNDECODABLE, detail)
    audio = Audio(np.clip(np.frombuffer(result.stdout, dtype=np.float32), -1.0, 1.0))
    if audio.duration_s > raw_limit:
        raise AudioError(AudioProblem.TOO_LONG, f"{audio.duration_s:.0f} s")
    return audio


def check_duration(audio: Audio, max_seconds: float = MAX_SECONDS) -> Audio:
    """Enforce the voiced-speech limit (call after ``trim_silence``)."""
    if audio.duration_s > max_seconds:
        raise AudioError(AudioProblem.TOO_LONG, f"{audio.duration_s:.0f} s of speech")
    return audio


def _frame_rms(samples: np.ndarray, frame: int) -> np.ndarray:
    usable = len(samples) // frame * frame
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    frames = samples[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def trim_silence(audio: Audio, *, pad_s: float = PAD_S) -> Audio:
    """Drop leading/trailing silence. The threshold is relative to the loudest frame (quiet phones still pass)."""
    frame = int(FRAME_S * SAMPLE_RATE)
    rms = _frame_rms(audio.samples, frame)
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 10 ** (SILENCE_FLOOR_DB / 20):
        raise AudioError(AudioProblem.NO_SPEECH)
    threshold = max(10 ** (SILENCE_FLOOR_DB / 20), peak * 10 ** (-SILENCE_BELOW_PEAK_DB / 20))
    voiced = np.flatnonzero(rms > threshold)
    pad = int(pad_s * SAMPLE_RATE)
    start = max(int(voiced[0]) * frame - pad, 0)
    end = min((int(voiced[-1]) + 1) * frame + pad, len(audio.samples))
    return Audio(audio.samples[start:end])


def split(audio: Audio, *, max_s: float = WHISPER_WINDOW_S, search_s: float = CUT_SEARCH_S) -> list[Audio]:
    """Split into balanced chunks of at most ``max_s``, each cut at the quietest moment near its ideal position.

    Balanced targets (``len / n``) avoid a tiny last chunk, which Whisper tends to fill with invented text.
    """
    samples = audio.samples
    limit = int(max_s * SAMPLE_RATE)
    if len(samples) <= limit:
        return [audio]
    frame = int(FRAME_S * SAMPLE_RATE)
    search = int(search_s * SAMPLE_RATE)
    count = math.ceil(len(samples) / limit)
    cuts = [0]
    for k in range(1, count):
        target = len(samples) * k // count
        # never cut so early that the rest cannot fit in the remaining chunks (that would force a tiny tail)
        low = max(target - search, cuts[-1] + frame, len(samples) - (count - k) * limit)
        high = min(target + search, cuts[-1] + limit)  # never exceed the window from the previous cut
        rms = _frame_rms(samples[low:high], frame)
        cut = low + (int(np.argmin(rms)) * frame + frame // 2 if rms.size else (high - low))
        cuts.append(min(cut, cuts[-1] + limit))
    if (
        len(samples) - cuts[-1] > limit
    ):  # pragma: no cover - defensive; the cut bounds guarantee the tail fits
        cuts += list(range(cuts[-1] + limit, len(samples), limit))
    cuts.append(len(samples))
    return [Audio(samples[a:b]) for a, b in itertools.pairwise(cuts) if b > a]
