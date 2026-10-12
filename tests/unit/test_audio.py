import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kudiguard.speech.audio import (
    MAX_AUDIO_BYTES,
    RAW_SLACK_S,
    check_duration,
    decode,
    find_ffmpeg,
    split,
    trim_silence,
)
from kudiguard.speech.base import SAMPLE_RATE, ASRError, Audio, AudioError, AudioProblem

try:
    FFMPEG = find_ffmpeg()
except FileNotFoundError:  # pragma: no cover - ffmpeg is a documented requirement
    FFMPEG = ""
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def encode(tmp_path: Path, name: str, filter_: str, *args: str) -> bytes:
    """Generate audio with ffmpeg's lavfi sources (real files, as phones would send them)."""
    out = tmp_path / name
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", filter_, *args, "-y", str(out)],
        check=True,
    )
    return out.read_bytes()


def tone(seconds: float, *, amplitude: float = 0.3, gaps: tuple[tuple[float, float], ...] = ()) -> Audio:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    samples = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for start, end in gaps:
        samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = 0.0
    return Audio(samples)


def problem(info: pytest.ExceptionInfo[AudioError]) -> AudioProblem:
    return info.value.problem


# --- decode ---------------------------------------------------------------------------------------------


@needs_ffmpeg
async def test_decodes_whatsapp_opus_voice_note(tmp_path: Path) -> None:
    data = encode(
        tmp_path, "note.ogg", "sine=frequency=300:duration=3", "-ac", "2", "-ar", "48000", "-c:a", "libopus"
    )
    audio = await decode(data, ffmpeg=FFMPEG)
    assert audio.samples.dtype == np.float32
    assert audio.duration_s == pytest.approx(3.0, abs=0.1)
    assert 0.05 < float(np.abs(audio.samples).max()) <= 1.0


@needs_ffmpeg
@pytest.mark.parametrize(
    ("name", "codec"), [("a.m4a", "aac"), ("a.amr", "libopencore_amrnb"), ("a.wav", "pcm_s16le")]
)
async def test_decodes_other_phone_formats(tmp_path: Path, name: str, codec: str) -> None:
    args = ["-ar", "8000", "-ac", "1"] if "amr" in codec else []
    try:
        data = encode(tmp_path, name, "sine=frequency=300:duration=2", *args, "-c:a", codec)
    except subprocess.CalledProcessError:
        pytest.skip(f"encoder {codec} not in this ffmpeg build")
    assert (await decode(data, ffmpeg=FFMPEG)).duration_s == pytest.approx(2.0, abs=0.15)


@needs_ffmpeg
async def test_decodes_m4a_with_index_at_the_end(tmp_path: Path) -> None:
    """Android recorders / Safari MediaRecorder write the MP4 index last: needs a seekable input (M5 review)."""
    data = encode(tmp_path, "late.m4a", "sine=frequency=300:duration=40", "-c:a", "aac")
    assert data.find(b"moov") > data.find(b"mdat")  # index really is after the media data
    assert (await decode(data, ffmpeg=FFMPEG)).duration_s == pytest.approx(40.0, abs=0.2)


@needs_ffmpeg
async def test_raw_audio_far_too_long_is_rejected(tmp_path: Path) -> None:
    data = encode(tmp_path, "long.ogg", "sine=duration=40", "-c:a", "libopus")
    with pytest.raises(AudioError) as info:
        await decode(data, ffmpeg=FFMPEG, max_seconds=5)  # raw limit = 5 + RAW_SLACK_S
    assert problem(info) is AudioProblem.TOO_LONG and info.value.too_long


@needs_ffmpeg
async def test_leading_silence_does_not_count_against_the_limit(tmp_path: Path) -> None:
    data = encode(
        tmp_path, "padded.wav", "sine=frequency=300:duration=4,adelay=10000|10000,apad=pad_dur=2", "-ac", "1"
    )
    audio = await decode(data, ffmpeg=FFMPEG, max_seconds=5)
    assert audio.duration_s > 5  # 16 s raw...
    assert check_duration(trim_silence(audio), max_seconds=5).duration_s < 5  # ...but ~4 s of speech


def test_voiced_duration_limit() -> None:
    with pytest.raises(AudioError) as info:
        check_duration(tone(10.0), max_seconds=5)
    assert problem(info) is AudioProblem.TOO_LONG


def test_too_short_voiced_speech_is_no_speech() -> None:
    """An accidental 0.46 s tap was transcribed as garbage and answered (live, Hausa): under MIN_SECONDS of voiced
    speech (padding included) is NO_SPEECH, the existing localised "no speech" reply."""
    with pytest.raises(AudioError) as info:
        check_duration(trim_silence(tone(0.5)))
    assert problem(info) is AudioProblem.NO_SPEECH and info.value.detail == "too short"
    assert check_duration(trim_silence(tone(1.2))).duration_s == pytest.approx(1.2, abs=0.05)
    assert check_duration(tone(0.5), min_seconds=0.4).duration_s == pytest.approx(0.5)


@needs_ffmpeg
async def test_garbage_and_empty_input_rejected_with_reasons() -> None:
    with pytest.raises(AudioError) as garbage:
        await decode(b"definitely not audio" * 50, ffmpeg=FFMPEG)
    assert problem(garbage) is AudioProblem.UNDECODABLE
    with pytest.raises(AudioError) as empty:
        await decode(b"", ffmpeg=FFMPEG)
    assert problem(empty) is AudioProblem.EMPTY


async def test_oversized_upload_rejected_before_ffmpeg() -> None:
    with pytest.raises(AudioError) as info:
        await decode(b"0" * (MAX_AUDIO_BYTES + 1), ffmpeg="ffmpeg-is-never-called")
    assert problem(info) is AudioProblem.TOO_LARGE


@needs_ffmpeg
@pytest.mark.parametrize(
    "payload",
    [
        b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nhttp://127.0.0.1:9/secret.ts\n#EXT-X-ENDLIST\n",
        b"#EXTM3U\n#EXTINF:10,\nfile:C:/Windows/win.ini\n#EXT-X-ENDLIST\n",
        b"ffconcat version 1.0\nfile 'C:/Windows/win.ini'\n",
    ],
)
async def test_playlists_cannot_read_urls_or_local_files(payload: bytes) -> None:
    with pytest.raises(AudioError) as info:
        await decode(payload, ffmpeg=FFMPEG, timeout_s=5)
    assert problem(info) is AudioProblem.UNDECODABLE


async def test_missing_ffmpeg_is_a_system_error_not_bad_audio(tmp_path: Path) -> None:
    with pytest.raises(ASRError, match="ffmpeg"):
        await asyncio.wait_for(decode(b"x" * 100, ffmpeg=str(tmp_path / "no-ffmpeg.exe")), timeout=10)


@needs_ffmpeg
def test_decode_works_on_a_selector_event_loop(tmp_path: Path) -> None:
    """uvicorn uses SelectorEventLoop on Windows with --reload/--workers, where asyncio subprocesses fail."""
    data = encode(tmp_path, "note.ogg", "sine=duration=1", "-c:a", "libopus")
    loop = asyncio.SelectorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
    try:
        audio = loop.run_until_complete(decode(data, ffmpeg=FFMPEG))
    finally:
        loop.close()
    assert audio.duration_s == pytest.approx(1.0, abs=0.1)


@needs_ffmpeg
async def test_temp_file_is_removed(tmp_path: Path) -> None:
    data = encode(tmp_path, "note.ogg", "sine=duration=1", "-c:a", "libopus")
    work = tmp_path / "work"
    work.mkdir()
    await decode(data, ffmpeg=FFMPEG, temp_dir=work)
    with pytest.raises(AudioError):
        await decode(b"junk" * 100, ffmpeg=FFMPEG, temp_dir=work)
    assert list(work.iterdir()) == []


@needs_ffmpeg
async def test_decoding_uses_its_own_temp_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    from kudiguard.speech import audio as audio_module

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    seen: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def spy(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[call-overload]
        seen.append(name)
        return fd, name

    monkeypatch.setattr(audio_module.tempfile, "mkstemp", spy)
    data = encode(tmp_path, "note.ogg", "sine=duration=1", "-c:a", "libopus")
    await decode(data, ffmpeg=FFMPEG)
    (name,) = seen
    assert Path(name).parent == tmp_path / audio_module.TEMP_FOLDER and Path(name).name.startswith(
        "kg-audio-"
    )
    assert list((tmp_path / audio_module.TEMP_FOLDER).iterdir()) == []


def test_audio_temp_dir_falls_back_to_the_system_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import tempfile

    from kudiguard.speech.audio import TEMP_FOLDER, audio_temp_dir

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    assert audio_temp_dir() == tmp_path / TEMP_FOLDER
    monkeypatch.setattr(os, "getuid", lambda: -1, raising=False)  # POSIX: someone else owns the folder
    assert audio_temp_dir() is None
    monkeypatch.setattr(os, "getuid", lambda: (tmp_path / TEMP_FOLDER).stat().st_uid, raising=False)
    assert audio_temp_dir() == tmp_path / TEMP_FOLDER  # POSIX: ours
    (tmp_path / TEMP_FOLDER).rmdir()
    (tmp_path / TEMP_FOLDER).write_text("a file in the way", encoding="utf-8")
    assert audio_temp_dir() is None


def test_a_symlink_in_place_of_the_temp_folder_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import tempfile

    from kudiguard.speech.audio import TEMP_FOLDER, audio_temp_dir

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    try:
        os.symlink(elsewhere, tmp_path / TEMP_FOLDER, target_is_directory=True)
    except OSError:  # Windows without the symlink privilege: simulate what lstat would report
        real_lstat = Path.lstat

        def fake_lstat(self: Path) -> os.stat_result:
            info = real_lstat(self)
            if self.name == TEMP_FOLDER:
                return os.stat_result((0o120777, *tuple(info)[1:]))  # S_IFLNK
            return info

        (tmp_path / TEMP_FOLDER).mkdir()
        monkeypatch.setattr(Path, "lstat", fake_lstat)
    assert audio_temp_dir() is None


def test_stale_temp_audio_is_swept_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import os
    import tempfile
    import time

    from kudiguard.speech.audio import STALE_AUDIO_S, TEMP_FOLDER, sweep_stale_audio

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    own = tmp_path / TEMP_FOLDER
    own.mkdir()
    old = time.time() - STALE_AUDIO_S - 60
    stale = [tmp_path / "kg-audio-legacy", own / "kg-audio-crash"]
    keep = [own / "kg-audio-fresh", tmp_path / "other-app.tmp", tmp_path / "kg-audiox", own / "notes.txt"]
    for path in stale + keep:
        path.write_bytes(b"voice")
    for path in [*stale, *keep[1:]]:
        os.utime(path, (old, old))
    (tmp_path / "kg-audio-dir").mkdir()  # never a folder
    os.utime(tmp_path / "kg-audio-dir", (old, old))
    assert sweep_stale_audio() == 2
    assert not any(p.exists() for p in stale) and all(p.exists() for p in keep)
    assert (tmp_path / "kg-audio-dir").is_dir() and "deleted 2 leftover" in caplog.text
    assert sweep_stale_audio([tmp_path / "missing"]) == 0


def test_sweep_skips_files_it_cannot_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kudiguard.speech.audio import sweep_stale_audio

    (tmp_path / "kg-audio-busy").write_bytes(b"voice")

    def locked(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "unlink", locked)
    assert sweep_stale_audio([tmp_path], max_age_s=-60) == 0


def test_find_ffmpeg_explicit_missing() -> None:
    with pytest.raises(FileNotFoundError):
        find_ffmpeg("Z:/nope/ffmpeg.exe")


def test_raw_slack_is_meaningful() -> None:
    assert RAW_SLACK_S >= 10


# --- silence and chunking -------------------------------------------------------------------------------


def test_trim_silence_removes_edges_and_keeps_padding() -> None:
    trimmed = trim_silence(tone(4.0, gaps=((0.0, 1.0), (3.0, 4.0))))
    assert 2.0 <= trimmed.duration_s <= 2.5


def test_quiet_recordings_are_not_mistaken_for_silence() -> None:
    quiet = tone(3.0, amplitude=0.008)  # ~ -45 dBFS RMS: a phone held far away
    assert trim_silence(quiet).duration_s == pytest.approx(3.0, abs=0.1)


def test_silent_audio_has_no_speech() -> None:
    with pytest.raises(AudioError) as info:
        trim_silence(Audio(np.zeros(SAMPLE_RATE * 3, dtype=np.float32)))
    assert problem(info) is AudioProblem.NO_SPEECH


def test_short_audio_is_one_chunk() -> None:
    assert len(split(tone(10.0))) == 1


def test_long_audio_split_at_quiet_points_near_balanced_targets() -> None:
    audio = tone(75.0, gaps=((24.8, 25.2), (49.8, 50.2)))
    chunks = split(audio, max_s=30.0)
    assert len(chunks) == 3
    assert all(c.duration_s <= 30.0 for c in chunks)
    assert 24.8 <= chunks[0].duration_s <= 25.2  # cut inside the pause, not mid-"word"
    assert sum(len(c.samples) for c in chunks) == len(audio.samples)  # nothing lost or duplicated


@pytest.mark.parametrize("seconds", [30.2, 61.0, 89.9])
def test_no_tiny_last_chunk(seconds: float) -> None:
    chunks = split(tone(seconds), max_s=30.0)
    assert all(c.duration_s <= 30.0 for c in chunks)
    assert min(c.duration_s for c in chunks) >= seconds / len(chunks) - 3.5
    assert sum(len(c.samples) for c in chunks) == int(seconds * SAMPLE_RATE)


def _files_in(directory: Path) -> list[Path]:
    return list(directory.iterdir())


async def test_hung_ffmpeg_is_a_timeout_and_temp_file_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def hang(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(args)
        assert kwargs["timeout"] == 0.5
        raise subprocess.TimeoutExpired(args, 0.5)

    monkeypatch.setattr("kudiguard.speech.audio.subprocess.run", hang)
    with pytest.raises(AudioError) as info:
        await decode(b"not really audio", ffmpeg="ffmpeg", timeout_s=0.5, temp_dir=tmp_path)
    assert problem(info) is AudioProblem.TIMEOUT
    assert seen and _files_in(tmp_path) == []  # the upload never lingers on disk


@pytest.mark.parametrize("samples", [0, 1, 319])  # shorter than one 20 ms analysis frame
def test_audio_shorter_than_a_frame_has_no_speech(samples: int) -> None:
    with pytest.raises(AudioError) as info:
        trim_silence(Audio(np.full(samples, 0.5, dtype=np.float32)))
    assert problem(info) is AudioProblem.NO_SPEECH


@pytest.mark.parametrize("seconds", [30.01, 61.0, 95.3, 300.0])
@pytest.mark.parametrize("search_s", [0.0, 0.5, 3.0])
def test_split_never_exceeds_the_window_and_covers_everything(seconds: float, search_s: float) -> None:
    rng = np.random.default_rng(int(seconds * 100))
    audio = Audio((rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.1).astype(np.float32))
    chunks = split(audio, max_s=30.0, search_s=search_s)
    assert all(len(c.samples) <= 30 * SAMPLE_RATE for c in chunks)
    assert sum(len(c.samples) for c in chunks) == len(audio.samples)
    assert len(chunks) == int(np.ceil(len(audio.samples) / (30 * SAMPLE_RATE)))  # no fallback windows needed
