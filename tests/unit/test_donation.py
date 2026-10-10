"""Voice-note donation vault: encryption at rest, writes off the hot path, erasure, retention and orphan cleanup."""

import asyncio
import os
import time
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from kudiguard.donation import (
    DONATION_VERSION,
    SUFFIX,
    Donation,
    Sealer,
    VoiceIntegrityError,
    VoiceVault,
    new_sample_id,
)
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.memory import MemoryStore

SECRET = b"v" * MIN_SECRET_BYTES
AUDIO = b"OggS" + bytes(range(256)) * 4
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)  # a Thursday: week of Monday 2026-10-12


def donation(user: str = "u1", **overrides: object) -> Donation:
    values: dict[str, object] = {
        "user_hash": user, "channel": "whatsapp", "language": "ha", "audio": AUDIO,
        "hypothesis": "kira 08031234567 yanzu", "consent_version": DONATION_VERSION, "duration_s": 4.26,
    }  # fmt: skip
    values.update(overrides)
    return Donation(**values)  # type: ignore[arg-type]


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(UserHasher(SECRET))


@pytest.fixture
def vault(tmp_path: Path, store: MemoryStore) -> VoiceVault:
    return VoiceVault(tmp_path / "voice", store, Sealer(SECRET), clock=lambda: NOW)


# --- encryption -------------------------------------------------------------------------------------------------


def test_sealing_round_trips_and_hides_the_audio() -> None:
    sealer, sample_id = Sealer(SECRET), new_sample_id()
    blob = sealer.seal(sample_id, AUDIO)
    assert AUDIO[4:64] not in blob and blob.startswith(b"KGV1")
    assert sealer.open(sample_id, blob) == AUDIO
    assert sealer.seal(sample_id, AUDIO) != blob  # a fresh nonce per file


def test_tampering_another_id_or_another_key_is_detected() -> None:
    sealer, sample_id = Sealer(SECRET), new_sample_id()
    blob = bytearray(sealer.seal(sample_id, AUDIO))
    blob[-1] ^= 1
    with pytest.raises(VoiceIntegrityError, match="authentication"):
        sealer.open(sample_id, bytes(blob))
    good = sealer.seal(sample_id, AUDIO)
    with pytest.raises(VoiceIntegrityError):  # a file renamed to another sample's id
        sealer.open(new_sample_id(), good)
    with pytest.raises(VoiceIntegrityError):  # the server secret changed
        Sealer(b"w" * MIN_SECRET_BYTES).open(sample_id, good)
    for junk in (b"", b"KGV1short", b"RIFF" + good[4:]):
        with pytest.raises(VoiceIntegrityError, match="not a KudiGuard"):
            sealer.open(sample_id, junk)


def test_the_key_is_domain_separated_from_the_user_hash_secret() -> None:
    # Same secret, different purpose: knowing a user id never helps decrypt (and vice versa).
    assert (
        UserHasher(SECRET).user_id("+2348031234567") not in Sealer(SECRET).seal(new_sample_id(), AUDIO).hex()
    )


@pytest.mark.parametrize("bad", ["../etc/passwd", "ABCDEF" * 6, "", "a" * 31])
def test_sample_ids_are_validated_before_touching_files(vault: VoiceVault, bad: str) -> None:
    with pytest.raises(ValueError, match="invalid sample id"):
        vault.path(bad)
    with pytest.raises(ValueError, match="invalid sample id"):
        Sealer(SECRET).seal(bad, AUDIO)


# --- donate --------------------------------------------------------------------------------------------------------


async def test_donate_keeps_an_encrypted_file_and_a_minimal_row(
    vault: VoiceVault, store: MemoryStore
) -> None:
    sample_id = await vault.donate(donation())
    assert sample_id is not None
    path = vault.path(sample_id)
    assert path.name == f"{sample_id}{SUFFIX}" and AUDIO not in path.read_bytes()
    assert await vault.read(sample_id) == AUDIO
    [row] = await store.voice_samples()
    assert (row.id, row.user_hash, row.language, row.channel) == (sample_id, "u1", "ha", "whatsapp")
    assert row.week == date(2026, 10, 12)  # the Monday, never the time
    assert row.hypothesis == "kira [number] yanzu" and row.duration_s == 4.3
    assert row.consent_version == DONATION_VERSION
    assert not list(vault.directory.glob("*.part"))


async def test_unknown_duration_is_kept_as_unknown(vault: VoiceVault, store: MemoryStore) -> None:
    await vault.donate(donation(duration_s=None))
    assert (await store.voice_samples())[0].duration_s is None


@pytest.mark.parametrize(
    "overrides", [{"channel": "api"}, {"consent_version": "2020-01-01"}, {"consent_version": ""}]
)
async def test_donations_without_valid_consent_or_from_api_users_are_refused(
    vault: VoiceVault, store: MemoryStore, overrides: dict[str, object]
) -> None:
    assert await vault.donate(donation(**overrides)) is None
    assert await store.voice_samples() == [] and not vault.directory.exists()


async def test_a_failing_row_write_leaves_no_file_and_never_raises(
    vault: VoiceVault, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(sample: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "add_voice_sample", broken)
    assert await vault.donate(donation()) is None
    assert list(vault.directory.iterdir()) == []
    assert "RuntimeError" in caplog.text and "kira" not in caplog.text  # never the content


async def test_a_failing_file_write_leaves_nothing_behind(
    vault: VoiceVault, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_replace(src: object, dst: object) -> None:
        raise OSError("read-only")

    monkeypatch.setattr("kudiguard.donation.os.replace", broken_replace)
    assert await vault.donate(donation()) is None
    assert await store.voice_samples() == [] and list(vault.directory.iterdir()) == []


async def test_pending_writes_are_bounded(tmp_path: Path, store: MemoryStore) -> None:
    vault = VoiceVault(tmp_path / "v", store, Sealer(SECRET), max_pending=1)
    gate = asyncio.Event()
    original = vault._write

    def slow_write(sample_id: str, audio: bytes) -> None:
        while not gate.is_set():
            time.sleep(0.005)
        original(sample_id, audio)

    vault._write = slow_write  # type: ignore[method-assign]
    first = asyncio.create_task(vault.donate(donation()))
    await asyncio.sleep(0.02)
    assert await vault.donate(donation(user="u2")) is None  # skipped: one write already pending
    gate.set()
    assert await first is not None
    assert await vault.donate(donation(user="u3")) is not None  # the slot is free again


async def test_a_cancelled_caller_cannot_be_overtaken_by_erasure(
    vault: VoiceVault, store: MemoryStore
) -> None:
    """The write is shielded and serialised with erase: a withdrawal right after always removes it."""
    gate = asyncio.Event()
    original = vault._write

    def slow_write(sample_id: str, audio: bytes) -> None:
        while not gate.is_set():
            time.sleep(0.005)
        original(sample_id, audio)

    vault._write = slow_write  # type: ignore[method-assign]
    caller = asyncio.create_task(vault.donate(donation()))
    await asyncio.sleep(0.02)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    erase = asyncio.create_task(vault.erase("u1"))
    await asyncio.sleep(0.02)
    assert not erase.done()  # waits for the in-flight write
    gate.set()
    assert await erase == 1
    assert await store.voice_samples() == [] and list(vault.directory.iterdir()) == []
    assert vault._pending == 0


# --- erase, purge, read ----------------------------------------------------------------------------------------


async def test_erase_removes_only_that_donors_rows_and_files(vault: VoiceVault, store: MemoryStore) -> None:
    mine = [await vault.donate(donation()) for _ in range(2)]
    theirs = await vault.donate(donation(user="u2"))
    assert await vault.erase("u1") == 2
    assert [s.id for s in await store.voice_samples()] == [theirs]
    assert all(not vault.path(i).exists() for i in mine if i) and theirs and vault.path(theirs).exists()
    assert await vault.erase("u1") == 0  # idempotent


async def test_erase_is_robust_to_missing_and_undeletable_files(
    vault: VoiceVault, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = await vault.donate(donation())
    stuck = await vault.donate(donation())
    assert gone and stuck
    vault.path(gone).unlink()  # removed by hand already
    real_unlink = Path.unlink

    def locked(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith(stuck):
            raise PermissionError("in use")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    assert await vault.erase("u1") == 2  # rows gone; the stuck file is an orphan for the next purge
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert vault.path(stuck).exists()
    assert await vault.purge(date(2000, 1, 1), orphan_grace_s=0) == 0
    assert not vault.path(stuck).exists()


async def test_purge_removes_expired_weeks_and_old_orphans_only(tmp_path: Path, store: MemoryStore) -> None:
    clock = [datetime(2025, 10, 8, tzinfo=UTC)]  # Wednesday; week of Monday 2025-10-06
    vault = VoiceVault(tmp_path / "v", store, Sealer(SECRET), clock=lambda: clock[0])
    old = await vault.donate(donation())
    clock[0] = NOW
    new = await vault.donate(donation())
    fresh_orphan = vault.directory / f"{new_sample_id()}{SUFFIX}"
    fresh_orphan.write_bytes(b"x")
    stale_orphan = vault.directory / f"{new_sample_id()}.part"
    stale_orphan.write_bytes(b"x")
    os.utime(stale_orphan, (time.time() - 7200, time.time() - 7200))
    unrelated = vault.directory / "README.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    assert await vault.purge(date(2025, 10, 5)) == 0  # the week started after the cutoff
    assert await vault.purge(date(2025, 10, 6)) == 1  # a week starting on the cutoff day is expired
    assert [s.id for s in await store.voice_samples()] == [new]
    assert old and not vault.path(old).exists() and new and vault.path(new).exists()
    assert fresh_orphan.exists() and not stale_orphan.exists() and unrelated.exists()


async def test_purge_without_any_files_yet(vault: VoiceVault) -> None:
    assert await vault.purge(date(2026, 1, 1)) == 0


async def test_orphan_sweep_survives_files_vanishing(
    vault: VoiceVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    await vault.donate(donation())
    orphan = vault.directory / f"{new_sample_id()}{SUFFIX}"
    orphan.write_bytes(b"x")
    real_stat = Path.stat

    def racing(self: Path, **kwargs: object) -> os.stat_result:
        if self == orphan:
            raise FileNotFoundError("deleted meanwhile")
        return real_stat(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", racing)
    assert await vault.purge(date(2000, 1, 1), orphan_grace_s=0) == 0
    monkeypatch.undo()
    assert orphan.exists()  # skipped this time, retried by the next purge


async def test_read_detects_a_tampered_file(vault: VoiceVault) -> None:
    sample_id = await vault.donate(donation())
    assert sample_id
    blob = bytearray(vault.path(sample_id).read_bytes())
    blob[20] ^= 0xFF
    vault.path(sample_id).write_bytes(bytes(blob))
    with pytest.raises(VoiceIntegrityError):
        await vault.read(sample_id)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
async def test_files_are_private_to_the_service_user(vault: VoiceVault) -> None:  # pragma: no cover - posix
    sample_id = await vault.donate(donation())
    assert sample_id
    assert vault.path(sample_id).stat().st_mode & 0o077 == 0
    assert vault.directory.stat().st_mode & 0o077 == 0
