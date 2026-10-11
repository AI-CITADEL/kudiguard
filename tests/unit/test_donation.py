"""Voice-note donation vault: encryption at rest, bounded writes, withdrawal races, erasure, retention, orphans."""

import asyncio
import os
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from kudiguard import donation as donation_module
from kudiguard.config import Settings
from kudiguard.donation import (
    DONATION_VERSION,
    MARKER,
    SUFFIX,
    Donation,
    Sealer,
    VoiceIntegrityError,
    VoiceVault,
    new_sample_id,
)
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.base import utcnow
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


def gate_writes(vault: VoiceVault) -> threading.Event:
    """Block the vault's file writes until the returned event is set (never longer than 5 s)."""
    gate = threading.Event()
    original = vault._write

    def slow_write(sample_id: str, audio: bytes, expected: str) -> int | None:
        gate.wait(timeout=5)
        return original(sample_id, audio, expected)

    vault._write = slow_write  # type: ignore[method-assign]
    return gate


def sample_files(vault: VoiceVault) -> list[Path]:
    return [p for p in vault.directory.iterdir() if p.suffix in (SUFFIX, ".part")]


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
    with pytest.raises(VoiceIntegrityError):  # the key changed
        Sealer(b"w" * MIN_SECRET_BYTES).open(sample_id, good)
    for junk in (b"", b"KGV1short", b"RIFF" + good[4:]):
        with pytest.raises(VoiceIntegrityError, match="not a KudiGuard"):
            sealer.open(sample_id, junk)


def test_keys_must_be_long_and_are_domain_separated_from_user_ids() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        Sealer(b"short")
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
    assert (vault.directory / MARKER).read_text(encoding="utf-8") == await store.voice_vault_id()
    assert [p.name for p in sample_files(vault)] == [path.name]


async def test_unknown_duration_is_kept_as_unknown(vault: VoiceVault, store: MemoryStore) -> None:
    await vault.donate(donation(duration_s=None))
    assert (await store.voice_samples())[0].duration_s is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"channel": "api"},
        {"consent_version": "2020-01-01"},
        {"consent_version": ""},
        {"consent_version": "9999-12-31"},
    ],
)
async def test_donations_without_valid_consent_or_from_api_users_are_refused(
    vault: VoiceVault, store: MemoryStore, overrides: dict[str, object]
) -> None:
    assert await vault.donate(donation(**overrides)) is None
    assert await store.voice_samples() == [] and not vault.directory.exists()
    assert vault.dropped["consent"] == 1


async def test_a_failing_row_write_leaves_no_file_and_never_raises(
    vault: VoiceVault, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(sample: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "add_voice_sample", broken)
    assert await vault.donate(donation()) is None
    assert sample_files(vault) == []
    assert "error:RuntimeError" in caplog.text and "kira" not in caplog.text  # never the content
    assert vault.dropped["error:RuntimeError"] == 1


async def test_a_failing_file_write_leaves_nothing_behind(
    vault: VoiceVault, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_replace(src: object, dst: object) -> None:
        raise OSError("read-only")

    monkeypatch.setattr("kudiguard.donation.os.replace", broken_replace)
    assert await vault.donate(donation()) is None
    assert await store.voice_samples() == [] and sample_files(vault) == []


async def test_pending_writes_are_bounded(tmp_path: Path, store: MemoryStore) -> None:
    vault = VoiceVault(tmp_path / "v", store, Sealer(SECRET), max_pending=1)
    gate = gate_writes(vault)
    first = asyncio.create_task(vault.donate(donation()))
    await asyncio.sleep(0.02)
    assert await vault.donate(donation(user="u2")) is None  # skipped: one write already pending
    gate.set()
    assert await first is not None
    assert await vault.donate(donation(user="u3")) is not None  # the slot is free again
    assert vault.dropped["busy"] == 1


async def test_a_donors_daily_cap_and_the_folder_size_cap(tmp_path: Path, store: MemoryStore) -> None:
    vault = VoiceVault(tmp_path / "v", store, Sealer(SECRET), max_per_day=2, max_bytes=2600)
    assert all([await vault.donate(donation()) for _ in range(2)])
    assert await vault.donate(donation()) is None and vault.dropped["daily-cap"] == 1
    assert (
        await vault.donate(donation(user="u2")) is None and vault.dropped["full"] == 1
    )  # a third 1060-byte file would pass 2600
    assert len(await store.voice_samples()) == 2
    await vault.erase("u1")  # space is freed
    assert await vault.donate(donation(user="u2")) is not None


async def test_the_daily_cap_resets_each_day(vault: VoiceVault, monkeypatch: pytest.MonkeyPatch) -> None:
    vault._max_per_day = 1
    assert await vault.donate(donation())
    assert await vault.donate(donation()) is None
    tomorrow = utcnow() + timedelta(days=1)
    monkeypatch.setattr(donation_module, "utcnow", lambda: tomorrow)
    assert await vault.donate(donation(consented_at=tomorrow))


# --- withdrawal ordering ---------------------------------------------------------------------------------------


async def test_a_cancelled_caller_cannot_be_overtaken_by_erasure(
    vault: VoiceVault, store: MemoryStore
) -> None:
    """The write is shielded and serialised with erase: a withdrawal right after always removes it."""
    gate = gate_writes(vault)
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
    assert await store.voice_samples() == [] and sample_files(vault) == []
    assert vault._pending == 0


async def test_a_late_write_after_a_withdrawal_is_refused(vault: VoiceVault, store: MemoryStore) -> None:
    """A web background task can run after the user unticked: consent given before the withdrawal does not count."""
    before = utcnow() - timedelta(seconds=1)
    await vault.erase("u1")
    assert await vault.donate(donation(consented_at=before)) is None and vault.dropped["withdrawn"] == 1
    assert await store.voice_samples() == []
    assert await vault.donate(donation(consented_at=utcnow() + timedelta(seconds=1)))  # a new yes, later


async def test_withdrawal_tombstones_expire(vault: VoiceVault, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(donation_module, "TOMBSTONE_S", 0.0)
    await vault.erase("old")
    await vault.erase("new")  # prunes the expired one
    assert list(vault._withdrawn) == ["new"]


# --- erase, remove, wipe, purge, read --------------------------------------------------------------------------


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


async def test_remove_deletes_one_flagged_sample(vault: VoiceVault, store: MemoryStore) -> None:
    keep, flagged = await vault.donate(donation()), await vault.donate(donation())
    assert keep and flagged
    assert await vault.remove(flagged) and not await vault.remove(flagged)
    assert [s.id for s in await store.voice_samples()] == [keep] and not vault.path(flagged).exists()
    with pytest.raises(ValueError, match="invalid sample id"):
        await vault.remove("../x")


async def test_wipe_deletes_everything_but_the_marker(vault: VoiceVault, store: MemoryStore) -> None:
    for user in ("u1", "u2"):
        await vault.donate(donation(user=user))
    (vault.directory / f"{new_sample_id()}.part").write_bytes(b"debris")
    assert await vault.wipe() == (2, 0)
    assert await store.voice_samples() == [] and sample_files(vault) == []
    assert (vault.directory / MARKER).exists()


async def test_wipe_of_a_vault_that_never_wrote_anything(vault: VoiceVault) -> None:
    assert await vault.wipe() == (0, 0) and not vault.directory.exists()


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


async def test_another_databases_folder_is_never_swept_or_written(
    tmp_path: Path, store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    """Pointing KUDIGUARD_VOICE_DIR at a folder of another database (or an empty test one) must not delete real
    audio: the folder's marker must match the database."""
    real = VoiceVault(tmp_path / "v", MemoryStore(UserHasher(SECRET)), Sealer(SECRET))
    kept = await real.donate(donation())
    assert kept
    stranger = VoiceVault(tmp_path / "v", store, Sealer(SECRET))  # a different (empty) database
    assert await stranger.purge(date(2000, 1, 1), orphan_grace_s=0) == 0
    assert real.path(kept).exists() and "does not belong to this database" in caplog.text
    assert await stranger.donate(donation()) is None and stranger.dropped["error:RuntimeError"] == 1
    assert await stranger.wipe() == (0, 1) and real.path(kept).exists()  # reported, never deleted


async def test_an_unmarked_folder_with_audio_is_not_claimed(tmp_path: Path, store: MemoryStore) -> None:
    folder = tmp_path / "v"
    folder.mkdir()
    loose = folder / f"{new_sample_id()}{SUFFIX}"
    loose.write_bytes(b"x")
    os.utime(loose, (time.time() - 7200, time.time() - 7200))
    vault = VoiceVault(folder, store, Sealer(SECRET))
    assert await vault.purge(date(2000, 1, 1)) == 0 and loose.exists()
    assert await vault.donate(donation()) is None and not (folder / MARKER).exists()


async def test_an_unmarked_folder_without_audio_is_left_alone_by_purge(
    tmp_path: Path, store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    assert await VoiceVault(folder, store, Sealer(SECRET)).purge(date(2000, 1, 1)) == 0
    assert not (folder / MARKER).exists() and "does not belong" not in caplog.text  # claimed only by a write


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


def test_directory_fsync_is_best_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(donation_module.os, "open", lambda path, flags: 99)

    def failing_fsync(fd: int) -> None:
        calls.append("fsync")
        raise OSError("not supported")

    monkeypatch.setattr(donation_module.os, "fsync", failing_fsync)
    monkeypatch.setattr(donation_module.os, "close", lambda fd: calls.append("close"))
    donation_module._fsync_directory(tmp_path)  # POSIX path: opened, synced (error ignored), closed
    assert calls == ["fsync", "close"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
async def test_files_are_private_to_the_service_user(vault: VoiceVault) -> None:  # pragma: no cover - posix
    sample_id = await vault.donate(donation())
    assert sample_id
    assert vault.path(sample_id).stat().st_mode & 0o077 == 0
    assert vault.directory.stat().st_mode & 0o077 == 0


# --- found by mutation testing (docs/SPIKES.md, G) ----------------------------------------------------------------


def test_the_file_format_is_stable() -> None:
    """Sealed files must stay readable by later releases and operator tools: b"KGV1" | 12-byte nonce | AES-256-GCM
    under HKDF-SHA256(secret, info=b"kudiguard/voice-donation/aes-256-gcm/v1"), with b"KGV1" + id as associated data."""
    sample_id = new_sample_id()
    blob = Sealer(SECRET).seal(sample_id, AUDIO)
    assert len(blob) == 4 + 12 + len(AUDIO) + 16
    info = b"kudiguard/voice-donation/aes-256-gcm/v1"
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(SECRET)
    assert AESGCM(key).decrypt(blob[4:16], blob[16:], b"KGV1" + sample_id.encode()) == AUDIO


def test_shortest_valid_file_and_truncation() -> None:
    sealer, sample_id = Sealer(SECRET), new_sample_id()
    empty = sealer.seal(sample_id, b"")
    assert len(empty) == 32 and sealer.open(sample_id, empty) == b""
    with pytest.raises(VoiceIntegrityError, match="not a KudiGuard"):
        sealer.open(sample_id, empty[:-1])


def test_keys_of_32_bytes_or_more_are_accepted() -> None:
    with pytest.raises(ValueError):
        Sealer(b"v" * 31)
    for secret in (b"v" * 32, b"v" * 64):
        sample_id = new_sample_id()
        assert Sealer(secret).open(sample_id, Sealer(secret).seal(sample_id, AUDIO)) == AUDIO


def test_retention_matches_what_users_are_told() -> None:
    # "12 months" in the offer and on /privacy (the version is checked against voice.html in test_patterns)
    field = Settings.model_fields["voice_retention_days"]
    assert donation_module.VOICE_RETENTION_DAYS == field.default == 365


async def test_default_bounds_twenty_a_day_and_eight_pending(tmp_path: Path, store: MemoryStore) -> None:
    vault = VoiceVault(tmp_path / "v", store, Sealer(SECRET))
    assert all([await vault.donate(donation()) for _ in range(20)])
    assert await vault.donate(donation()) is None and vault.dropped["daily-cap"] == 1
    gate = gate_writes(vault)
    writes = [asyncio.create_task(vault.donate(donation(user=f"p{i}"))) for i in range(8)]
    await asyncio.sleep(0.02)
    assert await vault.donate(donation(user="p8")) is None and vault.dropped["busy"] == 1
    gate.set()
    assert all(await asyncio.gather(*writes))


async def test_the_pending_bound_survives_completed_and_cancelled_writes(
    tmp_path: Path, store: MemoryStore
) -> None:
    vault = VoiceVault(tmp_path / "v", store, Sealer(SECRET), max_pending=1)
    assert await vault.donate(donation())  # completed: its slot is released exactly once
    gate = gate_writes(vault)
    caller = asyncio.create_task(vault.donate(donation()))
    await asyncio.sleep(0.02)
    caller.cancel()  # the write goes on (shielded) and still holds the slot
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert await vault.donate(donation(user="u2")) is None
    gate.set()
    for _ in range(200):
        if vault._pending == 0:
            break
        await asyncio.sleep(0.01)
    assert await vault.donate(donation(user="u3")) is not None


async def test_consent_from_the_instant_of_a_withdrawal_does_not_count(
    vault: VoiceVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    moment = utcnow()
    monkeypatch.setattr(donation_module, "utcnow", lambda: moment)
    await vault.erase("u1")
    assert await vault.donate(donation(consented_at=moment)) is None


async def test_each_withdrawal_blocks_late_writes_for_one_hour(
    vault: VoiceVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = utcnow()
    clock = [start]
    monkeypatch.setattr(donation_module, "utcnow", lambda: clock[0])
    old_consent = start - timedelta(seconds=1)
    await vault.erase("a")
    clock[0] = start + timedelta(seconds=3599)
    await vault.erase("b")  # another donor's withdrawal keeps a's tombstone
    assert await vault.donate(donation(user="a", consented_at=old_consent)) is None
    clock[0] = start + timedelta(seconds=3600)
    await vault.erase("c")  # one hour on: a's tombstone has expired, b's has not
    assert await vault.donate(donation(user="a", consented_at=old_consent))
    assert await vault.donate(donation(user="b", consented_at=old_consent)) is None


async def test_orphans_age_out_after_one_hour(vault: VoiceVault, caplog: pytest.LogCaptureFixture) -> None:
    await vault.donate(donation())
    young, old = (vault.directory / f"{new_sample_id()}{SUFFIX}" for _ in range(2))
    for path, age in ((young, 3599.0), (old, 3600.3)):
        path.write_bytes(b"x")
        os.utime(path, (time.time() - age, time.time() - age))
    await vault.purge(date(2000, 1, 1))
    assert young.exists() and not old.exists()
    assert "removed 1 orphaned voice files" in caplog.text


async def test_nothing_is_reported_swept_when_there_is_nothing_to_sweep(
    tmp_path: Path, store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    assert await VoiceVault(tmp_path / "missing", store, Sealer(SECRET)).purge(date(2000, 1, 1)) == 0
    real = VoiceVault(tmp_path / "v", MemoryStore(UserHasher(SECRET)), Sealer(SECRET))
    assert await real.donate(donation())
    assert await VoiceVault(tmp_path / "v", store, Sealer(SECRET)).purge(date(2000, 1, 1)) == 0
    assert "orphaned" not in caplog.text


async def test_debris_from_an_interrupted_write_counts_as_audio(
    tmp_path: Path, store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    (unmarked / f"{new_sample_id()}.part").write_bytes(b"x")
    assert await VoiceVault(unmarked, store, Sealer(SECRET)).donate(donation()) is None  # never claimed
    assert not (unmarked / MARKER).exists()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / MARKER).write_text("another database", encoding="utf-8")
    (foreign / f"{new_sample_id()}.part").write_bytes(b"x")
    assert await VoiceVault(foreign, store, Sealer(SECRET)).wipe() == (0, 1)  # reported, never deleted
    assert "does not belong to this database" in caplog.text


async def test_the_size_cap_counts_stored_samples_only_and_is_inclusive(
    tmp_path: Path, store: MemoryStore
) -> None:
    audio = bytes(1035)  # a sealed size whose bits overlap its double (so `|` cannot stand in for `+`)
    folder = tmp_path / "v"
    assert await VoiceVault(folder, store, Sealer(SECRET)).donate(donation(audio=audio))
    [size] = [p.stat().st_size for p in folder.iterdir() if p.suffix == SUFFIX]
    (folder / f"{new_sample_id()}.part").write_bytes(b"x" * 10_000)  # debris and other files do not count
    (folder / "notes.txt").write_bytes(b"x" * 10_000)
    restarted = VoiceVault(folder, store, Sealer(SECRET), max_bytes=2 * size)  # measures the folder once
    assert await restarted.donate(donation(user="u2", audio=audio))  # exactly at the cap: kept
    assert await restarted.donate(donation(user="u3", audio=audio)) is None
    tight = VoiceVault(folder, store, Sealer(SECRET), max_bytes=3 * size - 1)
    assert await tight.donate(donation(user="u4", audio=audio)) is None  # the two stored samples count


async def test_a_vault_folder_may_be_nested_in_new_folders(tmp_path: Path, store: MemoryStore) -> None:
    vault = VoiceVault(tmp_path / "a" / "b" / "voice", store, Sealer(SECRET))
    assert await vault.donate(donation())


async def test_erase_goes_on_after_a_missing_or_locked_file(
    vault: VoiceVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    donated = [await vault.donate(donation()) for _ in range(3)]
    gone, stuck, last = sorted(i for i in donated if i)  # the order erasure deletes them in
    vault.path(gone).unlink()
    real_unlink = Path.unlink

    def locked(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith(stuck):
            raise PermissionError("in use")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    assert await vault.erase("u1") == 3
    assert not vault.path(last).exists()


async def test_the_orphan_sweep_goes_on_after_a_locked_file(
    vault: VoiceVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    await vault.donate(donation())
    orphans = [vault.directory / f"{new_sample_id()}{SUFFIX}" for _ in range(2)]
    for orphan in orphans:
        orphan.write_bytes(b"x")
    real_stat = Path.stat
    failed: list[Path] = []

    def locked(self: Path, **kwargs: object) -> os.stat_result:
        if self in orphans and not failed:
            failed.append(self)
            raise PermissionError("locked")
        return real_stat(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", locked)
    await vault.purge(date(2000, 1, 1), orphan_grace_s=0)
    monkeypatch.undo()
    assert [o.exists() for o in orphans] == [o in failed for o in orphans]
