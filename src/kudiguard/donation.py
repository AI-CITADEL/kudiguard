"""Opt-in voice-note donation: keep a consenting adult's own voice notes to test and improve speech recognition.

Separate from the main consent (NDPA 2023 ss.25, 26, 30): off by default (and the whole feature is off until the
operator sets ``KUDIGUARD_VOICE_DONATION=true`` after the DPIA sign-off), given only with an explicit yes that also
confirms 18+ and own voice (WhatsApp button, web checkbox or the tester sign-up form), recorded with its version, time
and source, withdrawn as easily as given, which also deletes everything already donated. See docs/DPIA_VOICE.md.

* Only voice notes received *after* opt-in. On WhatsApp only recordings made in the app (``audio.voice``) that are not
  marked forwarded; text is never donated.
* The original audio bytes are encrypted at rest with AES-256-GCM under a key derived (HKDF-SHA256, own ``info``
  label) from ``KUDIGUARD_VOICE_KEY`` or else the hash secret, in files named by a random id; the id is authenticated,
  so files cannot be swapped or altered unnoticed. Rows hold no timestamp (only the week), no phone number and a
  redacted speech-recognition hypothesis. Nobody uses the audio to identify speakers.
* Writes happen after the user's reply, never raise, are bounded (pending writes, per-donor daily cap, folder size),
  and are serialised with erasure; a withdrawal leaves a tombstone, so a write whose consent predates it is refused
  even if it arrives late (e.g. a web background task).
* Erasure and retention delete the row first, then the file. Files without a row are swept by the next purge, but only
  when the folder's marker matches the database (``Store.voice_vault_id``): never against another database.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from kudiguard.domain import LanguageCode
from kudiguard.patterns import week_of
from kudiguard.storage.base import Store, VoiceSample, utcnow

log = logging.getLogger(__name__)

DONATION_VERSION = "2026-10-10"  # bump when the donation offer text changes materially (also in voice.html)
VOICE_RETENTION_DAYS = 365  # 12 months, stated in the offer and on /privacy (the setting cannot exceed it)
# End users who can consent themselves (never an API partner's users).
DONATION_CHANNELS = frozenset({"whatsapp", "web"})
MAX_PENDING_WRITES = 8  # beyond this a sample is skipped (bounded memory: each holds up to one voice note)
MAX_PER_DONOR_PER_DAY = 20  # a donor's samples per UTC day (bounds one person's share and runaway clients)
MAX_FOLDER_BYTES = 5 * 1024**3  # stop collecting before the disk fills
ORPHAN_GRACE_S = 3600.0  # a file without a row this old is debris from a crash (never an in-flight write)
TOMBSTONE_S = 3600.0  # how long a withdrawal blocks late writes whose consent predates it
MIN_KEY_BYTES = 32
SUFFIX = ".kgv"
MARKER = ".vault"  # holds Store.voice_vault_id: binds the folder to its database
_MAGIC = b"KGV1"  # format/version tag, also authenticated
_NONCE_BYTES = 12
_KEY_INFO = b"kudiguard/voice-donation/aes-256-gcm/v1"  # domain separation from the user-id HMAC
_ID = re.compile(r"^[0-9a-f]{32}$")


class VoiceIntegrityError(ValueError):
    """A sample file is not ours, was altered, or belongs to another id (or the key changed)."""


def new_sample_id() -> str:
    return secrets.token_hex(16)


def _checked(sample_id: str) -> str:
    if not _ID.fullmatch(sample_id):
        raise ValueError("invalid sample id")
    return sample_id


class Sealer:
    """AES-256-GCM with a random 96-bit nonce per file; the sample id is bound as associated data."""

    def __init__(self, secret: bytes) -> None:
        if len(secret) < MIN_KEY_BYTES:
            raise ValueError(f"the voice key must be at least {MIN_KEY_BYTES} bytes")
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_KEY_INFO).derive(secret)
        self._aead = AESGCM(key)

    def seal(self, sample_id: str, data: bytes) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        return _MAGIC + nonce + self._aead.encrypt(nonce, data, _MAGIC + _checked(sample_id).encode())

    def open(self, sample_id: str, blob: bytes) -> bytes:
        if not blob.startswith(_MAGIC) or len(blob) < len(_MAGIC) + _NONCE_BYTES + 16:
            raise VoiceIntegrityError("not a KudiGuard voice sample")
        nonce, sealed = blob[len(_MAGIC) : len(_MAGIC) + _NONCE_BYTES], blob[len(_MAGIC) + _NONCE_BYTES :]
        try:
            return self._aead.decrypt(nonce, sealed, _MAGIC + _checked(sample_id).encode())
        except InvalidTag as exc:
            raise VoiceIntegrityError("voice sample failed authentication (altered, or another key)") from exc


@dataclass(frozen=True, slots=True)
class Donation:
    """One voice note to keep (built by the channel after the reply was sent)."""

    user_hash: str
    channel: str
    language: LanguageCode
    audio: bytes
    hypothesis: str
    consent_version: str
    duration_s: float | None = None
    consented_at: datetime = field(
        default_factory=utcnow
    )  # consent stood at this time (when the note arrived)


class VoiceVault:
    """Encrypted files under ``directory`` plus their ``Store`` rows. One instance per process."""

    def __init__(
        self,
        directory: Path,
        store: Store,
        sealer: Sealer,
        *,
        clock: Callable[[], datetime] = utcnow,
        max_pending: int = MAX_PENDING_WRITES,
        max_per_day: int = MAX_PER_DONOR_PER_DAY,
        max_bytes: int = MAX_FOLDER_BYTES,
    ) -> None:
        self.directory = directory
        self._store = store
        self._sealer = sealer
        self._clock = clock  # dates the week of a sample
        self._max_pending = max_pending
        self._max_per_day = max_per_day
        self._max_bytes = max_bytes
        self._pending = 0
        self._lock = asyncio.Lock()  # donate vs erase/purge: a withdrawal is never overtaken by a late write
        self._withdrawn: dict[str, datetime] = {}  # user hash -> last withdrawal (real time, see Donation)
        self._today: tuple[date, Counter[str]] = (date.min, Counter())
        self._bytes: int | None = None  # folder size, measured once, then tracked
        self.dropped: Counter[str] = Counter()  # samples not kept, by reason (logged; for monitoring)

    def path(self, sample_id: str) -> Path:
        return self.directory / f"{_checked(sample_id)}{SUFFIX}"

    async def donate(self, donation: Donation) -> str | None:
        """Keep one sample; returns its id, or None if it was not kept. Never raises (the reply is already out)."""
        if donation.channel not in DONATION_CHANNELS or donation.consent_version != DONATION_VERSION:
            self._drop("consent")
            return None
        if self._pending >= self._max_pending:
            self._drop("busy")
            return None
        self._pending += 1
        work = asyncio.ensure_future(self._keep(donation))
        try:
            # Shielded: if the caller is cancelled (shutdown), the write still completes under the lock, so a
            # following erase sees it; the outcome is reported by _keep itself.
            return await asyncio.shield(work)
        finally:
            if work.done():
                self._pending -= 1
            else:
                work.add_done_callback(lambda _: self._release())

    def _release(self) -> None:
        self._pending -= 1

    def _drop(self, reason: str) -> None:
        """Count and log (never the content) a sample that was not kept; the caller returns None."""
        self.dropped[reason] += 1
        log.warning("voice donation not kept: %s (%d so far)", reason, self.dropped[reason])

    async def _keep(self, donation: Donation) -> str | None:
        sample_id = new_sample_id()
        try:
            async with self._lock:
                withdrawn = self._withdrawn.get(donation.user_hash)
                if withdrawn is not None and withdrawn >= donation.consented_at:
                    self._drop("withdrawn")  # consent ended after this note arrived
                    return None
                if not self._under_daily_cap(donation.user_hash):
                    self._drop("daily-cap")
                    return None
                expected = await self._store.voice_vault_id()
                size = await asyncio.to_thread(self._write, sample_id, donation.audio, expected)
                if size is None:
                    self._drop("full")
                    return None
                try:
                    await self._store.add_voice_sample(
                        VoiceSample(
                            id=sample_id, user_hash=donation.user_hash, week=week_of(self._clock().date()),
                            language=donation.language, channel=donation.channel,
                            consent_version=donation.consent_version, hypothesis=donation.hypothesis,
                            duration_s=round(donation.duration_s, 1) if donation.duration_s is not None else None,
                        )
                    )  # fmt: skip
                except BaseException:
                    await asyncio.to_thread(self._unlink_all, [sample_id])  # no row, no file
                    raise
                self._today[1][donation.user_hash] += 1
        except Exception as exc:  # never the content: the audio and hypothesis stay out of logs
            self._drop(f"error:{type(exc).__name__}")
            return None
        return sample_id

    def _under_daily_cap(self, user_hash: str) -> bool:
        today = utcnow().date()
        if self._today[0] != today:
            self._today = (today, Counter())  # bounded: one day of donors
        return self._today[1][user_hash] < self._max_per_day

    async def erase(self, user_hash: str) -> int:
        """Withdrawal, STOP or DELETE: remove every sample of a donor (rows, then files) and block late writes whose
        consent predates this moment. Returns samples removed."""
        async with self._lock:
            now = utcnow()
            self._withdrawn = {
                u: t for u, t in self._withdrawn.items() if now - t < timedelta(seconds=TOMBSTONE_S)
            }
            self._withdrawn[user_hash] = now
            ids = await self._store.delete_voice_samples(user_hash)
            await asyncio.to_thread(self._unlink_all, ids)
        return len(ids)

    async def remove(self, sample_id: str) -> bool:
        """Delete one sample (row, then file), e.g. a child's or someone else's voice found by a reviewer."""
        _checked(sample_id)
        async with self._lock:
            found = await self._store.delete_voice_sample(sample_id)
            await asyncio.to_thread(self._unlink_all, [sample_id])
        return found

    async def wipe(self) -> int:
        """Delete every sample (before rotating the key, or to end the programme). Returns samples removed."""
        async with self._lock:
            ids = [s.id for s in await self._store.voice_samples()]
            for sample_id in ids:
                await self._store.delete_voice_sample(sample_id)
            expected = await self._store.voice_vault_id()
            await asyncio.to_thread(self._unlink_all, ids)
            await asyncio.to_thread(self._sweep, set(), 0.0, expected)
        return len(ids)

    async def purge(self, cutoff: date, *, orphan_grace_s: float = ORPHAN_GRACE_S) -> int:
        """Retention: remove samples from weeks starting on or before ``cutoff``, plus orphaned files."""
        async with self._lock:
            ids = await self._store.purge_voice_samples(cutoff)
            await asyncio.to_thread(self._unlink_all, ids)
            known = {s.id for s in await self._store.voice_samples()}
            expected = await self._store.voice_vault_id()
            orphans = await asyncio.to_thread(self._sweep, known, orphan_grace_s, expected)
        if orphans:
            log.warning("removed %d orphaned voice files", orphans)
        return len(ids)

    async def read(self, sample_id: str) -> bytes:
        """The original audio of one sample (operator tools only). Raises ``VoiceIntegrityError`` if tampered."""
        blob = await asyncio.to_thread(self.path(sample_id).read_bytes)
        return self._sealer.open(sample_id, blob)

    # --- files (run in a worker thread) ---------------------------------------------------------------------

    def _marker_matches(self, expected: str, *, claim: bool) -> bool:
        """The folder belongs to this database. An unmarked folder is claimed only if it holds no samples yet."""
        marker = self.directory / MARKER
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip() == expected
        if not claim or any(p.suffix in (SUFFIX, ".part") for p in self.directory.iterdir()):
            return False
        marker.write_text(expected, encoding="utf-8")
        return True

    def _write(self, sample_id: str, audio: bytes, expected: str) -> int | None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self._marker_matches(expected, claim=True):
            raise RuntimeError("voice folder belongs to another database (marker mismatch)")
        if self._bytes is None:
            self._bytes = sum(p.stat().st_size for p in self.directory.iterdir() if p.suffix == SUFFIX)
        blob = self._sealer.seal(sample_id, audio)
        if self._bytes + len(blob) > self._max_bytes:
            return None
        final = self.path(sample_id)
        partial = final.with_suffix(".part")
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, final)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        _fsync_directory(self.directory)
        self._bytes += len(blob)
        return len(blob)

    def _unlink_all(self, ids: list[str]) -> None:
        for sample_id in ids:
            path = self.path(sample_id)
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue  # already gone (manual cleanup, earlier crash)
            except OSError as exc:  # keep going: the orphan sweep retries this file later
                log.error("could not delete a voice file (%s)", type(exc).__name__)
                continue
            if self._bytes is not None:
                self._bytes -= size

    def _sweep(self, known: set[str], grace_s: float, expected: str) -> int:
        if not self.directory.is_dir():
            return 0
        if not self._marker_matches(expected, claim=False):
            if any(p.suffix in (SUFFIX, ".part") for p in self.directory.iterdir()):
                log.error(
                    "voice folder %s does not belong to this database: orphan sweep refused "
                    "(check KUDIGUARD_VOICE_DIR and KUDIGUARD_DB_PATH)",
                    self.directory,
                )
            return 0
        removed = 0
        now = time.time()
        for path in self.directory.iterdir():
            if path.suffix not in (SUFFIX, ".part") or path.stem in known:
                continue
            try:
                if now - path.stat().st_mtime >= grace_s:
                    path.unlink()
                    removed += 1
            except OSError:
                continue  # raced with another delete, or locked: next purge
        self._bytes = None  # re-measure on the next write
        return removed


def _fsync_directory(directory: Path) -> None:
    """Make the rename durable (POSIX); Windows cannot open a directory this way and needs no such step."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
