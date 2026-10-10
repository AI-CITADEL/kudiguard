"""Opt-in voice-note donation: keep a consenting adult's own voice notes to test and improve speech recognition.

Separate from the main consent (NDPA 2023 ss.25-26): off by default, given only with an explicit "Yes, I am 18+"
(WhatsApp button or an unticked web checkbox), recorded with its version and time, withdrawn as easily as given
("stop donating" / unticking the box), which also deletes everything already donated. See docs/DPIA_VOICE.md.

* Only voice notes received *after* opt-in, in the donor's own voice: forwarded WhatsApp voice notes (usually someone
  else's voice, e.g. a scammer's) are never kept, and text is never donated.
* The original audio bytes are encrypted at rest with AES-256-GCM under a key derived (HKDF-SHA256, own ``info``
  label) from the server's hash secret, in files named by a random id; the file's id is authenticated, so files
  cannot be swapped or altered unnoticed. The database row holds no timestamp (only the week), no phone number and a
  redacted speech-recognition hypothesis. Nobody uses the audio to identify speakers.
* Writes happen after the user's reply has been sent and never raise into the conversation; they are bounded
  (``MAX_PENDING_WRITES``) and serialised with erasure so a withdrawal can never be overtaken by a late write.
* Erasure and retention delete the row first, then the file; a file left behind by a crash has no row and is removed
  by the next purge (``purge`` sweeps orphans older than ``ORPHAN_GRACE_S``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
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
VOICE_RETENTION_DAYS = 365  # 12 months, stated in the offer and on /privacy
# End users who can consent themselves (never an API partner's users).
DONATION_CHANNELS = frozenset({"whatsapp", "web"})
MAX_PENDING_WRITES = 8  # beyond this a sample is skipped (bounded memory: each holds up to one voice note)
ORPHAN_GRACE_S = 3600.0  # a file without a row this old is debris from a crash (never an in-flight write)
SUFFIX = ".kgv"
_MAGIC = b"KGV1"  # format/version tag, also authenticated
_NONCE_BYTES = 12
_KEY_INFO = b"kudiguard/voice-donation/aes-256-gcm/v1"  # domain separation from the user-id HMAC
_ID = re.compile(r"^[0-9a-f]{32}$")


class VoiceIntegrityError(ValueError):
    """A sample file is not ours, was altered, or belongs to another id (or the server secret changed)."""


def new_sample_id() -> str:
    return secrets.token_hex(16)


def _checked(sample_id: str) -> str:
    if not _ID.fullmatch(sample_id):
        raise ValueError("invalid sample id")
    return sample_id


class Sealer:
    """AES-256-GCM with a random 96-bit nonce per file; the sample id is bound as associated data."""

    def __init__(self, secret: bytes) -> None:
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
    ) -> None:
        self.directory = directory
        self._store = store
        self._sealer = sealer
        self._clock = clock
        self._max_pending = max_pending
        self._pending = 0
        self._lock = asyncio.Lock()  # donate vs erase/purge: a withdrawal is never overtaken by a late write

    def path(self, sample_id: str) -> Path:
        return self.directory / f"{_checked(sample_id)}{SUFFIX}"

    async def donate(self, donation: Donation) -> str | None:
        """Keep one sample; returns its id, or None if it was not kept. Never raises (the reply is already out)."""
        if donation.channel not in DONATION_CHANNELS or donation.consent_version != DONATION_VERSION:
            log.warning("voice donation refused: channel or consent version not accepted")
            return None
        if self._pending >= self._max_pending:
            log.warning("voice donation skipped: %d writes pending", self._pending)
            return None
        self._pending += 1
        work = asyncio.ensure_future(self._keep(donation))
        try:
            # Shielded: if the caller is cancelled (turn deadline, shutdown), the write still completes under the
            # lock, so a following erase sees it; the outcome is reported by _keep itself.
            return await asyncio.shield(work)
        finally:
            if work.done():
                self._pending -= 1
            else:
                work.add_done_callback(lambda _: self._release())

    def _release(self) -> None:
        self._pending -= 1

    async def _keep(self, donation: Donation) -> str | None:
        sample_id = new_sample_id()
        try:
            async with self._lock:
                await asyncio.to_thread(self._write, sample_id, donation.audio)  # encrypt off the event loop
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
                    await asyncio.to_thread(self._unlink, sample_id)  # no row, no file
                    raise
        except Exception as exc:  # never the content: the audio and hypothesis stay out of logs
            log.error("voice donation not kept (%s)", type(exc).__name__)
            return None
        return sample_id

    async def erase(self, user_hash: str) -> int:
        """Withdrawal or DELETE: remove every sample of a donor (rows, then files). Returns samples removed."""
        async with self._lock:
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

    async def purge(self, cutoff: date, *, orphan_grace_s: float = ORPHAN_GRACE_S) -> int:
        """Retention: remove samples from weeks starting on or before ``cutoff``, plus orphaned files."""
        async with self._lock:
            ids = await self._store.purge_voice_samples(cutoff)
            await asyncio.to_thread(self._unlink_all, ids)
            known = {s.id for s in await self._store.voice_samples()}
            orphans = await asyncio.to_thread(self._sweep, known, orphan_grace_s)
        if orphans:
            log.warning("removed %d orphaned voice files", orphans)
        return len(ids)

    async def read(self, sample_id: str) -> bytes:
        """The original audio of one sample (operator tools only). Raises ``VoiceIntegrityError`` if tampered."""
        blob = await asyncio.to_thread(self.path(sample_id).read_bytes)
        return self._sealer.open(sample_id, blob)

    # --- files (run in a worker thread) ---------------------------------------------------------------------

    def _write(self, sample_id: str, audio: bytes) -> None:
        blob = self._sealer.seal(sample_id, audio)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
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

    def _unlink(self, sample_id: str) -> None:
        self.path(sample_id).unlink(missing_ok=True)  # already gone is fine (manual cleanup, earlier crash)

    def _unlink_all(self, ids: list[str]) -> None:
        for sample_id in ids:
            try:
                self._unlink(sample_id)
            except OSError as exc:  # keep going: the orphan sweep retries this file later
                log.error("could not delete a voice file (%s)", type(exc).__name__)

    def _sweep(self, known: set[str], grace_s: float) -> int:
        if not self.directory.is_dir():
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
        return removed
