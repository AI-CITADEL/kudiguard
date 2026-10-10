"""Operator tools for donated voice notes (``kudiguard.donation``): the transcription loop and data-subject requests.

    kudiguard voice export --out var/voice-export [--languages ha,yo] [--n 50] [--seed 1]
    kudiguard voice wer var/voice-export/manifest.csv
    kudiguard voice stats
    kudiguard voice access --user +234803... [--out var/voice-access]   (NDPA s.34: a donor's own samples)
    kudiguard voice erase --user web:<session id>                       (s.34/s.35 requests that come by email)
    kudiguard voice remove <id> [<id> ...]                              (a reviewer found a child's/another voice)
    kudiguard voice preconsent --csv signups.csv --version 2026-10-10    (tester sign-up form: 18+ and own voice)
    kudiguard voice wipe --all                                          (before rotating the key, or to end it)

Exports are decrypted copies of personal data: they go only to the git-ignored ``var/`` (next to the database) or
``exports/`` folders, never into a cloud-synced folder, are shared only with named transcribers under written
confidentiality who work on them locally (no copies abroad or in cloud drives), and are deleted as soon as the
references are typed (docs/DPIA_VOICE.md).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kudiguard.config import ROOT, Settings
from kudiguard.donation import DONATION_VERSION, Sealer, VoiceIntegrityError, VoiceVault
from kudiguard.privacy import UserHasher, normalise_phone
from kudiguard.speech.metrics import POOLED_METRICS
from kudiguard.storage.base import DELETION_MARKER_DAYS, Store, UserState, VoiceSample, aware
from kudiguard.storage.sqlite import SQLiteStore
from kudiguard.text import redact

MANIFEST = "manifest.csv"
COLUMNS = ("id", "language", "week", "duration_s", "asr_hypothesis", "reference")
# Cells a spreadsheet may evaluate: these first characters (incl. full-width forms) after any leading blanks
# (incl. no-break and ideographic spaces), or a leading tab/CR/LF. Same semantics as the review kit's cells.
FORMULA_PREFIXES = ("=", "+", "-", "@", "|", "\t", "\r", "\n", "\uff1d", "\uff0b", "\uff0d", "\uff20")
EXPORT_DIRS = ("var", "exports")  # git-ignored folders of the checkout (.gitignore)
# Folder names of sync clients: prefixes catch "OneDrive - Contoso", "Dropbox (Personal)", "iCloudDrive", …;
# exact names for the ambiguous short ones ("My Drive" is Google Drive's root, "Box" and "Box Sync" are Box's).
CLOUD_PREFIXES = ("onedrive", "dropbox", "icloud", "google drive", "googledrive")
CLOUD_SYNCED = ("my drive", "box", "box sync")
TRUE = frozenset({"yes", "y", "true", "1", "x", "ticked"})
REMINDER = (
    "PRIVACY: these are real people's voices (personal data, NDPA 2023). Share them only with named transcribers "
    "under written confidentiality, who work locally (no cloud drives, no copies abroad), and delete every copy "
    "once the references are typed."
)
_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"OggS", ".ogg"),
    (0, b"\x1aE\xdf\xa3", ".webm"),
    (4, b"ftyp", ".m4a"),
    (0, b"RIFF", ".wav"),
    (0, b"ID3", ".mp3"),
    (0, b"\xff\xfb", ".mp3"),
)


# --- pure helpers ---------------------------------------------------------------------------------------------


def audio_suffix(data: bytes) -> str:
    """File extension from the container's magic bytes, so transcribers can play the original file."""
    for offset, magic, suffix in _SIGNATURES:
        if data[offset : offset + len(magic)] == magic:
            return suffix
    return ".bin"


def _needs_escape(value: str) -> bool:
    if value.startswith(FORMULA_PREFIXES) or value.lstrip().startswith(FORMULA_PREFIXES):
        return True
    return value.startswith("'") and _needs_escape(value[1:])  # so that a real leading apostrophe round-trips


def spreadsheet_cell(value: str) -> str:
    """CSV formula-injection guard (OWASP): a leading apostrophe makes spreadsheets show text, not evaluate it."""
    return "'" + value if _needs_escape(value) else value


def plain_cell(value: str) -> str:
    """Undo ``spreadsheet_cell`` exactly: strip the apostrophe only where escaping added it."""
    return value[1:] if value.startswith("'") and _needs_escape(value[1:]) else value


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE


def choose(
    samples: Iterable[VoiceSample],
    *,
    languages: Collection[str] | None = None,
    per_language: int | None = None,
    seed: int | None = None,
) -> list[VoiceSample]:
    """A reproducible random sample of up to ``per_language`` samples per language, ordered by language and id."""
    by_language: dict[str, list[VoiceSample]] = defaultdict(list)
    for sample in samples:
        if languages is None or sample.language in languages:
            by_language[sample.language].append(sample)
    rng = random.Random(seed)
    chosen: list[VoiceSample] = []
    for language in sorted(by_language):
        pool = sorted(by_language[language], key=lambda s: s.id)
        chosen += (
            pool if per_language is None or per_language >= len(pool) else rng.sample(pool, per_language)
        )
    return sorted(chosen, key=lambda s: (s.language, s.id))


def _cloud_synced(folder: str) -> bool:
    name = folder.lower().strip()
    return name in CLOUD_SYNCED or name.startswith(CLOUD_PREFIXES)


def safe_out(out: Path, allowed: Iterable[Path] = ()) -> Path:
    """Only the git-ignored ``var/`` and ``exports/`` folders (or ``allowed``, e.g. the database's folder), and never a
    cloud-synced folder: decrypted audio must not be committed or uploaded anywhere."""
    resolved = out.resolve()
    roots = [(ROOT / name).resolve() for name in EXPORT_DIRS] + [a.resolve() for a in allowed]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{out}: export only under {', '.join(str(r) for r in roots)} (git-ignored)")
    if any(_cloud_synced(part) for part in resolved.parts):
        raise ValueError(f"{out}: is inside a cloud-synced folder; export to a local folder")
    if (resolved / MANIFEST).exists():
        raise ValueError(f"{resolved / MANIFEST} already exists: use a new folder")
    return resolved


async def export_samples(vault: VoiceVault, samples: Iterable[VoiceSample], out: Path) -> tuple[int, int]:
    """Decrypt each sample to ``out/audio/<id>.<ext>`` and write ``out/manifest.csv`` (UTF-8 BOM, formula-safe).
    Returns (exported, skipped): a sample deleted meanwhile or failing authentication is skipped, never fatal."""
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    skipped = 0
    for sample in samples:
        try:
            data = await vault.read(sample.id)
        except (FileNotFoundError, VoiceIntegrityError):
            skipped += 1
            continue
        (audio_dir / f"{sample.id}{audio_suffix(data)}").write_bytes(data)
        duration = "" if sample.duration_s is None else f"{sample.duration_s:.1f}"
        rows.append([sample.id, sample.language, sample.week.isoformat(), duration, sample.hypothesis, ""])
    with (out / MANIFEST).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows([spreadsheet_cell(cell) for cell in row] for row in rows)
    return len(rows), skipped


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"language", "asr_hypothesis", "reference"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: not a voice manifest (missing columns {sorted(missing)})")
        return [{k: plain_cell(v or "") for k, v in row.items() if k} for row in reader]


METRICS = POOLED_METRICS


def wer_by_language(rows: Iterable[Mapping[str, str]]) -> dict[str, dict[str, float]]:
    """Pooled WER (diacritics folded and kept) and CER per language, over rows with a typed reference
    (``kudiguard.speech.metrics``, the same scoring as scripts/eval_asr.py). ``all`` pools every language."""
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: {k: [0, 0] for k in METRICS})
    clips: Counter[str] = Counter()
    pending: Counter[str] = Counter()
    for row in rows:
        groups = (row.get("language") or "?", "all")
        reference = redact((row.get("reference") or "").strip())  # numbers as [number], like the hypothesis
        for group in groups:
            if not reference:
                pending[group] += 1
                continue
            clips[group] += 1
            for key, (score, tone_sensitive) in METRICS.items():
                edits, total = score(
                    reference, row.get("asr_hypothesis") or "", tone_sensitive=tone_sensitive
                )
                counts[group][key][0] += edits
                counts[group][key][1] += total
    report: dict[str, dict[str, float]] = {}
    for group in sorted(set(clips) | set(pending), key=lambda g: (g == "all", g)):
        entry: dict[str, float] = {"clips": clips[group], "without_reference": pending[group]}
        for key in METRICS:
            edits, total = counts[group][key]
            entry[key] = edits / total if total else float("nan")
        report[group] = entry
    return report


def format_wer(report: Mapping[str, Mapping[str, float]]) -> str:
    lines = ["language  clips  no-ref   WER   WER+d   CER"]
    for group, r in report.items():
        lines.append(
            f"{group:<8} {int(r['clips']):>6} {int(r['without_reference']):>7} "
            f"{r['wer']:>6.1%} {r['wer_diacritics']:>6.1%} {r['cer']:>6.1%}"
        )
    return "\n".join(lines)


def stats(samples: Iterable[VoiceSample]) -> dict[str, dict[str, Any]]:
    """Donors, samples and minutes of audio per language (and ``all``)."""
    donors: dict[str, set[str]] = defaultdict(set)
    count: dict[str, int] = defaultdict(int)
    seconds: dict[str, float] = defaultdict(float)
    for sample in samples:
        for group in (sample.language, "all"):
            donors[group].add(sample.user_hash)
            count[group] += 1
            seconds[group] += sample.duration_s or 0.0
    return {
        group: {
            "donors": len(donors[group]),
            "samples": count[group],
            "minutes": round(seconds[group] / 60, 1),
        }
        for group in sorted(count, key=lambda g: (g == "all", g))
    }


# --- commands -------------------------------------------------------------------------------------------------


async def _open(settings: Settings) -> tuple[Store, VoiceVault, UserHasher]:
    if not settings.db_path.is_file():
        raise ValueError(f"no evidence database at {settings.db_path} (check KUDIGUARD_DB_PATH)")
    hasher = UserHasher(settings.hash_secret.get_secret_value().encode())
    store = await SQLiteStore.open(settings.db_path, hasher=hasher)
    return store, VoiceVault(settings.voice_dir, store, Sealer(settings.voice_secret())), hasher


def cmd_export(settings: Settings, args: argparse.Namespace) -> int:
    out = safe_out(Path(args.out), [settings.db_path.parent])
    languages = frozenset(args.languages.split(",")) if args.languages else None

    async def run() -> tuple[int, int]:
        store, vault, _ = await _open(settings)
        try:
            chosen = choose(
                await store.voice_samples(), languages=languages, per_language=args.n, seed=args.seed
            )
            return await export_samples(vault, chosen, out)
        finally:
            await store.close()

    exported, skipped = asyncio.run(run())
    print(f"exported {exported} voice samples to {out} ({skipped} skipped: deleted or unreadable)")
    print(
        f"transcribers: fill the 'reference' column of {MANIFEST}, then run: kudiguard voice wer <manifest>"
    )
    print(REMINDER)
    return 0


def cmd_wer(settings: Settings, args: argparse.Namespace) -> int:
    print(format_wer(wer_by_language(read_manifest(Path(args.manifest)))))
    print("WER folds diacritics, WER+d keeps them (they change meaning in Yoruba and Igbo). " + REMINDER)
    return 0


def cmd_stats(settings: Settings, args: argparse.Namespace) -> int:
    async def run() -> list[VoiceSample]:
        store, _, _ = await _open(settings)
        try:
            return await store.voice_samples()
        finally:
            await store.close()

    for group, row in stats(asyncio.run(run())).items():
        print(f"{group}: {row['donors']} donors, {row['samples']} samples, {row['minutes']} min")
    return 0


def cmd_access(settings: Settings, args: argparse.Namespace) -> int:
    """Answer an access request: the donor's consent record and samples, and with --out a copy of their audio."""
    out = safe_out(Path(args.out), [settings.db_path.parent]) if args.out else None

    async def run() -> int:
        store, vault, hasher = await _open(settings)
        try:
            user_hash = hasher.user_id(args.user)
            user = await store.get_user(user_hash)
            if user is not None and user.donation_version:
                assert user.donation_at is not None
                print(
                    f"donation consent: version {user.donation_version}, given {user.donation_at.isoformat()} "
                    f"via {user.donation_source or 'chat'}"
                )
            else:
                print("donation consent: none on record")
            samples = await store.voice_samples(user_hash)
            for s in samples:
                print(f"  {s.id}  week of {s.week}  {s.language}  {s.channel}  consent {s.consent_version}")
            if out is not None:
                exported, _ = await export_samples(vault, samples, out)
                print(
                    f"copied {exported} samples to {out}: send them securely to the requester, then delete."
                )
            return len(samples)
        finally:
            await store.close()

    print(f"{asyncio.run(run())} donated voice samples for this user")
    return 0


def cmd_erase(settings: Settings, args: argparse.Namespace) -> int:
    async def run() -> int:
        store, vault, hasher = await _open(settings)
        try:
            user_hash = hasher.user_id(args.user)
            removed = await vault.erase(user_hash)
            user = await store.get_user(user_hash)
            if user is not None:
                user.donation_version, user.donation_at, user.donation_source = None, None, None
                await store.save_user(user)
            return removed
        finally:
            await store.close()

    print(f"deleted {asyncio.run(run())} donated voice samples and withdrew the donation consent")
    return 0


def cmd_remove(settings: Settings, args: argparse.Namespace) -> int:
    """Delete single samples a reviewer flagged (a child's voice, someone else's voice, off-purpose content)."""

    async def run() -> int:
        store, vault, _ = await _open(settings)
        try:
            return sum([await vault.remove(sample_id) for sample_id in args.ids])
        finally:
            await store.close()

    print(f"deleted {asyncio.run(run())} of {len(args.ids)} voice samples")
    return 0


PRECONSENT_COLUMNS = ("number", "donate_confirmed", "age_confirmed", "own_voice_confirmed", "consented_at")
# Older answers are not imported. One day shorter than a DELETE marker's life (the purge loop drops markers up to
# one 6-hour interval early), so a marker always outlives every form answer it must block.
PRECONSENT_MAX_AGE_DAYS = DELETION_MARKER_DAYS - 1
_PHONE_TEXT = re.compile(r"^(?:whatsapp:|tel:)?[\d\s()+.\-]+$", re.IGNORECASE)  # digits and separators only
_NIGERIAN_MOBILE = re.compile(r"^234[789][01]\d{8}$")
_LOCAL_WITHOUT_ZERO = re.compile(r"^[789][01]\d{8}$")  # a spreadsheet dropped the leading 0 of 0803...
_SCIENTIFIC = re.compile(r"\d[.,]?\d*[eE][+-]?\d+")  # 2.34803E+12: the spreadsheet already lost digits


def nigerian_mobile(raw: str) -> str | None:
    """``+234XXXXXXXXXX`` for a Nigerian mobile number in any common format, else None (never guessed)."""
    text = raw.strip()
    if not text or _SCIENTIFIC.search(text) or not _PHONE_TEXT.fullmatch(text):
        return None
    digits = normalise_phone(text)
    if digits.startswith("00"):  # international dialling prefix: 00 234 ...
        digits = digits[2:]
    if _LOCAL_WITHOUT_ZERO.fullmatch(digits):
        digits = "234" + digits
    return f"+{digits}" if _NIGERIAN_MOBILE.fullmatch(digits) else None


def read_preconsents(path: Path, version: str) -> list[tuple[str, datetime]]:
    """Rows of the tester sign-up export (``PRECONSENT_COLUMNS``). Every row must have a Nigerian mobile number, all
    three boxes ticked and a past, timezone-aware time; any bad row rejects the whole file (nothing half-imported).
    Bad rows are reported by line number, never by phone number."""
    if version != DONATION_VERSION:
        raise ValueError(
            f"the form must show the current donation text (version {DONATION_VERSION}), not {version}"
        )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(PRECONSENT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows, bad_numbers, unconfirmed = [], [], []
        for line, row in enumerate(reader, start=2):
            number = nigerian_mobile(row["number"] or "")
            try:
                at = aware(datetime.fromisoformat((row["consented_at"] or "").strip()))
            except ValueError:
                at = None
            boxes = ("donate_confirmed", "age_confirmed", "own_voice_confirmed")
            if number is None:
                bad_numbers.append(line)
            elif not all(truthy(row[box]) for box in boxes) or at is None or at > datetime.now(UTC):
                unconfirmed.append(line)
            else:
                rows.append((number, at))
    problems = []
    if bad_numbers:
        problems.append(
            f"rows {bad_numbers} have no valid Nigerian mobile number (format the column as text: a spreadsheet "
            "turns 0803... into 803... or 2.34803E+12)"
        )
    if unconfirmed:
        problems.append(
            f"rows {unconfirmed} lack a ticked donate, 18+ or own-voice box, or a past, timezone-aware consented_at"
        )
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}; nothing was imported")
    return rows


def preconsent_blocker(
    user: UserState | None, at: datetime, now: datetime, deleted_at: datetime | None = None
) -> str | None:
    """Why a form answer must not be recorded: it would override a later DELETE, STOP, withdrawal or decline (or an
    earlier chat consent), or it is too old to rely on. None = record it."""
    if now - at > timedelta(days=PRECONSENT_MAX_AGE_DAYS):
        return "too old"
    if deleted_at is not None and deleted_at >= at:
        return "deleted"
    if user is None:
        return None
    if user.stopped:
        return "stopped"
    if user.donation_version is not None:
        return "already consented"
    if user.donation_withdrawn_at is not None:
        return "declined or withdrew"
    if user.donation_offered_version is not None or user.donation_offered_at is not None:
        return "offered, no answer yet"
    if user.last_seen_at > at:
        return "active after the form"
    return None


def cmd_preconsent(settings: Settings, args: argparse.Namespace) -> int:
    """Record sign-up-form consent as if the tester had tapped "Yes, I am 18+" (source=form; not asked again).

    Never resurrects consent: rows whose user deleted their data, stopped, withdrew, declined, was offered or was
    active after the form are skipped (those testers are asked in the chat instead). A DELETE leaves only a coded
    marker for ``PRECONSENT_MAX_AGE_DAYS``, the same limit as the form's age, so a deleted tester is never re-enrolled.
    Rows never imply main consent: the tester still gets the welcome in the chat. A number listed twice counts once,
    with its most recent answer. A DELETE that lands while a row is being saved wins: the row is re-checked after the
    save and erased again."""
    freshest: dict[str, datetime] = {}
    rows = read_preconsents(Path(args.csv), args.version)
    for number, at in rows:
        freshest[number] = max(at, freshest.get(number, at))
    duplicates = len(rows) - len(freshest)

    async def run() -> tuple[int, int, Counter[str]]:
        store, _, hasher = await _open(settings)
        recorded_existing, recorded_new, skipped = 0, 0, Counter[str]()
        if duplicates:
            skipped["listed twice"] = duplicates
        now = datetime.now(UTC)
        try:
            for number, at in freshest.items():
                user_hash = hasher.user_id(number)
                user = await store.get_user(user_hash)
                if reason := preconsent_blocker(user, at, now, await store.deleted_at(user_hash)):
                    skipped[reason] += 1
                    continue
                base = user or UserState(user_hash, created_at=at, last_seen_at=at)
                await store.save_user(
                    dataclasses.replace(
                        base, donation_version=args.version, donation_at=at, donation_source="form",
                        donation_offered_at=at, donation_offered_version=args.version,
                    )
                )  # fmt: skip
                deleted = await store.deleted_at(user_hash)  # a DELETE between our check and our save wins
                if deleted is not None and deleted >= at:
                    await store.delete_user(user_hash)
                    skipped["deleted"] += 1
                    continue
                if user is None:
                    recorded_new += 1
                else:
                    recorded_existing += 1
            return recorded_existing, recorded_new, skipped
        finally:
            await store.close()

    existing, new, skipped = asyncio.run(run())
    print(
        f"recorded donation consent (source: form) for {existing + new} testers: "
        f"{existing} already known to KudiGuard, {new} new."
    )
    if skipped:
        details = ", ".join(f"{count} {reason}" for reason, count in sorted(skipped.items()))
        print(f"skipped {sum(skipped.values())} (never overriding a later choice): {details}")
    print("Delete the sign-up export now: it holds phone numbers. Only hashes were stored.")
    return 0


def cmd_wipe(settings: Settings, args: argparse.Namespace) -> int:
    """Delete every donated sample (rows and files): before rotating the key, or to end the programme."""
    if not args.all:
        raise ValueError("refusing to wipe without --all")

    async def run() -> tuple[int, int]:
        store, vault, _ = await _open(settings)
        try:
            return await vault.wipe()
        finally:
            await store.close()

    removed, remaining = asyncio.run(run())
    if remaining:
        print(
            f"ERROR: {remaining} voice files remain in {settings.voice_dir} after deleting {removed} samples. The "
            "folder does not belong to this database (or files are locked). Check KUDIGUARD_VOICE_DIR and "
            "KUDIGUARD_DB_PATH; do NOT rotate the key until this folder is empty (see docs/DEPLOY.md).",
            file=sys.stderr,
        )
        return 1
    print(f"deleted all {removed} donated voice samples; the folder holds no audio")
    return 0


def add_commands(commands: Any) -> None:
    """Register ``kudiguard voice ...`` on the main CLI's subparsers."""
    voice = commands.add_parser(
        "voice", help="donated voice notes: transcription loop and data-subject requests"
    )
    sub = voice.add_subparsers(dest="voice_command", required=True)

    export = sub.add_parser("export", help="decrypted sample + manifest for native-speaker transcription")
    export.add_argument("--out", required=True, help="a new folder under var/ or exports/ (git-ignored)")
    export.add_argument("--languages", help="comma-separated codes (default: all)")
    export.add_argument("--n", type=int, default=None, help="at most this many samples per language")
    export.add_argument("--seed", type=int, default=None)
    export.set_defaults(func=cmd_export)

    wer = sub.add_parser("wer", help="word error rate per language from a transcribed manifest")
    wer.add_argument("manifest")
    wer.set_defaults(func=cmd_wer)

    count = sub.add_parser("stats", help="donors, samples and minutes per language")
    count.set_defaults(func=cmd_stats)

    access = sub.add_parser("access", help="a donor's consent record and samples (access request)")
    access.add_argument("--user", required=True, help="their WhatsApp number, or web:<session id>")
    access.add_argument("--out", help="also copy their audio here (git-ignored folder)")
    access.set_defaults(func=cmd_access)

    erase = sub.add_parser("erase", help="delete a donor's samples and withdraw their consent (by request)")
    erase.add_argument("--user", required=True, help="their WhatsApp number, or web:<session id>")
    erase.set_defaults(func=cmd_erase)

    remove = sub.add_parser("remove", help="delete single samples flagged by a reviewer (child, other voice)")
    remove.add_argument("ids", nargs="+", help="sample ids from the manifest")
    remove.set_defaults(func=cmd_remove)

    pre = sub.add_parser("preconsent", help="record donation consent given on the tester sign-up form")
    pre.add_argument("--csv", required=True, help=",".join(PRECONSENT_COLUMNS))
    pre.add_argument("--version", required=True, help="the donation text version shown on the form")
    pre.set_defaults(func=cmd_preconsent)

    wipe = sub.add_parser("wipe", help="delete every donated sample (before rotating the key)")
    wipe.add_argument("--all", action="store_true", help="required: confirms deleting everything")
    wipe.set_defaults(func=cmd_wipe)
