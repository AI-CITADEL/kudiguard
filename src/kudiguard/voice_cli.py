"""Operator tools for donated voice notes (``kudiguard.donation``): the transcription loop and data-subject requests.

    kudiguard voice export --out var/voice-export [--languages ha,yo] [--n 50] [--seed 1]
    kudiguard voice wer var/voice-export/manifest.csv
    kudiguard voice stats
    kudiguard voice access --user +234803... [--out var/voice-access]   (NDPA s.34: a donor's own samples)
    kudiguard voice erase --user web:<session id>                       (s.34/s.35 requests that come by email)
    kudiguard voice remove <id> [<id> ...]                              (a reviewer found a child's/another voice)

Exports are decrypted copies of personal data: they go only to git-ignored folders of this checkout (``var/`` or
``exports/``) or outside it, are shared only with named transcribers under written confidentiality who work on them
locally (no copies abroad or in cloud drives), and are deleted as soon as the references are typed (docs/DPIA_VOICE.md).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

from kudiguard.config import ROOT, Settings
from kudiguard.donation import Sealer, VoiceIntegrityError, VoiceVault
from kudiguard.privacy import UserHasher
from kudiguard.speech.metrics import POOLED_METRICS
from kudiguard.storage.base import Store, VoiceSample
from kudiguard.storage.sqlite import SQLiteStore

MANIFEST = "manifest.csv"
COLUMNS = ("id", "language", "week", "duration_s", "asr_hypothesis", "reference")
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")  # same rule as the review kit's spreadsheet cells
IGNORED_DIRS = ("var", "exports")  # git-ignored folders of the checkout (.gitignore)
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


def spreadsheet_cell(value: str) -> str:
    """CSV formula-injection guard (OWASP): a leading apostrophe makes spreadsheets show text, not evaluate it."""
    return "'" + value if value.startswith(FORMULA_PREFIXES) else value


def plain_cell(value: str) -> str:
    """Undo ``spreadsheet_cell`` when reading a manifest back."""
    return value[1:] if value.startswith("'") and value[1:].startswith(FORMULA_PREFIXES) else value


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


def safe_out(out: Path, root: Path | None = None) -> Path:
    """Inside this checkout, only git-ignored folders: decrypted audio must never be committed."""
    root = root or ROOT
    resolved = out.resolve()
    if resolved.is_relative_to(root.resolve()) and not any(
        resolved.is_relative_to((root / name).resolve()) for name in IGNORED_DIRS
    ):
        raise ValueError(
            f"{out}: inside the repository use a git-ignored folder ({', '.join(IGNORED_DIRS)}/)"
        )
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
        reference = (row.get("reference") or "").strip()
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
    secret = settings.hash_secret.get_secret_value().encode()
    hasher = UserHasher(secret)
    store = await SQLiteStore.open(settings.db_path, hasher=hasher)
    return store, VoiceVault(settings.voice_dir, store, Sealer(secret)), hasher


def cmd_export(settings: Settings, args: argparse.Namespace) -> int:
    out = safe_out(Path(args.out))
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
    out = safe_out(Path(args.out)) if args.out else None

    async def run() -> int:
        store, vault, hasher = await _open(settings)
        try:
            user_hash = hasher.user_id(args.user)
            user = await store.get_user(user_hash)
            if user is not None and user.donation_version:
                assert user.donation_at is not None
                print(
                    f"donation consent: version {user.donation_version}, given {user.donation_at.isoformat()}"
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
                user.donation_version, user.donation_at = None, None
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
