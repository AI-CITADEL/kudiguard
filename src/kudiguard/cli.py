"""Command line: run the server and produce the validation evidence.

    kudiguard serve [--host 127.0.0.1] [--port 8000]
    kudiguard stats [--json] [--exclude-users FILE] [--channels whatsapp,web]
    kudiguard export --out evidence.jsonl
    kudiguard purge [--days 90]          (also donated voice notes past KUDIGUARD_VOICE_RETENTION_DAYS)
    kudiguard voice export|wer|stats|access|erase|remove|preconsent|wipe   (kudiguard.voice_cli)

Exports contain no phone numbers or provider ids (pseudonymous user hashes, redacted text).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from kudiguard import voice_cli
from kudiguard.config import ROOT, Settings
from kudiguard.donation import Sealer, VoiceVault
from kudiguard.envfile import load_env_file
from kudiguard.privacy import UserHasher
from kudiguard.reporting import USER_CHANNELS, build_report, format_report
from kudiguard.storage.base import utcnow
from kudiguard.storage.sqlite import SQLiteStore


async def _rows(settings: Settings) -> list[dict[str, object]]:
    if not settings.db_path.is_file():
        raise ValueError(f"no evidence database at {settings.db_path} (check KUDIGUARD_DB_PATH)")
    store = await SQLiteStore.open(settings.db_path, hasher=_hasher(settings))
    try:
        return await store.export_interactions()
    finally:
        await store.close()


def _hasher(settings: Settings) -> UserHasher:
    return UserHasher(settings.hash_secret.get_secret_value().encode())


def cmd_stats(settings: Settings, args: argparse.Namespace) -> int:
    rows = asyncio.run(_rows(settings))
    exclude: frozenset[str] = frozenset()
    if args.exclude_users:  # one phone number / id per line (team testers) → hashed here, never stored
        hasher = _hasher(settings)
        lines = Path(args.exclude_users).read_text(encoding="utf-8").splitlines()
        exclude = frozenset(hasher.user_id(line.strip()) for line in lines if line.strip())
    channels = frozenset(args.channels.split(",")) if args.channels else USER_CHANNELS
    report = build_report(rows, channels=channels, exclude_users=exclude)
    print(json.dumps(report.as_dict(), indent=2) if args.json else format_report(report))
    return 0


def cmd_export(settings: Settings, args: argparse.Namespace) -> int:
    rows = asyncio.run(_rows(settings))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} interactions to {out}")
    return 0


def cmd_patterns(settings: Settings, args: argparse.Namespace) -> int:
    async def rows() -> list[dict[str, object]]:
        if not settings.db_path.is_file():
            raise ValueError(f"no evidence database at {settings.db_path} (check KUDIGUARD_DB_PATH)")
        store = await SQLiteStore.open(settings.db_path, hasher=_hasher(settings))
        try:
            return await store.export_patterns()
        finally:
            await store.close()

    patterns = asyncio.run(rows())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in patterns:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"exported {len(patterns)} de-identified scam examples to {out}")
    return 0


def cmd_purge(settings: Settings, args: argparse.Namespace) -> int:
    async def run() -> tuple[int, int]:
        if not settings.db_path.is_file():  # never create an empty database (its vault would match no files)
            raise ValueError(f"no evidence database at {settings.db_path} (check KUDIGUARD_DB_PATH)")
        store = await SQLiteStore.open(settings.db_path, hasher=_hasher(settings))
        try:
            vault = VoiceVault(settings.voice_dir, store, Sealer(settings.voice_secret()))
            voice = await vault.purge((utcnow() - timedelta(days=settings.voice_retention_days)).date())
            return await store.purge_before(utcnow() - timedelta(days=args.days)), voice
        finally:
            await store.close()

    rows, voice = asyncio.run(run())
    print(
        f"removed {rows} rows older than {args.days} days; "
        f"{voice} donated voice samples older than {settings.voice_retention_days} days"
    )
    return 0


def redact_query_strings(record: logging.LogRecord) -> bool:
    """uvicorn access-log filter: drop query strings (Meta's handshake sends hub.verify_token in the URL)."""
    if isinstance(record.args, tuple) and len(record.args) >= 3 and isinstance(record.args[2], str):
        record.args = (*record.args[:2], record.args[2].split("?", 1)[0], *record.args[3:])
    return True


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import logging

    import uvicorn

    from kudiguard.api.app import create_app

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("uvicorn.access").addFilter(redact_query_strings)

    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, log_level=args.log_level, proxy_headers=False
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_env_file(ROOT / ".env")  # WHATSAPP_* and other non-prefixed settings (real environment wins)
    parser = argparse.ArgumentParser(prog="kudiguard", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the API, web page and WhatsApp webhook")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    stats = commands.add_parser("stats", help="validation statistics from the evidence log")
    stats.add_argument("--json", action="store_true")
    stats.add_argument("--exclude-users", help="file with team testers' numbers (one per line) to leave out")
    stats.add_argument(
        "--channels", help=f"comma-separated channels (default: {','.join(sorted(USER_CHANNELS))})"
    )
    stats.set_defaults(func=cmd_stats)

    export = commands.add_parser("export", help="privacy-safe JSONL export of all interactions")
    export.add_argument("--out", required=True)
    export.set_defaults(func=cmd_export)

    patterns = commands.add_parser("patterns", help="export the de-identified scam pattern library (JSONL)")
    patterns.add_argument("--out", required=True)
    patterns.set_defaults(func=cmd_patterns)

    purge = commands.add_parser("purge", help="delete evidence older than the retention period")
    purge.add_argument("--days", type=int, default=None)
    purge.set_defaults(func=cmd_purge)

    voice_cli.add_commands(commands)

    args = parser.parse_args(argv)
    settings = Settings()
    if getattr(args, "days", 0) is None:
        args.days = settings.retention_days
    if getattr(args, "days", 1) < 1:
        print("error: --days must be at least 1 (0 would delete everything)", file=sys.stderr)
        return 2
    try:
        result: int = args.func(settings, args)
    except ValueError as exc:  # configuration errors (e.g. missing KUDIGUARD_HASH_SECRET)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":  # pragma: no cover - `python -m kudiguard.cli`; main() is tested directly
    raise SystemExit(main())
