"""Command line: run the server and produce the validation evidence.

    kudiguard serve [--host 127.0.0.1] [--port 8000]
    kudiguard stats [--json] [--exclude-users FILE] [--channels whatsapp,web]
    kudiguard export --out evidence.jsonl
    kudiguard purge [--days 90]          (also donated voice notes, DELETE markers and scam examples past their periods)
    kudiguard review export [--out-dir var/review] [--n 50] [--languages ha,yo,ig] [--seed 0]
    kudiguard review score graded.csv [more.csv ...] [--markdown]
    kudiguard review check sheet.csv [more.csv ...]
    kudiguard voice export|wer|stats|access|erase|remove|preconsent|wipe   (kudiguard.voice_cli)

Exports contain no phone numbers or provider ids (pseudonymous user hashes, redacted text).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import httpx

from kudiguard import review, voice_cli
from kudiguard.config import ROOT, Settings
from kudiguard.donation import Sealer, VoiceVault
from kudiguard.envfile import load_env_file
from kudiguard.privacy import UserHasher
from kudiguard.reporting import USER_CHANNELS, build_report, format_report
from kudiguard.storage.base import DELETION_MARKER_DAYS, utcnow
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


def _excluded(settings: Settings, path: str | None) -> frozenset[str]:
    """Team testers' numbers / ids, one per line → user hashes (hashed here, never stored)."""
    if not path:
        return frozenset()
    hasher = _hasher(settings)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return frozenset(hasher.user_id(line.strip()) for line in lines if line.strip())


def cmd_stats(settings: Settings, args: argparse.Namespace) -> int:
    rows = asyncio.run(_rows(settings))
    exclude = _excluded(settings, args.exclude_users)
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
    from kudiguard.api.services import pattern_keep_from

    async def run() -> tuple[int, int, int, int]:
        if not settings.db_path.is_file():  # never create an empty database (its vault would match no files)
            raise ValueError(f"no evidence database at {settings.db_path} (check KUDIGUARD_DB_PATH)")
        store = await SQLiteStore.open(settings.db_path, hasher=_hasher(settings))
        try:
            vault = VoiceVault(settings.voice_dir, store, Sealer(settings.voice_secret()))
            voice = await vault.purge((utcnow() - timedelta(days=settings.voice_retention_days)).date())
            markers = await store.purge_deletion_markers(utcnow() - timedelta(days=DELETION_MARKER_DAYS))
            patterns = await store.purge_patterns_before(
                pattern_keep_from(utcnow(), settings.pattern_retention_days, interval_s=0)
            )
            return await store.purge_before(utcnow() - timedelta(days=args.days)), voice, markers, patterns
        finally:
            await store.close()

    rows, voice, markers, patterns = asyncio.run(run())
    print(
        f"removed {rows} rows older than {args.days} days; "
        f"{voice} donated voice samples older than {settings.voice_retention_days} days; "
        f"{markers} DELETE markers older than {DELETION_MARKER_DAYS} days; "
        f"{patterns} scam examples older than {settings.pattern_retention_days} days"
    )
    return 0


IGNORED_DIRS = (ROOT / "var", ROOT / "exports")  # in .gitignore: review sheets must never be committed


def _committable(path: Path) -> bool:
    path = path.resolve()
    return path.is_relative_to(ROOT) and not any(path.is_relative_to(d) for d in IGNORED_DIRS)


def _review_dir(settings: Settings, out_dir: str | None) -> Path:
    return (
        Path(out_dir) if out_dir else settings.db_path.parent / "review"
    )  # next to the database: var/review


def cmd_review_export(settings: Settings, args: argparse.Namespace) -> int:
    from kudiguard.api.services import build_knowledge

    if args.n < 1:
        raise ValueError("--n must be at least 1")
    languages = tuple(dict.fromkeys(code.strip() for code in args.languages.split(",") if code.strip()))
    if not languages:
        raise ValueError("--languages needs at least one language code, e.g. ha,yo,ig")
    rows = review.eligible(
        asyncio.run(_rows(settings)),
        languages=languages,
        exclude_users=_excluded(settings, args.exclude_users),
        since=date.fromisoformat(args.since) if args.since else None,
    )
    sample = review.stratified_sample(rows, args.n, seed=args.seed, languages=languages)

    async def sheet() -> tuple[list[dict[str, str]], bool]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            knowledge, embedder = await build_knowledge(settings, client)
            rows = await review.review_rows(sample, hasher=_hasher(settings), knowledge=knowledge)
            return rows, embedder is not None

    sheet_rows, semantic = asyncio.run(sheet())
    out_dir = _review_dir(settings, args.out_dir)
    try:
        paths = review.write_sheets(out_dir, sheet_rows, stem=f"review-{utcnow():%Y%m%d}")
    except OSError as exc:
        raise ValueError(f"cannot write review sheets to {out_dir} ({exc.strerror})") from exc
    print(f"sampled {len(sample)} of {len(rows)} eligible replies")
    for language, path in paths.items():
        print(f"  {language}: {sum(r['language'] == language for r in sheet_rows)} replies -> {path}")
    search = "semantic + keyword search" if semantic else "keyword search only (no embedding model)"
    print(f"facts re-derived with {search}")
    if len(sample) < args.n:
        print(f"warning: only {len(sample)} eligible replies (asked for {args.n})", file=sys.stderr)
    if _committable(out_dir):
        print(
            f"warning: {out_dir} is inside the repository and not ignored; keep review sheets under var/",
            file=sys.stderr,
        )
    print(
        "Send each reviewer only their language's file; delete every copy after scoring (docs/NATIVE_REVIEW.md)."
    )
    return 0


def cmd_review_score(settings: Settings, args: argparse.Namespace) -> int:
    card = review.score([review.read_sheet(Path(path)) for path in args.files])
    print(review.markdown_scorecard(card) if args.markdown else review.format_scorecard(card))
    return 0


def cmd_review_check(settings: Settings, args: argparse.Namespace) -> int:
    """Replies erased (DELETE) or expired since the export must be removed from every copy of the sheets."""

    async def ids() -> frozenset[int]:
        if not settings.db_path.is_file():
            raise ValueError(f"no evidence database at {settings.db_path} (check KUDIGUARD_DB_PATH)")
        store = await SQLiteStore.open(settings.db_path, hasher=_hasher(settings))
        try:
            return await store.interaction_ids()
        finally:
            await store.close()

    current = review.known_review_ids(asyncio.run(ids()), _hasher(settings))
    in_sheets = frozenset().union(*(review.read_review_ids(Path(path)) for path in args.files))
    stale = sorted(in_sheets - current)
    if not stale:
        print(f"all {len(in_sheets)} review ids are still in the database")
        return 0
    print("no longer in the database (erased by the user or expired): delete these rows from every copy")
    print("\n".join(stale))
    return 1


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


def utf8_output() -> None:
    """Names and messages in Hausa/Yoruba/Igbo must not crash a redirected or piped Windows console (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    utf8_output()
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

    reviews = commands.add_parser(
        "review", help="native-speaker review of real replies (docs/NATIVE_REVIEW.md)"
    )
    review_commands = reviews.add_subparsers(dest="review_command", required=True)
    sample = review_commands.add_parser("export", help="CSV sample of model-written replies for reviewers")
    sample.add_argument("--out-dir", help="folder for the sheets (default: review/ next to the database)")
    sample.add_argument("--n", type=int, default=review.DEFAULT_SAMPLE)
    sample.add_argument("--languages", default=",".join(review.DEFAULT_LANGUAGES))
    sample.add_argument("--exclude-users", help="file with team testers' numbers (one per line) to leave out")
    sample.add_argument("--seed", type=int, default=0, help="same database + seed = same sample")
    sample.add_argument("--since", help="only replies from this date on (YYYY-MM-DD)")
    sample.set_defaults(func=cmd_review_export)
    graded = review_commands.add_parser("score", help="validate graded sheets and summarise the scores")
    graded.add_argument("files", nargs="+")
    graded.add_argument("--markdown", action="store_true", help="table for docs/VALIDATION.md")
    graded.set_defaults(func=cmd_review_score)
    check = review_commands.add_parser("check", help="list sheet rows erased or expired since the export")
    check.add_argument("files", nargs="+")
    check.set_defaults(func=cmd_review_check)
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
