"""Command line: run the server and produce the validation evidence.

    kudiguard serve [--host 127.0.0.1] [--port 8000]
    kudiguard stats [--json] [--exclude-users FILE] [--channels whatsapp,web]
    kudiguard export --out evidence.jsonl
    kudiguard purge [--days 90]
    kudiguard review export [--out var/review/review.csv] [--n 50] [--languages ha,yo,ig] [--seed 0]
    kudiguard review score graded.csv [more.csv ...] [--markdown]

Exports contain no phone numbers or provider ids (pseudonymous user hashes, redacted text).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import httpx

from kudiguard import review
from kudiguard.config import ROOT, Settings
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
    async def run() -> int:
        store = await SQLiteStore.open(settings.db_path, hasher=_hasher(settings))
        try:
            return await store.purge_before(utcnow() - timedelta(days=args.days))
        finally:
            await store.close()

    print(f"removed {asyncio.run(run())} rows older than {args.days} days")
    return 0


REVIEW_DIR = ROOT / "var" / "review"
IGNORED_DIRS = (ROOT / "var", ROOT / "exports")  # in .gitignore: review sheets must never be committed


def _committable(path: Path) -> bool:
    path = path.resolve()
    return path.is_relative_to(ROOT) and not any(path.is_relative_to(d) for d in IGNORED_DIRS)


def cmd_review_export(settings: Settings, args: argparse.Namespace) -> int:
    from kudiguard.api.services import build_knowledge

    languages = tuple(code.strip() for code in args.languages.split(",") if code.strip())
    rows = review.eligible(
        asyncio.run(_rows(settings)),
        languages=languages,
        exclude_users=_excluded(settings, args.exclude_users),
        since=date.fromisoformat(args.since) if args.since else None,
    )
    sample = review.stratified_sample(rows, args.n, seed=args.seed, languages=languages)

    async def sheet() -> list[dict[str, str]]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            knowledge = await build_knowledge(settings, client)
            return await review.review_rows(sample, hasher=_hasher(settings), knowledge=knowledge)

    out = Path(args.out) if args.out else REVIEW_DIR / f"review-{utcnow():%Y%m%d}.csv"
    review.write_sheet(out, asyncio.run(sheet()))
    print(f"wrote {len(sample)} of {len(rows)} eligible replies to {out}")
    if len(sample) < args.n:
        print(f"warning: only {len(sample)} eligible replies (asked for {args.n})", file=sys.stderr)
    if _committable(out):
        print(
            f"warning: {out} is inside the repository and not ignored; keep review sheets under var/",
            file=sys.stderr,
        )
    print("Share only with the named reviewers; delete every copy after scoring (docs/NATIVE_REVIEW.md).")
    return 0


def cmd_review_score(settings: Settings, args: argparse.Namespace) -> int:
    card = review.score([review.read_sheet(Path(path)) for path in args.files])
    print(review.markdown_scorecard(card) if args.markdown else review.format_scorecard(card))
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

    reviews = commands.add_parser(
        "review", help="native-speaker review of real replies (docs/NATIVE_REVIEW.md)"
    )
    review_commands = reviews.add_subparsers(dest="review_command", required=True)
    sample = review_commands.add_parser("export", help="CSV sample of model-written replies for reviewers")
    sample.add_argument("--out", help=f"CSV path (default: {REVIEW_DIR}/review-<date>.csv, not committed)")
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
