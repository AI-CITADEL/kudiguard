"""SQLite Store: a single file, WAL mode, versioned migrations; every query runs off the event loop.

One connection guarded by a lock (SQLite serialises writers anyway; the bot's write rate is a few per second at most).
Multi-statement operations (erasure, retention, migrations) run in explicit ``BEGIN IMMEDIATE`` transactions:
Python's legacy sqlite3 mode does not open one for ``with conn:`` when ``isolation_level=None``.
Timestamps are stored as fixed-width UTC strings so text comparison equals time comparison.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from kudiguard.domain import Flag, Interaction, MessageKind, Outcome, Verdict
from kudiguard.privacy import UserHasher
from kudiguard.storage.base import StoreClosedError, UserState, aware, utcnow
from kudiguard.storage.export import export_row, protect
from kudiguard.text import redact

T = TypeVar("T")
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

MIGRATIONS: Sequence[str] = (
    # 1: initial schema
    """
    CREATE TABLE users (
        user_hash TEXT PRIMARY KEY,
        language TEXT,
        consented_at TEXT,
        consent_version TEXT,
        stopped INTEGER NOT NULL DEFAULT 0,
        last_interaction_id INTEGER,
        created_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL
    );
    CREATE INDEX ix_users_last_seen ON users (last_seen_at);
    CREATE TABLE processed_messages (
        message_key TEXT PRIMARY KEY,
        claimed_at TEXT NOT NULL
    );
    CREATE INDEX ix_processed_time ON processed_messages (claimed_at);
    CREATE TABLE interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel TEXT NOT NULL,
        message_key TEXT NOT NULL,
        user_hash TEXT NOT NULL,
        language TEXT NOT NULL,
        kind TEXT NOT NULL,
        outcome TEXT NOT NULL,
        intent TEXT,
        transcript TEXT,
        english TEXT,
        verdict TEXT,
        flags TEXT NOT NULL DEFAULT '[]',
        reply TEXT,
        reply_to INTEGER,
        decode_ms INTEGER,
        asr_ms INTEGER,
        llm_ms INTEGER,
        total_ms INTEGER,
        error TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX ix_interactions_user_time ON interactions (user_hash, created_at);
    CREATE INDEX ix_interactions_time ON interactions (created_at);
    CREATE TABLE feedback (
        interaction_id INTEGER PRIMARY KEY REFERENCES interactions (id) ON DELETE CASCADE,
        helpful INTEGER NOT NULL,
        at TEXT NOT NULL
    );
    """,
)


def _ts(value: datetime) -> str:
    return aware(value).strftime(_TIME_FORMAT)


def _dt(value: str | None) -> datetime | None:
    return datetime.strptime(value, _TIME_FORMAT).replace(tzinfo=UTC) if value else None


def _dt_required(value: str | None) -> datetime:
    if not value:
        raise ValueError("corrupt row: missing required timestamp")
    return datetime.strptime(value, _TIME_FORMAT).replace(tzinfo=UTC)


@contextmanager
def _transaction(db: sqlite3.Connection) -> Iterator[None]:
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:  # also covers a failed COMMIT, which would otherwise leave the transaction open
            db.execute("ROLLBACK")
        raise


class SQLiteStore:
    def __init__(self, connection: sqlite3.Connection, hasher: UserHasher) -> None:
        self._db = connection
        self._hasher = hasher
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    async def open(cls, path: Path, *, hasher: UserHasher) -> SQLiteStore:
        return cls(await asyncio.to_thread(_connect, path), hasher)

    async def _run(self, work: Callable[[sqlite3.Connection], T]) -> T:
        def locked() -> T:
            with self._lock:
                if self._closed:
                    raise StoreClosedError("store is closed")
                return work(self._db)

        return await asyncio.to_thread(locked)

    # --- users ----------------------------------------------------------------------------------------

    async def get_user(self, user_hash: str) -> UserState | None:
        row = await self._run(
            lambda db: db.execute("SELECT * FROM users WHERE user_hash = ?", (user_hash,)).fetchone()
        )
        if row is None:
            return None
        return UserState(
            user_hash=row["user_hash"],
            language=row["language"],
            consented_at=_dt(row["consented_at"]),
            consent_version=row["consent_version"],
            stopped=bool(row["stopped"]),
            last_interaction_id=row["last_interaction_id"],
            created_at=_dt_required(row["created_at"]),
            last_seen_at=_dt_required(row["last_seen_at"]),
        )

    async def save_user(self, state: UserState) -> None:
        values = (
            state.user_hash,
            state.language,
            _ts(state.consented_at) if state.consented_at else None,
            state.consent_version,
            int(state.stopped),
            state.last_interaction_id,
            _ts(state.created_at),
            _ts(state.last_seen_at),
        )
        await self._run(
            lambda db: db.execute(
                """INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (user_hash) DO UPDATE SET language = excluded.language,
                     consented_at = excluded.consented_at, consent_version = excluded.consent_version,
                     stopped = excluded.stopped, last_interaction_id = excluded.last_interaction_id,
                     last_seen_at = excluded.last_seen_at""",
                values,
            )
        )

    # --- messages and interactions --------------------------------------------------------------------

    async def claim_message(self, channel: str, message_id: str) -> bool:
        key = self._hasher.message_key(channel, message_id)
        cursor = await self._run(
            lambda db: db.execute(
                "INSERT OR IGNORE INTO processed_messages VALUES (?, ?)", (key, _ts(utcnow()))
            )
        )
        return cursor.rowcount == 1

    async def add_interaction(self, interaction: Interaction) -> int:
        i = protect(interaction, self._hasher)
        values = (
            i.channel, i.message_id, i.user_hash, i.language, i.kind.value, i.outcome.value, i.intent,
            i.transcript, i.english, i.verdict.value if i.verdict else None, json.dumps([f.value for f in i.flags]),
            i.reply, i.reply_to, i.decode_ms, i.asr_ms, i.llm_ms, i.total_ms, i.error, _ts(i.created_at),
        )  # fmt: skip
        cursor = await self._run(
            lambda db: db.execute(
                """INSERT INTO interactions (channel, message_key, user_hash, language, kind, outcome, intent,
                     transcript, english, verdict, flags, reply, reply_to, decode_ms, asr_ms, llm_ms, total_ms,
                     error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    async def get_interaction(self, interaction_id: int) -> Interaction | None:
        row = await self._run(
            lambda db: db.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,)).fetchone()
        )
        return _interaction(row) if row else None

    async def set_outcome(self, interaction_id: int, outcome: Outcome, error: str | None = None) -> None:
        await self._run(
            lambda db: db.execute(
                "UPDATE interactions SET outcome = ?, error = COALESCE(?, error) WHERE id = ?",
                (outcome.value, redact(error) if error else None, interaction_id),
            )
        )

    async def add_feedback(self, user_hash: str, interaction_id: int, helpful: bool) -> None:
        await self._run(
            lambda db: db.execute(
                """INSERT INTO feedback (interaction_id, helpful, at)
                   SELECT id, ?, ? FROM interactions WHERE id = ? AND user_hash = ?
                   ON CONFLICT (interaction_id) DO UPDATE SET helpful = excluded.helpful, at = excluded.at""",
                (int(helpful), _ts(utcnow()), interaction_id, user_hash),
            )
        )

    async def count_interactions_since(self, user_hash: str, since: datetime) -> int:
        row = await self._run(
            lambda db: db.execute(
                "SELECT COUNT(*) FROM interactions WHERE user_hash = ? AND created_at >= ?",
                (user_hash, _ts(since)),
            ).fetchone()
        )
        return int(row[0])

    async def count_active_users_since(self, since: datetime) -> int:
        row = await self._run(
            lambda db: db.execute(
                "SELECT COUNT(DISTINCT user_hash) FROM interactions INDEXED BY ix_interactions_time "
                "WHERE created_at >= ?",
                (_ts(since),),
            ).fetchone()
        )
        return int(row[0])

    # --- erasure and retention ------------------------------------------------------------------------

    async def delete_user(self, user_hash: str) -> int:
        def work(db: sqlite3.Connection) -> int:
            with _transaction(db):
                feedback = db.execute(
                    "DELETE FROM feedback WHERE interaction_id IN (SELECT id FROM interactions WHERE user_hash = ?)",
                    (user_hash,),
                ).rowcount
                turns = db.execute("DELETE FROM interactions WHERE user_hash = ?", (user_hash,)).rowcount
                users = db.execute("DELETE FROM users WHERE user_hash = ?", (user_hash,)).rowcount
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # erased pages must not linger in the WAL file
            return feedback + turns + users

        return await self._run(work)

    async def purge_before(self, cutoff: datetime) -> int:
        stamp = _ts(cutoff)

        def work(db: sqlite3.Connection) -> int:
            with _transaction(db):
                feedback = db.execute(
                    "DELETE FROM feedback WHERE interaction_id IN (SELECT id FROM interactions WHERE created_at < ?)",
                    (stamp,),
                ).rowcount
                turns = db.execute("DELETE FROM interactions WHERE created_at < ?", (stamp,)).rowcount
                users = db.execute("DELETE FROM users WHERE last_seen_at < ?", (stamp,)).rowcount
                claims = db.execute("DELETE FROM processed_messages WHERE claimed_at < ?", (stamp,)).rowcount
            return feedback + turns + users + claims

        return await self._run(work)

    # --- reporting ------------------------------------------------------------------------------------

    async def export_interactions(self) -> list[dict[str, Any]]:
        rows = await self._run(
            lambda db: db.execute(
                """SELECT i.*, f.helpful AS helpful FROM interactions i
                   LEFT JOIN feedback f ON f.interaction_id = i.id ORDER BY i.id"""
            ).fetchall()
        )
        return [
            export_row(_interaction(r), None if r["helpful"] is None else bool(r["helpful"])) for r in rows
        ]

    async def close(self) -> None:
        def work() -> None:
            with self._lock:
                if not self._closed:
                    self._closed = True
                    self._db.close()

        await asyncio.to_thread(work)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA secure_delete=ON")  # erased rows are overwritten, not left readable in free pages
    db.execute("PRAGMA busy_timeout=5000")
    version = db.execute("PRAGMA user_version").fetchone()[0]
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        try:
            db.executescript(f"BEGIN IMMEDIATE; {script}; PRAGMA user_version={number}; COMMIT;")
        except sqlite3.Error:
            if db.in_transaction:
                db.execute("ROLLBACK")
            db.close()
            raise
    return db


def _interaction(row: sqlite3.Row) -> Interaction:
    return Interaction(
        id=row["id"],
        channel=row["channel"],
        message_id=row["message_key"],
        user_hash=row["user_hash"],
        language=row["language"],
        kind=MessageKind(row["kind"]),
        outcome=Outcome(row["outcome"]),
        intent=row["intent"],
        transcript=row["transcript"],
        english=row["english"],
        verdict=Verdict(row["verdict"]) if row["verdict"] else None,
        flags=tuple(Flag(f) for f in json.loads(row["flags"])),
        reply=row["reply"],
        reply_to=row["reply_to"],
        decode_ms=row["decode_ms"],
        asr_ms=row["asr_ms"],
        llm_ms=row["llm_ms"],
        total_ms=row["total_ms"],
        error=row["error"],
        created_at=_dt_required(row["created_at"]),
    )
