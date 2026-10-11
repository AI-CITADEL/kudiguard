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
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeVar

from kudiguard.domain import Flag, Interaction, MessageKind, Outcome, Verdict
from kudiguard.patterns import ScamPattern, deidentify, week_of
from kudiguard.privacy import UserHasher
from kudiguard.storage.base import StoreClosedError, UserState, VoiceSample, aware, utcnow
from kudiguard.storage.export import export_row, protect
from kudiguard.text import REDACTION_VERSION, redact

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
    # 2: de-identified scam pattern library (no user link; random ids, WITHOUT ROWI***REMOVED*** no insertion-order leak)
    """
    CREATE TABLE scam_patterns (
        id TEXT PRIMARY KEY,
        week TEXT NOT NULL,
        language TEXT NOT NULL,
        channel TEXT NOT NULL,
        verdict TEXT NOT NULL,
        flags TEXT NOT NULL DEFAULT '[]',
        text TEXT NOT NULL,
        english TEXT
    ) WITHOUT ROWID;
    CREATE INDEX ix_patterns_week ON scam_patterns (week, text);
    """,
    # 3: opt-in voice-note donation: consent record per user, one row per donated sample (audio in encrypted files)
    """
    ALTER TABLE users ADD COLUMN donation_version TEXT;
    ALTER TABLE users ADD COLUMN donation_at TEXT;
    ALTER TABLE users ADD COLUMN donation_source TEXT;
    ALTER TABLE users ADD COLUMN donation_offered_at TEXT;
    ALTER TABLE users ADD COLUMN donation_offered_version TEXT;
    ALTER TABLE users ADD COLUMN donation_withdrawn_at TEXT;
    ALTER TABLE users ADD COLUMN welcome_version TEXT;
    ALTER TABLE users ADD COLUMN welcome_at TEXT;
    CREATE TABLE deletion_markers (marker TEXT PRIMARY KEY, deleted_at TEXT NOT NULL) WITHOUT ROWID;
    CREATE INDEX ix_deletion_markers_time ON deletion_markers (deleted_at);
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
    INSERT INTO meta VALUES ('voice_vault_id', lower(hex(randomblob(16))));
    CREATE TABLE voice_samples (
        id TEXT PRIMARY KEY,
        user_hash TEXT NOT NULL,
        week TEXT NOT NULL,
        language TEXT NOT NULL,
        channel TEXT NOT NULL,
        consent_version TEXT NOT NULL,
        hypothesis TEXT NOT NULL,
        duration_s REAL
    ) WITHOUT ROWID;
    CREATE INDEX ix_voice_user ON voice_samples (user_hash);
    CREATE INDEX ix_voice_week ON voice_samples (week);
    """,
)
_USER_COLUMNS = (
    "user_hash, language, consented_at, consent_version, stopped, last_interaction_id, created_at, last_seen_at, "
    "donation_version, donation_at, donation_source, donation_offered_at, donation_offered_version, "
    "donation_withdrawn_at, welcome_version, welcome_at"
)
_USER_SLOTS = ", ".join("?" * len(_USER_COLUMNS.split(",")))


def _user_values(state: UserState) -> tuple[object, ...]:
    """A users row in ``_USER_COLUMNS`` order."""
    return (
        state.user_hash, state.language, _ts(state.consented_at) if state.consented_at else None,
        state.consent_version, int(state.stopped), state.last_interaction_id, _ts(state.created_at),
        _ts(state.last_seen_at), state.donation_version, _ts(state.donation_at) if state.donation_at else None,
        state.donation_source, _ts(state.donation_offered_at) if state.donation_offered_at else None,
        state.donation_offered_version, _ts(state.donation_withdrawn_at) if state.donation_withdrawn_at else None,
        state.welcome_version, _ts(state.welcome_at) if state.welcome_at else None,
    )  # fmt: skip


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
            donation_version=row["donation_version"],
            donation_at=_dt(row["donation_at"]),
            donation_source=row["donation_source"],
            donation_offered_at=_dt(row["donation_offered_at"]),
            donation_offered_version=row["donation_offered_version"],
            donation_withdrawn_at=_dt(row["donation_withdrawn_at"]),
            welcome_version=row["welcome_version"],
            welcome_at=_dt(row["welcome_at"]),
        )

    async def save_user(self, state: UserState) -> None:
        values = _user_values(state)
        await self._run(
            lambda db: db.execute(
                f"""INSERT INTO users ({_USER_COLUMNS}) VALUES ({_USER_SLOTS})
                   ON CONFLICT (user_hash) DO UPDATE SET language = excluded.language,
                     consented_at = excluded.consented_at, consent_version = excluded.consent_version,
                     stopped = excluded.stopped, last_interaction_id = excluded.last_interaction_id,
                     last_seen_at = excluded.last_seen_at, donation_version = excluded.donation_version,
                     donation_at = excluded.donation_at, donation_source = excluded.donation_source,
                     donation_offered_at = excluded.donation_offered_at,
                     donation_offered_version = excluded.donation_offered_version,
                     donation_withdrawn_at = excluded.donation_withdrawn_at,
                     welcome_version = excluded.welcome_version, welcome_at = excluded.welcome_at""",
                values,
            )
        )

    async def record_form_consent(self, state: UserState, expected: UserState | None) -> bool:
        if expected is None:
            values = _user_values(state)
            cursor = await self._run(
                lambda db: db.execute(
                    f"INSERT INTO users ({_USER_COLUMNS}) VALUES ({_USER_SLOTS}) ON CONFLICT (user_hash) DO NOTHING",
                    values,
                )
            )
            return int(cursor.rowcount) == 1
        params = (
            state.donation_version, _ts(state.donation_at) if state.donation_at else None, state.donation_source,
            _ts(state.donation_offered_at) if state.donation_offered_at else None, state.donation_offered_version,
            state.user_hash, _ts(expected.last_seen_at),
        )  # fmt: skip
        cursor = await self._run(
            lambda db: db.execute(
                """UPDATE users SET donation_version = ?, donation_at = ?, donation_source = ?,
                     donation_offered_at = ?, donation_offered_version = ?
                   WHERE user_hash = ? AND last_seen_at = ? AND stopped = 0 AND donation_version IS NULL
                     AND donation_withdrawn_at IS NULL AND donation_offered_version IS NULL""",
                params,
            )
        )
        return int(cursor.rowcount) == 1

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

    async def count_active_users_since(self, since: datetime, channels: Collection[str] | None = None) -> int:
        query = "SELECT COUNT(DISTINCT user_hash) FROM interactions INDEXED BY ix_interactions_time WHERE created_at >= ?"
        params: list[str] = [_ts(since)]
        if channels is not None:
            wanted = sorted(set(channels))
            if not wanted:
                return 0
            query += f" AND channel IN ({', '.join('?' * len(wanted))})"
            params += wanted
        row = await self._run(lambda db: db.execute(query, params).fetchone())
        return int(row[0])

    # --- erasure and retention ------------------------------------------------------------------------

    async def delete_user(self, user_hash: str) -> int:
        marker = self._hasher.deletion_marker(user_hash)

        def work(db: sqlite3.Connection) -> int:
            with _transaction(db):
                scams = db.execute(
                    "SELECT created_at, transcript FROM interactions WHERE user_hash = ? AND transcript IS NOT NULL "
                    "AND verdict IN ('likely_scam', 'caution')",
                    (user_hash,),
                ).fetchall()
                patterns = 0
                for row in scams:  # examples taken from this user's stored messages (still matchable to them)
                    week = week_of(_dt_required(row["created_at"]).date()).isoformat()
                    patterns += db.execute(
                        "DELETE FROM scam_patterns WHERE week = ? AND text = ?",
                        (week, deidentify(row["transcript"])),
                    ).rowcount
                feedback = db.execute(
                    "DELETE FROM feedback WHERE interaction_id IN (SELECT id FROM interactions WHERE user_hash = ?)",
                    (user_hash,),
                ).rowcount
                turns = db.execute("DELETE FROM interactions WHERE user_hash = ?", (user_hash,)).rowcount
                users = db.execute("DELETE FROM users WHERE user_hash = ?", (user_hash,)).rowcount
                voice = db.execute("DELETE FROM voice_samples WHERE user_hash = ?", (user_hash,)).rowcount
                db.execute(
                    "INSERT INTO deletion_markers VALUES (?, ?) ON CONFLICT (marker) DO UPDATE SET deleted_at = "
                    "excluded.deleted_at",
                    (marker, _ts(utcnow())),
                )
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # erased pages must not linger in the WAL file
            return patterns + feedback + turns + users + voice

        return await self._run(work)

    async def delete_voice_samples(self, user_hash: str) -> list[str]:
        def work(db: sqlite3.Connection) -> list[str]:
            with _transaction(db):
                ids = [
                    r[0] for r in db.execute("SELECT id FROM voice_samples WHERE user_hash = ?", (user_hash,))
                ]
                db.execute("DELETE FROM voice_samples WHERE user_hash = ?", (user_hash,))
            if ids:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return sorted(ids)

        return await self._run(work)

    async def delete_voice_sample(self, sample_id: str) -> bool:
        def work(db: sqlite3.Connection) -> bool:
            removed = db.execute("DELETE FROM voice_samples WHERE id = ?", (sample_id,)).rowcount == 1
            if removed:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return removed

        return await self._run(work)

    async def purge_voice_samples(self, cutoff: date) -> list[str]:
        stamp = cutoff.isoformat()

        def work(db: sqlite3.Connection) -> list[str]:
            with _transaction(db):
                ids = [r[0] for r in db.execute("SELECT id FROM voice_samples WHERE week <= ?", (stamp,))]
                db.execute("DELETE FROM voice_samples WHERE week <= ?", (stamp,))
            return sorted(ids)

        return await self._run(work)

    async def deleted_at(self, user_hash: str) -> datetime | None:
        marker = self._hasher.deletion_marker(user_hash)
        row = await self._run(
            lambda db: db.execute(
                "SELECT deleted_at FROM deletion_markers WHERE marker = ?", (marker,)
            ).fetchone()
        )
        return _dt(row[0]) if row else None

    async def purge_deletion_markers(self, cutoff: datetime) -> int:
        cursor = await self._run(
            lambda db: db.execute("DELETE FROM deletion_markers WHERE deleted_at < ?", (_ts(cutoff),))
        )
        return int(cursor.rowcount)

    async def purge_before(self, cutoff: datetime) -> int:
        stamp = _ts(cutoff)

        def work(db: sqlite3.Connection) -> int:
            with _transaction(db):
                feedback = db.execute(
                    "DELETE FROM feedback WHERE interaction_id IN (SELECT id FROM interactions WHERE created_at < ?)",
                    (stamp,),
                ).rowcount
                turns = db.execute("DELETE FROM interactions WHERE created_at < ?", (stamp,)).rowcount
                users = db.execute(  # a donor's row is their consent record: kept while samples exist
                    "DELETE FROM users WHERE last_seen_at < ? AND user_hash NOT IN (SELECT user_hash FROM voice_samples)",
                    (stamp,),
                ).rowcount
                claims = db.execute("DELETE FROM processed_messages WHERE claimed_at < ?", (stamp,)).rowcount
            return feedback + turns + users + claims

        return await self._run(work)

    # --- reporting ------------------------------------------------------------------------------------

    async def add_pattern(self, pattern: ScamPattern) -> None:
        values = (
            pattern.id, pattern.week.isoformat(), pattern.language, pattern.channel, pattern.verdict.value,
            json.dumps([f.value for f in pattern.flags]), pattern.text, pattern.english,
        )  # fmt: skip
        await self._run(
            lambda db: db.execute("INSERT INTO scam_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)
        )

    async def purge_patterns_before(self, cutoff: date) -> int:
        cursor = await self._run(
            lambda db: db.execute("DELETE FROM scam_patterns WHERE week < ?", (week_of(cutoff).isoformat(),))
        )
        return int(cursor.rowcount)

    async def export_patterns(self) -> list[dict[str, Any]]:
        rows = await self._run(
            lambda db: db.execute("SELECT * FROM scam_patterns ORDER BY week, id").fetchall()
        )
        return [
            ScamPattern(
                language=r["language"], channel=r["channel"], verdict=Verdict(r["verdict"]),
                flags=tuple(Flag(f) for f in json.loads(r["flags"])), text=r["text"], english=r["english"],
                week=date.fromisoformat(r["week"]), id=r["id"],
            ).as_row()
            for r in rows
        ]  # fmt: skip

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

    async def interaction_ids(self) -> frozenset[int]:
        rows = await self._run(lambda db: db.execute("SELECT id FROM interactions").fetchall())
        return frozenset(int(r["id"]) for r in rows)

    async def add_voice_sample(self, sample: VoiceSample) -> None:
        values = (
            sample.id, sample.user_hash, week_of(sample.week).isoformat(), sample.language, sample.channel,
            sample.consent_version, redact(sample.hypothesis), sample.duration_s,
        )  # fmt: skip

        def work(db: sqlite3.Connection) -> None:
            try:
                db.execute("INSERT INTO voice_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)
            except sqlite3.IntegrityError as exc:  # same contract as the memory store
                raise ValueError("duplicate voice sample id") from exc

        await self._run(work)

    async def voice_vault_id(self) -> str:
        row = await self._run(
            lambda db: db.execute("SELECT value FROM meta WHERE key = 'voice_vault_id'").fetchone()
        )
        return str(row[0])

    async def voice_samples(self, user_hash: str | None = None) -> list[VoiceSample]:
        query = "SELECT * FROM voice_samples"
        params: tuple[str, ...] = ()
        if user_hash is not None:
            query, params = query + " WHERE user_hash = ?", (user_hash,)
        rows = await self._run(lambda db: db.execute(query + " ORDER BY week, id", params).fetchall())
        return [
            VoiceSample(
                id=r["id"], user_hash=r["user_hash"], week=date.fromisoformat(r["week"]), language=r["language"],
                channel=r["channel"], consent_version=r["consent_version"], hypothesis=r["hypothesis"],
                duration_s=r["duration_s"],
            )
            for r in rows
        ]  # fmt: skip

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
    try:
        _reredact(db)
    except sqlite3.Error:
        db.close()
        raise
    return db


def _reredact(db: sqlite3.Connection) -> None:
    """Once per ``REDACTION_VERSION``: redact every stored free text again, and de-identify every stored scam example
    again, with the current patterns (rows written by an older release may hold numbers it missed). Pattern texts are
    re-derived the same way as a new user's (``deidentify``), so DELETE still finds the examples taken from a user's
    messages. One transaction; secure_delete overwrites the old values and the WAL is truncated afterwards."""
    row = db.execute("SELECT value FROM meta WHERE key = 'redaction_version'").fetchone()
    if row is not None and int(row[0]) >= REDACTION_VERSION:
        return
    with _transaction(db):
        for r in db.execute("SELECT id, transcript, english, reply, error FROM interactions").fetchall():
            texts = [r["transcript"], r["english"], r["reply"], r["error"]]
            if (new := [redact(t) if t else t for t in texts]) != texts:
                db.execute(
                    "UPDATE interactions SET transcript = ?, english = ?, reply = ?, error = ? WHERE id = ?",
                    (*new, r["id"]),
                )
        for r in db.execute("SELECT id, hypothesis FROM voice_samples").fetchall():
            if (hypothesis := redact(r["hypothesis"])) != r["hypothesis"]:
                db.execute("UPDATE voice_samples SET hypothesis = ? WHERE id = ?", (hypothesis, r["id"]))
        for r in db.execute("SELECT id, text, english FROM scam_patterns").fetchall():
            new = [deidentify(r["text"]), deidentify(r["english"]) if r["english"] else r["english"]]
            if new != [r["text"], r["english"]]:
                db.execute("UPDATE scam_patterns SET text = ?, english = ? WHERE id = ?", (*new, r["id"]))
        db.execute(
            "INSERT INTO meta VALUES ('redaction_version', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (str(REDACTION_VERSION),),
        )
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # the old values must not linger in the WAL file


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
