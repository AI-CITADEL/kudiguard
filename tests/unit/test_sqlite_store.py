import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kudiguard.domain import Interaction, MessageKind
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage import sqlite as sqlite_module
from kudiguard.storage.base import StoreClosedError, UserState
from kudiguard.storage.sqlite import MIGRATIONS, SQLiteStore, _dt, _transaction, _ts

HASHER = UserHasher(b"s" * MIN_SECRET_BYTES)


def turn(n: int, user: str = "u1") -> Interaction:
    return Interaction(
        channel="whatsapp", message_id=f"m{n}", user_hash=user, language="en", kind=MessageKind.TEXT
    )


async def test_data_survives_restart_and_migrations_run_once(tmp_path: Path) -> None:
    path = tmp_path / "data" / "kg.db"  # parent directory is created
    store = await SQLiteStore.open(path, hasher=HASHER)
    await store.save_user(UserState("u1", language="ha"))
    interaction_id = await store.add_interaction(turn(1))
    await store.close()

    reopened = await SQLiteStore.open(path, hasher=HASHER)
    user = await reopened.get_user("u1")
    assert user is not None and user.language == "ha"
    assert await reopened.get_interaction(interaction_id) is not None
    await reopened.close()

    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    db.close()


async def test_existing_database_migrates_to_voice_donation(tmp_path: Path) -> None:
    """A production database at schema 2 (before donation) keeps its users and gains the new columns/table."""
    path = tmp_path / "kg.db"
    db = sqlite3.connect(path)
    for number, script in enumerate(MIGRATIONS[:2], start=1):
        db.executescript(f"BEGIN; {script}; PRAGMA user_version={number}; COMMIT;")
    db.execute(
        "INSERT INTO users VALUES ('u1', 'ig', ?, '2026-10-08', 0, 7, ?, ?)",
        (_ts(datetime(2026, 10, 9, tzinfo=UTC)),) * 3,
    )
    db.commit()
    db.close()

    store = await SQLiteStore.open(path, hasher=HASHER)
    user = await store.get_user("u1")
    assert user is not None and user.language == "ig" and user.last_interaction_id == 7
    assert (user.donation_version, user.donation_at, user.donation_offered_at) == (None, None, None)
    user.donation_version, user.donation_at = "2026-10-10", datetime(2026, 10, 10, tzinfo=UTC)
    await store.save_user(user)
    reloaded = await store.get_user("u1")
    assert reloaded is not None and reloaded.donation_version == "2026-10-10"
    assert await store.voice_samples() == []
    await store.close()
    check = sqlite3.connect(path)
    assert check.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS) == 3
    check.close()


async def test_concurrent_writes_all_land(tmp_path: Path) -> None:
    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    ids = await asyncio.gather(*(store.add_interaction(turn(n)) for n in range(200)))
    assert len(set(ids)) == 200
    assert len(await store.export_interactions()) == 200
    await store.close()


def test_timestamps_round_trip_and_sort_as_text() -> None:
    whole = datetime(2026, 10, 7, 10, 0, 0, tzinfo=UTC)
    fraction = whole + timedelta(microseconds=123)
    assert _dt(_ts(whole)) == whole and _dt(_ts(fraction)) == fraction
    assert _ts(whole) < _ts(fraction)  # fixed width: text order == time order
    lagos = whole.astimezone(datetime.now().astimezone().tzinfo)
    assert _ts(lagos) == _ts(whole)  # always stored as UTC
    assert _dt(None) is None


async def test_user_state_is_upserted_not_duplicated(tmp_path: Path) -> None:
    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    for language in ("ha", "yo", "en"):
        await store.save_user(UserState("u1", language=language))
    db = sqlite3.connect(tmp_path / "kg.db")
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    db.close()
    user = await store.get_user("u1")
    assert user is not None and user.language == "en"
    await store.close()


def test_transaction_rolls_back_on_failure(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "t.db", isolation_level=None)
    db.execute("CREATE TABLE t (x INTEGER)")
    try:
        with _transaction(db):
            db.execute("INSERT INTO t VALUES (1)")
            raise RuntimeError("crash mid-erasure")
    except RuntimeError:
        pass
    assert db.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    db.close()


async def test_failed_migration_leaves_version_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "kg.db"
    await (await SQLiteStore.open(path, hasher=HASHER)).close()
    monkeypatch.setattr(sqlite_module, "MIGRATIONS", (*MIGRATIONS, "CREATE TABLE ok (x); THIS IS NOT SQL;"))
    with pytest.raises(sqlite3.Error):
        await SQLiteStore.open(path, hasher=HASHER)
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert db.execute("SELECT name FROM sqlite_master WHERE name = 'ok'").fetchone() is None
    db.close()


async def test_use_after_close_is_a_clear_error(tmp_path: Path) -> None:
    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    await store.close()
    await store.close()  # idempotent
    with pytest.raises(StoreClosedError):
        await store.get_user("u1")


async def test_corrupt_required_timestamp_raises(tmp_path: Path) -> None:
    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    await store.save_user(UserState("u1"))
    db = sqlite3.connect(tmp_path / "kg.db")
    db.execute("UPDATE users SET created_at = '' ")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="corrupt"):
        await store.get_user("u1")
    await store.close()


async def test_erased_text_is_not_left_readable_on_disk(tmp_path: Path) -> None:
    from kudiguard.domain import Interaction

    store = await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    secret_words = "zebra-umbrella-pancake"
    await store.add_interaction(
        Interaction(
            channel="w",
            message_id="m",
            user_hash="u1",
            language="en",
            kind=MessageKind.TEXT,
            transcript=secret_words,
        )
    )
    await store.delete_user("u1")
    await store.close()

    def leftover() -> bool:
        return any(secret_words.encode() in f.read_bytes() for f in tmp_path.glob("kg.db*"))

    assert not await asyncio.to_thread(leftover)


def test_failure_after_the_transaction_already_ended_is_not_masked(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "t.db", isolation_level=None)
    db.execute("CREATE TABLE t (x INTEGER)")
    with pytest.raises(RuntimeError, match="original"), _transaction(db):
        db.execute("INSERT INTO t VALUES (1)")
        db.execute("ROLLBACK")  # e.g. SQLite rolled back on its own (disk full, I/O error)
        raise RuntimeError("original")  # must surface, not "cannot rollback - no transaction is active"
    assert not db.in_transaction and db.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    db.close()


async def test_migration_failing_outside_a_transaction_closes_and_raises_the_real_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "kg.db"
    await (await SQLiteStore.open(path, hasher=HASHER)).close()
    monkeypatch.setattr(sqlite_module, "MIGRATIONS", (*MIGRATIONS, "COMMIT; THIS IS NOT SQL"))
    with pytest.raises(sqlite3.OperationalError, match="syntax error"):
        await SQLiteStore.open(path, hasher=HASHER)
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    db.close()
    path.unlink()  # the failed opener closed its connection (an open handle would block this on Windows)
