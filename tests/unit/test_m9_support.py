"""Settings, rate limiting, validation reporting, service wiring and the CLI."""

import asyncio
import json
from pathlib import Path

import pytest

from kudiguard import cli
from kudiguard.api.ratelimit import RateLimiter
from kudiguard.api.services import Services, build_services, purge_loop
from kudiguard.config import Settings
from kudiguard.domain import Interaction, MessageKind, Outcome, Verdict
from kudiguard.knowledge.file_pack import KnowledgeConfigError
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.reporting import build_report, format_report, percentile
from kudiguard.storage.sqlite import SQLiteStore

ROOT = Path(__file__).resolve().parents[2]
SECRET = "s" * MIN_SECRET_BYTES


# --- settings ----------------------------------------------------------------------------------------------------


def test_settings_parse_csv_and_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUDIGUARD_API_KEYS", f"bank:{'a' * 24}, ngo:{'b' * 30}")
    monkeypatch.setenv("KUDIGUARD_PRELOAD_LANGUAGES", "ha,yo")
    monkeypatch.setenv("KUDIGUARD_PACKS", "")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.api_key_map() == {"a" * 24: "bank", "b" * 30: "ngo"}
    assert settings.preload_languages == ("ha", "yo") and settings.packs == ()


@pytest.mark.parametrize("entry", ["nokey", ":" + "a" * 30, "bank:short"])
def test_malformed_api_keys(entry: str) -> None:
    with pytest.raises(ValueError):
        Settings(api_keys=(entry,), _env_file=None).api_key_map()  # type: ignore[call-arg]


def test_secrets_are_not_printed() -> None:
    settings = Settings(hash_secret=SECRET, llm_api_key="llm-key", _env_file=None)  # type: ignore[call-arg]
    assert SECRET not in repr(settings) and "llm-key" not in repr(settings)


# --- rate limiter ------------------------------------------------------------------------------------------------


def test_token_bucket_refills_over_time() -> None:
    now = [0.0]
    limiter = RateLimiter(60, burst=2, clock=lambda: now[0])
    assert limiter.allow("k") and limiter.allow("k") and not limiter.allow("k")
    assert limiter.retry_after("k") == 1
    now[0] += 1.0  # one token per second at 60/min
    assert limiter.allow("k") and not limiter.allow("k")
    assert limiter.allow("other")  # independent keys


def test_limiter_memory_is_bounded() -> None:
    limiter = RateLimiter(1, max_keys=2)
    for key in ("a", "b", "c"):
        limiter.allow(key)
    assert limiter.allow("a")  # evicted, so it starts with a full bucket again
    with pytest.raises(ValueError):
        RateLimiter(0)


# --- reporting ---------------------------------------------------------------------------------------------------


def row(**values: object) -> dict[str, object]:
    base: dict[str, object] = {"channel": "whatsapp", "user": "u1", "language": "ha", "kind": "text",
                               "outcome": "answered", "verdict": None, "helpful": None, "total_ms": 1000,
                               "created_at": "2026-10-09T10:00:00+00:00"}  # fmt: skip
    return base | values


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([5, 1, 3], 50) == 3 and percentile([1, 2, 3, 4], 95) == 4 and percentile([7], 0) == 7


def test_build_report() -> None:
    rows = [
        row(verdict="likely_scam", helpful=True),
        row(user="u2", language="en", kind="audio", total_ms=7000, helpful=False, verdict="caution"),
        row(user="u2", channel="web", total_ms=3000, created_at="2026-10-10T08:00:00+00:00"),
        row(user="team", verdict="likely_scam"),
        row(user="biz", channel="api"),
    ]
    report = build_report(rows, exclude_users=frozenset({"team"}))
    assert report.interactions == 3 and report.unique_users == 2
    assert report.users_by_channel == {"web": 1, "whatsapp": 2}
    assert report.by_language == {"ha": 2, "en": 1} and report.by_verdict == {"likely_scam": 1, "caution": 1}
    assert (report.feedback_given, report.helpful, report.helpful_rate) == (2, 1, 0.5)
    assert report.latency_ms["audio"] == {"n": 1, "p50": 7000.0, "p95": 7000.0}
    assert report.per_day == {"2026-10-09": 2, "2026-10-10": 1}
    text = format_report(report)
    assert "unique users: 2" in text and "Helpful: 1/2 (50%)" in text
    assert json.dumps(report.as_dict())
    assert build_report([]).helpful_rate is None and "n/a" in format_report(build_report([]))


# --- services ----------------------------------------------------------------------------------------------------


def base_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {"hash_secret": SECRET, "db_path": tmp_path / "kg.db", "speech": False,
                                 "embeddings": False, "whatsapp_enabled": False, "_env_file": None}  # fmt: skip
    return Settings(**(values | overrides))  # type: ignore[arg-type]


async def test_build_services_wires_a_working_engine(tmp_path: Path) -> None:
    services = await build_services(base_settings(tmp_path, packs=("kudiguard-core",)))
    try:
        assert services.conversation is None and not services.engine.has_speech
        assert any(t["id"] == "flag-asks-credentials" for t in services.engine.topics())
        assert len(services.background) == 1  # retention purge loop
    finally:
        await services.aclose()
    assert services.background[0].done()


async def test_build_services_with_whatsapp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {"WHATSAPP_PHONE_NUMBER_ID": "P", "WHATSAPP_ACCESS_TOKEN": "t",
                        "WHATSAPP_APP_SECRET": "s", "WHATSAPP_VERIFY_TOKEN": "v"}.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    services = await build_services(base_settings(tmp_path, whatsapp_enabled=True, packs=("kudiguard-core",)))
    try:
        assert services.conversation is not None and services.whatsapp is not None
        assert services.donations is not None and services.conversation._donations is services.donations
        assert services.donations.directory == Settings().voice_dir  # isolated per test (conftest)
        assert not services.conversation._collecting  # off until the DPIA is signed off
    finally:
        await services.aclose()
    services = await build_services(
        base_settings(tmp_path, whatsapp_enabled=True, packs=("kudiguard-core",), voice_donation=True,
                      storage_location="nigeria")
    )  # fmt: skip
    try:
        assert services.conversation is not None and services.conversation._collecting
        assert services.conversation._storage == "nigeria"
    finally:
        await services.aclose()


def test_voice_donation_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from kudiguard.hosting import StorageLocation

    defaults = Settings(_env_file=None, hash_secret=SECRET)  # type: ignore[call-arg]
    assert defaults.voice_donation is False and defaults.voice_retention_days == 365
    assert defaults.storage_location is StorageLocation.OUTSIDE_NIGERIA  # never "in Nigeria" by default
    assert defaults.voice_secret() == SECRET.encode()
    own_key = Settings(_env_file=None, hash_secret=SECRET, voice_key="k" * 40)  # type: ignore[call-arg]
    assert own_key.voice_secret() == b"k" * 40
    blank_key = Settings(_env_file=None, hash_secret=SECRET, voice_key="")  # type: ignore[call-arg]
    assert blank_key.voice_secret() == SECRET.encode()  # KUDIGUARD_VOICE_KEY= as in .env.example
    with pytest.raises(ValidationError):
        Settings(_env_file=None, voice_retention_days=366)  # type: ignore[call-arg]
    monkeypatch.setenv("KUDIGUARD_STORAGE_LOCATION", "nigeria")
    assert Settings(_env_file=None).storage_location is StorageLocation.NIGERIA  # type: ignore[call-arg]
    monkeypatch.setenv("KUDIGUARD_STORAGE_LOCATION", "lagos")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert (
        StorageLocation.NIGERIA.english == "in Nigeria"
        and "outside" in StorageLocation.OUTSIDE_NIGERIA.english
    )


async def test_one_failing_purge_step_never_skips_the_others(caplog: pytest.LogCaptureFixture) -> None:
    from datetime import datetime, timedelta

    from kudiguard.storage.base import DELETION_MARKER_DAYS, utcnow

    caplog.set_level("INFO", logger="kudiguard.api.services")
    marker_cutoffs: list[datetime] = []

    class Store:
        async def purge_before(self, cutoff: object) -> int:
            raise RuntimeError("locked")

        async def purge_patterns_before(self, cutoff: object) -> int:
            return 1

        async def purge_deletion_markers(self, cutoff: datetime) -> int:
            marker_cutoffs.append(cutoff)
            return 2

    class Vault:
        async def purge(self, cutoff: object) -> int:
            raise OSError("disk")

    task = asyncio.create_task(purge_loop(Store(), 90, interval_s=3600, donations=Vault()))  # type: ignore[arg-type]
    await asyncio.sleep(0.02)
    task.cancel()
    assert len(marker_cutoffs) == 1  # ran although evidence (before) and voice (after) failed
    expected = utcnow() - timedelta(days=DELETION_MARKER_DAYS) + timedelta(hours=1)  # one interval early
    assert abs((marker_cutoffs[0] - expected).total_seconds()) < 5
    assert "retention purge failed (evidence)" in caplog.text
    assert "retention purge failed (donated voice notes)" in caplog.text
    assert "retention purge removed 3 rows" in caplog.text


async def test_purge_loop_expires_donated_voice_notes_on_their_own_clock() -> None:
    from datetime import date, timedelta

    from kudiguard.storage.base import utcnow

    cutoffs: list[date] = []

    class Store:
        async def purge_before(self, cutoff: object) -> int:
            return 0

        async def purge_patterns_before(self, cutoff: object) -> int:
            return 0

    class Vault:
        async def purge(self, cutoff: date) -> int:
            cutoffs.append(cutoff)
            return 2

    task = asyncio.create_task(
        purge_loop(Store(), 90, interval_s=0.01, donations=Vault(), voice_retention_days=365)  # type: ignore[arg-type]
    )
    await asyncio.sleep(0.03)
    task.cancel()
    assert cutoffs and cutoffs[0] == (utcnow() - timedelta(days=365)).date()


async def test_build_services_without_whatsapp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
    services = await build_services(base_settings(tmp_path, whatsapp_enabled=True, packs=("kudiguard-core",)))
    try:
        assert services.conversation is None
    finally:
        await services.aclose()


async def test_build_services_fails_fast_and_cleans_up(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least"):
        await build_services(base_settings(tmp_path, hash_secret="short"))
    with pytest.raises(KnowledgeConfigError, match="unknown knowledge packs"):  # after the HTTP client opened
        await build_services(base_settings(tmp_path, packs=("no-such-pack",)))


async def test_purge_loop_runs_and_survives_errors() -> None:
    calls: list[str] = []

    class Store:
        async def purge_before(self, cutoff: object) -> int:
            calls.append("purge")
            if len(calls) == 1:
                raise RuntimeError("locked")
            return 3

    task = asyncio.create_task(purge_loop(Store(), 90, interval_s=0.01))  # type: ignore[arg-type]
    await asyncio.sleep(0.05)
    task.cancel()
    assert len(calls) >= 2


async def test_aclose_survives_failing_closers(tmp_path: Path) -> None:
    services = await build_services(base_settings(tmp_path, packs=("kudiguard-core",)))

    async def broken() -> None:
        raise RuntimeError("already closed")

    services.closers.append(broken)
    await services.aclose()  # logs, does not raise


def test_services_dataclass_defaults() -> None:
    assert Services.__dataclass_fields__["conversation"].default is None


# --- CLI -----------------------------------------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "kg.db"
    monkeypatch.setenv("KUDIGUARD_HASH_SECRET", SECRET)
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(db))
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)

    async def seed() -> None:
        store = await SQLiteStore.open(db, hasher=UserHasher(SECRET.encode()))
        hasher = UserHasher(SECRET.encode())
        for n, user in enumerate(("+2348031234567", "+2348099999999")):
            await store.add_interaction(
                Interaction(channel="whatsapp", message_id=f"m{n}", user_hash=hasher.user_id(user), language="ha",
                            kind=MessageKind.TEXT, outcome=Outcome.ANSWERED, verdict=Verdict.LIKELY_SCAM, total_ms=500)
            )  # fmt: skip
        await store.close()

    asyncio.run(seed())
    return tmp_path


def test_cli_stats_export_and_purge(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stats"]) == 0
    assert "unique users: 2" in capsys.readouterr().out
    testers = cli_env / "team.txt"
    testers.write_text("08031234567\n\n", encoding="utf-8")  # local format: same user as +234…
    assert cli.main(["stats", "--json", "--exclude-users", str(testers)]) == 0
    assert json.loads(capsys.readouterr().out)["unique_users"] == 1
    out = cli_env / "evidence.jsonl"
    assert cli.main(["export", "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and "2348031234567" not in out.read_text(encoding="utf-8")
    assert (
        cli.main(["purge"]) == 0
        and "removed 0 rows older than 90 days" in capsys.readouterr().out.splitlines()[-1]
    )
    assert cli.main(["stats", "--channels", "api"]) == 0
    assert "unique users: 0" in capsys.readouterr().out


def test_cli_reports_configuration_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)
    monkeypatch.setenv("KUDIGUARD_HASH_SECRET", "short")
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(tmp_path / "x.db"))
    assert cli.main(["stats"]) == 2


def test_cli_serve_runs_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    called: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)
    monkeypatch.setenv("KUDIGUARD_HASH_SECRET", SECRET)
    monkeypatch.setenv("KUDIGUARD_WHATSAPP_ENABLED", "false")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: called.update(kwargs))
    assert cli.main(["serve", "--port", "9000"]) == 0
    assert called["port"] == 9000 and called["host"] == "127.0.0.1"


def test_rows_without_latency_are_still_counted() -> None:
    report = build_report([row(total_ms=None), row(user="u2")])
    assert report.interactions == 2 and report.latency_ms["text"]["n"] == 1


async def test_purge_loop_logs_removals_quietly_when_nothing_to_do() -> None:
    results = iter([0, 5])

    class Store:
        async def purge_before(self, cutoff: object) -> int:
            return next(results, 0)

    task = asyncio.create_task(purge_loop(Store(), 90, interval_s=0.01))  # type: ignore[arg-type]
    await asyncio.sleep(0.04)
    task.cancel()


@pytest.mark.parametrize("preload_failures", [{"ha": "missing model"}, {}])
async def test_build_services_with_embeddings_and_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preload_failures: dict[str, str]
) -> None:
    """Production paths, with the heavy model libraries replaced at their boundaries."""
    from kudiguard.api import services as services_module
    from kudiguard.knowledge import embeddings
    from kudiguard.speech import natlas_asr

    created: dict[str, object] = {}

    class FakeEmbedder:
        def __init__(self, cache_dir: Path) -> None:
            created["embedder"] = cache_dir

    class FakeASR:
        def __init__(self, models: object, *, loader: object) -> None:
            created["asr_models"] = models

        async def preload(self, languages: object) -> dict[str, str]:
            return preload_failures  # failures are logged, not fatal

    class FakeKnowledge:
        sources = ()

        async def refresh_all(self) -> dict[str, str]:
            return {"partner-api": "unreachable"}  # logged, not fatal

    monkeypatch.setattr(embeddings, "FastEmbedder", FakeEmbedder)
    monkeypatch.setattr(natlas_asr, "NatlasASR", FakeASR)
    monkeypatch.setattr(natlas_asr, "ct2_loader", lambda **kwargs: object())
    monkeypatch.setattr(services_module, "find_ffmpeg", lambda explicit=None: "ffmpeg")
    monkeypatch.setattr(services_module, "knowledge_base_from_packs", lambda *a, **k: FakeKnowledge())
    services = await build_services(base_settings(tmp_path, embeddings=True, speech=True))
    try:
        assert services.engine.has_speech and "embedder" in created and "ha" in created["asr_models"]  # type: ignore[operator]
    finally:
        await services.aclose()


async def test_missing_ffmpeg_means_text_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kudiguard.api import services as services_module

    def no_ffmpeg(explicit: str | None = None) -> str:
        raise FileNotFoundError("ffmpeg not found")

    monkeypatch.setattr(services_module, "find_ffmpeg", no_ffmpeg)
    services = await build_services(base_settings(tmp_path, speech=True, packs=("kudiguard-core",)))
    try:
        assert not services.engine.has_speech  # server still starts; voice disabled with a warning
    finally:
        await services.aclose()


def test_cli_guards(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["purge", "--days", "0"]) == 2
    assert "at least 1" in capsys.readouterr().err
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(cli_env / "missing.db"))
    assert cli.main(["stats"]) == 2
    assert "no evidence database" in capsys.readouterr().err
    assert not (cli_env / "missing.db").exists()  # never silently created


async def test_missing_embedding_model_falls_back_to_keyword_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kudiguard.knowledge import embeddings

    class Unavailable:
        def __init__(self, **kwargs: object) -> None:
            raise ValueError("Could not load model BAAI/bge-small-en-v1.5 from any source")

    monkeypatch.setattr(embeddings, "FastEmbedder", Unavailable)
    services = await build_services(base_settings(tmp_path, embeddings=True, packs=("kudiguard-core",)))
    try:
        assert any(t["id"] == "flag-asks-credentials" for t in services.engine.topics())
    finally:
        await services.aclose()


async def test_broken_pack_is_still_a_startup_error_with_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kudiguard.knowledge import embeddings

    monkeypatch.setattr(embeddings, "FastEmbedder", lambda **kwargs: None)
    with pytest.raises(KnowledgeConfigError):
        await build_services(base_settings(tmp_path, embeddings=True, packs=("no-such-pack",)))


def test_access_log_drops_query_strings() -> None:
    import logging

    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                               ("1.2.3.4", "GET", "/webhooks/whatsapp?hub.verify_token=SECRET", "1.1", 200), None)  # fmt: skip
    assert cli.redact_query_strings(record) and "SECRET" not in record.getMessage()
    other = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "plain", None, None)
    assert cli.redact_query_strings(other) and other.getMessage() == "plain"


def test_cli_patterns_export(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import date

    from kudiguard.patterns import ScamPattern

    async def seed() -> None:
        store = await SQLiteStore.open(cli_env / "kg.db", hasher=UserHasher(SECRET.encode()))
        await store.add_pattern(ScamPattern.from_check(language="ha", channel="whatsapp", verdict=Verdict.CAUTION,
                                                       flags=(), text="ka turo min lambar", english=None,
                                                       day=date(2026, 10, 8)))  # fmt: skip
        await store.close()

    asyncio.run(seed())
    out = cli_env / "patterns.jsonl"
    assert cli.main(["patterns", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["language"] == "ha"
    assert "1 de-identified" in capsys.readouterr().out


def test_cli_patterns_needs_an_existing_database(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(cli_env / "missing.db"))
    assert cli.main(["patterns", "--out", str(cli_env / "p.jsonl")]) == 2
