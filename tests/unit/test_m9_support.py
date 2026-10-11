"""Settings, rate limiting, validation reporting, service wiring and the CLI."""

import asyncio
import json
import logging
import os
import re
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
from kudiguard.text import RedactingFilter

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

    from kudiguard.api.services import PURGE_INTERVAL_S, PURGE_MARGIN
    from kudiguard.storage.base import DELETION_MARKER_DAYS, utcnow

    # Markers must outlive every importable form answer (29 days) even when dropped early.
    assert PURGE_INTERVAL_S < 24 * 3600
    assert timedelta(days=DELETION_MARKER_DAYS) - timedelta(
        seconds=PURGE_INTERVAL_S
    ) - PURGE_MARGIN > timedelta(days=29)
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
    expected = utcnow() - timedelta(days=DELETION_MARKER_DAYS) + timedelta(hours=1) + PURGE_MARGIN  # early
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
    handlers = logging.getLogger().handlers
    before = {id(h): list(h.filters) for h in handlers}
    own = logging.getLogger("kudiguard.test_serve")
    own_handler = logging.NullHandler()
    own.addHandler(own_handler)
    try:
        assert cli.main(["serve", "--port", "9000"]) == 0
        assert called["port"] == 9000 and called["host"] == "127.0.0.1"
        assert called["log_config"] == cli.serve_log_config()
        assert all(any(isinstance(f, RedactingFilter) for f in h.filters) for h in [*handlers, own_handler])
        cli.redact_every_handler()  # idempotent: one filter per handler
        assert sum(isinstance(f, RedactingFilter) for f in own_handler.filters) == 1
    finally:
        own.removeHandler(own_handler)
        for handler in handlers:
            handler.filters = before.get(id(handler), [])


@pytest.fixture
def uvicorn_logging() -> object:
    """uvicorn's loggers as uvicorn configures them, restored afterwards."""
    names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    saved = {n: (logging.getLogger(n).handlers[:], logging.getLogger(n).propagate, logging.getLogger(n).level)
             for n in names}  # fmt: skip
    yield
    for name, (handlers, propagate, level) in saved.items():
        logger = logging.getLogger(name)
        logger.handlers, logger.propagate, logger.level = handlers, propagate, level


@pytest.mark.usefixtures("uvicorn_logging")
def test_uvicorn_own_handlers_redact_tracebacks_and_access_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """uvicorn's loggers do not propagate to the root: the redaction must sit on uvicorn's own handlers, as uvicorn
    configures them from ``serve_log_config``."""
    import uvicorn

    async def app(scope: object, receive: object, send: object) -> None:  # pragma: no cover - never called
        raise AssertionError

    uvicorn.Config(
        app, log_config=cli.serve_log_config(), log_level="info"
    )  # configures logging, as uvicorn.run
    try:
        raise ValueError("bad request from 08031234567")
    except ValueError:
        logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:5000",
        "GET",
        "/check/0803-123-4567?hub.verify_token=SECRET",
        "1.1",
        200,
    )
    out = capsys.readouterr()
    assert "Exception in ASGI application" in out.err and "ValueError: bad request from [number]" in out.err
    assert "Traceback" in out.err and '"GET /check/[number] HTTP/1.1" 200' in out.out
    assert "0803" not in out.err + out.out and "SECRET" not in out.out


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

    from fakes import BagOfWordsEmbedder

    class FakeEmbedder(BagOfWordsEmbedder):
        DIM = 384  # the shipped guard classifier's embedding size

        def __init__(self, cache_dir: Path) -> None:
            super().__init__()
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
        assert services.engine.grounding is not None  # on by default, reusing the search model
        assert services.engine.classifier is not None  # shadow by default, same model
        assert services.engine.grounding.threshold == services.settings.grounding_threshold
        assert services.engine.grounding.budget_s == services.settings.grounding_budget_ms / 1000
    finally:
        await services.aclose()


async def test_grounding_check_needs_embeddings_and_can_be_turned_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from fakes import BagOfWordsEmbedder
    from kudiguard.knowledge import embeddings

    with caplog.at_level("WARNING", logger="kudiguard.api.services"):
        services = await build_services(base_settings(tmp_path, packs=("kudiguard-core",)))  # no embeddings
    await services.aclose()
    assert services.engine.grounding is None and "grounding check disabled" in caplog.text

    monkeypatch.setattr(embeddings, "FastEmbedder", lambda cache_dir: BagOfWordsEmbedder())
    caplog.set_level("INFO", logger="kudiguard.api.services")
    for mode in ("shadow", "enforce", "off"):
        settings = base_settings(tmp_path, embeddings=True, grounding_mode=mode, packs=("kudiguard-core",))
        services = await build_services(settings)
        await services.aclose()
        assert (services.engine.grounding is not None) is (mode != "off")
        if services.engine.grounding is not None:  # shadow checks drain before the checker's worker stops
            closers = services.closers
            assert closers.index(services.engine.aclose) > closers.index(services.engine.grounding.aclose)
            assert services.engine.grounding.wait_when_busy is (mode == "enforce")
        assert services.engine.grounding_mode == mode
    assert "evidence sentences embedded" in caplog.text  # pack facts pre-embedded at startup


def test_grounding_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from kudiguard.safety.grounding import DEFAULT_BUDGET_S, DEFAULT_THRESHOLD

    defaults = Settings(_env_file=None)  # type: ignore[call-arg]
    assert defaults.grounding_mode == "shadow" and defaults.grounding_threshold == DEFAULT_THRESHOLD
    assert defaults.grounding_budget_ms == DEFAULT_BUDGET_S * 1000  # config.py keeps literal copies
    monkeypatch.setenv("KUDIGUARD_GROUNDING_MODE", "enforce")
    monkeypatch.setenv("KUDIGUARD_GROUNDING_THRESHOLD", "0.8")
    tuned = Settings(_env_file=None)  # type: ignore[call-arg]
    assert tuned.grounding_mode == "enforce" and tuned.grounding_threshold == 0.8
    monkeypatch.setenv("KUDIGUARD_GROUNDING_MODE", "on")
    with pytest.raises(ValueError, match="grounding_mode"):
        Settings(_env_file=None)  # type: ignore[call-arg]
    monkeypatch.setenv("KUDIGUARD_GROUNDING_MODE", "shadow")
    monkeypatch.setenv("KUDIGUARD_GROUNDING_THRESHOLD", "1.5")
    with pytest.raises(ValueError, match="grounding_threshold"):
        Settings(_env_file=None)  # type: ignore[call-arg]


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


def test_log_grounding_records_counts_per_turn(caplog: pytest.LogCaptureFixture) -> None:
    from kudiguard.api.services import log_grounding
    from kudiguard.safety.grounding import Claim, GroundingReport

    report = GroundingReport("x", (Claim("a b c", 0.9, True), Claim("Pay the NFIU.", 0.2, False)))
    with caplog.at_level("INFO", logger="kudiguard.api.services"):
        log_grounding("wamid.1", report, 31)
        log_grounding("wamid.2", None, 150)
    assert "grounding turn=wamid.1 grounded=False unsupported_claims=1 claims=2 ms=31" in caplog.text
    assert "grounding turn=wamid.2 checked=no ms=150" in caplog.text and "NFIU" not in caplog.text


def test_guard_classifier_wiring(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from fakes import BagOfWordsEmbedder
    from kudiguard.api.services import build_guard_classifier
    from kudiguard.languages import LanguageRegistry
    from kudiguard.safety.guard_classifier import DEFAULT_BUDGET_S

    class Bge(BagOfWordsEmbedder):  # the shipped artefact's dimension
        DIM = 384

    registry = LanguageRegistry.load(Path(__file__).resolve().parents[2] / "data" / "languages")
    embedder = Bge()
    defaults = base_settings(tmp_path)
    assert defaults.guard_classifier_mode == "shadow" and defaults.guard_classifier_threshold is None
    assert defaults.guard_classifier_pidgin_mode is None
    assert defaults.guard_classifier_budget_ms == DEFAULT_BUDGET_S * 1000
    built = build_guard_classifier(defaults, registry, embedder)
    assert (
        built is not None and built.threshold == built.model.threshold and built.budget_s == DEFAULT_BUDGET_S
    )
    tuned = build_guard_classifier(
        base_settings(tmp_path, guard_classifier_threshold=0.9), registry, embedder
    )
    assert tuned is not None and tuned.threshold == 0.9
    assert (
        build_guard_classifier(base_settings(tmp_path, guard_classifier_mode="off"), registry, embedder)
        is None
    )
    with caplog.at_level("WARNING", logger="kudiguard.api.services"):
        assert build_guard_classifier(defaults, registry, None) is None
        missing = base_settings(tmp_path, guard_classifier_path=tmp_path / "none.npz")
        assert build_guard_classifier(missing, registry, embedder) is None
    assert "needs the embedding model" in caplog.text and "no usable artefact" in caplog.text
    pidgin_only = base_settings(tmp_path, guard_classifier_mode="off", guard_classifier_pidgin_mode="enforce")
    assert build_guard_classifier(pidgin_only, registry, embedder) is not None


def test_guard_classifier_mismatch_disables_the_layer_with_one_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from fakes import BagOfWordsEmbedder
    from kudiguard.api.services import build_guard_classifier
    from kudiguard.languages import LanguageRegistry

    registry = LanguageRegistry.load(Path(__file__).resolve().parents[2] / "data" / "languages")
    with caplog.at_level("ERROR", logger="kudiguard.api.services"):
        assert build_guard_classifier(base_settings(tmp_path), registry, BagOfWordsEmbedder()) is None
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "384-dim embeddings, the embedder gives 512" in errors[0].getMessage()


def test_env_example_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A .env copied from .env.example must start: empty optional values mean "unset", not a parse error."""
    example = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text(example, encoding="utf-8")
    for name in list(os.environ):
        if name.startswith(("KUDIGUARD_", "WHATSAPP_")):
            monkeypatch.delenv(name)
    settings = Settings(_env_file=env)  # type: ignore[call-arg]
    assert settings.guard_classifier_threshold is None and settings.guard_classifier_pidgin_mode is None
    env.write_text(example + "\nKUDIGUARD_GUARD_CLASSIFIER_THRESHOLD=\nKUDIGUARD_GUARD_CLASSIFIER_PIDGIN_MODE= \n",
                   encoding="utf-8")  # fmt: skip
    blank = Settings(_env_file=env)  # type: ignore[call-arg]
    assert blank.guard_classifier_threshold is None and blank.guard_classifier_pidgin_mode is None
    env.write_text(example + "\nKUDIGUARD_GUARD_CLASSIFIER_THRESHOLD=0.7\n", encoding="utf-8")
    assert Settings(_env_file=env).guard_classifier_threshold == 0.7  # type: ignore[call-arg]


def test_env_example_documents_every_operator_setting() -> None:
    """Review R: each setting an operator may need appears in .env.example (commented lines show the default), and
    every KUDIGUARD_* name there is a real setting."""
    example = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    named = {m.lower() for m in re.findall(r"^#? ?KUDIGUARD_([A-Z0-9_]+)=", example, flags=re.MULTILINE)}
    assert named <= set(Settings.model_fields), named - set(Settings.model_fields)
    required = {
        "trust_proxy_headers", "public_url", "active_user_cap", "web_session_cap", "per_user_per_hour",
        "api_rate_per_minute", "web_rate_per_minute", "turn_deadline_s", "db_path", "embeddings_cache", "ffmpeg",
        "data_dir", "packs", "allow_sample_packs", "llm_api_key", "model_cache", "guard_classifier_path", "speech",
        "preload_languages", "default_language", "whatsapp_enabled",
    }  # fmt: skip
    assert required <= named, required - named
    assert "Cloudflare" in example and not re.search(r"^HF_HOME=", example, flags=re.MULTILINE)


async def test_grounding_warm_up_failure_is_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from types import SimpleNamespace

    from kudiguard.api.services import build_grounding
    from kudiguard.languages import LanguageRegistry

    class Broken:
        def embed_documents(self, texts: object) -> object:
            raise RuntimeError("onnx")

        def embed_query(self, text: str) -> object:
            raise RuntimeError("onnx")

    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    entry = SimpleNamespace(facts=("Banks never ask for your PIN.",), actions=())
    knowledge = SimpleNamespace(sources=(SimpleNamespace(entries=(entry,)),))
    with caplog.at_level("WARNING", logger="kudiguard.api.services"):
        grounding = await build_grounding(base_settings(tmp_path), registry, Broken(), knowledge)  # type: ignore[arg-type]
    assert grounding is not None and "grounding evidence not pre-embedded: onnx" in caplog.text
    await grounding.aclose()


async def test_evidence_is_purged_one_pass_early_so_up_to_n_days_holds() -> None:
    """Messages are dropped one interval (plus margin) before they turn N days old: "up to 90 days" stays true until
    the next pass, while the server runs."""
    from datetime import datetime, timedelta

    from kudiguard.api.services import PURGE_MARGIN, _purge_once
    from kudiguard.storage.base import utcnow

    cutoffs: list[datetime] = []

    class Store:
        async def purge_before(self, cutoff: datetime) -> int:
            cutoffs.append(cutoff)
            return 0

        async def purge_patterns_before(self, cutoff: object) -> int:
            return 0

        async def purge_deletion_markers(self, cutoff: object) -> int:
            return 0

    now = utcnow()
    await _purge_once(Store(), now, 90, 730, 3600, None, 365)  # type: ignore[arg-type]
    assert cutoffs == [now - timedelta(days=90) + timedelta(seconds=3600) + PURGE_MARGIN]


def test_cli_purge_also_expires_scam_examples(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import date

    from kudiguard import cli
    from kudiguard.domain import Flag, Verdict
    from kudiguard.patterns import ScamPattern
    from kudiguard.storage.sqlite import SQLiteStore

    async def seed() -> None:
        store = await SQLiteStore.open(cli_env / "kg.db", hasher=UserHasher(SECRET.encode()))
        await store.add_pattern(ScamPattern.from_check(language="en", channel="whatsapp", verdict=Verdict.CAUTION,
                                                       flags=(Flag.URGENCY,), text="pay now", english="pay now",
                                                       day=date(2020, 1, 6)))  # fmt: skip
        await store.close()

    asyncio.run(seed())
    assert cli.main(["purge"]) == 0
    assert "1 scam examples older than 730 days" in capsys.readouterr().out


async def test_the_evidence_purge_never_cuts_inside_the_licence_window(tmp_path: Path) -> None:
    """Users active 28-29.9 days ago still count toward the cap after a purge pass, even at the shortest retention."""
    from datetime import timedelta

    from kudiguard.admission import Admission, AtCapacity
    from kudiguard.api.services import PURGE_INTERVAL_S, _purge_once
    from kudiguard.storage.base import utcnow

    store = await SQLiteStore.open(tmp_path / "p30.db", hasher=UserHasher(SECRET.encode()))
    now = utcnow()
    for n, age_h in enumerate((30 * 24 - 2, 30 * 24 - 6, 10)):
        await store.add_interaction(
            Interaction(channel="whatsapp", message_id=f"m{n}", user_hash=f"u{n}", language="en",
                        kind=MessageKind.TEXT, outcome=Outcome.ANSWERED, created_at=now - timedelta(hours=age_h))
        )  # fmt: skip
    since = now - timedelta(days=30)
    await _purge_once(store, now, 30, 730, PURGE_INTERVAL_S, None, 365)
    assert await store.count_active_users_since(since) == 3
    with pytest.raises(AtCapacity):
        await Admission(store, 3).reserve("new", "whatsapp")
    await store.close()


def test_cli_purge_keeps_the_licence_window_while_the_cap_is_on(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["purge", "--days", "20"]) == 2
    assert "at least 30" in capsys.readouterr().err
