"""Builds the running system from ``Settings``: one place where adapters are chosen and wired.

Tests (and other hosts, e.g. a future call-centre worker) construct ``Services`` directly with fakes instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import httpx

from kudiguard.admission import Admission
from kudiguard.config import Settings
from kudiguard.donation import VOICE_RETENTION_DAYS, Sealer, VoiceVault
from kudiguard.engine import Engine
from kudiguard.knowledge.embeddings import Embedder
from kudiguard.knowledge.file_pack import KnowledgeConfigError
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import LLM
from kudiguard.llm.openai_compat import OpenAICompatLLM
from kudiguard.patterns import PATTERN_RETENTION_DAYS
from kudiguard.pipeline import Conversation, Limits
from kudiguard.privacy import UserHasher
from kudiguard.safety.grounding import GroundingChecker, GroundingMode, GroundingReport
from kudiguard.safety.guard_classifier import ClassifierMode, ClassifierModel, UnsafeAdviceClassifier
from kudiguard.safety.output_guard import OutputGuard
from kudiguard.speech.audio import find_ffmpeg, sweep_stale_audio
from kudiguard.storage.base import ACTIVE_USER_WINDOW_DAYS, DELETION_MARKER_DAYS, Store, utcnow
from kudiguard.storage.sqlite import SQLiteStore

log = logging.getLogger(__name__)

PURGE_INTERVAL_S = 6 * 3600
# A pass may run late (sleep drift, slow disk): DELETE markers expire this much earlier.
PURGE_MARGIN = timedelta(hours=1)


@dataclass
class Services:
    settings: Settings
    registry: LanguageRegistry
    engine: Engine
    store: Store
    hasher: UserHasher
    llm: LLM
    conversation: Conversation | None = None  # present when a chat channel (WhatsApp) is configured
    whatsapp: object | None = None  # channels.whatsapp.WhatsAppConfig when configured
    donations: VoiceVault | None = None  # opt-in voice-note donation (WhatsApp and the web page)
    closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    background: list[asyncio.Task[None]] = field(default_factory=list)
    # The one licence-cap gate for every channel (WhatsApp, website, partner API); new-user gates plug in here.
    admission: Admission = field(init=False)

    def __post_init__(self) -> None:
        self.admission = Admission(
            self.store, self.settings.active_user_cap, shares={"web": self.settings.web_session_cap}
        )

    async def aclose(self) -> None:
        for task in self.background:
            task.cancel()
        for task in self.background:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self.conversation is not None:
            await self.conversation.drain()
        for close in reversed(self.closers):
            try:
                await close()
            except Exception:
                log.exception("error while shutting down")


async def purge_loop(
    store: Store,
    retention_days: int,
    interval_s: float = PURGE_INTERVAL_S,
    pattern_retention_days: int = PATTERN_RETENTION_DAYS,
    donations: VoiceVault | None = None,
    voice_retention_days: int = VOICE_RETENTION_DAYS,
) -> None:
    """Delete evidence (and, on their own clocks, scam examples and donated voice notes) older than their retention
    periods (NDPA s.24), every ``interval_s``."""
    while True:
        removed = await _purge_once(
            store,
            utcnow(),
            retention_days,
            pattern_retention_days,
            interval_s,
            donations,
            voice_retention_days,
        )
        if removed:
            log.info("retention purge removed %d rows", removed)
        await asyncio.sleep(interval_s)


async def build_knowledge(
    settings: Settings, client: httpx.AsyncClient
) -> tuple[KnowledgeBase, Embedder | None]:
    """The knowledge base exactly as the bot searches it, and its embedder (None = keyword search only).

    Also used to re-derive grounding for native-speaker reviews; the embedder can be shared by other checks."""

    def knowledge_with(embedder: Embedder | None) -> KnowledgeBase:
        return knowledge_base_from_packs(
            settings.data_dir / "packs",
            client=client,
            include=settings.packs or None,
            embedder=embedder,
            allow_samples=settings.allow_sample_packs,
            allow_live=settings.allow_live_packs,
        )

    knowledge, embedder = None, None
    if settings.embeddings:
        try:
            from kudiguard.knowledge.embeddings import FastEmbedder

            candidate: Embedder = FastEmbedder(cache_dir=settings.embeddings_cache)
            knowledge, embedder = knowledge_with(candidate), candidate
        except KnowledgeConfigError:
            raise  # a broken pack is a configuration error, not a missing model
        except Exception as exc:  # model not cached / not installed: keyword search still works
            log.warning("semantic search disabled (keyword search only): %s", exc)
    if knowledge is None:
        knowledge = knowledge_with(None)
    if failed := await knowledge.refresh_all():
        log.warning("knowledge sources unavailable at startup: %s", sorted(failed))
    return knowledge, embedder


def pattern_keep_from(now: datetime, retention_days: int, interval_s: float = PURGE_INTERVAL_S) -> date:
    """The first day whose scam examples may stay until the next pass: every example still kept then is younger
    than ``retention_days`` (the promise on /privacy and in the consent text), whatever its time of day.

    Examples carry only their week (Monday), so a week is dropped as soon as its Monday falls before this day; the
    next pass may run ``interval_s`` (plus ``PURGE_MARGIN``) later, so that much is taken off in advance."""
    next_pass = now + timedelta(seconds=interval_s) + PURGE_MARGIN
    return (next_pass - timedelta(days=retention_days)).date() + timedelta(days=1)


async def _purge_once(
    store: Store,
    now: datetime,
    retention_days: int,
    pattern_retention_days: int,
    interval_s: float,
    donations: VoiceVault | None,
    voice_retention_days: int,
) -> int:
    """One retention pass; each step on its own, so one failing never skips the others."""
    # DELETE markers are dropped one interval (plus a margin for slow passes) early, so none ever outlives
    # DELETION_MARKER_DAYS (/privacy), while still outliving every importable form answer (29 days).
    marker_cutoff = now - timedelta(days=DELETION_MARKER_DAYS) + timedelta(seconds=interval_s) + PURGE_MARGIN
    pattern_cutoff = pattern_keep_from(now, pattern_retention_days, interval_s)
    # Messages likewise: "up to N days" holds until the next pass, so purge one interval (plus margin) early, but
    # never inside the licence window: the cap counts active users from these rows (retention_days >= 31).
    evidence_cutoff = min(
        now - timedelta(days=retention_days) + timedelta(seconds=interval_s) + PURGE_MARGIN,
        now - timedelta(days=ACTIVE_USER_WINDOW_DAYS),
    )
    steps: list[tuple[str, Callable[[], Awaitable[int]]]] = [
        ("evidence", lambda: store.purge_before(evidence_cutoff)),
        (
            "scam examples",
            lambda: store.purge_patterns_before(pattern_cutoff),
        ),
        ("deletion markers", lambda: store.purge_deletion_markers(marker_cutoff)),
    ]
    if donations is not None:
        vault = donations
        steps.append(
            ("donated voice notes", lambda: vault.purge((now - timedelta(days=voice_retention_days)).date()))
        )
    removed = 0
    for name, work in steps:
        try:
            removed += await work()
        except Exception:
            log.exception("retention purge failed (%s)", name)
    return removed


def log_grounding(turn_id: str | None, report: GroundingReport | None, ms: int) -> None:
    """Per-turn grounding result for the shadow review (>= 100 real replies before enforce): counts only, no text.

    ``turn_id`` is the hashed message key (``UserHasher.message_key``), never a raw provider id.
    TODO(merge with E's pipeline): store grounded / unsupported_claims / grounding_ms on the turn's interaction row
    (needs a store migration); until then the logs carry them, keyed by the hashed message key.
    """
    if report is None:
        log.info("grounding turn=%s checked=no ms=%d", turn_id, ms)
    else:
        log.info("grounding turn=%s grounded=%s unsupported_claims=%d claims=%d ms=%d", turn_id, report.grounded,
                 len(report.unsupported), len(report.claims), ms)  # fmt: skip


def build_guard_classifier(
    settings: Settings, registry: LanguageRegistry, embedder: Embedder | None
) -> UnsafeAdviceClassifier | None:
    """The learned second guard layer (English), reusing the knowledge-search embedder; None when off/unavailable.

    The artefact is validated here once (checksum, features, embedding model and dimension): a mismatch disables
    the layer with one error at startup, never one per reply."""
    pidgin = settings.guard_classifier_pidgin_mode or settings.guard_classifier_mode
    if {settings.guard_classifier_mode, pidgin} == {"off"}:
        return None
    if embedder is None:
        log.warning("guard classifier disabled: it needs the embedding model (KUDIGUARD_EMBEDDINGS)")
        return None
    try:
        model = ClassifierModel.load(settings.guard_classifier_path)
        classifier = UnsafeAdviceClassifier(
            model,
            embedder,
            OutputGuard(registry),
            threshold=settings.guard_classifier_threshold,
            budget_s=settings.guard_classifier_budget_ms / 1000,
        )
    except Exception as exc:  # optional layer: a bad artefact or embedder must not stop the service
        log.error(
            "guard classifier disabled: no usable artefact at %s (%s)", settings.guard_classifier_path, exc
        )
        return None
    log.info("guard classifier (%s, Pidgin %s): version %s", settings.guard_classifier_mode, pidgin,
             model.version)  # fmt: skip
    return classifier


async def build_grounding(
    settings: Settings, registry: LanguageRegistry, embedder: Embedder | None, knowledge: KnowledgeBase
) -> GroundingChecker | None:
    """The live grounding check (English only, no extra model call), reusing the knowledge-search embedder.

    Pack facts are embedded once here, so a turn embeds only the reply and the user's message (knowledge-index
    vectors are per entry, not per sentence, and cannot be reused). Partner API entries are cached on first use.
    """
    if GroundingMode(settings.grounding_mode) is GroundingMode.OFF:
        return None
    if embedder is None:
        log.warning("grounding check disabled: it needs the embedding model (KUDIGUARD_EMBEDDINGS)")
        return None
    grounding = GroundingChecker(
        None,
        embedder,
        registry,
        threshold=settings.grounding_threshold,
        budget_s=settings.grounding_budget_ms / 1000,
        wait_when_busy=GroundingMode(settings.grounding_mode) is GroundingMode.ENFORCE,
    )
    texts = [
        text
        for source in knowledge.sources
        for entry in getattr(source, "entries", ())
        for text in (*entry.facts, *entry.actions)
    ]
    started = time.perf_counter()
    try:
        warmed = await asyncio.to_thread(grounding.warm, texts)
    except Exception as exc:  # not fatal: evidence is then embedded (and cached) on first use
        log.warning("grounding evidence not pre-embedded: %s", exc)
    else:
        log.info("grounding (%s): %d evidence sentences embedded in %.1f s", settings.grounding_mode, warmed,
                 time.perf_counter() - started)  # fmt: skip
    return grounding


async def build_services(settings: Settings) -> Services:
    """Production wiring. Fails fast on missing secrets or broken configuration."""
    from kudiguard.channels.whatsapp import WhatsAppConfig, WhatsAppMessenger

    hasher = UserHasher(settings.hash_secret.get_secret_value().encode())
    settings.api_key_map()  # validate early
    await asyncio.to_thread(sweep_stale_audio)  # plaintext audio left in the temp dir by a crash mid-decode
    registry = LanguageRegistry.load(settings.data_dir / "languages")
    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    closers: list[Callable[[], Awaitable[None]]] = [client.aclose]
    try:
        knowledge, embedder = await build_knowledge(settings, client)
        api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        llm = OpenAICompatLLM(settings.llm_url, client=client, api_key=api_key)

        asr = None
        ffmpeg = None
        if settings.speech:
            from kudiguard.speech.natlas_asr import NatlasASR, ct2_loader

            try:
                ffmpeg = find_ffmpeg(settings.ffmpeg)
            except (FileNotFoundError, RuntimeError) as exc:
                log.warning("voice disabled (text only): ffmpeg not found (%s)", exc)
            else:
                asr = NatlasASR(
                    registry.asr_models(),
                    loader=ct2_loader(
                        cache_dir=settings.model_cache, whisper_languages=registry.asr_whisper_languages()
                    ),
                )
                if failed_models := await asr.preload(settings.preload_languages):
                    log.warning("speech models not preloaded: %s", failed_models)
        grounding = await build_grounding(settings, registry, embedder, knowledge)
        if grounding is not None:
            closers.append(grounding.aclose)
        classifier = build_guard_classifier(settings, registry, embedder)
        if classifier is not None:
            closers.append(classifier.aclose)
        engine = Engine(
            registry=registry,
            llm=llm,
            knowledge=knowledge,
            asr=asr,
            ffmpeg=ffmpeg,
            grounding=grounding,
            grounding_mode=GroundingMode(settings.grounding_mode),
            on_grounding=log_grounding,
            classifier=classifier,
            classifier_mode=ClassifierMode(settings.guard_classifier_mode),
            classifier_pidgin_mode=(
                ClassifierMode(settings.guard_classifier_pidgin_mode)
                if settings.guard_classifier_pidgin_mode
                else None
            ),
            compose_policy=settings.compose_policy([lang.code for lang in registry.enabled]),
        )
        closers.append(engine.aclose)  # closers run in reverse: shadow checks drain before the checker stops

        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        store = await SQLiteStore.open(settings.db_path, hasher=hasher)
        closers.append(store.close)

        services = Services(settings, registry, engine, store, hasher, llm, closers=closers)
        # Always built (erasure and retention must work even while collection is off).
        services.donations = VoiceVault(
            settings.voice_dir, store, Sealer(settings.voice_secret()), max_per_day=settings.voice_max_per_day,
            max_bytes=int(settings.voice_max_gb * 1024**3),
        )  # fmt: skip
        whatsapp = None
        if settings.whatsapp_enabled:
            try:
                whatsapp = WhatsAppConfig.from_env()
            except ValueError as exc:
                log.warning("WhatsApp channel disabled: %s", exc)
        if whatsapp is not None:
            limits = Limits(
                per_user_per_hour=settings.per_user_per_hour,
                active_user_cap=settings.active_user_cap,
                turn_deadline_s=settings.turn_deadline_s,
                default_language=settings.default_language,
                retention_days=settings.retention_days,
                pattern_retention_days=settings.pattern_retention_days,
            )
            services.whatsapp = whatsapp
            services.conversation = Conversation(
                engine=engine,
                store=store,
                messenger=WhatsAppMessenger(whatsapp, client=client),
                hasher=hasher,
                limits=limits,
                privacy_url=f"{settings.public_url.rstrip('/')}/privacy" if settings.public_url else "",
                donations=services.donations,
                accept_donations=settings.voice_donation,
                storage=settings.storage_location,
                admission=services.admission,
            )
        services.background.append(
            asyncio.create_task(
                purge_loop(
                    store,
                    settings.retention_days,
                    pattern_retention_days=settings.pattern_retention_days,
                    donations=services.donations,
                    voice_retention_days=settings.voice_retention_days,
                )
            )
        )
        return services
    except BaseException:
        for close in reversed(closers):
            with contextlib.suppress(Exception):
                await close()
        raise
