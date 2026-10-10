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
from datetime import timedelta

import httpx

from kudiguard.config import Settings
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
from kudiguard.speech.audio import find_ffmpeg
from kudiguard.storage.base import Store, utcnow
from kudiguard.storage.sqlite import SQLiteStore

log = logging.getLogger(__name__)

PURGE_INTERVAL_S = 6 * 3600


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
    closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    background: list[asyncio.Task[None]] = field(default_factory=list)

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
) -> None:
    """Delete evidence (and, on their own clock, scam examples) older than their retention periods (NDPA)."""
    while True:
        try:
            now = utcnow()
            removed = await store.purge_before(now - timedelta(days=retention_days))
            removed += await store.purge_patterns_before(
                (now - timedelta(days=pattern_retention_days)).date()
            )
            if removed:
                log.info("retention purge removed %d rows", removed)
        except Exception:
            log.exception("retention purge failed")
        await asyncio.sleep(interval_s)


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
    registry = LanguageRegistry.load(settings.data_dir / "languages")
    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    closers: list[Callable[[], Awaitable[None]]] = [client.aclose]
    try:

        def knowledge_with(embedder: object | None) -> KnowledgeBase:
            return knowledge_base_from_packs(
                settings.data_dir / "packs",
                client=client,
                include=settings.packs or None,
                embedder=embedder,  # type: ignore[arg-type]
                allow_samples=settings.allow_sample_packs,
            )

        knowledge = None
        embedder = None
        if settings.embeddings:
            try:
                from kudiguard.knowledge.embeddings import FastEmbedder

                candidate = FastEmbedder(cache_dir=settings.embeddings_cache)
                knowledge = knowledge_with(candidate)
                embedder = candidate
            except KnowledgeConfigError:
                raise  # a broken pack is a configuration error, not a missing model
            except Exception as exc:  # model not cached / not installed: keyword search still works
                log.warning("semantic search disabled (keyword search only): %s", exc)
        if knowledge is None:
            knowledge = knowledge_with(None)
        if failed := await knowledge.refresh_all():
            log.warning("knowledge sources unavailable at startup: %s", sorted(failed))
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
        )
        closers.append(engine.aclose)  # closers run in reverse: shadow checks drain before the checker stops

        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        store = await SQLiteStore.open(settings.db_path, hasher=hasher)
        closers.append(store.close)

        services = Services(settings, registry, engine, store, hasher, llm, closers=closers)
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
            )
            services.whatsapp = whatsapp
            services.conversation = Conversation(
                engine=engine,
                store=store,
                messenger=WhatsAppMessenger(whatsapp, client=client),
                hasher=hasher,
                limits=limits,
                privacy_url=f"{settings.public_url.rstrip('/')}/privacy" if settings.public_url else "",
            )
        services.background.append(
            asyncio.create_task(
                purge_loop(
                    store, settings.retention_days, pattern_retention_days=settings.pattern_retention_days
                )
            )
        )
        return services
    except BaseException:
        for close in reversed(closers):
            with contextlib.suppress(Exception):
                await close()
        raise
