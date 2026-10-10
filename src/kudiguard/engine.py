"""The check engine: a message (text or transcript) in one language in, a structured, grounded answer out.

Stateless and channel-agnostic: WhatsApp, the web page, the public REST API and (later) a call-centre agent all call
the same engine. Conversation state, consent, commands and rendering live in ``kudiguard.pipeline``.

Flow (PLAN §3): greeting/content-free shortcut → ANALYSE (N-ATLAS, JSON) → translate fallback → deterministic verdict
policy (``safety.decision``) → knowledge search → COMPOSE (N-ATLAS) → clean + output guard, retry once → fixed
fallback text. The model never decides the verdict and never writes contacts; it can only make the answer better.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from kudiguard.commands import CommandMatcher
from kudiguard.domain import Flag, LanguageCode, Outcome, Verdict
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.models import Contact, KnowledgeResult, Query
from kudiguard.knowledge.service import KnowledgeBase
from kudiguard.languages import PIVOT, LanguageRegistry
from kudiguard.llm.base import DEFAULT_TIMEOUTS, LLM, ChatRequest, Completion, LLMError
from kudiguard.prompts import (
    MAX_ENTRIES,
    MAX_STEPS,
    Analysis,
    Intent,
    analyse_request,
    clean_reply,
    clean_translation,
    compose_request,
    parse_analysis,
    steps_request,
    translate_request,
    warn_family_request,
)
from kudiguard.render import REPORT_VERDICTS
from kudiguard.safety.decision import decide, is_content_free, resolve_analysis
from kudiguard.safety.output_guard import OutputGuard, OutputIssue
from kudiguard.speech.base import ASR
from kudiguard.speech.service import SpeechResult, transcribe_upload

log = logging.getLogger(__name__)

RETRY_TEMPERATURE = 0.5  # a retry at the base temperature (0, greedy) would repeat the blocked output exactly
# Minimum time left in the turn to start a model step (else: skip it and use the deterministic/fixed path).
MIN_ANALYSE_S = 2.0
MIN_GENERATE_S = 3.0  # a short reply takes ~2-3 s on one GPU
MIN_RETRY_S = 5.0  # a retry is optional: only when there is clearly time


class AnswerKind(StrEnum):
    CHECK = "check"  # verdict + body (model-composed or fixed fallback)
    GREETING = "greeting"  # greetings/thanks: fixed reply, no model call
    CONTENT_FREE = "content_free"  # emoji, "?", single characters: fixed reply, no model call


@dataclass(frozen=True, slots=True)
class Answer:
    language: LanguageCode
    kind: AnswerKind
    outcome: Outcome
    verdict: Verdict = Verdict.NOT_A_CHECK
    flags: tuple[Flag, ...] = ()
    intent: Intent | None = None
    english: str = ""  # English pivot text ("" when no usable translation)
    body: str = ""  # in the user's language: cleaned model text or fixed fallback text
    contacts: tuple[Contact, ...] = ()  # verified report channels, scam verdicts only
    actions: tuple[str, ...] = ()  # English next steps from the knowledge base
    sources: tuple[str, ...] = ()  # knowledge entry ids used for grounding
    issues: tuple[OutputIssue, ...] = ()  # output-guard findings that forced a retry or fallback
    llm_ms: int = 0


@dataclass(frozen=True, slots=True)
class FollowUp:
    text: str
    outcome: Outcome
    llm_ms: int = 0


@dataclass(slots=True)
class _Generation:
    text: str | None = None
    ms: int = 0
    issues: tuple[OutputIssue, ...] = ()
    unavailable: bool = False  # the model could not be reached (skip further calls this turn)


class Engine:
    def __init__(
        self,
        *,
        registry: LanguageRegistry,
        llm: LLM,
        knowledge: KnowledgeBase,
        asr: ASR | None = None,
        ffmpeg: str | None = None,
        temp_dir: Path | None = None,
        max_concurrent_transcriptions: int = 4,
    ) -> None:
        self.registry = registry
        self.messages = Messages(registry)
        # One limit for every caller (WhatsApp, API, web): decoding + recognition are CPU-bound. A transcription
        # cancelled by a deadline finishes in its worker thread, so briefly more may run under timeouts.
        self.speech_slots = asyncio.Semaphore(max_concurrent_transcriptions)
        self._llm = llm
        self._knowledge = knowledge
        self._asr = asr
        self._ffmpeg = ffmpeg
        self._temp_dir = temp_dir
        self._guard = OutputGuard(registry)
        self._matcher = CommandMatcher(registry)

    @property
    def has_speech(self) -> bool:
        return self._asr is not None and self._ffmpeg is not None

    def topics(self) -> list[dict[str, Any]]:
        """Knowledge entries answers are grounded on (title, source pack, flags), for API discovery."""
        topics: list[dict[str, Any]] = []
        for source in self._knowledge.sources:
            for entry in getattr(source, "entries", ()):
                topics.append(
                    {
                        "id": entry.id,
                        "title": entry.title,
                        "pack": source.manifest.id,
                        "flags": [f.value for f in entry.flags],
                    }
                )
        return topics

    # --- speech -------------------------------------------------------------------------------------------

    async def transcribe(self, data: bytes, language: LanguageCode) -> SpeechResult:
        """Raises ``AudioError`` (user-facing) or ``ASRError`` (system fault)."""
        if self._asr is None or self._ffmpeg is None:
            raise RuntimeError("speech is not configured (ASR and ffmpeg are required)")
        async with self.speech_slots:
            return await transcribe_upload(
                self._asr, data, language, ffmpeg=self._ffmpeg, temp_dir=self._temp_dir
            )

    # --- check --------------------------------------------------------------------------------------------

    async def check(self, text: str, language: LanguageCode, *, deadline: float | None = None) -> Answer:
        """``deadline`` (``time.monotonic()`` seconds): model steps that no longer fit are skipped, so the
        deterministic verdict with fixed text is always delivered in time."""
        lexicon = self.registry.lexicon_for(language)
        if is_content_free(text):
            return Answer(language, AnswerKind.CONTENT_FREE, Outcome.FIXED_REPLY)
        greeting = self._matcher.match(text).greeting_language is not None
        if greeting and not decide(text, Analysis("", Intent.GREETING, ()), lexicon).flags:
            return Answer(language, AnswerKind.GREETING, Outcome.FIXED_REPLY, intent=Intent.GREETING)

        analysis, ms, model_down = await self.analyse(text, language, deadline=deadline)
        assessment = decide(text, analysis, lexicon)
        if analysis.intent is Intent.GREETING and not assessment.flags:
            return Answer(
                language, AnswerKind.GREETING, Outcome.FIXED_REPLY, intent=Intent.GREETING, llm_ms=ms
            )

        knowledge = await self._knowledge.search(
            Query(text=analysis.english, flags=assessment.flags, language=language)
        )
        verdict = assessment.verdict
        contacts = tuple(c for c in knowledge.contacts if c.verified) if verdict in REPORT_VERDICTS else ()
        generation = _Generation(unavailable=model_down)
        if analysis.english and not model_down:
            request = compose_request(
                language_name=self.registry.get(language).name,
                english=analysis.english,
                verdict=verdict,
                flags=assessment.flags,
                knowledge=knowledge,
            )
            generation = await self._generate(request, language, deadline)
        body, outcome = generation.text, Outcome.ANSWERED
        if body is None:
            body, outcome = self._fallback_body(verdict, language), Outcome.FALLBACK
        return Answer(
            language=language,
            kind=AnswerKind.CHECK,
            outcome=outcome,
            verdict=verdict,
            flags=assessment.flags,
            intent=analysis.intent,
            english=analysis.english,
            body=body,
            contacts=contacts,
            actions=_actions(knowledge),
            sources=tuple(item.entry.id for item in knowledge.entries[:MAX_ENTRIES]),
            issues=generation.issues,
            llm_ms=ms + generation.ms,
        )

    async def analyse(
        self, text: str, language: LanguageCode, *, deadline: float | None = None
    ) -> tuple[Analysis, int, bool]:
        """ANALYSE → translate fallback → policy resolution (shared with ``scripts/eval_prompts.py``).

        Returns (analysis, model ms, model unavailable). Never raises for model failures: the rules still run.
        """
        name = self.registry.get(language).name
        ms = 0
        try:
            completion = await self._complete(analyse_request(text, name), deadline, MIN_ANALYSE_S)
        except LLMError as exc:
            log.warning("analyse failed, rules only: %s", exc)
            return resolve_analysis(_NO_ANALYSIS, text, language), ms, True
        ms += completion.latency_ms + completion.queued_ms
        analysis = parse_analysis(completion.text, original=text, source_language=language) or _NO_ANALYSIS
        fallback: str | None = None
        if language != PIVOT and (analysis.untranslated or not analysis.english):
            try:
                translated = await self._complete(translate_request(text, name), deadline, MIN_ANALYSE_S)
                ms += translated.latency_ms + translated.queued_ms
                fallback = clean_translation(translated.text) or None
            except LLMError as exc:
                log.warning("translate fallback failed: %s", exc)
            if fallback is None:  # no usable English: rules on the original text only, no model flags
                analysis = dataclasses.replace(analysis, english="", flags=(), untranslated=True)
        return resolve_analysis(analysis, text, language, fallback_translation=fallback), ms, False

    async def _complete(self, request: ChatRequest, deadline: float | None, minimum_s: float) -> Completion:
        """One model call capped to the time left; raises ``LLMError`` without calling when too little is left."""
        if deadline is not None:
            left = deadline - time.monotonic()
            if left < minimum_s:
                raise LLMError(f"{request.purpose}: skipped, {left:.1f}s left in the turn")
            budget = min(request.timeout_s or DEFAULT_TIMEOUTS[request.purpose], left)
            request = dataclasses.replace(request, timeout_s=budget)
        return await self._llm.complete(request)

    async def _generate(
        self, request: ChatRequest, language: LanguageCode, deadline: float | None = None
    ) -> _Generation:
        """One model reply, cleaned and guarded; retried once (warmer) on a guard hit; None text = use fixed text."""
        result = _Generation()
        for attempt in range(2):
            current = request if attempt == 0 else dataclasses.replace(request, temperature=RETRY_TEMPERATURE)
            try:
                completion = await self._complete(
                    current, deadline, MIN_GENERATE_S if attempt == 0 else MIN_RETRY_S
                )
            except LLMError as exc:
                log.warning("%s failed: %s", request.purpose, exc)
                result.unavailable = True
                return result
            result.ms += completion.latency_ms + completion.queued_ms
            text = clean_reply(completion.text, truncated=completion.truncated)
            issues = self._guard.check(text, language)
            if not issues:
                result.text = text
                return result
            result.issues += issues
            log.info("%s blocked by output guard (attempt %d): %s", request.purpose, attempt + 1, issues)
        return result

    def _fallback_body(self, verdict: Verdict, language: LanguageCode) -> str:
        if verdict in REPORT_VERDICTS:
            return self.messages.text(language, Msg.FALLBACK_SCAM)
        if verdict is Verdict.NO_FLAGS:
            return ""  # the fixed headline already says "no red flags found, still confirm with your bank"
        return self.messages.text(language, Msg.FALLBACK_ANSWER)

    # --- follow-ups (reply buttons) -----------------------------------------------------------------------

    async def next_steps(
        self, english: str, flags: Sequence[Flag], language: LanguageCode, *, deadline: float | None = None
    ) -> FollowUp:
        """ "What should I do?": the knowledge base's steps for this situation, in the user's language."""
        knowledge = await self._knowledge.search(Query(text=english, flags=tuple(flags), language=language))
        actions = _actions(knowledge)[:MAX_STEPS]
        if not actions:
            return FollowUp(self.messages.text(language, Msg.FALLBACK_STEPS), Outcome.FALLBACK)
        if language == PIVOT:
            return FollowUp("\n".join(f"{n}. {a}" for n, a in enumerate(actions, 1)), Outcome.ANSWERED)
        name = self.registry.get(language).name
        generation = await self._generate(
            steps_request(language_name=name, actions=actions), language, deadline
        )
        if generation.text is None:
            return FollowUp(self.messages.text(language, Msg.FALLBACK_STEPS), Outcome.FALLBACK, generation.ms)
        return FollowUp(generation.text, Outcome.ANSWERED, generation.ms)

    async def report_contacts(
        self, english: str, flags: Sequence[Flag], language: LanguageCode
    ) -> tuple[Contact, ...]:
        """ "Where do I report?": verified contacts, the ones relevant to these red flags first."""
        knowledge = await self._knowledge.search(Query(text=english, flags=tuple(flags), language=language))
        return tuple(c for c in knowledge.contacts if c.verified)

    async def warning(
        self, english: str, flags: Sequence[Flag], language: LanguageCode, *, deadline: float | None = None
    ) -> FollowUp:
        """ "Warn my family": a short forwardable warning in the user's language."""
        ms = 0
        if english:
            name = self.registry.get(language).name
            request = warn_family_request(language_name=name, english=english, flags=tuple(flags))
            generation = await self._generate(request, language, deadline)
            if generation.text is not None:
                return FollowUp(generation.text, Outcome.ANSWERED, generation.ms)
            ms = generation.ms
        return FollowUp(self.messages.text(language, Msg.WARN_FAMILY_FALLBACK), Outcome.FALLBACK, ms)


_NO_ANALYSIS = Analysis(english="", intent=Intent.OTHER, flags=(), untranslated=True)


def _actions(knowledge: KnowledgeResult) -> tuple[str, ...]:
    return tuple(dict.fromkeys(a for item in knowledge.entries[:MAX_ENTRIES] for a in item.entry.actions))
