"""The check engine: a message (text or transcript) in one language in, a structured, grounded answer out.

Stateless and channel-agnostic: WhatsApp, the web page, the public REST API and (later) a call-centre agent all call
the same engine. Conversation state, consent, commands and rendering live in ``kudiguard.pipeline``.

Flow (PLAN §3): greeting/content-free shortcut → ANALYSE (N-ATLAS, JSON) → translate fallback → deterministic verdict
policy (``safety.decision``) → knowledge search → COMPOSE (N-ATLAS) → clean + output guard, retry once → grounding
check (``safety.grounding``; shadow: run after the reply, logged; enforce: unsupported steps dropped or fixed
text, no retry).
The model never decides the verdict and never writes contacts; it can only make the answer better.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable, Sequence
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
from kudiguard.llm.base import LLM, ChatRequest, Completion, LLMError, complete_within
from kudiguard.prompts import (
    MAX_ENTRIES,
    MAX_STEPS,
    SYSTEM_PROMPTS,
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
from kudiguard.reply_translation import (
    DEFAULT_COMPOSE_STRATEGY,
    RETRYABLE_PROBLEMS,
    ComposePolicy,
    ComposeStrategy,
    problem_kind,
    reply_translate_request,
    strip_reply_tags,
    translation_problems,
)
from kudiguard.reply_translation import SYSTEM_PROMPTS as TRANSLATION_PROMPTS
from kudiguard.safety.abuse import (
    INJECTIONS,
    Category,
    InputScreen,
    ScreenMode,
    TranscriptFilter,
    log_screening,
)
from kudiguard.safety.decision import decide, is_content_free, resolve_analysis
from kudiguard.safety.grounding import (
    GroundingChecker,
    GroundingMode,
    GroundingReport,
    evidence_for,
    without_unsupported_steps,
)
from kudiguard.safety.guard_classifier import ClassifierMode, UnsafeAdviceClassifier, is_pidgin
from kudiguard.safety.output_guard import OutputGuard, OutputIssue
from kudiguard.safety.reply_screen import ReplyScreen
from kudiguard.safety.scam_rules import assess
from kudiguard.speech.base import ASR, AudioError, AudioProblem
from kudiguard.speech.service import SpeechResult, transcribe_upload

log = logging.getLogger(__name__)

GroundingHook = Callable[[str | None, GroundingReport | None, int], None]

RETRY_TEMPERATURE = 0.5  # a retry at the base temperature (0, greedy) would repeat the blocked output exactly
# Minimum time left in the turn to start a model step (else: skip it and use the deterministic/fixed path).
MIN_ANALYSE_S = 2.0
MIN_GENERATE_S = 3.0  # a short reply takes ~2-3 s on one GPU
MIN_RETRY_S = 5.0  # a retry is optional: only when there is clearly time
# english_first: a reply translation takes p50 2-3 s, p95 ~5 s, up to 9 s for long drafts (docs/SPIKES.md H). With
# less than ~5 s left a translation would often be cut off by the deadline (and wasted GPU time), so english_first
# starts only with MIN_GENERATE_S + MIN_TRANSLATE_S (8 s) left, else it sends fixed text in the language at once.
MIN_TRANSLATE_S = 5.0
SHADOW_DRAIN_S = 1.0  # shutdown wait for shadow checks when the checker has no budget
MIN_GROUNDING_S = 0.5  # the English check takes tens of ms (CPU); it must never make a turn miss its deadline


class AnswerKind(StrEnum):
    CHECK = "check"  # verdict + body (model-composed or fixed fallback)
    GREETING = "greeting"  # greetings/thanks: fixed reply, no model call
    CONTENT_FREE = "content_free"  # emoji, "?", single characters: fixed reply, no model call
    # crisis / abuse / scam-tool request (safety.abuse, enforce mode): fixed reply in ``body``, no model call
    SCREENED = "screened"


# The fixed reply for each screen that answers with it alone (crisis: the crisis reply first, then the rules'
# check if the message is also a scam; injection: the rules' answer).
SCREEN_REPLY: dict[Category, Msg] = {
    Category.ABUSE: Msg.ABUSE_REPLY,
    Category.SCAM_TOOL: Msg.SCAM_TOOL_REPLY,
}


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
    grounding_ms: int = 0  # wall time the grounding check added (0 = not run), separate from model time
    grounded: bool | None = None  # grounding verdict on the composed reply (None = not checked)
    unsupported_claims: int = 0
    # english_first: the checked English reply that was translated (kept when the translation was rejected and
    # ``body`` is fixed text); "" for direct replies
    reply_english: str = ""
    # english_first: why translations were rejected ("negation_lost", "steps", ...), in order; () = none rejected
    translation_rejections: tuple[str, ...] = ()
    # abuse-screen categories found in the message (shadow and enforce), in priority order
    screened: tuple[str, ...] = ()
    screen_action: str | None = None  # the category acted on (enforce mode), e.g. "crisis", "injection"
    # sent before `body` (the crisis reply, when a crisis message is also a scam to warn about); "" = none
    preface: str = ""


@dataclass(frozen=True, slots=True)
class FollowUp:
    text: str
    outcome: Outcome
    llm_ms: int = 0
    translation_rejections: tuple[str, ...] = ()  # english_first: as ``Answer.translation_rejections``


@dataclass(slots=True)
class _Generation:
    text: str | None = None
    ms: int = 0
    issues: tuple[OutputIssue, ...] = ()
    unavailable: bool = False  # the model could not be reached (skip further calls this turn)
    english: str | None = None  # english_first: the checked English reply that was translated
    rejections: tuple[str, ...] = ()  # english_first: why translations were rejected
    report: GroundingReport | None = (
        None  # enforce-mode grounding of the composed text (English when translated)
    )
    grounding_ms: int = 0


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
        grounding: GroundingChecker | None = None,
        grounding_mode: GroundingMode = GroundingMode.SHADOW,
        on_grounding: GroundingHook | None = None,
        classifier: UnsafeAdviceClassifier | None = None,
        classifier_mode: ClassifierMode = ClassifierMode.SHADOW,
        classifier_pidgin_mode: ClassifierMode | None = None,
        compose_policy: ComposePolicy | None = None,
        screen: InputScreen | None = None,
        reply_screen: ReplyScreen | None = None,
        transcripts: TranscriptFilter | None = None,
    ) -> None:
        """``compose_policy``: direct or English-first replies per language (default:
        ``reply_translation.DEFAULT_COMPOSE_STRATEGY``, Yoruba/Igbo English-first, Hausa direct).
        ``screen``, ``reply_screen``, ``transcripts``: the malicious-use defences (docs/ABUSE.md); None = each at
        its default modes (as in production with no settings changed)."""
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
        self.grounding = grounding if grounding_mode is not GroundingMode.OFF else None
        self.grounding_mode = grounding_mode
        # Hook for persisting results (e.g. onto the stored interaction): called with (turn_id, report, ms) after
        # every check, also the shadow checks that finish after the reply was sent.
        self.on_grounding = on_grounding
        # Pidgin replies arrive as "en": ``classifier_pidgin_mode`` (None = the same mode) lets them run in shadow
        # while English is enforced, or the other way round
        self.classifier_mode = classifier_mode
        self.classifier_pidgin_mode = (
            classifier_mode if classifier_pidgin_mode is None else classifier_pidgin_mode
        )
        modes = {self.classifier_mode, self.classifier_pidgin_mode}
        self.classifier = classifier if modes != {ClassifierMode.OFF} else None
        self._shadow: set[asyncio.Task[None]] = set()
        self._matcher = CommandMatcher(registry)
        self.compose_policy = compose_policy or ComposePolicy.parse(DEFAULT_COMPOSE_STRATEGY)
        self.screen = screen or InputScreen(registry)
        self.reply_screen = reply_screen or ReplyScreen(registry, SYSTEM_PROMPTS + TRANSLATION_PROMPTS)
        self.transcripts = transcripts or TranscriptFilter(registry)

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
            speech = await transcribe_upload(
                self._asr, data, language, ffmpeg=self._ffmpeg, temp_dir=self._temp_dir
            )
        # Whisper-style text invented on music or noise ("thanks for watching") is not a message (docs/ABUSE.md)
        filtered = self.transcripts.filter(speech.transcript.text, language)
        if not filtered.removed:
            return speech
        if filtered.empty:
            raise AudioError(AudioProblem.NO_SPEECH, "only recogniser hallucinations")
        transcript = dataclasses.replace(speech.transcript, text=filtered.text)
        return dataclasses.replace(speech, transcript=transcript, filtered=True)

    # --- check --------------------------------------------------------------------------------------------

    async def check(
        self, text: str, language: LanguageCode, *, deadline: float | None = None, turn_id: str | None = None
    ) -> Answer:
        """``deadline`` (``time.monotonic()`` seconds): model steps that no longer fit are skipped, so the
        deterministic verdict with fixed text is always delivered in time."""
        lexicon = self.registry.lexicon_for(language)
        if is_content_free(text):
            return Answer(language, AnswerKind.CONTENT_FREE, Outcome.FIXED_REPLY)
        greeting = self._matcher.match(text).greeting_language is not None
        if greeting and not decide(text, Analysis("", Intent.GREETING, ()), lexicon).flags:
            return Answer(language, AnswerKind.GREETING, Outcome.FIXED_REPLY, intent=Intent.GREETING)

        screening = self.screen.screen(text, language, flagged=bool(assess([text], lexicon).flags))
        log_screening(screening, turn_id, language)
        screened = tuple(c.value for c in screening.detected)
        action = screening.enforced.value if screening.enforced else None
        if screening.enforced is not None and (key := SCREEN_REPLY.get(screening.enforced)) is not None:
            body = self.messages.text(language, key)
            return Answer(language, AnswerKind.SCREENED, Outcome.FIXED_REPLY, body=body, screened=screened,
                          screen_action=action)  # fmt: skip
        # Crisis: the full check continues after the crisis reply (a victim who "gave them my OTP" needs the model's
        # reading of the message: the rules alone missed it live); only a message with nothing to warn about gets
        # the crisis reply alone.
        crisis = screening.enforced is Category.CRISIS
        if screening.enforced in INJECTIONS:
            # The rules answer alone (with a translation for the rules): an injected message never reaches a model
            # that writes the reply.
            analysis, ms = await self._rules_only_analysis(text, language, deadline)
            model_down = True
        else:
            analysis, ms, model_down = await self.analyse(text, language, deadline=deadline)
        assessment = decide(text, analysis, lexicon)
        if (
            crisis and assessment.verdict not in REPORT_VERDICTS
        ):  # nothing to warn about: the crisis reply alone
            body = self.messages.text(language, Msg.CRISIS_REPLY)
            return Answer(language, AnswerKind.SCREENED, Outcome.FIXED_REPLY, body=body, screened=screened,
                          screen_action=action, llm_ms=ms)  # fmt: skip
        if screening.enforced is Category.INJECTION and assessment.verdict is Verdict.NO_FLAGS:
            # A message that clearly tries to steer the checker is itself a warning sign: never "no red flags".
            # (Not for the broad, suspect phrases: genuine messages use them too.)
            assessment = dataclasses.replace(assessment, verdict=Verdict.CAUTION)
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
        report: GroundingReport | None = None
        grounding_ms = 0
        if analysis.english and not model_down:
            evidence = evidence_for(knowledge, verdict=verdict, flags=assessment.flags)
            generation = await self._compose(
                language, analysis.english, verdict, assessment.flags, knowledge, deadline, evidence, turn_id,
                sources=(text, analysis.english),
            )  # fmt: skip
            report, grounding_ms = generation.report, generation.grounding_ms
            if generation.text is not None:
                # Shadow checks see the English text when there is one (english_first: the draft that was
                # translated), else the reply in its own language.
                checked, checked_language = (
                    (generation.english, PIVOT) if generation.english else (generation.text, language)
                )
                self._classify_in_background(checked, checked_language, turn_id)
                if self.grounding_mode is GroundingMode.SHADOW:
                    self._ground_in_background(checked, checked_language, evidence, analysis.english, turn_id)
        if generation.text is not None:
            body, outcome = generation.text, Outcome.ANSWERED
        elif screening.enforced in INJECTIONS and verdict is Verdict.NOT_A_CHECK:
            # not "try again later" (a retry is screened again) and not the greeting (it would look ignored)
            body, outcome = self.messages.text(language, Msg.SCREEN_NEUTRAL), Outcome.FIXED_REPLY
        else:
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
            grounding_ms=grounding_ms,
            grounded=None if report is None else report.grounded,
            unsupported_claims=0 if report is None else len(report.unsupported),
            reply_english=generation.english or "",
            translation_rejections=generation.rejections,
            screened=screened,
            screen_action=action,
            preface=self.messages.text(language, Msg.CRISIS_REPLY) if crisis else "",
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

    async def _rules_only_analysis(
        self, text: str, language: LanguageCode, deadline: float | None
    ) -> tuple[Analysis, int]:
        """Injection screen in enforce mode: no ANALYSE (its intent could lower the verdict) and no COMPOSE. A message
        not in English is translated first, because the English rules need English: the translation is only ever
        read by the rules (which also run on the original), never shown, so a manipulated translation can at worst
        add nothing."""
        if language == PIVOT:
            return resolve_analysis(_NO_ANALYSIS, text, language), 0
        name = self.registry.get(language).name
        try:
            completion = await self._complete(translate_request(text, name), deadline, MIN_ANALYSE_S)
        except LLMError as exc:
            log.warning("rules-only translation failed: %s", exc)
            return _NO_ANALYSIS, 0
        fallback = clean_translation(completion.text) or None
        analysis = resolve_analysis(_NO_ANALYSIS, text, language, fallback_translation=fallback)
        return analysis, completion.latency_ms + completion.queued_ms

    async def _complete(self, request: ChatRequest, deadline: float | None, minimum_s: float) -> Completion:
        """One model call capped to the time left; raises ``LLMError`` without calling when too little is left."""
        return await complete_within(self._llm, request, deadline, minimum_s)

    # --- compose strategies (reply_translation) ---------------------------------------------------------------

    async def _compose(
        self,
        language: LanguageCode,
        english: str,
        verdict: Verdict,
        flags: tuple[Flag, ...],
        knowledge: KnowledgeResult,
        deadline: float | None,
        evidence: Sequence[str] = (),
        turn_id: str | None = None,
        *,
        sources: Sequence[str] = (),
    ) -> _Generation:
        """The reply body: composed in ``language`` (direct) or in English, checked, then translated.

        English-first: the output guard and the classifier in enforce mode run on the English draft inside
        ``_generate(…, PIVOT)``; grounding in enforce mode runs on it here (unsupported steps trimmed from the English
        before translating); shadow-mode checks run in ``check`` on ``_Generation.english``."""
        direct = self.compose_policy.for_language(language) is ComposeStrategy.DIRECT
        target = self.registry.get(language).name
        request = compose_request(
            language_name=target if direct else self.registry.get(PIVOT).name,
            english=english,
            verdict=verdict,
            flags=flags,
            knowledge=knowledge,
        )
        if direct:
            generation = await self._generate(request, language, deadline, sources)
            return await self._enforce_grounding(generation, language, evidence, english, deadline, turn_id)
        if deadline is not None and deadline - time.monotonic() < MIN_GENERATE_S + MIN_TRANSLATE_S:
            # Not enough time for both steps: fixed text in the user's language now, rather than an English reply
            # (the user chose a language) or a translation cut off by the deadline.
            log.warning("english_first: %.1fs left, using fixed text", deadline - time.monotonic())
            # Not a model failure: no call is made for lack of time; "unavailable" gives the same fixed-text path.
            return _Generation(unavailable=True, rejections=("no_time",))
        draft = await self._generate(request, PIVOT, deadline, sources)
        draft = await self._enforce_grounding(draft, PIVOT, evidence, english, deadline, turn_id)
        if draft.text is None:
            return draft
        translated = await self._translate_reply(
            reply_translate_request(draft.text, target), draft.text, language, deadline, sources
        )
        translated.ms += draft.ms
        translated.issues = draft.issues + translated.issues
        translated.english = draft.text
        translated.report, translated.grounding_ms = draft.report, draft.grounding_ms
        return translated

    async def _enforce_grounding(
        self,
        generation: _Generation,
        language: LanguageCode,
        evidence: Sequence[str],
        message: str,
        deadline: float | None,
        turn_id: str | None,
    ) -> _Generation:
        """Enforce mode: ground the composed text; an ungrounded English reply loses its unsupported steps, any
        other ungrounded reply becomes fixed text (None)."""
        if generation.text is None or self.grounding_mode is not GroundingMode.ENFORCE:
            return generation
        report, generation.grounding_ms = await self._ground(
            generation.text, language, evidence, message, deadline, turn_id
        )
        generation.report = report
        if report is not None and not report.grounded:
            generation.issues += (OutputIssue.UNGROUNDED,)
            generation.text = (
                without_unsupported_steps(generation.text, report) if language == PIVOT else None
            )
        return generation

    async def _translate_reply(
        self,
        request: ChatRequest,
        english: str,
        language: LanguageCode,
        deadline: float | None,
        sources: Sequence[str] = (),
    ) -> _Generation:
        """The English reply in ``language``: cleaned, guarded (the language's patterns: a second net) and checked
        for fidelity (``translation_problems``). Only a formatting failure (echoed tags, untranslated echo, length)
        is retried, warmer; a guard hit or a content failure (lost/added negation, steps, numbers, links) goes
        straight to fixed text: a warm retry let "do not forget to steal money" through in review."""
        result = _Generation()
        for attempt in range(2):
            current = request if attempt == 0 else dataclasses.replace(request, temperature=RETRY_TEMPERATURE)
            try:
                completion = await self._complete(
                    current, deadline, MIN_TRANSLATE_S if attempt == 0 else MIN_RETRY_S
                )
            except LLMError as exc:
                log.warning("reply translation failed: %s", exc)
                result.unavailable = True
                return result
            result.ms += completion.latency_ms + completion.queued_ms
            text = clean_reply(strip_reply_tags(completion.text), truncated=completion.truncated)
            issues = self._checks(text, language, sources)
            blocking = tuple(i for i in issues if self._blocks(i))  # log-only outcome claims do not reject
            problems = translation_problems(english, text, language)
            result.issues += issues
            if not blocking and not problems:
                if issues:
                    log.info("reply translation outcome claim (log only): %s", issues)
                result.text = text
                return result
            result.rejections += tuple(dict.fromkeys(problem_kind(p) for p in problems)) + (
                ("guard",) if blocking else ()
            )
            log.info("reply translation rejected (attempt %d): %s %s", attempt + 1, issues, problems)
            if blocking or not all(problem_kind(p) in RETRYABLE_PROBLEMS for p in problems):
                break
        return result

    async def _generate(
        self,
        request: ChatRequest,
        language: LanguageCode,
        deadline: float | None = None,
        sources: Sequence[str] = (),
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
            issues = self._checks(text, language, sources)
            if not any(self._blocks(i) for i in issues) and await self._classified_unsafe(text, language):
                issues += (OutputIssue.CLASSIFIED_UNSAFE,)
            if not any(self._blocks(i) for i in issues):
                if issues:  # outcome claims outside enforce mode: recorded, not blocking (yet)
                    log.info("%s outcome claim (log only): %s", request.purpose, issues)
                    result.issues += issues
                result.text = text
                return result
            result.issues += issues
            log.info("%s blocked by output guard (attempt %d): %s", request.purpose, attempt + 1, issues)
        return result

    def _checks(self, text: str, language: LanguageCode, sources: Sequence[str]) -> tuple[OutputIssue, ...]:
        """The output guard, then the malicious-use reply checks (persona, prompt leak, echo of `sources`)."""
        return self._guard.check(text, language) + self.reply_screen.check(text, language, sources)

    def _classifier_mode(self, text: str, language: LanguageCode) -> ClassifierMode:
        if self.classifier is None or not self.classifier.supports(language):
            return ClassifierMode.OFF
        return self.classifier_pidgin_mode if is_pidgin(text) else self.classifier_mode

    async def _classified_unsafe(self, text: str, language: LanguageCode) -> bool:
        """Enforce mode only: the learned layer's verdict on a reply the regex guard let through. Waits in the
        classifier's queue within its budget, then fails open."""
        if self.classifier is None or self._classifier_mode(text, language) is not ClassifierMode.ENFORCE:
            return False
        try:
            verdict = await self.classifier.check(text, language, wait=True)
        except Exception as exc:  # optional layer: the regex verdict stands
            log.warning("guard classifier not run, regex verdict only: %s", exc)
            return False
        log.info("guard_classifier mode=enforce flagged=%s p=%.2f ms=%d units=%d unscored=%d", verdict.flagged,
                 verdict.probability, verdict.ms, verdict.scored, verdict.unscored)  # fmt: skip
        return verdict.flagged

    def _classify_in_background(self, text: str, language: LanguageCode, turn_id: str | None) -> None:
        """Shadow mode: score the sent reply after the fact and log it (counts only, never the text)."""
        classifier = self.classifier
        if classifier is None or self._classifier_mode(text, language) is not ClassifierMode.SHADOW:
            return

        async def run() -> None:
            try:
                verdict = await classifier.check(text, language, wait=False)
            except Exception as exc:
                log.warning("guard classifier not run (shadow): %s", exc)
                return
            log.info("guard_classifier turn=%s mode=shadow flagged=%s p=%.2f ms=%d units=%d unscored=%d", turn_id,
                     verdict.flagged, verdict.probability, verdict.ms, verdict.scored, verdict.unscored)  # fmt: skip

        task = asyncio.create_task(run())
        self._shadow.add(task)
        task.add_done_callback(self._shadow.discard)

    def _ground_in_background(
        self, text: str, language: LanguageCode, evidence: Sequence[str], message: str, turn_id: str | None
    ) -> None:
        """Shadow mode: the check runs after the reply is sent, so it adds no time to the turn."""
        if self.grounding is None or not self.grounding.supports(language):
            return

        async def run() -> None:
            await self._ground(text, language, evidence, message, None, turn_id)

        task = asyncio.create_task(run())
        self._shadow.add(task)
        task.add_done_callback(self._shadow.discard)

    async def aclose(self) -> None:
        """Let running shadow checks finish (bounded by the check's budget) before the checker shuts down."""
        if self._shadow:
            budgets = [getattr(self.grounding, "budget_s", None), getattr(self.classifier, "budget_s", None)]
            budget = max((b for b in budgets if b), default=SHADOW_DRAIN_S)
            _, pending = await asyncio.wait(set(self._shadow), timeout=budget)
            for task in pending:  # shutting down: a check past its budget is not worth waiting for
                task.cancel()

    async def _ground(
        self,
        text: str,
        language: LanguageCode,
        evidence: Sequence[str],
        message: str,
        deadline: float | None,
        turn_id: str | None = None,
    ) -> tuple[GroundingReport | None, int]:
        """The grounding check and its wall time in ms; None when it is off, unsupported, out of time or failed.

        Fails open: the reply already passed cleaning and the output guard, so a check that cannot run must not
        replace a good answer. An ungrounded reply is not retried: the same facts tend to produce the same invention
        (and a warmer retry invents more); the fixed text is safe and immediate.
        """
        if self.grounding is None or not self.grounding.supports(language):
            return None, 0
        if deadline is not None and deadline - time.monotonic() < MIN_GROUNDING_S:
            log.warning("grounding check skipped: too little time left in the turn")
            return None, 0
        start = time.perf_counter()
        report: GroundingReport | None = None
        try:
            report = await self.grounding.check(text, language, evidence, message=message, deadline=deadline)
        except Exception as exc:  # optional step: any failure (incl. its time cap) sends the guarded reply
            if self.grounding_mode is GroundingMode.SHADOW:  # the reply has already gone out
                log.warning("grounding check not run (shadow): %s", exc)
            else:
                log.warning("grounding check failed, sending the guarded reply: %s", exc)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log.info(
            "grounding_ms=%d language=%s mode=%s grounded=%s unsupported=%d/%d min_support=%.2f",
            elapsed_ms,
            language,
            self.grounding_mode,
            None if report is None else report.grounded,
            0 if report is None else len(report.unsupported),
            0 if report is None else len(report.claims),
            min((c.support for c in report.claims), default=1.0) if report else 0.0,
        )
        if self.on_grounding is not None:
            try:
                self.on_grounding(turn_id, report, elapsed_ms)
            except Exception:
                log.exception("grounding hook failed")
        return report, elapsed_ms

    def _blocks(self, issue: OutputIssue) -> bool:
        return issue is not OutputIssue.OUTCOME_CLAIM or self.grounding_mode is GroundingMode.ENFORCE

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
        numbered = "\n".join(f"{n}. {a}" for n, a in enumerate(actions, 1))
        if language == PIVOT:
            return FollowUp(numbered, Outcome.ANSWERED)
        name = self.registry.get(language).name
        if self.compose_policy.for_language(language) is ComposeStrategy.ENGLISH_FIRST:
            # The knowledge base's English steps, translated and checked like a reply (negation kept, no new numbers)
            generation = await self._translate_reply(
                reply_translate_request(numbered, name), numbered, language, deadline
            )
        else:
            generation = await self._generate(
                steps_request(language_name=name, actions=actions), language, deadline
            )
        rejections = generation.rejections
        if generation.text is None:
            text = self.messages.text(language, Msg.FALLBACK_STEPS)
            return FollowUp(text, Outcome.FALLBACK, generation.ms, rejections)
        return FollowUp(generation.text, Outcome.ANSWERED, generation.ms, rejections)

    async def report_contacts(
        self, english: str, flags: Sequence[Flag], language: LanguageCode
    ) -> tuple[Contact, ...]:
        """ "Where do I report?": verified contacts, the ones relevant to these red flags first."""
        knowledge = await self._knowledge.search(Query(text=english, flags=tuple(flags), language=language))
        return tuple(c for c in knowledge.contacts if c.verified)

    async def warning(
        self, english: str, flags: Sequence[Flag], language: LanguageCode, *, deadline: float | None = None
    ) -> FollowUp:
        """ "Warn my family": a short forwardable warning in the user's language. A message with injected
        instructions (injection screen in enforce mode) gets the fixed warning: the model never sees it."""
        ms = 0
        rejections: tuple[str, ...] = ()
        if english and self._injection_enforced(english):
            english = ""
        if english:
            name = self.registry.get(language).name
            sources = (english,)
            if self.compose_policy.for_language(language) is ComposeStrategy.ENGLISH_FIRST:
                # Written and guarded in English, then translated and checked like a reply
                request = warn_family_request(
                    language_name=self.registry.get(PIVOT).name, english=english, flags=tuple(flags)
                )
                generation = await self._generate(request, PIVOT, deadline, sources)
                if generation.text is not None:
                    draft = generation.text
                    translated = await self._translate_reply(
                        reply_translate_request(draft, name), draft, language, deadline, sources
                    )
                    translated.ms += generation.ms
                    generation = translated
            else:
                request = warn_family_request(language_name=name, english=english, flags=tuple(flags))
                generation = await self._generate(request, language, deadline, sources)
            rejections = generation.rejections
            if generation.text is not None:
                return FollowUp(generation.text, Outcome.ANSWERED, generation.ms, rejections)
            ms = generation.ms
        text = self.messages.text(language, Msg.WARN_FAMILY_FALLBACK)
        return FollowUp(text, Outcome.FALLBACK, ms, rejections)

    def _injection_enforced(self, english: str) -> bool:
        if self.screen.modes.injection is not ScreenMode.ENFORCE or not self.screen.injected(english, PIVOT):
            return False
        log.info("abuse_screen follow-up: injection in the checked message, fixed text")
        return True


_NO_ANALYSIS = Analysis(english="", intent=Intent.OTHER, flags=(), untranslated=True)


def _actions(knowledge: KnowledgeResult) -> tuple[str, ...]:
    return tuple(dict.fromkeys(a for item in knowledge.entries[:MAX_ENTRIES] for a in item.entry.actions))
