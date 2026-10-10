"""Grounding check: is every sentence of a composed reply supported by the evidence it was composed from?

The reply (in English, or back-translated to English) is split into claims; each claim is compared by meaning
(embedding cosine) with the evidence sentences: the knowledge-base facts and actions the model was shown, the user's
own message, the verdict and red flags, and generic framing that is always acceptable (empathy, "ask your bank",
the reviewed English fixed strings). A claim is supported when its best match reaches the threshold and its specifics
also appear in the evidence: numbers (wrong days, hours, amounts) and names (agencies: "EFCC", "Fraud Bureau"). Those
are the commonest inventions and the ones meaning alone misses: "report it to the EFCC" reads like "report it to
your bank".

What it catches: invented facts, agencies, outcomes ("you will get your money back") and advice the knowledge base
does not give. What it cannot catch: a negated paraphrase of a real fact ("share the code with your bank") scores
like the fact itself; the output guard's safety-claim patterns cover the dangerous cases of that.

Calibration (``data/eval/grounding_claims.yaml``, ``tests/integration/test_grounding_quality.py``): bge-small, English.
Cross-lingual embedders were measured for direct Hausa/Yoruba/Igbo checking and were not good enough (docs/SPIKES.md),
so non-English replies are checked only through a ``Translator`` (N-ATLAS back-translation: offline evaluation, not
the live path, where it would cost a second model call per reply).

Scope: COMPOSE replies only. Follow-ups are not checked: English "what to do" steps are the knowledge base's actions
verbatim, and translated steps and family warnings could only be checked through back-translation.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kudiguard.domain import Flag, LanguageCode, Verdict
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.embeddings import CachedEmbedder, Embedder
from kudiguard.knowledge.models import KnowledgeResult
from kudiguard.languages import PIVOT, LanguageRegistry
from kudiguard.llm.base import LLM, LLMError, complete_within
from kudiguard.prompts import FLAG_TEXT, clean_translation, facts_block, translate_request
from kudiguard.text import fold

log = logging.getLogger(__name__)

# bge-small-en-v1.5 cosine, claim vs best evidence sentence. On data/eval/grounding_claims.yaml every supported
# paraphrase scored >= 0.765 and most inventions < 0.75 (the specifics check catches most of the rest). Set below the
# lowest paraphrase on purpose: a false flag replaces a good answer, and N-ATLAS paraphrases more loosely than the set.
DEFAULT_THRESHOLD = 0.75
MIN_CLAIM_WORDS = 3  # "Stay calm." / "Be careful!" carry no checkable fact
MIN_TRANSLATE_S = 2.0  # back-translation is a full model call
DEFAULT_BUDGET_S = 0.15  # live cap on scoring; measured p50 12 ms / p95 15 ms per English reply (SPIKES S8)

# Always acceptable, whatever the facts: COMPOSE is told to explain calmly and to suggest the bank when the facts
# do not cover the question.
GENERIC_FRAMING: tuple[str, ...] = (
    "I understand your worry.",
    "You did the right thing to check first.",
    "If the facts do not cover your question, ask your bank directly.",
)
# The meaning of each verdict, as the reply may restate it (the fixed headline already says it in full).
VERDICT_LINE: dict[Verdict, str] = {
    Verdict.LIKELY_SCAM: "This looks like a scam.",
    Verdict.CAUTION: "There are warning signs. Check with your bank directly before you act.",
    Verdict.NO_FLAGS: "No warning signs were found. Still confirm with your bank before you act.",
}
# Reviewed English fixed text: safe generic advice ("call your bank on the number printed on your card").
FIXED_ADVICE: tuple[Msg, ...] = (Msg.FALLBACK_SCAM, Msg.FALLBACK_STEPS, Msg.WARN_FAMILY_FALLBACK)

_MARKUP = re.compile(r"[_~`]|\*(?!\d)")  # WhatsApp markup, not USSD codes (*346#)
_LEADING_SYMBOLS = re.compile(r"^[^\w\[(\"'“‘]+")  # emoji, bullets, arrows  # noqa: RUF001
_LIST_MARKER = re.compile(r"^(?:step\s*)?\(?\d{1,2}\s*[.):](?!\d)\s*", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")
_LETTER_WORD = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")  # "don't" is one word
_DIGITS = re.compile(r"\d+(?:[.,]\d+)*")
# Everyday banking terms a reply may use without the facts naming them; agencies (EFCC, CBN, SEC, ...) are not here.
COMMON_ACRONYMS = frozenset(
    {"OTP", "PIN", "BVN", "NIN", "ATM", "POS", "SMS", "USSD", "CVV", "ID", "APK", "KYC"}
)
# "one" is left out: "no one", "the one" are not quantities.
_NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "hundred": "100", "thousand": "1000", "million": "1000000",
}  # fmt: skip


class GroundingError(RuntimeError):
    """The check could not run (no translation, out of time). Callers fail open: the reply already passed the guard."""


@dataclass(frozen=True, slots=True)
class Claim:
    text: str  # one English sentence/claim from the reply
    support: float  # best cosine similarity against the evidence
    supported: bool
    unmatched: tuple[str, ...] = ()  # numbers and names in the claim that no evidence contains


@dataclass(frozen=True, slots=True)
class GroundingReport:
    english: str  # the reply in English (back-translated, or as-is for English)
    claims: tuple[Claim, ...]
    ms: int = 0  # wall time of the check, incl. back-translation

    @property
    def grounded(self) -> bool:
        return all(c.supported for c in self.claims)

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(c.text for c in self.claims if not c.supported)


class Translator(Protocol):
    async def to_english(self, text: str, language: LanguageCode, *, deadline: float | None = None) -> str:
        """English rendering of ``text``. Raises ``GroundingError``."""
        ...


# --- pure helpers ---------------------------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    """Sentences of a reply, without list markers, bullets, leading emoji or WhatsApp markup."""
    sentences: list[str] = []
    for raw in _MARKUP.sub("", unicodedata.normalize("NFKC", text)).splitlines():
        line = _LEADING_SYMBOLS.sub("", raw.strip())
        line = _LEADING_SYMBOLS.sub("", _LIST_MARKER.sub("", line))
        sentences.extend(s.strip() for s in _SENTENCE_BREAK.split(line) if s.strip())
    return sentences


def is_trivial(sentence: str) -> bool:
    """Filler or a lead-in ("What to do:") rather than a checkable claim."""
    return len(_LETTER_WORD.findall(sentence)) < MIN_CLAIM_WORDS or sentence.rstrip().endswith(":")


def split_claims(english: str) -> list[str]:
    """The checkable claims of an English reply: one per sentence or list item, filler skipped."""
    return [s for s in _sentences(english) if not is_trivial(s)]


def numbers_in(text: str) -> set[str]:
    """Quantities written as digits ("24", "N5,000") or English number words ("three"), normalised to digits."""
    found = {d.replace(",", "") for d in _DIGITS.findall(text)}
    found.update(_NUMBER_WORDS[w] for w in _LETTER_WORD.findall(fold(text)) if w in _NUMBER_WORDS)
    return found


def names_in(text: str) -> set[str]:
    """Names a claim commits to, lower-cased: acronyms ("EFCC") and runs of capitalised words ("Fraud Bureau"),
    not the sentence's first word nor single capitalised words ("Nigeria", "WhatsApp" are too common to check)."""
    words = list(_LETTER_WORD.finditer(text))
    names = {w.group().lower() for w in words if len(w.group()) > 1 and w.group().isupper()}
    names -= {a.lower() for a in COMMON_ACRONYMS}
    run: list[re.Match[str]] = []
    for word in [*words, None]:
        if word is not None and word.group()[0].isupper():
            if run and not text[run[-1].end() : word.start()].isspace():
                run = _close_run(run, names)
            run.append(word)
        else:
            run = _close_run(run, names)
    return names


def _close_run(run: list[re.Match[str]], names: set[str]) -> list[re.Match[str]]:
    if len(run) > 1:
        names.update(w.group().lower() for w in run if w.start() > 0)
    return []


def evidence_units(evidence: Iterable[str]) -> list[str]:
    """Each evidence text plus its single sentences: a claim usually restates one sentence of a longer fact."""
    units: dict[str, None] = {}
    for text in evidence:
        if text.strip():
            units[text.strip()] = None
            units.update(dict.fromkeys(_sentences(text)))
    return list(units)


def evidence_for(
    knowledge: KnowledgeResult,
    *,
    verdict: Verdict | None = None,
    flags: Sequence[Flag] = (),
    message: str = "",
) -> list[str]:
    """What a COMPOSE reply may say: the facts and actions it was shown (``prompts.facts_block``), the verdict's
    meaning, the red flags found and the user's (English) message, which replies restate. ``GroundingChecker``
    adds the generic framing and fixed advice itself."""
    facts, actions = facts_block(knowledge)
    evidence = [*facts, *actions]
    if verdict is not None and verdict in VERDICT_LINE:
        evidence.append(VERDICT_LINE[verdict])
    evidence.extend(f"The message {FLAG_TEXT[f]}." for f in flags)
    if message.strip():
        evidence.append(message.strip())
    return evidence


def score_claims(
    claims: Sequence[str],
    evidence: Sequence[str],
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[Claim, ...]:
    """Best cosine of each claim against the evidence sentences (CPU work: run it off the event loop)."""
    if not claims:
        return ()
    units = evidence_units(evidence)
    known = " ".join(s for text in evidence for s in _sentences(text))  # without list markers ("1.")
    known_numbers = numbers_in(known)
    known_words = {w.lower() for w in _LETTER_WORD.findall(known)}
    support = np.zeros(len(claims), dtype=np.float32)
    if units:  # one batch: behind a CachedEmbedder only the reply (and the user's message) reach the model
        vectors = embedder.embed_documents([*claims, *units])
        support = (vectors[: len(claims)] @ vectors[len(claims) :].T).max(axis=1)
    scored = []
    for text, best in zip(claims, support, strict=True):
        unmatched = tuple(sorted((numbers_in(text) - known_numbers) | (names_in(text) - known_words)))
        score = round(float(best), 4)
        scored.append(Claim(text, score, score >= threshold and not unmatched, unmatched))
    return tuple(scored)


# --- adapters ---------------------------------------------------------------------------------------------------


class LLMTranslator:
    """Back-translation with N-ATLAS (``prompts.translate_request``): one extra model call per reply, so it is the
    offline/evaluation path, not the live default."""

    def __init__(self, llm: LLM, registry: LanguageRegistry, *, minimum_s: float = MIN_TRANSLATE_S) -> None:
        self._llm = llm
        self._registry = registry
        self._minimum_s = minimum_s

    async def to_english(self, text: str, language: LanguageCode, *, deadline: float | None = None) -> str:
        request = translate_request(text, self._registry.get(language).name)
        try:
            completion = await complete_within(self._llm, request, deadline, self._minimum_s)
        except LLMError as exc:
            raise GroundingError(f"back-translation failed: {exc}") from exc
        english = clean_translation(completion.text)
        if not english:
            raise GroundingError("back-translation was empty")
        return english


class GroundingChecker:
    """Checks a reply against its evidence. English is checked directly (milliseconds, CPU); other languages need a
    ``Translator``: by default N-ATLAS back-translation when ``llm`` is given. With neither, only English is
    supported (``supports``): the live configuration.

    Evidence vectors are cached by text (``CachedEmbedder``; ``warm`` fills it at startup), so a turn embeds only the
    reply's sentences and the user's message, in one batch off the event loop. ``budget_s`` caps that scoring step
    (not a back-translation, which is bounded by the deadline): past it the check raises and the caller fails open.
    """

    def __init__(
        self,
        llm: LLM | None,
        embedder: Embedder,
        registry: LanguageRegistry,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        translator: Translator | None = None,
        budget_s: float | None = None,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        if budget_s is not None and budget_s <= 0:
            raise ValueError(f"budget_s must be positive, got {budget_s}")
        self.threshold = threshold
        self.budget_s = budget_s
        self._embedder = embedder if isinstance(embedder, CachedEmbedder) else CachedEmbedder(embedder)
        if translator is None and llm is not None:
            translator = LLMTranslator(llm, registry)
        self._translator = translator
        messages = Messages(registry)
        self.framing: tuple[str, ...] = (*GENERIC_FRAMING, *(messages.text(PIVOT, m) for m in FIXED_ADVICE))

    def supports(self, language: LanguageCode) -> bool:
        return language == PIVOT or self._translator is not None

    def warm(self, evidence: Iterable[str]) -> int:
        """Embed evidence ahead of time (e.g. every pack fact at startup; blocking CPU work). Returns the count."""
        units = evidence_units([*evidence, *self.framing])
        self._embedder.embed_documents(units)
        return len(units)

    async def check(
        self, reply: str, language: LanguageCode, evidence: Sequence[str], *, deadline: float | None = None
    ) -> GroundingReport:
        """Raises ``GroundingError`` when the reply cannot be checked (unsupported language, no translation, no time)."""
        start = time.perf_counter()
        if language == PIVOT:
            english = reply
        elif self._translator is None:
            raise GroundingError(f"no translator for {language!r}: only English replies can be checked")
        else:
            english = await self._translator.to_english(reply, language, deadline=deadline)
        claims = split_claims(english)
        work = asyncio.to_thread(
            score_claims, claims, [*evidence, *self.framing], self._embedder, threshold=self.threshold
        )
        limits = [
            s
            for s in (self.budget_s, None if deadline is None else deadline - time.monotonic())
            if s is not None
        ]
        try:  # on timeout the worker thread finishes in the background (and still fills the cache)
            scored = await asyncio.wait_for(work, min(limits, default=None))
        except TimeoutError as exc:
            raise GroundingError("out of time for the grounding check") from exc
        report = GroundingReport(english, scored, int((time.perf_counter() - start) * 1000))
        log.info(
            "grounding %s: %d claims, %d unsupported (min support %.2f), %d ms",
            language,
            len(scored),
            len(report.unsupported),
            min((c.support for c in scored), default=1.0),
            report.ms,
        )
        return report
