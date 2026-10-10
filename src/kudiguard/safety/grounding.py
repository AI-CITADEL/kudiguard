"""Grounding check: is every sentence of a composed reply supported by what the model was given?

The reply (English, or back-translated to English) is split into claims. Each claim is compared by meaning (embedding
cosine) with the evidence sentences: the knowledge-base facts and actions the model was shown, the verdict, the red
flags, generic framing that is always acceptable (empathy, "ask your bank", the reviewed English fixed strings).
A claim is supported when

* its best match **of the same polarity** reaches the threshold ("share the code" is not supported by "do not share
  the code"; negation is detected from markers such as not/never/don't/avoid), and
* its specifics appear in the evidence or in the user's message: numbers (days, hours, amounts, normalised: "50k",
  "N50,000" and "fifty thousand" are the same) and names (acronyms such as EFCC, multi-word names as phrases, and
  known single-word agencies such as Police).

The user's message counts as evidence for specifics only (a reply may name the bank or the amount the user named),
and for meaning only when the claim is explicitly attributed ("They said your account will be blocked..."): otherwise
a reply could repeat the scammer's claims as fact and pass.

Limits: meaning similarity cannot see every contradiction ("invest only half of your savings" sits close to real
advice). The output guard's unsafe-instruction patterns are the deterministic safety net for the dangerous cases
(share the code, pay the fee, click the link, install the app, send the money back). Calibration and held-out results:
``data/eval/grounding_claims.yaml``, docs/SPIKES.md S8. Only English is checked live; other languages need a
``Translator`` (N-ATLAS back-translation: offline evaluation, it would cost a second model call per reply).

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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from kudiguard.domain import Flag, LanguageCode, Verdict
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.embeddings import CachedEmbedder, Embedder, Matrix
from kudiguard.knowledge.models import KnowledgeResult
from kudiguard.languages import PIVOT, LanguageRegistry
from kudiguard.llm.base import LLM, LLMError, complete_within
from kudiguard.prompts import FLAG_TEXT, clean_translation, facts_block, translate_request
from kudiguard.text import fold

log = logging.getLogger(__name__)

# bge-small-en-v1.5 cosine, claim vs best same-polarity evidence sentence. Chosen on the dev split of
# data/eval/grounding_claims.yaml only (docs/SPIKES.md S8), below the lowest supported paraphrase on purpose: a false
# flag replaces a good answer, and N-ATLAS paraphrases more loosely than the set.
DEFAULT_THRESHOLD = 0.73
DEFAULT_BUDGET_S = 0.15  # live cap on scoring (the check fails open past it)
MIN_CLAIM_WORDS = 3  # "Stay calm." / "Be careful!" carry no checkable fact
MIN_TRANSLATE_S = 2.0  # back-translation is a full model call
MAX_MESSAGE_SENTENCES = 10  # the user's message: only its first sentences are embedded (latency)
MAX_SENTENCE_CHARS = 300
EMBED_CHUNK = 16  # texts per model call, sorted by length so padding stays small


class GroundingMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"  # check and record, never change the reply (default until enforce is validated)
    ENFORCE = "enforce"  # unsupported claims: drop the step, or fall back to fixed text


# Always acceptable, whatever the facts: COMPOSE is told to explain calmly and to suggest the bank when the facts
# do not cover the question. No "don't worry": it overlaps the guard's "nothing to worry about".
GENERIC_FRAMING: tuple[str, ...] = (
    "I understand your worry.",
    "I am sorry this happened to you.",
    "You did the right thing to check first.",
    "Stay calm and take your time before you act.",
    "Tell your family and friends about this trick.",
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
# Everyday terms a reply may use without the facts naming them; agencies (EFCC, CBN, SEC, ...) are not here.
COMMON_ACRONYMS = frozenset(
    {"OTP", "PIN", "BVN", "NIN", "ATM", "POS", "SMS", "USSD", "CVV", "ID", "APK", "KYC", "OK", "NO", "NOT",
     "STOP", "DELETE", "SCAM", "ALERT", "URGENT", "NEVER", "DO"}
)  # fmt: skip
# Single words that name an authority when capitalised ("the Police will refund you").
KNOWN_AGENCIES = frozenset({"police", "interpol", "customs", "immigration", "army", "navy", "court", "fbi"})

_MARKUP = re.compile(r"[_~`]|\*(?!\d)")  # WhatsApp markup, not USSD codes (*346#)
_LEADING_SYMBOLS = re.compile(r"^[^\w\[(\"'“‘]+")  # emoji, bullets, arrows  # noqa: RUF001
_LIST_MARKER = re.compile(
    r"^(?:\d\ufe0f?\u20e3|(?:step\s*)?\(?\d{1,2}\s*(?:[.):]|\s[-\u2013\u2014])(?!\d))\s*", re.IGNORECASE
)  # "1.", "2)", "Step 3:", "1 - ", keycap "1️⃣"
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")
_ABBREVIATION = re.compile(r"(?:^|\s)(?:e\.g|i\.e|mr|mrs|ms|dr|no|vs)\.$", re.IGNORECASE)
_LETTER_WORD = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")  # "don't" is one word
_AMOUNT = re.compile(r"(\d+(?:[.,]\d+)*)(?:\s*(k|m|bn|thousand|million|billion)(?![a-z]))?", re.IGNORECASE)
_SCALE = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000, "bn": 10**9, "billion": 10**9}
_SMALL = {
    w: n
    for n, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
    )
}
_TENS = {
    w: 10 * n
    for n, w in enumerate(
        ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    )
    if n > 1
}
_UNITS = frozenset(
    [
        "minute",
        "minutes",
        "hour",
        "hours",
        "hr",
        "hrs",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "time",
        "times",
        "naira",
        "kobo",
        "percent",
    ]
)  # a number word counts only before a unit ("seven days"), never "no one" / "four digits"
_NEGATION = re.compile(
    r"\b(?:not|no|never|\w+n't|dont|cannot|nobody|nothing|none|without|avoid|stop|ignore|refuse|neither|nor)\b"
)
_ATTRIBUTION = re.compile(
    r"\b(?:they|he|she|the (?:caller|sender|message|person|man|woman|text|sms|scammers?)|someone|scammers?)\b"
    r"[^.]{0,40}\b(?:said|says|say|claim|claims|claimed|told|tells|asked|asks|want|wants|wanted|promise|promises|"
    r"promised|threaten|threatens|threatened)\b",
    re.IGNORECASE,
)


class GroundingError(RuntimeError):
    """The check could not run (busy, no translation, out of time). Callers fail open: the reply passed the guard."""


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


def _strip_line(raw: str) -> str:
    line = _LEADING_SYMBOLS.sub("", raw.strip())
    return _LEADING_SYMBOLS.sub("", _LIST_MARKER.sub("", line))


def _sentences(text: str) -> list[str]:
    """Sentences of a reply, without list markers, bullets, leading emoji or WhatsApp markup."""
    sentences: list[str] = []
    for raw in _MARKUP.sub("", unicodedata.normalize("NFKC", text)).splitlines():
        pieces: list[str] = []
        for piece in _SENTENCE_BREAK.split(_strip_line(raw)):
            if pieces and _ABBREVIATION.search(pieces[-1]):  # "e.g. by SMS" is one sentence
                pieces[-1] = f"{pieces[-1]} {piece}"
            else:
                pieces.append(piece)
        sentences.extend(p.strip() for p in pieces if p.strip())
    return sentences


def _number_value(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.4g}"


def numbers_in(text: str) -> set[str]:
    """Quantities, normalised: digits ("24", "N5,000" -> 5000, "50k" -> 50000, "2 million") and English number
    words before a unit or with a scale ("seven days" -> 7, "five thousand" -> 5000)."""
    found = {
        _number_value(float(m.group(1).replace(",", "")) * _SCALE.get((m.group(2) or "").lower(), 1))
        for m in _AMOUNT.finditer(text)
        if m.group(1).replace(",", "").count(".") <= 1
    }
    words = _LETTER_WORD.findall(fold(text))
    total = current = 0
    started = scaled = False
    previous = ""
    for word in [*words, ""]:
        if word in _SMALL or word in _TENS:
            current += _SMALL.get(word, 0) + _TENS.get(word, 0)
            started = True
        elif word == "hundred" and (started or previous == "a"):  # "a hundred", "two hundred"
            current, started, scaled = (current or 1) * 100, True, True
        elif word in ("thousand", "million") and (started or previous == "a"):  # not "2 million" (digits)
            total, current, started, scaled = total + (current or 1) * _SCALE[word], 0, True, True
        else:
            if started and (scaled or word in _UNITS):
                found.add(_number_value(total + current))
            total = current = 0
            started = scaled = False
        previous = word
    return found


def names_in(text: str) -> set[str]:
    """Names a claim commits to, lower-cased: acronyms ("efcc"), runs of capitalised words as phrases ("fraud
    bureau"; the sentence's first word is left out) and known single-word agencies ("police" when capitalised)."""
    words = list(_LETTER_WORD.finditer(text))
    names = {w.group().lower() for w in words if len(w.group()) > 1 and w.group().isupper()}
    names -= {a.lower() for a in COMMON_ACRONYMS}
    names |= {
        w.group().lower() for w in words if w.group()[0].isupper() and w.group().lower() in KNOWN_AGENCIES
    }
    run: list[re.Match[str]] = []
    for word in [*words, None]:
        joined = word is not None and bool(run) and text[run[-1].end() : word.start()] == " "
        if word is not None and word.group()[0].isupper() and (joined or not run):
            run.append(word)
            continue
        _close_run(run, names)
        run = [word] if word is not None and word.group()[0].isupper() else []
    return names


def _close_run(run: list[re.Match[str]], names: set[str]) -> None:
    if len(run) > 1:
        phrase = [w.group().lower() for w in run if w.start() > 0 and w.group() not in COMMON_ACRONYMS]
        if phrase:
            names.add(" ".join(phrase))


def is_trivial(sentence: str) -> bool:
    """Filler or a lead-in ("What to do:") rather than a checkable claim. Never when it names a number or a name."""
    if any(ch.isdigit() for ch in sentence) or names_in(sentence):
        return False
    return len(_LETTER_WORD.findall(sentence)) < MIN_CLAIM_WORDS or sentence.rstrip().endswith(":")


def split_claims(english: str) -> list[str]:
    """The checkable claims of an English reply: one per sentence or list item, filler skipped."""
    return [s for s in _sentences(english) if not is_trivial(s)]


def negated(sentence: str) -> bool:
    return bool(_NEGATION.search(fold(sentence)))


def attributed(sentence: str) -> bool:
    """Reported speech about the scammer ("They said...", "The caller asked..."), not advice or fact."""
    return bool(_ATTRIBUTION.search(sentence))


def evidence_units(evidence: Iterable[str]) -> list[str]:
    """The single sentences of the evidence (polarity is judged per sentence), without duplicates."""
    units: dict[str, None] = {}
    for text in evidence:
        units.update(dict.fromkeys(_sentences(text)))
    return list(units)


def message_sentences(message: str) -> list[str]:
    """The user's message as bounded sentences: long forwarded messages must not blow the latency budget."""
    return [s[:MAX_SENTENCE_CHARS] for s in _sentences(message)[:MAX_MESSAGE_SENTENCES]]


def evidence_for(
    knowledge: KnowledgeResult, *, verdict: Verdict | None = None, flags: Sequence[Flag] = ()
) -> list[str]:
    """What a COMPOSE reply may say: the facts and actions it was shown (``prompts.facts_block``), the verdict's
    meaning and the red flags found. No user text: evidence vectors are cached (pass the user's message to
    ``check(message=...)``). ``GroundingChecker`` adds the generic framing and fixed advice itself."""
    facts, actions = facts_block(knowledge)
    evidence = [*facts, *actions]
    if verdict is not None and verdict in VERDICT_LINE:
        evidence.append(VERDICT_LINE[verdict])
    evidence.extend(f"The message {FLAG_TEXT[f]}." for f in flags)
    return evidence


def _embed_by_length(embedder: Embedder, texts: Sequence[str]) -> Matrix:
    """Embed in length-sorted chunks (padding follows the longest text in a batch), rows in input order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    chunks = [
        embedder.embed_documents([texts[i] for i in order[n : n + EMBED_CHUNK]])
        for n in range(0, len(order), EMBED_CHUNK)
    ]
    vectors = np.concatenate(chunks)
    result = np.empty_like(vectors)
    result[order] = vectors
    return result


def _best(similarity: Matrix, mask: Sequence[bool]) -> float:
    picked = similarity[np.asarray(mask, dtype=bool)]
    return float(picked.max()) if picked.size else 0.0


def score_claims(
    claims: Sequence[str],
    evidence: Sequence[str],
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    message: str = "",
) -> tuple[Claim, ...]:
    """Score each claim against the evidence (see the module docstring). CPU work: run it off the event loop.

    With a ``CachedEmbedder``, evidence vectors come from the cache and only the claims and the message's sentences
    are embedded (directly, never cached: they are user-specific)."""
    if not claims:
        return ()
    units = evidence_units(evidence)
    said = message_sentences(message) if any(attributed(c) for c in claims) else []  # only they need it
    fresh = _embed_by_length(
        embedder.inner if isinstance(embedder, CachedEmbedder) else embedder, [*claims, *said]
    )
    claim_vectors, said_vectors = fresh[: len(claims)], fresh[len(claims) :]
    unit_vectors = (
        embedder.embed_documents(units) if units else np.zeros((0, fresh.shape[1]), dtype=np.float32)
    )
    unit_polarity = [negated(u) for u in units]
    known = " ".join([*units, message])
    known_numbers = numbers_in(known)
    known_words = " " + " ".join(w.lower() for w in _LETTER_WORD.findall(known)) + " "
    scored = []
    for text, vector in zip(claims, claim_vectors, strict=True):
        polarity = negated(text)
        best = _best(unit_vectors @ vector, [p == polarity for p in unit_polarity])
        if said and attributed(text):
            best = max(best, float((said_vectors @ vector).max()))
        names = {n for n in names_in(text) if f" {n} " not in known_words}
        unmatched = tuple(sorted((numbers_in(text) - known_numbers) | names))
        score = round(best, 4)
        scored.append(Claim(text, score, score >= threshold and not unmatched, unmatched))
    return tuple(scored)


_STEP_NUMBER = re.compile(r"^(\s*\**)(\d{1,2})(?=\**\s*[.)])")


def without_unsupported_steps(reply: str, report: GroundingReport) -> str | None:
    """``reply`` minus the numbered steps holding its unsupported claims (renumbered), or None when an unsupported
    claim is outside a step or nothing checkable would remain. English replies only (claims are reply sentences)."""
    bad = set(report.unsupported)
    kept: list[str] = []
    dropped: set[str] = set()
    for line in reply.splitlines():
        stripped = _LEADING_SYMBOLS.sub("", _MARKUP.sub("", unicodedata.normalize("NFKC", line)).strip())
        hits = set(split_claims(line)) & bad
        if hits and _LIST_MARKER.match(stripped):
            dropped |= hits
            continue
        kept.append(line)
    if dropped != bad or not split_claims("\n".join(kept)):
        return None
    step = 0
    renumbered = []
    for line in kept:
        if match := _STEP_NUMBER.match(line):
            step += 1
            line = f"{match.group(1)}{step}{line[match.end() :]}"
        renumbered.append(line)
    return "\n".join(renumbered).strip()


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
    reply's sentences and the user's message. Scoring runs on one dedicated worker thread; a check that finds it
    busy (a previous one still running after its timeout) is skipped rather than queued, so it never competes with
    knowledge search. ``budget_s`` caps the scoring step (not a back-translation, which is bounded by the deadline).
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
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grounding")
        self._running: Future[tuple[Claim, ...]] | None = None

    def supports(self, language: LanguageCode) -> bool:
        return language == PIVOT or self._translator is not None

    def warm(self, evidence: Iterable[str]) -> int:
        """Embed evidence ahead of time (pack facts at startup; blocking CPU work). Returns the sentence count.
        Framing, verdict lines and red-flag texts are always included."""
        extra = [*self.framing, *VERDICT_LINE.values(), *(f"The message {t}." for t in FLAG_TEXT.values())]
        units = evidence_units([*evidence, *extra])
        self._embedder.embed_documents(units)
        return len(units)

    async def aclose(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)

    async def check(
        self,
        reply: str,
        language: LanguageCode,
        evidence: Sequence[str],
        *,
        message: str = "",
        deadline: float | None = None,
    ) -> GroundingReport:
        """``message``: the user's English message (specifics, and attributed claims). Raises ``GroundingError`` when
        the reply cannot be checked (unsupported language, no translation, busy, out of time)."""
        start = time.perf_counter()
        if language == PIVOT:
            english = reply
        elif self._translator is None:
            raise GroundingError(f"no translator for {language!r}: only English replies can be checked")
        else:
            english = await self._translator.to_english(reply, language, deadline=deadline)
        if self._running is not None and not self._running.done():
            raise GroundingError("busy: a previous check is still running")
        claims = split_claims(english)
        self._running = self._worker.submit(
            score_claims,
            claims,
            [*evidence, *self.framing],
            self._embedder,
            threshold=self.threshold,
            message=message,
        )
        left = None if deadline is None else deadline - time.monotonic()
        limits = [s for s in (self.budget_s, left) if s is not None]
        try:  # on timeout the worker finishes in the background; checks are skipped until it does
            scored = await asyncio.wait_for(asyncio.wrap_future(self._running), min(limits, default=None))
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
