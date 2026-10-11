"""Malicious-use screens on what users send (docs/ABUSE.md).

Deterministic, per-language phrase lists (``data/languages/<code>.yaml`` under ``abuse``, the pivot's always merged
in) matched on normalised text, plus language-independent role markup (chat-template tokens and their look-alikes,
fake "system:" turns). No model call, microseconds per message.

* ``InputScreen`` sorts a message into at most one acted-on category, in priority order:
  crisis (first-person self-harm; checked with every language's phrases) > scam_tool (write/translate/polish scam
  content) > abuse (insults, hate, sexual messages at the bot; only when the rules find no red flag, so forwarded
  sextortion or threat scams are still checked) > injection (override, role-play, verdict change, prompt extraction).
  Each category has its own mode: ``off`` (not checked), ``shadow`` (detected and logged, reply unchanged) or
  ``enforce`` (acted on by the engine: a fixed reply, or for injection a rules-only answer without any model call).
* ``TranscriptFilter`` drops transcript segments speech recognition invents on music, noise or silence ("thanks for
  watching", subtitle credits, "[Music]"); a voice note left with no words is treated as no speech.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from kudiguard.domain import LanguageCode
from kudiguard.languages import AbusePatterns, LanguageRegistry
from kudiguard.safety.scam_rules import Patterns
from kudiguard.text import normalize

log = logging.getLogger(__name__)

MAX_SCREEN_CHARS = (
    4000  # longer input is screened on its first part (bounded regex work; prompts cut at 2000)
)


class ScreenMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class Category(StrEnum):
    CRISIS = "crisis"
    SCAM_TOOL = "scam_tool"
    ABUSE = "abuse"
    INJECTION = "injection"


PRIORITY = (Category.CRISIS, Category.SCAM_TOOL, Category.ABUSE, Category.INJECTION)

# Chat-template tokens (Llama-3 <|eot_id|>, ChatML <|im_start|>, look-alike bars), Llama-2 [INST]/<<SYS>>, our own
# prompt tags, markdown "### Instruction" headers and fake role turns at a line start ("system:", "assistant:").
_BAR = "[|" + "".join(map(chr, (0xA6, 0xFF5C, 0x2502, 0x2223, 0x1C0))) + "]"  # | and its look-alikes
ROLE_MARKUP = re.compile(
    rf"<\s*{_BAR}[^<>]{{0,40}}{_BAR}\s*>"
    r"|<\s*/?\s*(?:system|assistant|user|developer|tool|message|facts|actions|instructions?|s)\s*>"
    r"|\[\s*/?\s*inst\s*\]|<<\s*/?\s*sys\s*>>"
    r"|(?:begin_of_text|end_of_text|start_header_id|end_header_id|eot_id|im_start|im_end)"
    r"|^\s*#{2,}\s*(?:system|instructions?|response|assistant)\b"
    r"|^\s*(?:system|assistant|developer)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ScreenModes:
    crisis: ScreenMode = ScreenMode.ENFORCE
    scam_tool: ScreenMode = ScreenMode.ENFORCE
    abuse: ScreenMode = ScreenMode.ENFORCE
    injection: ScreenMode = ScreenMode.ENFORCE

    def of(self, category: Category) -> ScreenMode:
        mode: ScreenMode = getattr(self, category.value)
        return mode


@dataclass(frozen=True, slots=True)
class Screening:
    detected: tuple[Category, ...] = ()  # every category found (shadow and enforce)
    enforced: Category | None = None  # the one the engine acts on

    @property
    def shadowed(self) -> tuple[Category, ...]:
        return tuple(c for c in self.detected if c is not self.enforced)


def _search(patterns: Patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def has_role_markup(text: str) -> bool:
    return ROLE_MARKUP.search(unicodedata.normalize("NFKC", text[:MAX_SCREEN_CHARS])) is not None


class InputScreen:
    def __init__(self, registry: LanguageRegistry, modes: ScreenModes | None = None) -> None:
        self._registry = registry
        self.modes = modes or ScreenModes()
        self._everywhere = registry.all_abuse_patterns()

    def screen(self, text: str, language: LanguageCode, *, flagged: bool = False) -> Screening:
        """``flagged``: the scam rules found a red flag (then abuse is not screened: it is a message to check)."""
        head = text[:MAX_SCREEN_CHARS]
        folded = normalize(head)
        patterns = self._registry.abuse_patterns(language)
        detected: list[Category] = []
        for category in PRIORITY:
            if self.modes.of(category) is ScreenMode.OFF or (category is Category.ABUSE and flagged):
                continue
            if self._matches(category, folded, head, patterns):
                detected.append(category)
        enforced = next((c for c in detected if self.modes.of(c) is ScreenMode.ENFORCE), None)
        return Screening(tuple(detected), enforced)

    def injected(self, text: str, language: LanguageCode) -> bool:
        """Injection phrases or role markup, whatever the mode (pack screening, follow-up buttons)."""
        head = text[:MAX_SCREEN_CHARS]
        patterns = self._registry.abuse_patterns(language)
        return self._matches(Category.INJECTION, normalize(head), head, patterns)

    def _matches(self, category: Category, folded: str, raw: str, patterns: AbusePatterns) -> bool:
        if category is Category.CRISIS:
            return _search(self._everywhere.crisis, folded)
        if category is Category.INJECTION:
            return _search(patterns.injection, folded) or has_role_markup(raw)
        return _search(getattr(patterns, category.value), folded)


def log_screening(screening: Screening, turn_id: str | None, language: LanguageCode) -> None:
    """Categories only, never the text (a crisis message is sensitive personal data)."""
    if screening.detected:
        log.info("abuse_screen turn=%s language=%s enforced=%s shadow=%s", turn_id, language,
                 screening.enforced, ",".join(screening.shadowed) or "-")  # fmt: skip


# --- speech-recognition hallucinations ------------------------------------------------------------------------

# A short bracketed/starred annotation ("[Music]", "(laughs)", "*applause*", "♪ ... ♪"): never something the user said.
_ANNOTATION = re.compile(r"[\[(*♪♫]\s*[^\[\]()*♪♫\n]{0,40}?\s*[\])*♪♫]")
_SEGMENT = re.compile(r"(?:[^.!?\n]|[.!?](?=[^\s.!?]))+[.!?]*")  # "Amara.org" stays one segment
_EDGE = " \t.,;:!?-\"'…"
_WORD = re.compile(r"[^\W_]")


@dataclass(frozen=True, slots=True)
class FilteredTranscript:
    text: str
    removed: int  # segments/annotations dropped (0 = unchanged)

    @property
    def empty(self) -> bool:
        return _WORD.search(self.text) is None


class TranscriptFilter:
    def __init__(self, registry: LanguageRegistry, mode: ScreenMode = ScreenMode.ENFORCE) -> None:
        self._registry = registry
        self.mode = mode

    def filter(self, text: str, language: LanguageCode) -> FilteredTranscript:
        """The transcript without hallucinated segments. Shadow: computed and logged, ``text`` returned as is."""
        if self.mode is ScreenMode.OFF:
            return FilteredTranscript(text, 0)
        patterns = self._registry.abuse_patterns(language).asr_hallucinations
        dropped = 0

        def hallucinated(fragment: str) -> bool:
            core = normalize(fragment).strip(_EDGE)
            return bool(core) and any(p.fullmatch(core) for p in patterns)

        def annotation(match: re.Match[str]) -> str:
            nonlocal dropped
            inner = match.group()
            # "♪ ... ♪" is sung; "[Music]", "(kiɗa)" are labels: dropped. Other brackets may be speech: kept.
            if inner[0] in "♪♫" or hallucinated(inner[1:-1]):
                dropped += 1
                return " "
            return inner

        kept: list[str] = []
        for segment in _SEGMENT.findall(_ANNOTATION.sub(annotation, text)):
            if hallucinated(segment):
                dropped += 1
            elif segment.strip(_EDGE):
                kept.append(segment.strip())
        result = FilteredTranscript(" ".join(kept), dropped)
        if result.removed:
            log.info("asr_hallucination mode=%s removed=%d empty=%s", self.mode, result.removed, result.empty)
        if self.mode is ScreenMode.SHADOW:
            return FilteredTranscript(text, 0)
        return result
