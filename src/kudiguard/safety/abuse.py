"""Malicious-use screens on what users send (docs/ABUSE.md).

Deterministic, per-language phrase lists (``data/languages/<code>.yaml`` under ``abuse``, the pivot's always merged
in) matched on normalised text, plus language-independent role markup (chat-template tokens and their look-alikes,
fake "system:" turns). No model call, microseconds per message.

* ``InputScreen`` sorts a message into at most one acted-on category, in priority order:
  crisis (first-person self-harm; checked with every language's phrases) > scam_tool (write/translate/polish scam
  content) > abuse (insults, hate, sexual messages at the bot; only when the rules find no red flag, so forwarded
  sextortion or threat scams are still checked) > injection (high-precision override, role, verdict-change and
  prompt-extraction phrases and chat-template tokens) > injection_suspect (broad phrases that genuine users also
  write, and injection phrases inside reported or quoted content: "my boss said ignore the previous instructions").
  Each category has its own mode: ``off`` (not checked), ``shadow`` (detected and logged, reply unchanged) or
  ``enforce`` (acted on by the engine: crisis help first, a fixed reply for abuse and scam-tool requests, a
  rules-only answer for injection).
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
from kudiguard.languages import PIVOT, AbusePatterns, LanguageRegistry
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
    INJECTION_SUSPECT = "injection_suspect"


PRIORITY = (
    Category.CRISIS,
    Category.SCAM_TOOL,
    Category.ABUSE,
    Category.INJECTION,
    Category.INJECTION_SUSPECT,
)
INJECTIONS = frozenset({Category.INJECTION, Category.INJECTION_SUSPECT})

_BAR = "[|" + "".join(map(chr, (0xA6, 0xFF5C, 0x2502, 0x2223, 0x1C0))) + "]"  # | and its look-alikes
# Chat-template tokens (Llama-3 <|eot_id|>, ChatML <|im_start|>, look-alike bars): no genuine SMS contains them, so
# they always count, reported or not.
TEMPLATE_TOKENS = re.compile(
    rf"<\s*{_BAR}[^<>]{{0,40}}{_BAR}\s*>"
    r"|(?:begin_of_text|end_of_text|start_header_id|end_header_id|eot_id|im_start|im_end)",
    re.IGNORECASE,
)
# Other role markup (Llama-2 [INST]/<<SYS>>, our prompt tags, "<system>", "### Instruction:", a "system:" turn at a line
# start): strong only when nothing in the message says it is someone else's text ("The SMS said: <SYSTEM> ...").
SOFT_MARKUP = re.compile(
    r"<\s*/?\s*(?:system|assistant|user|developer|tool|message|facts|actions|instructions?|s)\s*>"
    r"|\[\s*/?\s*inst\s*\]|<<\s*/?\s*sys\s*>>"
    r"|^\s*#{2,}\s*(?:system|instructions?|response|assistant)\s*:"
    r"|^\s*(?:system|assistant|developer)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
# Quoted spans in normalised text: "...", “...”, and '...' opened after a space or colon.
_LQ, _RQ = chr(0x201C), chr(0x201D)  # curly double quotes
_QUOTED = re.compile(
    '"[^"\n]{3,}"|' + _LQ + "[^" + _RQ + "\n]{3,}" + _RQ + r"|(?:(?<=[\s:])|^)'[^'\n]{3,}'(?=[\s.,!?)]|$)"
)


@dataclass(frozen=True, slots=True)
class ScreenModes:
    crisis: ScreenMode = ScreenMode.ENFORCE
    scam_tool: ScreenMode = ScreenMode.ENFORCE
    abuse: ScreenMode = ScreenMode.ENFORCE
    injection: ScreenMode = ScreenMode.ENFORCE
    injection_suspect: ScreenMode = ScreenMode.SHADOW

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
    """Chat-template tokens or other role markup (reported or not)."""
    raw = unicodedata.normalize("NFKC", text[:MAX_SCREEN_CHARS])
    return TEMPLATE_TOKENS.search(raw) is not None or SOFT_MARKUP.search(raw) is not None


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
        found = {
            Category.CRISIS: lambda: self.crisis_in(folded),
            Category.SCAM_TOOL: lambda: _search(patterns.scam_tool, folded),
            Category.ABUSE: lambda: not flagged and _search(patterns.abuse, folded),
            Category.INJECTION: lambda: self._strong(folded, head, patterns),
            Category.INJECTION_SUSPECT: lambda: self._suspect(folded, head, patterns),
        }
        detected = [c for c in PRIORITY if self.modes.of(c) is not ScreenMode.OFF and found[c]()]
        if Category.INJECTION in detected and Category.INJECTION_SUSPECT in detected:
            detected.remove(Category.INJECTION_SUSPECT)  # one injection category per message
        enforced = next((c for c in detected if self.modes.of(c) is ScreenMode.ENFORCE), None)
        return Screening(tuple(detected), enforced)

    def crisis(self, text: str) -> bool:
        """The crisis screen alone, in enforce mode only (used before any other gate: docs/ABUSE.md B2)."""
        return self.modes.crisis is ScreenMode.ENFORCE and self.crisis_in(normalize(text[:MAX_SCREEN_CHARS]))

    def crisis_language(self, text: str) -> LanguageCode | None:
        """The language whose own crisis phrases match (the pivot's last): the reply for a user we do not know yet."""
        folded = normalize(text[:MAX_SCREEN_CHARS])
        codes = sorted((lang.code for lang in self._registry.enabled), key=lambda code: code == PIVOT)
        return next((c for c in codes if _search(self._registry.own_abuse_patterns(c).crisis, folded)), None)

    def crisis_in(self, folded: str) -> bool:
        return _search(self._everywhere.crisis, folded)

    def injected(self, text: str, language: LanguageCode) -> bool:
        """The high-precision injection signals, whatever the mode (pack screening, follow-up buttons)."""
        head = text[:MAX_SCREEN_CHARS]
        return self._strong(normalize(head), head, self._registry.abuse_patterns(language))

    def _strong(self, folded: str, raw: str, patterns: AbusePatterns) -> bool:
        nfkc = unicodedata.normalize("NFKC", raw)
        if TEMPLATE_TOKENS.search(nfkc):
            return True
        if SOFT_MARKUP.search(nfkc) and not _search(patterns.injection_reported, folded):
            return True
        quoted = [m.span() for m in _QUOTED.finditer(folded)]
        for pattern in patterns.injection:
            for match in pattern.finditer(folded):
                start = match.start()
                reported = any(c.search(folded, 0, start) for c in patterns.injection_reported)
                if not reported and not any(a <= start < b for a, b in quoted):
                    return True
        return False

    def _suspect(self, folded: str, raw: str, patterns: AbusePatterns) -> bool:
        """Broad phrases, and strong signals that sit in reported or quoted content."""
        return (
            _search(patterns.injection_suspect, folded)
            or has_role_markup(raw)
            or _search(patterns.injection, folded)
        )


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
        patterns = self._registry.abuse_patterns(language)
        dropped = 0

        def matches(fragment: str, terms: Patterns) -> bool:
            core = normalize(fragment).strip(_EDGE)
            return bool(core) and any(p.fullmatch(core) for p in terms)

        def annotation(match: re.Match[str]) -> str:
            nonlocal dropped
            inner = match.group()
            # "♪ ... ♪" is sung; "[Music]", "(kiɗa)" are labels: dropped. Other brackets may be speech: kept.
            labels = patterns.asr_annotations + patterns.asr_hallucinations
            if inner[0] in "♪♫" or matches(inner[1:-1], labels):
                dropped += 1
                return " "
            return inner

        kept: list[tuple[str, bool]] = []  # (segment, only kept when other speech exists)
        for segment in _SEGMENT.findall(_ANNOTATION.sub(annotation, text)):
            if matches(segment, patterns.asr_hallucinations):
                dropped += 1
            elif segment.strip(_EDGE):
                kept.append((segment.strip(), matches(segment, patterns.asr_alone)))
        if all(alone for _, alone in kept):  # nothing but a lone "You.": noise
            dropped += len(kept)
            kept = []
        result = FilteredTranscript(" ".join(s for s, _ in kept), dropped)
        if result.removed:
            log.info("asr_hallucination mode=%s removed=%d empty=%s", self.mode, result.removed, result.empty)
        if self.mode is ScreenMode.SHADOW:
            return FilteredTranscript(text, 0)
        return result
