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
from collections.abc import Iterable
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
    CRISIS_OTHER = "crisis_other"  # someone else is at risk ("my brother says he will kill himself")
    CRISIS = "crisis"
    SCAM_TOOL = "scam_tool"
    ABUSE = "abuse"
    INJECTION = "injection"
    INJECTION_SUSPECT = "injection_suspect"


PRIORITY = (
    Category.CRISIS_OTHER,  # first: a message about someone else gets the reply framed for helping them
    Category.CRISIS,
    Category.SCAM_TOOL,
    Category.ABUSE,
    Category.INJECTION,
    Category.INJECTION_SUSPECT,
)
INJECTIONS = frozenset({Category.INJECTION, Category.INJECTION_SUSPECT})
CRISES = frozenset({Category.CRISIS, Category.CRISIS_OTHER})

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

# --- obfuscation: what the screens see besides the plain normalised text ---------------------------------------
# Cyrillic and Greek letters that look Latin (a Cyrillic "i" in "ignore"). Only these code points change: Hausa, Yoruba and
# Igbo letters and tone marks are Latin (with combining marks) and pass through untouched.
_HOMOGLYPHS = str.maketrans(
    {
        **dict.fromkeys(map(chr, (0x430, 0x3B1)), "a"), **dict.fromkeys(map(chr, (0x435, 0x3B5)), "e"),
        **dict.fromkeys(map(chr, (0x43E, 0x3BF)), "o"), **dict.fromkeys(map(chr, (0x440, 0x3C1)), "p"),
        chr(0x441): "c", chr(0x443): "y", **dict.fromkeys(map(chr, (0x445, 0x3C7)), "x"),
        **dict.fromkeys(map(chr, (0x456, 0x3B9)), "i"), chr(0x458): "j", chr(0x455): "s", chr(0x501): "d",
        chr(0x3BA): "k", chr(0x3BD): "v", chr(0x3C4): "t", chr(0x3C5): "u",
        **dict.fromkeys(map(chr, (0x410, 0x391)), "A"), **dict.fromkeys(map(chr, (0x412, 0x392)), "B"),
        **dict.fromkeys(map(chr, (0x415, 0x395)), "E"), **dict.fromkeys(map(chr, (0x41A, 0x39A)), "K"),
        **dict.fromkeys(map(chr, (0x41C, 0x39C)), "M"), **dict.fromkeys(map(chr, (0x41D, 0x397)), "H"),
        **dict.fromkeys(map(chr, (0x41E, 0x39F)), "O"), **dict.fromkeys(map(chr, (0x420, 0x3A1)), "P"),
        chr(0x421): "C", **dict.fromkeys(map(chr, (0x422, 0x3A4)), "T"), **dict.fromkeys(map(chr, (0x425, 0x3A7)), "X"),
        **dict.fromkeys(map(chr, (0x406, 0x399)), "I"), chr(0x408): "J", chr(0x405): "S", chr(0x39D): "N",
    }
)  # fmt: skip
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "!": "i", "@": "a", "$": "s",
                       "|": "l"})  # fmt: skip
_LEET_TOKEN = re.compile(r"[A-Za-z0-9!@$|]+")
_SPACED_LETTERS = re.compile(r"(?:\b\w ){4,}\w\b")  # "I g n o r e"
_SPACED_GAP = re.compile(r"\b(\w) (?=\w\b)")


# a reply that repeats the user's self-harm words back to them ("you want to kill yourself"), for scrubbing stored replies
_ECHOED_SELF_HARM = re.compile(
    r"\b(?:kill\w*\W+(?:your|my|him|her)\W*sel(?:f|ves)|suicid\w*|end\w*\W+(?:your|my)\W+(?:own\W+)?life|want\w*\W+to\W+die)\b"
)


def _clean(text: str) -> str:
    """Without invisible format characters (zero-width spaces/joiners, BOM) and with look-alike letters mapped."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf").translate(_HOMOGLYPHS)


def _unleet(match: re.Match[str]) -> str:
    token = match.group()
    if any(ch.isalpha() for ch in token) and any(not ch.isalpha() for ch in token):
        return token.translate(_LEET)  # "k!ll", "1gn0r3", "pr3v10us"
    return token


def variants(text: str) -> list[str]:
    """The normalised text, plus de-obfuscated readings when they differ: leetspeak folded ("1gn0r3" -> "ignore")
    and letter-spaced words joined ("I g n o r e  p r e v i o u s" -> "ignore previous")."""
    clean = _clean(text[:MAX_SCREEN_CHARS])
    readings = [normalize(clean), normalize(_LEET_TOKEN.sub(_unleet, clean))]
    if _SPACED_LETTERS.search(clean):
        readings.append(normalize(_SPACED_GAP.sub(r"\1", clean)))
    return list(dict.fromkeys(readings))


@dataclass(frozen=True, slots=True)
class ScreenModes:
    crisis: ScreenMode = ScreenMode.ENFORCE  # also the mode of crisis_other
    scam_tool: ScreenMode = ScreenMode.ENFORCE
    abuse: ScreenMode = ScreenMode.ENFORCE
    injection: ScreenMode = ScreenMode.ENFORCE
    injection_suspect: ScreenMode = ScreenMode.SHADOW

    def of(self, category: Category) -> ScreenMode:
        mode: ScreenMode = getattr(self, "crisis" if category in CRISES else category.value)
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


def _search_any(patterns: Patterns, readings: Iterable[str]) -> bool:
    return any(_search(patterns, reading) for reading in readings)


def has_role_markup(text: str) -> bool:
    """Chat-template tokens or other role markup (reported or not)."""
    raw = unicodedata.normalize("NFKC", _clean(text[:MAX_SCREEN_CHARS]))
    return TEMPLATE_TOKENS.search(raw) is not None or SOFT_MARKUP.search(raw) is not None


class InputScreen:
    def __init__(self, registry: LanguageRegistry, modes: ScreenModes | None = None) -> None:
        self._registry = registry
        self.modes = modes or ScreenModes()
        self._everywhere = registry.all_abuse_patterns()

    def screen(self, text: str, language: LanguageCode, *, flagged: bool = False) -> Screening:
        """``flagged``: the scam rules found a red flag (then abuse is not screened: it is a message to check)."""
        head = text[:MAX_SCREEN_CHARS]
        readings = variants(head)
        folded = readings[0]
        patterns = self._registry.abuse_patterns(language)
        found = {
            Category.CRISIS_OTHER: lambda: _search_any(self._everywhere.crisis_other, readings),
            Category.CRISIS: lambda: _search_any(self._everywhere.crisis, readings),
            Category.SCAM_TOOL: lambda: (
                _search_any(patterns.scam_tool, readings)
                and not _search(patterns.scam_tool_defensive, folded)
            ),
            Category.ABUSE: lambda: not flagged and _search(patterns.abuse, folded),
            Category.INJECTION: lambda: any(self._strong(r, head, patterns) for r in readings),
            Category.INJECTION_SUSPECT: lambda: self._suspect(readings, head, patterns),
        }
        detected = [c for c in PRIORITY if self.modes.of(c) is not ScreenMode.OFF and found[c]()]
        for broad, specific in (
            (Category.INJECTION_SUSPECT, Category.INJECTION),
            (Category.CRISIS, Category.CRISIS_OTHER),
        ):
            if broad in detected and specific in detected:
                detected.remove(broad)  # one category of each kind per message
        enforced = next((c for c in detected if self.modes.of(c) is ScreenMode.ENFORCE), None)
        return Screening(tuple(detected), enforced)

    def crisis(self, text: str) -> Category | None:
        """The crisis screens alone (crisis_other first), in enforce mode only: used before any other gate."""
        if self.modes.crisis is not ScreenMode.ENFORCE:
            return None
        readings = variants(text)
        if _search_any(self._everywhere.crisis_other, readings):
            return Category.CRISIS_OTHER
        return Category.CRISIS if _search_any(self._everywhere.crisis, readings) else None

    def crisis_language(self, text: str) -> LanguageCode | None:
        """The language whose own crisis phrases match (the pivot's last): the reply for a user we do not know yet."""
        readings = variants(text)
        codes = sorted((lang.code for lang in self._registry.enabled), key=lambda code: code == PIVOT)
        for code in codes:
            own = self._registry.own_abuse_patterns(code)
            if _search_any(own.crisis + own.crisis_other, readings):
                return code
        return None

    def scrub_crisis(self, text: str) -> str:
        """``text`` without the sentences a crisis phrase matches in (a stored reply that quotes the user)."""
        phrases = self._everywhere.crisis + self._everywhere.crisis_other

        def quotes_crisis(sentence: str) -> bool:
            readings = variants(sentence)
            return _search_any(phrases, readings) or any(_ECHOED_SELF_HARM.search(r) for r in readings)

        lines = []
        for line in text.split("\n"):
            sentences = re.split(r"(?<=[a-z\)][.!?])\s+", line)  # not after a step number ("1. Do not ...")
            lines.append(" ".join(s for s in sentences if not quotes_crisis(s)))
        return "\n".join(line for line in lines if line.strip())

    def injected(self, text: str, language: LanguageCode, *, markup: bool = True) -> bool:
        """The high-precision injection signals, whatever the mode (pack screening, follow-up buttons).
        ``markup=False``: chat-template tokens and phrases only (pack facts may mention "<system> tags" as text)."""
        head = text[:MAX_SCREEN_CHARS]
        patterns = self._registry.abuse_patterns(language)
        return any(self._strong(r, head, patterns, markup=markup) for r in variants(head))

    def _strong(self, folded: str, raw: str, patterns: AbusePatterns, *, markup: bool = True) -> bool:
        nfkc = unicodedata.normalize("NFKC", _clean(raw))
        if TEMPLATE_TOKENS.search(nfkc):
            return True
        if markup and SOFT_MARKUP.search(nfkc) and not _search(patterns.injection_reported, folded):
            return True
        quoted = [m.span() for m in _QUOTED.finditer(folded)]
        for pattern in patterns.injection:
            for match in pattern.finditer(folded):
                start = match.start()
                reported = any(c.search(folded, 0, start) for c in patterns.injection_reported)
                if not reported and not any(a <= start < b for a, b in quoted):
                    return True
        return False

    def _suspect(self, readings: list[str], raw: str, patterns: AbusePatterns) -> bool:
        """Broad phrases, and strong signals that sit in reported or quoted content."""
        return (
            _search_any(patterns.injection_suspect, readings)
            or has_role_markup(raw)
            or _search_any(patterns.injection, readings)
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
