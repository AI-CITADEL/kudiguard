"""The verdict policy in one place: how rules, the model's analysis and fallbacks combine.

Used by the pipeline and the prompt evaluation so production and measurement can never drift apart.

Policy (eval E1-E3, M3 reviews):
* English input: the user's own text is the English text (the model's "translation" invented facts).
* If the analyse step failed and the plain-translation fallback was used, that step's flags are discarded.
* Deterministic rules run on the original and the English text; model flags are only used when the rules found
  nothing and the message may be a check (``prompts.select_flags``).
* A message is a check if the model says ``scam_check``, the rules found any flag, or it is essentially a bare
  phone number / link / email (the user wants it checked; we never vouch for it: at best "no red flags found").
* Content-free messages (emoji, "?") and greetings never reach the model (``is_content_free``; pipeline).
* A general question (intent ``question``, no check phrase) that only mentions a red flag is capped at CAUTION.
* Model-only flags on a message the model labelled ``other`` are capped at CAUTION: only rule-detected flags can
  produce LIKELY_SCAM without the model also recognising a check.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata

from kudiguard.domain import LanguageCode, ScamAssessment, Verdict
from kudiguard.languages import PIVOT
from kudiguard.prompts import Analysis, Intent, select_flags, strip_contacts
from kudiguard.safety.scam_rules import Lexicon, assess, has_check_intent

_WORD = re.compile(r"[^\W_]{2,}")  # a real word: two or more letters/digits
LOOKUP_MAX_WORDS = 3  # "is this number a scam" style messages that are mostly a number/link
LOOKUP_MAX_CHARS = 200


def is_content_free(text: str) -> bool:
    """Emoji, punctuation or single characters only: answer with fixed text, never with the model (edge eval)."""
    return not _WORD.search(text)


def is_lookup(text: str) -> bool:
    """A message that is essentially a phone number, link or email: the user wants it checked."""
    # Cheap exits: lookups are short; this also bounds regex work on pathological input.
    if len(text) > LOOKUP_MAX_CHARS or len(text.split()) > LOOKUP_MAX_WORDS + 3:
        return False
    canonical = " ".join(unicodedata.normalize("NFKC", text).split())
    remainder = strip_contacts(canonical)
    return remainder != canonical and len(_WORD.findall(remainder)) < LOOKUP_MAX_WORDS


def resolve_analysis(
    analysis: Analysis, original: str, language: LanguageCode, *, fallback_translation: str | None = None
) -> Analysis:
    """Apply the English-source rule and the translation fallback to a parsed analysis."""
    if language == PIVOT:
        return dataclasses.replace(analysis, english=original, untranslated=False)
    if fallback_translation is not None:
        return dataclasses.replace(analysis, english=fallback_translation, untranslated=False, flags=())
    return analysis


def decide(original: str, analysis: Analysis, lexicon: Lexicon) -> ScamAssessment:
    texts = [original, analysis.english]
    rules = assess(texts, lexicon)
    flags = select_flags(rules.flags, analysis)
    is_check = analysis.intent is Intent.SCAM_CHECK or bool(rules.flags) or is_lookup(original)
    # Score the texts again (not just the flags) so the lexicon's check-intent phrases still count.
    result = assess(texts, lexicon, flags, is_check=is_check)
    if result.verdict is not Verdict.LIKELY_SCAM:
        return result
    model_only = bool(flags) and not rules.flags
    if model_only and analysis.intent is not Intent.SCAM_CHECK:
        return dataclasses.replace(result, verdict=Verdict.CAUTION)
    # A general question that only *mentions* a red flag ("do banks ever ask for OTP?") is not a reported scam.
    if analysis.intent is Intent.QUESTION and not has_check_intent(texts, lexicon):
        return dataclasses.replace(result, verdict=Verdict.CAUTION)
    return result
