"""Malicious-use checks on what the model writes, after cleaning and the output guard (docs/ABUSE.md).

Deterministic and cheap; a hit is handled like an output-guard hit (one warmer retry, then fixed text), so a false
positive costs a fixed reply, never an unsafe one.

* ``PERSONA``: the reply claims to be a bank, customer care or an official ("I am GTBank support"): the result of a
  role-play injection. Phrases per language (``abuse.persona_claims``).
* ``PROMPT_LEAK``: the reply repeats a run of ``LEAK_WORDS`` words from one of KudiGuard's own system prompts, or the
  prompt scaffolding ("<facts>", "Task:", "Warning signs found:").
* ``ECHO``: the reply copies a run of ``echo_words`` words from the user's message (or its English translation): a
  "repeat after me", a polished copy of a scam, a translated payload passed through.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

from kudiguard.domain import LanguageCode
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.abuse import ScreenMode
from kudiguard.safety.output_guard import OutputIssue
from kudiguard.text import fold, normalize

log = logging.getLogger(__name__)

LEAK_WORDS = 8
DEFAULT_ECHO_WORDS = 12
_WORDS = re.compile(r"[^\W_]+")
_SCAFFOLD = re.compile(
    r"<\s*/?\s*(?:facts|actions|message)\s*>|\btask\s*:|\bwarning signs (?:found\s*)?:|\breply in [a-z]+\s*\.",
)


def _words(text: str) -> list[str]:
    return _WORDS.findall(fold(text))


def _grams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class ReplyScreen:
    def __init__(
        self,
        registry: LanguageRegistry,
        prompts: Iterable[str] = (),
        *,
        mode: ScreenMode = ScreenMode.ENFORCE,
        echo_words: int = DEFAULT_ECHO_WORDS,
    ) -> None:
        if echo_words < 4:
            raise ValueError("echo_words must be at least 4 (shorter runs are ordinary phrases)")
        self._registry = registry
        self.mode = mode
        self._echo_words = echo_words
        self._prompt_grams: set[tuple[str, ...]] = set()
        for prompt in prompts:
            self._prompt_grams |= _grams(_words(prompt), LEAK_WORDS)

    def check(
        self, text: str, language: LanguageCode, sources: Sequence[str] = ()
    ) -> tuple[OutputIssue, ...]:
        """Issues to act on (enforce mode); shadow mode logs them and returns ()."""
        if self.mode is ScreenMode.OFF:
            return ()
        issues = self.find(text, language, sources)
        if issues:
            log.info("reply_screen mode=%s issues=%s", self.mode, ",".join(issues))
        return issues if self.mode is ScreenMode.ENFORCE else ()

    def find(self, text: str, language: LanguageCode, sources: Sequence[str] = ()) -> tuple[OutputIssue, ...]:
        folded = normalize(text)
        words = _words(text)
        issues: list[OutputIssue] = []
        if any(p.search(folded) for p in self._registry.abuse_patterns(language).persona_claims):
            issues.append(OutputIssue.PERSONA)
        if _SCAFFOLD.search(folded) or _grams(words, LEAK_WORDS) & self._prompt_grams:
            issues.append(OutputIssue.PROMPT_LEAK)
        reply_grams = _grams(words, self._echo_words)
        if any(reply_grams & _grams(_words(source), self._echo_words) for source in sources if source):
            issues.append(OutputIssue.ECHO)
        return tuple(issues)
