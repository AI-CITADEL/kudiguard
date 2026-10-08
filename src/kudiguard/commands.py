"""Recognise whole-message commands, greetings and language choices, in every configured language.

Phrases live in ``data/languages/<code>.yaml`` under ``commands`` (configuration, not code). Matching is on the
canonical form (case, accents, tone marks and punctuation ignored), and commands from *every* language apply to every
user: someone set to Hausa may still type "STOP".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from kudiguard.domain import LanguageCode
from kudiguard.languages import LanguageRegistry
from kudiguard.text import tokens

MAX_COMMAND_CHARS = 60  # longer messages are content, never commands
_BRAND = "kudiguard"  # "hi kudiguard" is still a greeting


class Command(StrEnum):
    STOP = "stop"
    DELETE = "delete"
    LANGUAGE_MENU = "language"


@dataclass(frozen=True, slots=True)
class Match:
    command: Command | None = None
    language: LanguageCode | None = None  # a language chosen by name ("Hausa", "Yorùbá")
    greeting_language: LanguageCode | None = None  # the message is only greetings/thanks in this language


def canonical(text: str) -> str:
    return " ".join(tokens(text))


class CommandMatcher:
    def __init__(self, registry: LanguageRegistry) -> None:
        self._commands: dict[str, Command] = {}
        self._languages: dict[str, LanguageCode] = {}
        self._greetings: list[tuple[LanguageCode, re.Pattern[str]]] = []
        for lang in registry.enabled:
            spec = lang.commands
            for command, phrases in (
                (Command.STOP, spec.stop),
                (Command.DELETE, spec.delete),
                (Command.LANGUAGE_MENU, spec.language),
            ):
                for phrase in phrases:
                    key = canonical(phrase)
                    if key and self._commands.setdefault(key, command) is not command:
                        raise ValueError(f"{lang.code}.yaml: {phrase!r} is configured for two commands")
            for name in (lang.name, lang.native_name):
                if key := canonical(name):
                    self._languages[key] = lang.code
            phrases = sorted({canonical(g) for g in spec.greetings} - {""}, key=len, reverse=True)
            if phrases:
                alternatives = "|".join(re.escape(p) for p in [*phrases, _BRAND])
                self._greetings.append((lang.code, re.compile(rf"(?:(?:{alternatives})(?: |$))+")))

    def match(self, text: str | None) -> Match:
        if not text or len(text) > MAX_COMMAND_CHARS:
            return Match()
        key = canonical(text)
        if not key:
            return Match()
        if command := self._commands.get(key):
            return Match(command=command)
        if language := self._languages.get(key):
            return Match(language=language)
        for code, pattern in self._greetings:
            if pattern.fullmatch(key + " ") and key != _BRAN***REMOVED***
                return Match(greeting_language=code)
        return Match()
