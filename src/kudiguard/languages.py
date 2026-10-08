"""Language registry: every supported language is one YAML file in ``data/languages/<code>.yaml``.

Adding a language is configuration only. The pivot language (English) is the language every message is
translated into, so its lexicon is always merged with the user's language lexicon.
"""

from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from kudiguard.domain import LanguageCode
from kudiguard.safety.scam_rules import Lexicon, Patterns, compile_term

PIVOT: LanguageCode = "en"
BUTTON_PREFIX = "btn_"
BUTTON_MAX_CHARS = 20  # WhatsApp interactive reply-button title limit


class LanguageConfigError(ValueError):
    pass


def _placeholders(code: LanguageCode, key: str, template: str) -> frozenset[str]:
    """Named ``{fields}`` of a message template; malformed templates fail loudly at load time."""
    try:
        fields = [name for _, name, _, _ in string.Formatter().parse(template) if name is not None]
    except ValueError as exc:
        raise LanguageConfigError(f"{code}.yaml: {key!r} is not a valid template: {exc}") from exc
    if any(not name.isidentifier() for name in fields):
        raise LanguageConfigError(f"{code}.yaml: {key!r} placeholders must be names, e.g. {{amount}}")
    return frozenset(fields)


class LanguageStatus(StrEnum):
    STABLE = "stable"
    BETA = "beta"
    DISABLED = "disabled"


class ReviewStatus(StrEnum):
    PENDING = "pending-native-review"
    REVIEWED = "reviewed"


class _LexiconSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flags: dict[str, list[str]] = Field(default_factory=dict)
    check_intent: list[str] = Field(default_factory=list)


class _OutputGuardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    safety_claims: list[str] = Field(default_factory=list)  # "this is safe / not a scam" in this language
    refusals: list[str] = Field(default_factory=list)  # "I can't help" in this language


@dataclass(frozen=True, slots=True)
class GuardPatterns:
    safety_claims: Patterns
    refusals: Patterns


def _ui_length(text: str) -> int:
    """Length as WhatsApp counts it: UTF-16 code units of the NFC form (emoji count 2, tone marks may add 1)."""
    return len(unicodedata.normalize("NFC", text).encode("utf-16-le")) // 2


class LanguageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Annotated[str, StringConstraints(pattern=r"^[a-z]{2,3}$")]
    name: str
    native_name: str
    order: int
    status: LanguageStatus
    asr_model: str
    review: dict[str, ReviewStatus] = Field(default_factory=dict)  # e.g. {lexicon: ..., strings: ...}
    lexicon: _LexiconSpec = Field(default_factory=_LexiconSpec)
    output_guard: _OutputGuardSpec = Field(default_factory=_OutputGuardSpec)
    strings: dict[str, str] = Field(default_factory=dict)  # fixed UI text; keys must match the pivot's


class LanguageRegistry:
    def __init__(
        self,
        languages: list[LanguageConfig],
        lexicons: dict[LanguageCode, Lexicon],
        guards: dict[LanguageCode, GuardPatterns] | None = None,
    ) -> None:
        self._by_code = {lang.code: lang for lang in languages}
        self._own_lexicons = lexicons
        self._own_guards = guards or {}
        self._merged: dict[LanguageCode, Lexicon] = {}
        self.enabled: tuple[LanguageConfig, ...] = tuple(
            sorted(
                (lang for lang in languages if lang.status is not LanguageStatus.DISABLED),
                key=lambda x: x.order,
            )
        )

    @classmethod
    def load(cls, directory: Path) -> LanguageRegistry:
        languages: list[LanguageConfig] = []
        lexicons: dict[LanguageCode, Lexicon] = {}
        guards: dict[LanguageCode, GuardPatterns] = {}
        for path in sorted(directory.glob("*.yaml")):
            try:
                lang = LanguageConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
                lexicons[lang.code] = Lexicon.from_terms(lang.lexicon.flags, lang.lexicon.check_intent)
                guards[lang.code] = GuardPatterns(
                    tuple(compile_term(t) for t in lang.output_guard.safety_claims),
                    tuple(compile_term(t) for t in lang.output_guard.refusals),
                )
            except (ValidationError, ValueError, yaml.YAMLError) as exc:
                raise LanguageConfigError(f"{path.name}: {exc}") from exc
            if path.stem != lang.code:
                raise LanguageConfigError(f"{path.name}: file name must match code {lang.code!r}")
            languages.append(lang)

        if PIVOT not in lexicons:
            raise LanguageConfigError(f"pivot language {PIVOT!r} must be configured in {directory}")
        pivot_keys = next(lang for lang in languages if lang.code == PIVOT).strings.keys()
        for lang in languages:
            if missing := sorted(pivot_keys - lang.strings.keys()):
                raise LanguageConfigError(f"{lang.code}.yaml: missing strings {missing}")
            if extra := sorted(lang.strings.keys() - pivot_keys):
                raise LanguageConfigError(f"{lang.code}.yaml: strings not in pivot {extra}")
        pivot = next(lang for lang in languages if lang.code == PIVOT)
        expected = {key: _placeholders(PIVOT, key, text) for key, text in pivot.strings.items()}
        for lang in languages:
            for key, text in lang.strings.items():
                if _placeholders(lang.code, key, text) != expected[key]:
                    raise LanguageConfigError(
                        f"{lang.code}.yaml: {key!r} placeholders differ from the pivot's {sorted(expected[key])}"
                    )
                if key.startswith(BUTTON_PREFIX) and (size := _ui_length(text)) > BUTTON_MAX_CHARS:
                    raise LanguageConfigError(
                        f"{lang.code}.yaml: {key!r} is {size} chars; buttons allow {BUTTON_MAX_CHARS}"
                    )
        orders = [lang.order for lang in languages]
        if len(orders) != len(set(orders)):
            raise LanguageConfigError(f"duplicate 'order' values in {directory}")
        return cls(languages, lexicons, guards)

    def __contains__(self, code: object) -> bool:
        return code in self._by_code

    def get(self, code: LanguageCode) -> LanguageConfig:
        return self._by_code[code]

    def lexicon_for(self, code: LanguageCode) -> Lexicon:
        """User-language lexicon merged with the pivot lexicon (cached)."""
        if code not in self._merged:
            own = self._own_lexicons[code]
            pivot = self._own_lexicons[PIVOT]
            self._merged[code] = own if code == PIVOT else pivot.merge(own)
        return self._merged[code]

    def guard_patterns(self, code: LanguageCode) -> GuardPatterns:
        """Output-guard patterns for ``code`` merged with the pivot's (replies may mix in English)."""
        own = self._own_guards.get(code, GuardPatterns((), ()))
        if code == PIVOT:
            return own
        pivot = self._own_guards.get(PIVOT, GuardPatterns((), ()))
        return GuardPatterns(pivot.safety_claims + own.safety_claims, pivot.refusals + own.refusals)
