"""Language registry: every supported language is one YAML file in ``data/languages/<code>.yaml``.

Adding a language is configuration only. The pivot language (English) is the language every message is
translated into, so its lexicon is always merged with the user's language lexicon.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from kudiguard.domain import LanguageCode
from kudiguard.safety.scam_rules import Lexicon

PIVOT: LanguageCode = "en"


class LanguageConfigError(ValueError):
    pass


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
    strings: dict[str, str] = Field(default_factory=dict)  # fixed UI text; keys must match the pivot's


class LanguageRegistry:
    def __init__(self, languages: list[LanguageConfig], lexicons: dict[LanguageCode, Lexicon]) -> None:
        self._by_code = {lang.code: lang for lang in languages}
        self._own_lexicons = lexicons
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
        for path in sorted(directory.glob("*.yaml")):
            try:
                lang = LanguageConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
                lexicons[lang.code] = Lexicon.from_terms(lang.lexicon.flags, lang.lexicon.check_intent)
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
        orders = [lang.order for lang in languages]
        if len(orders) != len(set(orders)):
            raise LanguageConfigError(f"duplicate 'order' values in {directory}")
        return cls(languages, lexicons)

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
