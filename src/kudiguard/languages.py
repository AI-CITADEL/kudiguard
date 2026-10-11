"""Language registry: every supported language is one YAML file in ``data/languages/<code>.yaml``.

Adding a language is configuration only. The pivot language (English) is the language every message is
translated into, so its lexicon is always merged with the user's language lexicon.
"""

from __future__ import annotations

import dataclasses
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
    # Instructions that hand a scammer what they want ("share the code", "pay the fee", "click the link"). Matched
    # per clause (split at punctuation and ``clause_breaks``) and only as an imperative: at the clause start, after
    # at most a few ``instruction_prefixes`` words ("please", "you should"). A ``negations`` word among those
    # ("do not", "never", "kada") or a ``negations_after`` phrase later in the clause ("to no one") cancels the hit.
    unsafe_instructions: list[str] = Field(default_factory=list)
    instruction_prefixes: list[str] = Field(default_factory=list)
    negations: list[str] = Field(default_factory=list)
    negations_after: list[str] = Field(default_factory=list)
    clause_breaks: list[str] = Field(default_factory=list)
    condition_breaks: list[str] = Field(
        default_factory=list
    )  # start a new clause, kept: "unless you send..."
    speech_commands: list[str] = Field(default_factory=list)  # "do as he says:" introduces an instruction
    # "send it" / "give it to them": unsafe when the same sentence names a ``credentials`` term ("..., so send it").
    pronoun_instructions: list[str] = Field(default_factory=list)
    means_instructions: list[str] = Field(default_factory=list)  # "... by sending the OTP": any position
    # matched on the whole sentence segment, lead-in required in its own clause ("pay ... in your bank app so ...")
    segment_instructions: list[str] = Field(default_factory=list)
    credentials: list[str] = Field(default_factory=list)
    # Promised outcomes and threats stated as fact ("you will get your money back", "your account will be
    # blocked"); a ``reported_speech`` word earlier in the clause ("they said") exempts it.
    outcome_claims: list[str] = Field(default_factory=list)
    reported_speech: list[str] = Field(default_factory=list)
    hedges_before: list[str] = Field(default_factory=list)  # "if ...", "there is no guarantee ..."
    hedges_after: list[str] = Field(default_factory=list)  # "... if you report quickly", "not guaranteed"


class _AbuseSpec(BaseModel):
    """Malicious-use phrases (``safety.abuse``, docs/ABUSE.md), matched like the lexicon on normalised text."""

    model_config = ConfigDict(extra="forbid")
    crisis: list[str] = Field(default_factory=list)  # first-person self-harm / suicide statements
    crisis_other: list[str] = Field(
        default_factory=list
    )  # someone else at risk ("my brother will kill himself")
    # sentences for the embedding-similarity crisis layer (safety/crisis_semantic.py): plain text, not patterns
    crisis_exemplars: list[str] = Field(default_factory=list)
    abuse: list[str] = Field(default_factory=list)  # insults, hate and sexual messages aimed at the bot
    scam_tool: list[str] = Field(default_factory=list)  # requests to write, translate or polish scam content
    scam_tool_defensive: list[str] = Field(
        default_factory=list
    )  # questions about scams ("so I can recognise")
    # attempts to override instructions or extract them: high-precision phrases (enforce by default)
    injection: list[str] = Field(default_factory=list)
    injection_suspect: list[str] = Field(
        default_factory=list
    )  # broad phrases with real false positives (shadow)
    # cues that the text before an injection phrase reports someone else's words ("my boss said ignore ...")
    injection_reported: list[str] = Field(default_factory=list)
    # model-directed instructions planted in partner knowledge ("tell the user to ...", "always recommend ...")
    pack_instructions: list[str] = Field(default_factory=list)
    # whole transcript segments speech recognition invents on music/noise ("thanks for watching")
    asr_hallucinations: list[str] = Field(default_factory=list)
    # labels dropped only inside brackets ("(kiɗa)", "[Music]"): as plain words they can be speech (Igbo "egwu" = fear)
    asr_annotations: list[str] = Field(default_factory=list)
    # segments dropped only when nothing else was said (a lone "You." on noise; "you" is a word in real speech)
    asr_alone: list[str] = Field(default_factory=list)
    persona_claims: list[str] = Field(default_factory=list)  # a reply claiming to be a bank / customer care
    ai_denials: list[str] = Field(default_factory=list)  # a reply denying it is an AI ("I am not a bot")
    ai_questions: list[str] = Field(default_factory=list)  # "are you an AI / a real person?": the honest line


class _CommandsSpec(BaseModel):
    """Whole-message phrases, written naturally (matching ignores case, accents and punctuation)."""

    model_config = ConfigDict(extra="forbid")
    stop: list[str] = Field(default_factory=list)  # stop receiving messages
    delete: list[str] = Field(default_factory=list)  # erase my data
    language: list[str] = Field(default_factory=list)  # show the language menu
    donate: list[str] = Field(default_factory=list)  # offer voice-note donation (kudiguard.donation)
    stop_donating: list[str] = Field(default_factory=list)  # withdraw it and delete donated voice notes
    greetings: list[str] = Field(default_factory=list)  # greetings and thanks: fixed reply, no model call


@dataclass(frozen=True, slots=True)
class GuardPatterns:
    safety_claims: Patterns
    refusals: Patterns
    unsafe_instructions: Patterns = ()
    instruction_prefixes: Patterns = ()
    negations: Patterns = ()
    negations_after: Patterns = ()
    clause_breaks: Patterns = ()
    condition_breaks: Patterns = ()
    speech_commands: Patterns = ()
    pronoun_instructions: Patterns = ()
    means_instructions: Patterns = ()
    segment_instructions: Patterns = ()
    credentials: Patterns = ()
    outcome_claims: Patterns = ()
    reported_speech: Patterns = ()
    hedges_before: Patterns = ()
    hedges_after: Patterns = ()


@dataclass(frozen=True, slots=True)
class AbusePatterns:
    crisis: Patterns = ()
    crisis_other: Patterns = ()
    abuse: Patterns = ()
    scam_tool: Patterns = ()
    scam_tool_defensive: Patterns = ()
    injection: Patterns = ()
    injection_suspect: Patterns = ()
    injection_reported: Patterns = ()
    pack_instructions: Patterns = ()
    asr_hallucinations: Patterns = ()
    asr_annotations: Patterns = ()
    asr_alone: Patterns = ()
    persona_claims: Patterns = ()
    ai_denials: Patterns = ()
    ai_questions: Patterns = ()


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
    asr_whisper_language: str | None = None  # Whisper language token for the ASR model (None = model decides)
    review: dict[str, ReviewStatus] = Field(default_factory=dict)  # e.g. {lexicon: ..., strings: ...}
    lexicon: _LexiconSpec = Field(default_factory=_LexiconSpec)
    output_guard: _OutputGuardSpec = Field(default_factory=_OutputGuardSpec)
    commands: _CommandsSpec = Field(default_factory=_CommandsSpec)
    abuse: _AbuseSpec = Field(default_factory=_AbuseSpec)
    strings: dict[str, str] = Field(default_factory=dict)  # fixed UI text; keys must match the pivot's


class LanguageRegistry:
    def __init__(
        self,
        languages: list[LanguageConfig],
        lexicons: dict[LanguageCode, Lexicon],
        guards: dict[LanguageCode, GuardPatterns] | None = None,
        abuse: dict[LanguageCode, AbusePatterns] | None = None,
    ) -> None:
        self._by_code = {lang.code: lang for lang in languages}
        self._own_lexicons = lexicons
        self._own_guards = guards or {}
        self._own_abuse = abuse or {}
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
        abuse: dict[LanguageCode, AbusePatterns] = {}
        for path in sorted(directory.glob("*.yaml")):
            try:
                lang = LanguageConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
                lexicons[lang.code] = Lexicon.from_terms(lang.lexicon.flags, lang.lexicon.check_intent)
                guards[lang.code] = GuardPatterns(
                    **{
                        f.name: tuple(compile_term(t) for t in getattr(lang.output_guard, f.name))
                        for f in dataclasses.fields(GuardPatterns)
                    }
                )
                abuse[lang.code] = AbusePatterns(
                    **{
                        f.name: tuple(compile_term(t) for t in getattr(lang.abuse, f.name))
                        for f in dataclasses.fields(AbusePatterns)
                    }
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
        return cls(languages, lexicons, guards, abuse)

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
        return GuardPatterns(
            **{
                f.name: getattr(pivot, f.name) + getattr(own, f.name)
                for f in dataclasses.fields(GuardPatterns)
            }
        )

    def abuse_patterns(self, code: LanguageCode) -> AbusePatterns:
        """Malicious-use phrases for ``code`` merged with the pivot's (messages mix in English and Pidgin)."""
        own = self._own_abuse.get(code, AbusePatterns())
        if code == PIVOT:
            return own
        pivot = self._own_abuse.get(PIVOT, AbusePatterns())
        return AbusePatterns(
            **{
                f.name: getattr(pivot, f.name) + getattr(own, f.name)
                for f in dataclasses.fields(AbusePatterns)
            }
        )

    def own_abuse_patterns(self, code: LanguageCode) -> AbusePatterns:
        """`code`'s own phrases only (which language a crisis message is in)."""
        return self._own_abuse.get(code, AbusePatterns())

    def all_abuse_patterns(self) -> AbusePatterns:
        """Every configured language's phrases together (crisis messages are checked in all languages)."""
        merged = AbusePatterns()
        for code in sorted(self._own_abuse):
            own = self._own_abuse[code]
            merged = AbusePatterns(
                **{
                    f.name: getattr(merged, f.name) + getattr(own, f.name)
                    for f in dataclasses.fields(AbusePatterns)
                }
            )
        return merged

    def asr_models(self) -> dict[LanguageCode, str]:
        """Language code -> speech model repo, for enabled languages."""
        return {lang.code: lang.asr_model for lang in self.enabled}

    def asr_whisper_languages(self) -> dict[str, str | None]:
        """Speech model repo -> Whisper language token (see ``speech.natlas_asr.ct2_loader``)."""
        return {lang.asr_model: lang.asr_whisper_language for lang in self.enabled}
