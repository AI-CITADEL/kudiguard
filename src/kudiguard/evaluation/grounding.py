"""Grounding for the answer eval: workstream B's check (``kudiguard.safety.grounding``) when it is installed, else a
stand-in with the same names and contract, so the merge is a switch, not a rewrite (``make_grounding``).

The evidence is what the model was shown (B's ``evidence_for``): the facts and actions of ``prompts.facts_block``,
the verdict's meaning and the red flags; the user's English message goes to ``check(message=...)`` (B uses it for
specifics a reply may repeat, e.g. the amount the user named). Once B merges, the eval engine must be built like
production (grounding on): ungrounded English replies are then replaced before they reach the user, so the English
grounded rate is ~100% by construction and the measure that matters is the other languages (via back-translation).
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

import numpy as np

from kudiguard.domain import Flag, LanguageCode, Verdict
from kudiguard.evaluation.answers import Specifics, names_in, numbers_in, split_sentences, words_in
from kudiguard.knowledge.embeddings import Embedder
from kudiguard.knowledge.models import KnowledgeResult
from kudiguard.languages import PIVOT, LanguageRegistry
from kudiguard.llm.base import LLM, LLMError
from kudiguard.prompts import FLAG_TEXT, clean_translation, facts_block, translate_request

GROUNDING_MODULE = "kudiguard.safety.grounding"  # workstream B
DEFAULT_SUPPORT_THRESHOLD = (
    0.75  # stand-in only; mirrors B's DEFAULT_THRESHOLD (``support_threshold`` reads B's)
)
MIN_CLAIM_WORDS = 3  # "Stay calm." carries no checkable fact
VERDICT_LINE: dict[Verdict, str] = {  # as in B
    Verdict.LIKELY_SCAM: "This looks like a scam.",
    Verdict.CAUTION: "There are warning signs. Check with your bank directly before you act.",
    Verdict.NO_FLAGS: "No warning signs were found. Still confirm with your bank before you act.",
}


class GroundingError(RuntimeError):
    """The check could not run (no usable back-translation, out of time): the answer is left unscored."""


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    support: float
    supported: bool
    unmatched: tuple[str, ...] = ()  # numbers and names no evidence contains


@dataclass(frozen=True, slots=True)
class GroundingReport:
    english: str
    claims: tuple[Claim, ...]
    ms: int = 0

    @property
    def grounded(self) -> bool:
        return all(c.supported for c in self.claims)

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(c.text for c in self.claims if not c.supported)


def split_claims(english: str) -> list[str]:
    """Checkable claims: sentences with at least ``MIN_CLAIM_WORDS`` words, lead-ins ("What to do:") skipped."""
    return [
        s
        for s in split_sentences(english)
        if len([w for w in s.split() if any(ch.isalpha() for ch in w)]) >= MIN_CLAIM_WORDS
        and not s.rstrip().endswith(":")
    ]


def evidence_for(
    knowledge: KnowledgeResult,
    *,
    verdict: Verdict | None = None,
    flags: Sequence[Flag] = (),
) -> list[str]:
    """What a COMPOSE reply may say (mirrors B): facts and actions shown, verdict line, red flags."""
    facts, actions = facts_block(knowledge)
    evidence = [*facts, *actions]
    if verdict is not None and verdict in VERDICT_LINE:
        evidence.append(VERDICT_LINE[verdict])
    evidence.extend(f"The message {FLAG_TEXT[f]}." for f in flags)
    return evidence


class Grounding(Protocol):
    def evidence(
        self, knowledge: KnowledgeResult, *, verdict: Verdict, flags: Sequence[Flag]
    ) -> list[str]: ...

    async def ground(
        self, reply: str, language: LanguageCode, evidence: Sequence[str], *, message: str = ""
    ) -> GroundingReport:
        """``message``: the user's English message. Raises ``GroundingError``."""
        ...


class CheckerGrounding:
    """Adapter over workstream B's module: ``evidence_for`` and ``GroundingChecker.check``."""

    def __init__(self, module: ModuleType, checker: Any) -> None:
        self._module = module
        self._checker = checker

    def evidence(self, knowledge: KnowledgeResult, *, verdict: Verdict, flags: Sequence[Flag]) -> list[str]:
        return list(self._module.evidence_for(knowledge, verdict=verdict, flags=flags))

    async def ground(
        self, reply: str, language: LanguageCode, evidence: Sequence[str], *, message: str = ""
    ) -> GroundingReport:
        try:
            report = await self._checker.check(reply, language, list(evidence), message=message)
        except self._module.GroundingError as exc:
            raise GroundingError(str(exc)) from exc
        claims = tuple(
            Claim(str(c.text), float(c.support), bool(c.supported), tuple(getattr(c, "unmatched", ())))
            for c in report.claims
        )
        return GroundingReport(str(report.english), claims, int(getattr(report, "ms", 0)))


class FallbackGrounding:
    """Stand-in until B merges: plain back-translation (N-ATLAS), then a claim is supported when it paraphrases an
    evidence sentence (cosine >= ``threshold``) and its names and numbers appear in the evidence or the message."""

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        registry: LanguageRegistry,
        *,
        threshold: float | None = None,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._registry = registry
        self.threshold = support_threshold() if threshold is None else threshold

    def evidence(self, knowledge: KnowledgeResult, *, verdict: Verdict, flags: Sequence[Flag]) -> list[str]:
        return evidence_for(knowledge, verdict=verdict, flags=flags)

    async def ground(
        self, reply: str, language: LanguageCode, evidence: Sequence[str], *, message: str = ""
    ) -> GroundingReport:
        english = reply if language == PIVOT else await self._translate(reply, language)
        claims = split_claims(english)
        units = list(dict.fromkeys([*evidence, *(s for text in evidence for s in split_sentences(text))]))
        support = np.zeros(len(claims), dtype=np.float32)
        if claims and units:
            vectors = self._embedder.embed_documents([*claims, *units])
            support = (vectors[: len(claims)] @ vectors[len(claims) :].T).max(axis=1)
        known = " ".join([*evidence, message])  # the message counts for specifics only (as in B)
        known_numbers, known_words = numbers_in(known), words_in(known)
        scored = []
        for text, best in zip(claims, support, strict=True):
            unmatched = tuple(sorted((numbers_in(text) - known_numbers) | (names_in(text) - known_words)))
            value = round(float(best), 4)
            scored.append(Claim(text, value, value >= self.threshold and not unmatched, unmatched))
        return GroundingReport(english, tuple(scored))

    async def _translate(self, reply: str, language: LanguageCode) -> str:
        request = translate_request(reply, self._registry.get(language).name)
        try:
            english = clean_translation((await self._llm.complete(request)).text)
        except LLMError as exc:
            raise GroundingError(f"back-translation failed: {exc}") from exc
        if not english:
            raise GroundingError("back-translation was empty")
        return english


def _workstream_b() -> ModuleType | None:
    try:
        return importlib.import_module(GROUNDING_MODULE)
    except ImportError:
        return None


def make_grounding(llm: LLM, embedder: Embedder, registry: LanguageRegistry) -> Grounding:
    """Workstream B's ``GroundingChecker`` when installed, else ``FallbackGrounding``. ``llm`` should be the plain
    backend (no evaluation sampling overrides), so translator noise stays out of an A/B comparison."""
    module = _workstream_b()
    if module is None:
        return FallbackGrounding(llm, embedder, registry)
    return CheckerGrounding(module, module.GroundingChecker(llm, embedder, registry))


def text_specifics() -> tuple[Specifics, Specifics]:
    """(names_in, numbers_in): B's when installed, else the scorer's mirrors."""
    module = _workstream_b()
    if module is None:
        return names_in, numbers_in
    return module.names_in, module.numbers_in


def support_threshold() -> float:
    """The grounding threshold in use: B's ``DEFAULT_THRESHOLD`` when installed, else the stand-in's mirror."""
    module = _workstream_b()
    return DEFAULT_SUPPORT_THRESHOLD if module is None else float(module.DEFAULT_THRESHOLD)
