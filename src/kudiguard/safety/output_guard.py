"""Checks every generated reply before it reaches a user.

Language-specific phrases live in ``data/languages/<code>.yaml`` under ``output_guard`` (config, not code).
A hit means the pipeline retries once, then falls back to fixed text: the model must never tell a user that
something is safe or not a scam, and a refusal is never a useful answer.
"""

from __future__ import annotations

from enum import StrEnum

from kudiguard.domain import LanguageCode
from kudiguard.languages import LanguageRegistry
from kudiguard.text import normalize


class OutputIssue(StrEnum):
    SAFETY_CLAIM = "safety_claim"
    REFUSAL = "refusal"
    EMPTY = "empty"
    UNGROUNDED = "ungrounded"  # a claim not supported by the evidence (safety.grounding); fixed text instead


class OutputGuard:
    def __init__(self, registry: LanguageRegistry) -> None:
        self._registry = registry

    def check(self, text: str, language: LanguageCode) -> tuple[OutputIssue, ...]:
        folded = normalize(text)
        if not folded:
            return (OutputIssue.EMPTY,)
        patterns = self._registry.guard_patterns(language)
        issues = []
        if any(p.search(folded) for p in patterns.safety_claims):
            issues.append(OutputIssue.SAFETY_CLAIM)
        if any(p.search(folded) for p in patterns.refusals):
            issues.append(OutputIssue.REFUSAL)
        return tuple(issues)
