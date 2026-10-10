"""Checks every generated reply before it reaches a user.

Language-specific phrases live in ``data/languages/<code>.yaml`` under ``output_guard`` (config, not code).
A hit means the pipeline retries once, then falls back to fixed text: the model must never tell a user that
something is safe or not a scam, never tell them to do what the scammer wants (share a code, pay a fee, click a link,
install an app, send money back), and a refusal is never a useful answer.

Unsafe instructions are matched as imperatives; a hit is dropped when a negation ("do not", "kada", "ma se") comes
earlier in the same sentence. That is deliberately lenient: "Don't wait, share the code" is missed rather than
blocking every "do not share the code"; the grounding check and the fixed headline are the other layers.
"""

from __future__ import annotations

from enum import StrEnum

from kudiguard.domain import LanguageCode
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.scam_rules import Patterns
from kudiguard.text import normalize


class OutputIssue(StrEnum):
    SAFETY_CLAIM = "safety_claim"
    UNSAFE_INSTRUCTION = "unsafe_instruction"
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
        if _unsafe_instruction(folded, patterns.unsafe_instructions, patterns.negations):
            issues.append(OutputIssue.UNSAFE_INSTRUCTION)
        return tuple(issues)


_SENTENCE_ENDS = ".!?;\n"


def _unsafe_instruction(folded: str, instructions: Patterns, negations: Patterns) -> bool:
    for pattern in instructions:
        for match in pattern.finditer(folded):
            before = folded[: match.start()]
            clause = before[
                max(before.rfind(c) for c in _SENTENCE_ENDS) + 1 :
            ]  # this sentence, up to the hit
            if not any(n.search(clause) for n in negations):
                return True
    return False
