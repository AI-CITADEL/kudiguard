"""Checks every generated reply before it reaches a user.

Language-specific phrases live in ``data/languages/<code>.yaml`` under ``output_guard`` (config, not code).
A hit means the pipeline retries once, then falls back to fixed text: the model must never tell a user that
something is safe or not a scam, never tell them to do what the scammer wants (share a code, pay a fee, click a link,
install an app, send money back), and a refusal is never a useful answer.

Unsafe instructions are matched per clause (split at punctuation and words such as "and", "but", "just", "so",
"then") and only as imperatives: the verb at the clause start, after at most ``MAX_PREFIX_WORDS`` words from the
language's ``instruction_prefixes`` ("please", "you should", "do not"). So "Don't worry, just send them the OTP" and
"As they said, send the OTP" are caught, while "Ask your bank if they can refund it" and "They said you should send
the OTP" (a subject before the verb: description, not advice) are not. A negation among those prefix words ("do not",
"never", "kada") or a ``negations_after`` phrase in the clause ("send the code to no one") cancels the hit. A bare
"send it" counts when the sentence names a credential ("He said you should send the OTP, so send it").

``outcome_claims`` (promised outcomes and threats stated as fact: "you will get your money back", "your account will
be blocked") are reported as ``OUTCOME_CLAIM``; callers decide whether that blocks (enforce) or is only logged.
"""

from __future__ import annotations

import re
from enum import StrEnum

from kudiguard.domain import LanguageCode
from kudiguard.languages import GuardPatterns, LanguageRegistry
from kudiguard.safety.scam_rules import Patterns
from kudiguard.text import normalize

MAX_PREFIX_WORDS = 4  # "please do not ever share ..." / "you need to send ..."


class OutputIssue(StrEnum):
    SAFETY_CLAIM = "safety_claim"
    UNSAFE_INSTRUCTION = "unsafe_instruction"
    OUTCOME_CLAIM = "outcome_claim"  # blocking only in enforce mode (see engine)
    REFUSAL = "refusal"
    EMPTY = "empty"
    UNGROUNDED = "ungrounded"  # a claim not supported by the evidence (safety.grounding)


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
        unsafe = outcome = False
        for sentence in _SENTENCE_BREAK.split(folded):
            names_credential = any(c.search(sentence) for c in patterns.credentials)
            for clause in _clauses(sentence, patterns):
                instructions = patterns.unsafe_instructions
                if names_credential:
                    instructions += patterns.pronoun_instructions
                unsafe = unsafe or _unsafe_instruction(clause, instructions, patterns)
            outcome = outcome or _outcome_claim(sentence, patterns)
        if unsafe:
            issues.append(OutputIssue.UNSAFE_INSTRUCTION)
        if outcome:
            issues.append(OutputIssue.OUTCOME_CLAIM)
        return tuple(issues)


_SENTENCE_BREAK = re.compile(r"[.!?](?=\s|$)|\n")
_CLAUSE_BREAK = re.compile(r"[;:]|,(?=\s)")
_WORD = re.compile(r"[\w']+")


def _clauses(sentence: str, patterns: GuardPatterns) -> list[str]:
    clauses = [c.strip() for c in _CLAUSE_BREAK.split(sentence)]
    for breaker in patterns.clause_breaks:
        clauses = [part.strip() for c in clauses for part in breaker.split(c)]
    return [c for c in clauses if c]


def _unsafe_instruction(clause: str, instructions: Patterns, patterns: GuardPatterns) -> bool:
    allowed = patterns.instruction_prefixes + patterns.negations
    for pattern in instructions:
        for match in pattern.finditer(clause):
            prefix = [w for w in _WORD.findall(clause[: match.start()]) if not w.isdigit()]
            if len(prefix) > MAX_PREFIX_WORDS or not all(
                any(p.fullmatch(w) for p in allowed) for w in prefix
            ):
                continue  # a subject or other words before the verb: not an instruction to the user
            if any(n.fullmatch(w) for n in patterns.negations for w in prefix):
                continue
            if any(n.search(clause, match.end()) for n in patterns.negations_after):
                continue
            return True
    return False


def _outcome_claim(sentence: str, patterns: GuardPatterns) -> bool:
    for pattern in patterns.outcome_claims:
        for match in pattern.finditer(sentence):
            if not any(r.search(sentence, 0, match.start()) for r in patterns.reported_speech):
                return True
    return False
