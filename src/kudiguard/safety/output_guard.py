"""Checks every generated reply before it reaches a user.

Language-specific phrases live in ``data/languages/<code>.yaml`` under ``output_guard`` (config, not code).
A hit means the pipeline retries once, then falls back to fixed text: the model must never tell a user that
something is safe or not a scam, never tell them to do what the scammer wants (share a code, pay a fee, click a link,
install an app, send money back), and a refusal is never a useful answer.

Unsafe instructions are matched per clause (split at punctuation, ``clause_breaks`` such as "and", "but", "just",
and before ``condition_breaks`` such as "unless") and only as instructions to the user: everything before the verb
must be lead-in from ``instruction_prefixes`` ("please", "make sure you", "you can safely", "I recommend you",
"unless you"). A subject or other words before the verb make it a description ("They said you should send the OTP",
"Ask your bank if they can refund it"). A negation in that lead-in ("do not", "never", "kada") or a
``negations_after`` phrase ("send the code to no one") cancels the hit. Questions are not instructions. After a
colon, text introduced by reported speech ("Scammers often say: send the code") or by a negated lead-in ("Never tell
the caller: here is my PIN") is not one either, but "Do as he says: send him the OTP" is. A bare "send it" counts when
the sentence names a credential, unless it goes to a legitimate recipient ("send it to your bank").

``outcome_claims`` (promised outcomes and threats stated as fact: "you will get your money back", "your money is
safe") are reported as ``OUTCOME_CLAIM`` unless reported or hedged; callers decide whether that blocks (enforce) or
is only logged. The probe set in ``tests/fixtures/guard_probes.yaml`` pins the expected behaviour.
"""

from __future__ import annotations

import itertools
import re
from enum import StrEnum

from kudiguard.domain import LanguageCode
from kudiguard.languages import GuardPatterns, LanguageRegistry
from kudiguard.safety.scam_rules import Patterns
from kudiguard.text import normalize

NEGATION_WINDOW = 4  # lead-in words in which a negation counts ("please do not ever share")


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
        sentences = [(m.group(1), m.group(2).startswith("?")) for m in _SENTENCE.finditer(folded)]
        if any(_unsafe_sentence(s, patterns) for s, question in sentences if not question):
            issues.append(OutputIssue.UNSAFE_INSTRUCTION)
        if any(_outcome_claim(s, patterns) for s, _ in sentences):
            issues.append(OutputIssue.OUTCOME_CLAIM)
        return tuple(issues)


_SENTENCE = re.compile(r"([^\n]*?)([.!?]+['\"\u201d\u2019)]*(?=\s|$)|\n|$)")
_CLAUSE_BREAK = re.compile(r";|,(?=\s)")
_WORD = re.compile(r"[\w']+")


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text) if not w.isdigit()]


def _matches_any(patterns: Patterns, word: str) -> bool:
    return any(p.fullmatch(word) for p in patterns)


def _unsafe_sentence(sentence: str, patterns: GuardPatterns) -> bool:
    if not sentence.strip():
        return False
    instructions = patterns.unsafe_instructions
    if any(c.search(sentence) for c in patterns.credentials):
        instructions += patterns.pronoun_instructions
    segments = sentence.split(":")
    for index, segment in enumerate(segments):
        if index and _quoted_or_negated(segments[index - 1], patterns):
            continue
        for clause in _clauses(segment, patterns):
            if _reported_condition(clause, segment, patterns):
                continue  # "They said your account will be blocked unless you send the OTP" reports the threat
            if _unsafe_instruction(clause, instructions, patterns) or _means_instruction(clause, patterns):
                return True
    return False


def _means_instruction(clause: str, patterns: GuardPatterns) -> bool:
    """ "Verify your account by sending the OTP": the means is the instruction, whatever comes before it."""
    for pattern in patterns.means_instructions:
        for match in pattern.finditer(clause):
            if not any(_matches_any(patterns.negations, w) for w in _words(clause[: match.start()])):
                return True
    return False


def _reported_condition(clause: str, segment: str, patterns: GuardPatterns) -> bool:
    if not any(c.match(clause) for c in patterns.condition_breaks):
        return False
    before = segment[: segment.find(clause)]
    return any(r.search(before) for r in patterns.reported_speech)


def _quoted_or_negated(intro: str, patterns: GuardPatterns) -> bool:
    """The text before a colon introduces reported speech ("scammers often say:") or a script the user must not
    say ("never tell the caller:"); "do as he says:" introduces an instruction."""
    clauses = _clauses(intro, patterns)
    words = _words(clauses[-1]) if clauses else []
    if not words:
        return False
    reported = _matches_any(patterns.reported_speech, words[-1]) and not _matches_any(
        patterns.speech_commands, words[0]
    )
    return reported or any(_matches_any(patterns.negations, w) for w in words[:NEGATION_WINDOW])


def _clauses(segment: str, patterns: GuardPatterns) -> list[str]:
    clauses = [c.strip() for c in _CLAUSE_BREAK.split(segment)]
    for breaker in patterns.clause_breaks:
        clauses = [part.strip() for c in clauses for part in breaker.split(c)]
    for condition in patterns.condition_breaks:  # kept at the start of the next clause ("unless you send...")
        clauses = [part.strip() for c in clauses for part in _split_before(condition, c)]
    return [c for c in clauses if c]


def _split_before(pattern: re.Pattern[str], text: str) -> list[str]:
    starts = [m.start() for m in pattern.finditer(text)]
    bounds = [0, *starts, len(text)]
    return [text[a:b] for a, b in itertools.pairwise(bounds) if text[a:b].strip()]


def _is_lead_in(prefix: str, patterns: GuardPatterns) -> bool:
    """Whether everything before the verb is lead-in ("please make sure you"), consumed phrase by phrase."""
    text = " ".join(_words(prefix))
    allowed = patterns.instruction_prefixes + patterns.negations
    position = 0
    while position < len(text):
        end = max((m.end() for p in allowed if (m := p.match(text, position))), default=position)
        if end == position:
            return False
        position = end + 1  # the joining space
    return True


def _unsafe_instruction(clause: str, instructions: Patterns, patterns: GuardPatterns) -> bool:
    for pattern in instructions:
        for match in pattern.finditer(clause):
            prefix = clause[: match.start()]
            if not _is_lead_in(prefix, patterns):
                continue  # a subject or other words before the verb: not an instruction to the user
            if any(_matches_any(patterns.negations, w) for w in _words(prefix)[-NEGATION_WINDOW:]):
                continue
            if any(n.search(clause, match.end()) for n in patterns.negations_after):
                continue
            return True
    return False


def _outcome_claim(sentence: str, patterns: GuardPatterns) -> bool:
    for pattern in patterns.outcome_claims:
        for match in pattern.finditer(sentence):
            before = any(
                r.search(sentence, 0, match.start())
                for r in patterns.reported_speech + patterns.hedges_before
            )
            after = any(h.search(sentence, match.end()) for h in patterns.hedges_after)
            if not (before or after):
                return True
    return False
