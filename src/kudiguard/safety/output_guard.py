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
from dataclasses import dataclass
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
    # the learned second layer (safety.guard_classifier), enforce mode
    CLASSIFIED_UNSAFE = "classified_unsafe"
    REFUSAL = "refusal"
    EMPTY = "empty"
    UNGROUNDED = "ungrounded"  # a claim not supported by the evidence (safety.grounding)
    # malicious-use reply checks (safety.reply_screen, docs/ABUSE.md)
    PERSONA = "persona"  # claims to be a bank / customer care
    PROMPT_LEAK = "prompt_leak"  # repeats KudiGuard's own instructions
    ECHO = "echo"  # copies a long run of the user's message


class OutputGuard:
    def __init__(self, registry: LanguageRegistry) -> None:
        self._registry = registry

    def check(self, text: str, language: LanguageCode) -> tuple[OutputIssue, ...]:
        folded = normalize(text)
        if not folded:
            return (OutputIssue.EMPTY,)
        patterns = self._registry.guard_patterns(language)
        issues = []
        sentences = [(m.group(1), m.group(2).startswith("?")) for m in _SENTENCE.finditer(folded)]
        if any(_safety_claim(s, patterns) for s, question in sentences if not question):
            issues.append(OutputIssue.SAFETY_CLAIM)
        if any(p.search(folded) for p in patterns.refusals):
            issues.append(OutputIssue.REFUSAL)
        # quoted speech ("Fraudsters say \"send your PIN\" to scare you") is not an instruction to the user; a quote
        # the user is told to say is also read as its own clause ('Write "here is my password" in the chat')
        quoted = _unify_quotes(folded)
        unquoted = [
            (m.group(1), m.group(2).startswith("?"))
            for variant in (_mask_speech(quoted), _split_told(quoted))
            for m in _SENTENCE.finditer(variant)
        ]
        if any(_unsafe_sentence(s, patterns) for s, question in unquoted if not question):
            issues.append(OutputIssue.UNSAFE_INSTRUCTION)
        if any(_outcome_claim(s, patterns) for s, _ in sentences):
            issues.append(OutputIssue.OUTCOME_CLAIM)
        return tuple(issues)

    def units(self, text: str, language: LanguageCode) -> list[Unit]:
        """The sentences of a reply and, when a sentence has several, its clauses, for a second (learned) layer.
        ``gated`` marks units a classifier must not block on its own: questions, units negated in their opening
        window, reported speech ("they will probably ask you to ...", "dem fit tell you make you ..."), subordinate
        clauses ("if they ask you to ..."), descriptions ("you send them the code and they ..."), fragments under
        three words, units that start inside double quotes (a quoted script), text before a colon that only
        introduces what follows ("Scammers often say:", "Tell the caller this:") and quoted or negated scripts after
        one ("Tell him: no, ..."). An instruction before a colon ("Reply with your card details: ...") is scored."""
        patterns = self._registry.guard_patterns(language)
        folded = _unify_quotes(normalize(text))
        quoted = _speech_spans(folded)
        units: list[Unit] = []
        for match in _SENTENCE.finditer(folded):
            raw, question = match.group(1), match.group(2).startswith("?")
            sentence = raw.strip()
            if not sentence:
                continue
            start = match.start(1) + len(raw) - len(raw.lstrip())
            in_quotes = _inside(_first_word(folded, start), quoted)
            gated = question or in_quotes or ":" in sentence or _gated(sentence, sentence, patterns)
            units.append(Unit(sentence, gated))
            segments = sentence.split(":")
            clauses: list[tuple[str, bool]] = []
            for index, segment in enumerate(segments):
                script = bool(index) and _quoted_or_negated(segments[index - 1], patterns)
                intro = index < len(segments) - 1 and _introduces(segment, patterns)
                clauses += [(c, script or intro) for c in _clauses(segment, patterns)]
            if len(clauses) > 1:
                cursor = 0
                for clause, off in clauses:
                    offset = sentence.find(clause, cursor)
                    cursor = max(offset, cursor)
                    inside = _inside(_first_word(folded, start + cursor), quoted)
                    units.append(
                        Unit(clause, question or off or inside or _gated(clause, sentence, patterns))
                    )
        return units


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of the text inside closed pairs of double quotes (straight or curly) on one line. An unclosed
    quote marks nothing: a stray quote must not hide the rest of a reply."""
    spans = [(m.start(1), m.end(1)) for m in _CURLY_QUOTED.finditer(text)]
    spans += [(m.start(1), m.end(1)) for m in _STRAIGHT_QUOTED.finditer(text)]
    return spans


def _first_word(text: str, position: int) -> int:
    match = _WORD.search(text, position)
    return match.start() if match else position


def _speech_spans(text: str) -> list[tuple[int, int]]:
    """The closed quotes that are someone else's speech: a quote that opens a sentence or follows a word that
    introduces speech ("say", "write", "like", "message"; see ``_INTRODUCERS``). Not speech: a quote the USER is told
    to say ("Tell him: \"Here is my OTP\"", "Just say \"...\" and read it out", "Text \"your PIN\" to the agent" - the
    clause opens with an imperative speech verb) or one followed by "is what you should do" / "and send it". Other
    quotes are never speech, so quoting the object ("send \"the code\" to him") hides nothing."""
    spans = []
    for start, end in sorted(_quoted_spans(text)):
        before = text[: start - 1].rstrip()
        words = _words(before[-40:])
        introduced = (
            not before or before.endswith((".", "!", "?", ":", "\n")) or (words and words[-1] in _INTRODUCERS)
        )
        clause = _CLAUSE_START.split(before)[-1]
        told = _TOLD_TO_SAY.match(clause) is not None
        refusal = (
            _REFUSAL_SCRIPT.match(text, start) is not None
        )  # 'Tell him "I don't give my PIN"': a refusal
        if (introduced and not told and not _ADOPTED.match(text, end + 1)) or (told and refusal):
            spans.append((start, end))
    return spans


def _split_told(text: str) -> str:
    """``text`` with quoted speech masked and every other quote of 3+ words opened by ";" instead of its mark (and its
    closing mark blanked), so it is checked as its own clause: the user is told to say it."""
    chars = list(text)
    speech = set(_speech_spans(text))
    for start, end in _quoted_spans(text):
        clause = _CLAUSE_START.split(text[: start - 1])[-1]
        if _NEGATED_LEAD.search(clause):  # 'Never tell him "my PIN is 1234"': a warning
            continue
        told = _TOLD_TO_SAY.match(clause) is not None  # told to say it: checked even when it opens with "no"
        if ((start, end) not in speech or told) and len(_words(text[start:end])) >= 3:
            chars[start - 1], chars[end] = ";", " "
    return _mask_speech("".join(chars))


def _unify_quotes(text: str) -> str:
    """Single quotes used as quotation marks (folding turns curly ones into "'") become double quotes; apostrophes
    ("don't", "the bank's") stay. Same length, so positions do not move."""
    return _SINGLE_QUOTED.sub(lambda m: '"' + m.group(1) + '"', text)


# "never tell him", "do not text anyone", "you should never tell anyone"
# ("don't forget / hesitate to tell him ..." inverts the negation: an instruction)
_NEGATED_LEAD = re.compile(
    r"\b(?:not|never|no|don't|dont)\s+"
    r"(?:(?!(?:forget|hesitate|fail|neglect|wait|delay|be|worry|panic|stop)\b)[\w']+\s+){0,2}?"
    r"(?:tell|say|reply|answer|text|write|message|type|send|respond|read)\b"
)
# an elided opener ("'cause", "'em", "'til") is an apostrophe, not a quotation mark
_SINGLE_QUOTED = re.compile(
    r"(?<![\w'])'(?!(?:cause|cos|coz|em|til|tis|twas|n|bout|round)\b)(?=\w)([^\n]*?[\w.!?,])'(?![\w'])"
)
_REFUSAL_SCRIPT = re.compile(
    r"\W*(?:[\w']+\W+){0,3}?(?:no|not|never|don't|dont|won't|wont|can't|cannot|refuse)\b"
)


# the clause tells the user to say or write the quote: "tell him:", "just say", "reply to the caller:", "read out"
_CLAUSE_START = re.compile(r"[.!?\n]\s*")
_TOLD_TO_SAY = re.compile(
    r"\W*(?:(?:when|if|once|after|as\s+soon\s+as|before)\b[^,]*,\s*)?"
    r"(?:(?:please|kindly|just|then|now|simply|quickly|so)\s+)*"
    r"(?:tell|say|reply|answer|text|write|message|read(?:\s+out)?|type|send|respond)\b"
)
# '"Send the OTP to the agent now," is what you should do', '"..." and send it to him'
_ADOPTED = re.compile(
    r"\W*(?:is|are)\s+what\s+you\s+(?:should|must|need\s+to|have\s+to)\s+(?:do|say|send)"
    r"|\W*and\s+(?:send|read|give|type|enter|forward)\s+(?:it|them|this)\b"
)


def _mask_speech(text: str) -> str:
    """``text`` with quoted speech (``_speech_spans``) replaced by "quote", keeping a sentence end inside the quote
    ('said: "Pay." Then ...')."""
    result, last = [], 0
    for start, end in _speech_spans(text):
        inner = text[start:end].rstrip()
        ending = inner[len(inner.rstrip(".!?")) :]
        result.append(text[last:start] + "quote" + ending)
        last = end
    return "".join(result) + text[last:]


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= position < b for a, b in spans)


_CURLY_QUOTED = re.compile("\u201c([^\u201d\n]*)\u201d")
_STRAIGHT_QUOTED = re.compile(r'"([^"\n]*)"')


def _introduces(intro: str, patterns: GuardPatterns) -> bool:
    """Text before a colon that only introduces what follows: reported speech or a negated script (see
    ``_quoted_or_negated``), ending in a word such as "this", "example", "say" or "below", or a speech verb with no
    secret or money in it ("Tell the agent: ..."; but "Reply with your card details: ..." is an instruction)."""
    words = _words(intro)
    if len(words) < 3 or _quoted_or_negated(intro, patterns) or words[-1] in _INTRODUCERS:
        return True
    return words[0] in _SPEECH_VERBS and not _INTRO_OBJECT.search(intro)


_SPEECH_VERBS = frozenset(["tell", "say", "reply", "answer", "text", "write", "ask", "message"])
_INTRO_OBJECT = re.compile(
    r"\b(?:otp|pin|bvn|nin|password|passcode|cvv|code|token|card|details|money|fee|account|number|payment)\b"
)


_INTRODUCERS = frozenset(
    [
        "this",
        "these",
        "following",
        "below",
        "example",
        "e.g",
        "eg",
        "like",
        "say",
        "says",
        "said",
        "saying",
        "tell",
        "tells",
        "told",
        "reply",
        "answer",
        "write",
        "writes",
        "wrote",
        "text",
        "texts",
        "message",
        "messages",
        "script",
        "words",
        "steps",
        "tips",
        "signs",
        "ways",
    ]
)


@dataclass(frozen=True, slots=True)
class Unit:
    text: str  # folded (``text.normalize``)
    gated: bool  # question, negated or reported: not an instruction on its own


def _gated(unit: str, sentence: str, patterns: GuardPatterns) -> bool:
    words = _words(unit)
    if len(words) < 3 or _SUBORDINATE_OR_DESCRIPTION.match(unit):
        return True
    opening = words[:NEGATION_WINDOW]
    if any(_matches_any(patterns.negations, w) for w in opening):
        return True
    if _reported_condition(unit, sentence, patterns):
        return True
    return bool(_REPORTED_START.match(unit))


# "if they ask you to share the code", "once you send it", "you send them the code and they ...": a condition or a
# description of what happens, not an instruction (English)
_SUBORDINATE_OR_DESCRIPTION = re.compile(
    r"(?:if|once|when|whenever|after|before|as soon as|the moment|until|because|since)\b"
    r"|you\s+(?:send|share|give|tell|pay|transfer|enter|type|click|install|reply|\w+ed)\b"
)
# "they said ...", "the caller asked you to ...", "he will probably ask you to ...", Pidgin "dem go ask you ...",
# "dem fit talk say ...": reported, not advice (English and Pidgin)
_REPORTED_START = re.compile(
    r"(?:they|he|she|someone|scammers?|fraudsters?|dem|im|the\s+[\w']+)\s+(?:[\w']+\s+){0,2}"
    r"(?:said|say|says|told|tell|tells|talk|asked|ask|asks|want|wants|wanted|claim\w*|promise[sd]?|threaten\w*"
    r"|warn\w*|write|writes|wrote|text|texts|message|messages)\b"
)


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
        if _means_instruction(segment, patterns) or _segment_instruction(segment, patterns):
            return True
        for clause in _clauses(segment, patterns):
            if _reported_condition(clause, segment, patterns):
                continue  # "They said your account will be blocked unless you send the OTP" reports the threat
            if _unsafe_instruction(clause, instructions, patterns):
                return True
    return False


def _safety_claim(sentence: str, patterns: GuardPatterns) -> bool:
    """A claim that something is safe, unless a negation opens its window ("Never assume it's safe to pay")."""
    for pattern in patterns.safety_claims:
        for match in pattern.finditer(sentence):
            # its own clause only: "Don't worry, this is safe" is still a claim
            before = _words(_CLAUSE_BREAK.split(sentence[: match.start()])[-1])[-NEGATION_WINDOW:]
            if not any(_matches_any(patterns.negations, w) for w in before):
                return True
    return False


def _segment_instruction(segment: str, patterns: GuardPatterns) -> bool:
    """Instructions whose tell-tale part may sit past a clause break ("pay the fee in your bank app so your account
    is not blocked"): matched on the segment, but the lead-in rule applies to the clause the verb starts in."""
    for pattern in patterns.segment_instructions:
        for match in pattern.finditer(segment):
            own_clause = (_clauses(segment[: match.start()], patterns) or [""])[-1]
            if own_clause and _reported_condition(own_clause, segment, patterns):
                continue
            if _is_lead_in(own_clause, patterns) and not any(
                _matches_any(patterns.negations, w) for w in _words(own_clause)
            ):
                return True
    return False


def _means_instruction(segment: str, patterns: GuardPatterns) -> bool:
    """ "Verify your account by sending the OTP": the means is the instruction, whatever comes before it. Matched on
    the whole segment (its exceptions may look past "and": "... BVN and other documents your bank asks for"); a
    negation counts only in the means' own clause."""
    for pattern in patterns.means_instructions:
        for match in pattern.finditer(segment):
            own_clause = (_clauses(segment[: match.start()], patterns) or [""])[-1]
            if own_clause and _reported_condition(own_clause, segment, patterns):
                continue  # "They said ... unless you verify by sending the OTP" reports the threat
            if not any(_matches_any(patterns.negations, w) for w in _words(own_clause)):
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
