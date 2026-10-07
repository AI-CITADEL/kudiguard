"""Deterministic scam red-flag detection.

The lexicon (data, per language) finds flags; this module owns the *policy*: flag weights and verdict
thresholds. LLM-extracted flags are merged in via ``extra_flags`` but can never lower a verdict.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from kudiguard.domain import Flag, ScamAssessment, Verdict
from kudiguard.text import fold, normalize

# Policy. 3 = decisive on its own, 2 = strong, 1 = amplifier (only matters in combination or on request).
WEIGHTS: Mapping[Flag, int] = MappingProxyType(
    {
        Flag.ASKS_CREDENTIALS: 3,
        Flag.REMOTE_ACCESS: 3,
        Flag.GUARANTEED_RETURNS: 3,
        Flag.WRONG_TRANSFER: 3,
        Flag.ACCOUNT_THREAT: 2,
        Flag.UPFRONT_FEE: 2,
        Flag.PRIZE_LOTTERY: 2,
        Flag.FAKE_PAYMENT: 2,
        Flag.MONEY_REQUEST: 2,
        Flag.SUSPICIOUS_LINK: 2,
        Flag.IMPERSONATION: 1,
        Flag.URGENCY: 1,
        Flag.SECRECY: 1,
    }
)
SCAM_THRESHOLD = 3
CAUTION_THRESHOLD = 2

_FLAG_ORDER = {flag: i for i, flag in enumerate(Flag)}
Patterns = tuple[re.Pattern[str], ...]


def _compile(term: str) -> re.Pattern[str]:
    # Compare against .lower() so regex escapes such as \S and \W are not mistaken for capitals.
    if fold(term) != term.lower():
        raise ValueError(f"term {term!r} must be written without accents (write {fold(term)!r})")
    try:
        raw = re.compile(term)
    except re.error as exc:
        raise ValueError(f"invalid regex {term!r}: {exc}") from exc
    if raw.fullmatch(""):
        raise ValueError(f"term {term!r} matches the empty string")
    return re.compile(rf"(?<!\w)(?:{term})(?!\w)")


@dataclass(frozen=True, slots=True)
class Lexicon:
    flags: Mapping[Flag, Patterns]
    check_intent: Patterns

    @classmethod
    def from_terms(cls, flags: Mapping[str, Iterable[str]], check_intent: Iterable[str]) -> Lexicon:
        compiled: dict[Flag, Patterns] = {}
        for name, terms in flags.items():
            try:
                flag = Flag(name)
            except ValueError:
                raise ValueError(
                    f"unknown flag {name!r}; expected one of {[f.value for f in Flag]}"
                ) from None
            compiled[flag] = tuple(_compile(t) for t in terms)
        return cls(MappingProxyType(compiled), tuple(_compile(t) for t in check_intent))

    def merge(self, other: Lexicon) -> Lexicon:
        flags = {f: self.flags.get(f, ()) + other.flags.get(f, ()) for f in {*self.flags, *other.flags}}
        return Lexicon(MappingProxyType(flags), self.check_intent + other.check_intent)


def assess(
    texts: Sequence[str | None],
    lexicon: Lexicon,
    extra_flags: Iterable[Flag | str] = (),
    *,
    is_check: bool = False,
) -> ScamAssessment:
    """Score red flags across ``texts`` (e.g. original transcript + English translation).

    ``extra_flags``: flags from another detector (the LLM); unknown values raise ``ValueError``.
    ``is_check``: caller already knows the user wants a check (e.g. a forwarded message).
    """
    corpus = [normalize(t) for t in texts if t and t.strip()]

    found = {Flag(f) for f in extra_flags}
    for flag, patterns in lexicon.flags.items():
        if flag not in found and any(p.search(text) for p in patterns for text in corpus):
            found.add(flag)

    flags = tuple(sorted(found, key=lambda f: (-WEIGHTS[f], _FLAG_ORDER[f])))
    score = sum(WEIGHTS[f] for f in flags)
    checking = is_check or any(p.search(text) for p in lexicon.check_intent for text in corpus)

    if score >= SCAM_THRESHOL***REMOVED***
        verdict = Verdict.LIKELY_SCAM
    elif score >= CAUTION_THRESHOLD or (checking and score > 0):
        verdict = Verdict.CAUTION
    elif checking:
        verdict = Verdict.NO_FLAGS
    else:
        verdict = Verdict.NOT_A_CHECK
    return ScamAssessment(verdict=verdict, flags=flags, score=score)
