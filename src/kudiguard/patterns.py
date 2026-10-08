"""De-identified scam pattern library: what scammers say, kept to protect others and to improve KudiGuard.

When a check ends 🔴/🟠, a copy of the scam *content* is stored separately from the evidence log: no user hash,
message id or interaction id, and only the week (Monday) rather than the time, under a random id. Most numbers, emails
and card numbers are masked; links are reduced to their host (paths, queries and credentials can identify a victim).
Names in free text cannot be removed reliably, so this is *de-identified*, not anonymous: it is personal data while it
can still be matched to the evidence log (90 days), so a user's DELETE also removes examples taken from their stored
messages (``storage``), and examples are kept for ``PATTERN_RETENTION_DAYS``, as disclosed in the consent text and on
/privacy.

Uses (disclosed): protecting others and improving KudiGuard's detection: new lexicon phrasings and blind test cases
(written by people, never auto-tuned), scam-trend reports and, after a separate review, training its models.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

from kudiguard.domain import Flag, LanguageCode, Verdict
from kudiguard.text import redact

PATTERN_RETENTION_DAYS = 730
MAX_PATTERN_CHARS = 2000
PATTERN_CHANNELS = frozenset({"whatsapp", "web"})  # end users who consented (API callers' users did not)
_TLDS = "com|ng|net|org|ly|me|io|co|app|xyz|info|biz|link|site|online|top|live|shop|store|click|club|gov|edu|uk|us|tk|ml|ga"
_URL = re.compile(
    rf"(?i)(?:\b(?:https?|ftp)://|\bwww\.|(?<![@\w.-])(?=(?:[a-z0-9-]+\.)+(?:{_TLDS})\b))"  # scheme, www. or bare host
    rf"(?:[^\s/@?#<>\"']*@)?"  # credentials (user:pass@) are dropped
    rf"((?:[a-z0-9-]+\.)+[a-z]{{2,24}})"  # the host we keep
    rf"(?::\d+)?[^\s<>\"']*"  # port, path, query: dropped
)


def week_of(day: date) -> date:
    """The Monday of ``day``'s ISO week: coarse enough that a pattern does not pin down a single conversation."""
    return day - timedelta(days=day.weekday())


_HOST_DIGITS = re.compile(r"\d{6,}")
_MASKED_URL = re.compile(
    r"(?i)\b(?:https?|ftp)://\S*\[email\]\S*"
)  # credentials@host already masked as [email]


def _host(host: str) -> str:
    """Lower-case host without www.; long digit runs (an account or phone number in a domain) are masked."""
    return _HOST_DIGITS.sub("#", host.lower().removeprefix("www."))


def deidentify(text: str) -> str:
    """Mask identifiers (most numbers, emails, cards) and reduce links to their host; cap the length."""
    text = unicodedata.normalize("NFKC", text)[: MAX_PATTERN_CHARS * 2]  # bound the regex work first
    text = redact(text)  # first: digits glued to a domain ("…4567pay.ng") are masked before links are read
    text = _MASKED_URL.sub("[link]", text)
    text = _URL.sub(lambda m: f"[link: {_host(m.group(1))}]", text)
    return " ".join(text.split())[:MAX_PATTERN_CHARS]


@dataclass(frozen=True, slots=True)
class ScamPattern:
    language: LanguageCode
    channel: str
    verdict: Verdict
    flags: tuple[Flag, ...]
    text: str  # de-identified user wording (transcript or message)
    english: str | None  # de-identified English translation
    week: date  # Monday of the ISO week
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def from_check(
        cls,
        *,
        language: LanguageCode,
        channel: str,
        verdict: Verdict,
        flags: tuple[Flag, ...],
        text: str,
        english: str | None,
        day: date,
    ) -> ScamPattern:
        return cls(
            language=language,
            channel=channel,
            verdict=verdict,
            flags=flags,
            text=deidentify(text),
            english=deidentify(english) if english else None,
            week=week_of(day),
        )

    def as_row(self) -> dict[str, object]:
        return {
            "id": self.id,
            "week": self.week.isoformat(),
            "language": self.language,
            "channel": self.channel,
            "verdict": self.verdict.value,
            "flags": [f.value for f in self.flags],
            "text": self.text,
            "english": self.english,
        }
