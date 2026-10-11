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
    rf"(?i)(?:\b(?:https?|ftp)://|\bwww\.|(?<![@\w.#-])(?=(?:[a-z0-9-]+\.)+(?:{_TLDS})\b))"  # scheme, www. or bare host
    rf"(?:[^\s/@?#<>\"']*@)?"  # credentials (user:pass@) are dropped
    rf"((?:[a-z0-9-]+\.)+[a-z]{{2,24}})"  # the host we keep
    rf"(?::\d+)?[^\s<>\"']*"  # port, path, query: dropped
)


def week_of(day: date) -> date:
    """The Monday of ``day``'s ISO week: coarse enough that a pattern does not pin down a single conversation."""
    return day - timedelta(days=day.weekday())


_HOST_DIGITS = re.compile(r"\d{6,}")
_WWW = re.compile(r"^(?:www\.)+")  # every leading "www.", so a host read again stays the same
_MASKED_URL = re.compile(
    r"(?i)\b(?:https?|ftp)://(?:\S*\[email\]|[^\s/?#]*\[number\])\S*"
)  # credentials, a number in the host
_LINK_MARKER = re.compile(r"(\[link(?:: [^\]\s]*)?\])")  # a link already reduced: not read as text again
# An address whose domain an older release reduced to a link ("j @ [link: x.com]", "john(at)[link: gmail.com]").
_EMAIL_BEFORE_LINK = re.compile(
    r"(?i)(?<![\w.+-])(?:[\w.+-]+|\"[^\"\n]+\")(?:\s*(?:@|\(at\)|\[at\]|\{at\}|<at>)\s*|\s+at\s+)\[link: [^\]\s]*\]"
)


def _host(host: str) -> str | None:
    """Lower-case host without www.; None when it holds a number ("pay-0803-123-4567.ng"); long digit runs that
    ``redact`` reads as amounts ("pay123456k.ng") are masked."""
    host = redact(_WWW.sub("", host.lower()))
    return None if "[" in host else _HOST_DIGITS.sub("#", host)


def _link(host: str) -> str:
    kept = _host(host)
    return "[link]" if kept is None else f"[link: {kept}]"


def deidentify(text: str) -> str:
    """Mask identifiers (most numbers, emails, cards) and reduce links to their host; cap the length.

    Idempotent, so stored examples can be de-identified again: a link already reduced is not read as text again, only
    its host is checked again (an older release kept "[link: pay-0803.123.4567.ng]")."""
    text = " ".join(unicodedata.normalize("NFKC", text)[: MAX_PATTERN_CHARS * 2].split())  # bounded, one form
    parts = _LINK_MARKER.split(text)
    for i, part in enumerate(parts):
        if i % 2:  # a reduced link
            parts[i] = _link(part[len("[link: ") : -1]) if part.startswith("[link: ") else part
        else:  # free text; first redact: digits glued to a domain ("…4567pay.ng") are masked before links are read
            part = _MASKED_URL.sub("[link]", redact(part))
            parts[i] = _URL.sub(lambda m: _link(m.group(1)), part)
    text = _MASKED_URL.sub("[link]", _EMAIL_BEFORE_LINK.sub("[email]", "".join(parts)))  # "j @ [link: x.com]"
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
            "text": redact(self.text),  # again on export: examples stored before a redaction fix
            "english": redact(self.english) if self.english else self.english,
        }
