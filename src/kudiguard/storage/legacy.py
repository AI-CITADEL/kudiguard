"""Stored text written by releases before ``REDACTION_VERSION`` 2 (up to commit b4f96f8): re-redaction on open.

That release redacted with a handful of regular expressions that missed many spellings of a number or an email. Its
``redact`` and ``deidentify`` are frozen here (``v1_deidentify``), byte for byte, for one purpose: to find the
interaction each stored scam example was taken from (same week, ``v1_deidentify(stored transcript) == example``), so
the example can be re-derived from the re-redacted transcript exactly as DELETE re-derives it, and DELETE keeps finding
it. Never used to redact anything new.
"""

from __future__ import annotations

import re
import unicodedata

from kudiguard.patterns import MAX_PATTERN_CHARS
from kudiguard.text import redact

# --- frozen: kudiguard.text.redact at b4f96f8 -----------------------------------------------------------------------
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w+])(?:\+|0)[\d ()-]{8,}\d")
_LONG_NUMBER = re.compile(
    r"(?<![\w₦$£€*])(?<!\d[.,])(?<!\b[Nn] )(?<![₦$] )\d{6,}(?!\w|[.,]\d|#|\s?(?:naira|kobo|k)\b)",
    re.IGNORECASE,
)
_CARD = re.compile(r"(?<!\w)\d{4}(?:[ -]?\d{4}){2,3}(?!\w)")
_GROUPED = re.compile(r"(?<![\w₦$£€.,*:])(?<!\b[Nn] )(?<![₦$] )\d{2,4}(?:[ -]\d{2,5}){2,4}(?![\w#:]|[.,]\d)")
_SPOKEN = re.compile(r"(?<![\w.,])\d(?:[ -]\d){9,}(?![\w.,])")


def _mask_phone(match: re.Match[str]) -> str:
    return "[number]" if sum(ch.isdigit() for ch in match.group()) >= 10 else match.group()


def _v1_redact(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub("[number]", text)
    text = _PHONE.sub(_mask_phone, text)
    text = _GROUPED.sub(_mask_phone, text)
    text = _SPOKEN.sub("[number]", text)
    return _LONG_NUMBER.sub("[number]", text)


# --- frozen: kudiguard.patterns.deidentify at b4f96f8 ---------------------------------------------------------------
_TLDS = "com|ng|net|org|ly|me|io|co|app|xyz|info|biz|link|site|online|top|live|shop|store|click|club|gov|edu|uk|us|tk|ml|ga"
_URL = re.compile(
    rf"(?i)(?:\b(?:https?|ftp)://|\bwww\.|(?<![@\w.-])(?=(?:[a-z0-9-]+\.)+(?:{_TLDS})\b))"
    rf"(?:[^\s/@?#<>\"']*@)?"
    rf"((?:[a-z0-9-]+\.)+[a-z]{{2,24}})"
    rf"(?::\d+)?[^\s<>\"']*"
)
_HOST_DIGITS = re.compile(r"\d{6,}")
_MASKED_URL = re.compile(r"(?i)\b(?:https?|ftp)://\S*\[email\]\S*")


def v1_deidentify(text: str) -> str:
    """The scam example b4f96f8 stored for a transcript it stored (frozen; for matching only)."""
    text = unicodedata.normalize("NFKC", text)[: MAX_PATTERN_CHARS * 2]
    text = _v1_redact(text)
    text = _MASKED_URL.sub("[link]", text)
    text = _URL.sub(
        lambda m: f"[link: {_HOST_DIGITS.sub('#', m.group(1).lower().removeprefix('www.'))}]", text
    )
    return " ".join(text.split())[:MAX_PATTERN_CHARS]


# --- re-redaction ----------------------------------------------------------------------------------------------------
# Digits left next to a mask by the old patterns ("card [number] 1234"): part of the masked number. A group followed
# by ":" or by "," or "." and a digit is a time or an amount ("[number] 10:30", "₦5,000 [number]") and stays.
_LEFTOVER = re.compile(r"(?:[ \-]?+\d++)++(?![\d:]|[.,]\d)")
_MASK = "[number]"


def _absorb(text: str) -> str:
    """Merge digit groups that touch a ``[number]`` into it. Linear: each piece is matched once from each end."""
    pieces = text.split(_MASK)
    for i, piece in enumerate(pieces):
        if i > 0 and (after := _LEFTOVER.match(piece)):  # digits right after a mask
            piece = piece[after.end() :]
        if i < len(pieces) - 1 and (
            before := _LEFTOVER.match(piece[::-1])
        ):  # digits right before the next one
            piece = piece[: len(piece) - before.end()]
        pieces[i] = piece
    return _MASK.join(pieces)


def reredact(text: str) -> str:
    """A stored free text, redacted with today's rules; digits the old rules left next to a mask go with it."""
    return _absorb(redact(text))
