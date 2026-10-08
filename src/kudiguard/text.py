"""Text normalisation shared by rule matching and knowledge search."""

from __future__ import annotations

import re
import unicodedata

# Letters NFKD leaves intact (Hausa hooked letters) and typographic quotes.
_FOLD = str.maketrans({"ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y", "’": "'", "‘": "'", "ʼ": "'"})  # noqa: RUF001 - curly quotes are intentional
# Spelled-out acronyms: o.t.p / b-v-n (2+ separators) and o t p (3+ single letters). Note: "u.s.a." -> "usa."
_SPELLED_PUNCT = re.compile(r"\b(?:[^\W\d_][.\-]){2,}[^\W\d_]\b")
_SPELLED_SPACE = re.compile(r"\b(?:[^\W\d_] ){2,}[^\W\d_]\b")
_SPACES = re.compile(r"\s+")


def fold(text: str) -> str:
    """Lower-case and strip accents/tone marks (the canonical form lexicon terms are written in)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower().translate(_FOLD)


def normalize(text: str) -> str:
    """``fold`` + join spelled-out acronyms + collapse whitespace."""
    text = _SPACES.sub(" ", fold(text)).strip()
    text = _SPELLED_PUNCT.sub(lambda m: re.sub(r"[.\-]", "", m.group()), text)
    return _SPELLED_SPACE.sub(lambda m: m.group().replace(" ", ""), text)


_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str, stopwords: frozenset[str] = frozenset()) -> list[str]:
    """Normalised word tokens, minus ``stopwords``."""
    return [t for t in _TOKEN.findall(normalize(text)) if t not in stopwords]


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w+])(?:\+|0)[\d ()-]{8,}\d")  # +234 803 123 4567, 0803-123-4567
# Account/BVN/card numbers. Not amounts: "N500000", "N 500000", "₦ 500000", "500000 naira", "500000k", "2.5000000",
# nor USSD fields: "*737*1*500000#".
_LONG_NUMBER = re.compile(
    r"(?<![\w₦$£€*])(?<!\d[.,])(?<!\b[Nn] )(?<![₦$] )\d{6,}(?!\w|[.,]\d|#|\s?(?:naira|kobo|k)\b)",
    re.IGNORECASE,
)
_CARD = re.compile(r"(?<!\w)\d{4}(?:[ -]?\d{4}){2,3}(?!\w)")  # 5399 8312 3456 7890
_MIN_PHONE_DIGITS = 10


def _mask_phone(match: re.Match[str]) -> str:
    digits = sum(ch.isdigit() for ch in match.group())
    return "[number]" if digits >= _MIN_PHONE_DIGITS else match.group()


def redact(text: str) -> str:
    """Mask personal identifiers before text leaves our system (emails, phone and account numbers).

    Amounts (N50,000 / ₦5000000), years, dates and USSD codes are kept: they carry meaning, not identity.
    """
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub("[number]", text)
    text = _PHONE.sub(_mask_phone, text)
    return _LONG_NUMBER.sub("[number]", text)
