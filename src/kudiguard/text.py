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


_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")  # anchored: linear on long inputs
# +234 803 123 4567, (0803) 123 4567; only a whole number (never starting or ending inside a longer run of groups).
_PHONE = re.compile(r"(?<![\w+])(?<!\d[ .,:\-])(?:\+|0)[\d ()-]{8,}\d(?!:|[ .,\-]?\d)")
# Account/BVN/card numbers. Not amounts: "N500000", "N 500000", "₦ 500000", "500000 naira", "500000k", "2.5000000",
# nor USSD fields: "*737*1*500000#".
_LONG_NUMBER = re.compile(
    r"(?<![\w₦$£€*])(?<!\d[.,])(?<!\b[Nn] )(?<![₦$] )\d{6,}(?!\w|[.,]\d|#|\s?(?:naira|kobo|k)\b)",
    re.IGNORECASE,
)
_MIN_PHONE_DIGITS = 10
# A run of digit groups of any size joined by single spaces, dots or dashes, however a number is split or read out:
# "8 031 234 567", "0803.123.4567", "2 2 3 1 5 6 …", "5399 8312 3456 7890 1234". Masked whole at >= 10 digits.
# A run starts only where a number starts (not inside a word, a USSD code "*737*…", a time "10:30 …" or after
# another group), and ends before a word, "#", ":" or a decimal comma. A currency sign in front ("₦", "N", "$") and a
# unit after it ("naira", "kobo", "k") are seen by ``_mask_run``, which keeps amounts, dates and times.
_RUN = re.compile(
    r"(?:(?P<currency>(?:[₦$£€]|\b[Nn])\s?)|(?<![\w*])(?<!\d[ .,:\-]))"
    r"(?P<number>\d+(?:[ .\-]\d+)*)"
    r"(?:(?=(?P<unit>\s?(?:naira|kobo|k)\b))|(?![\w#:]|[.,]\d))",
    re.IGNORECASE,
)
# An amount after a currency sign: thousands in groups of three, or plain digits, never starting with 0 (a phone).
_AMOUNT = re.compile(r"[1-9]\d{0,2}(?: \d{3})+(?:\.\d{1,2})?|[1-9]\d*(?:\.\d{1,2})?")
_DATE = re.compile(
    r"(?P<y>(?:19|20)\d\d)([.\-])(?P<m>\d\d?)\2(?P<d>\d\d?)|(?P<d2>\d\d?)([.\-])(?P<m2>\d\d?)\6(?:19|20)?\d\d"
)
_TIME = re.compile(r"(?P<h>\d\d?)\.(?P<min>\d\d)")


def _date_or_time(token: str) -> bool:
    if date := _DATE.fullmatch(token):
        month, day = int(date["m"] or date["m2"]), int(date["d"] or date["d2"])
        return 1 <= month <= 12 and 1 <= day <= 31
    time = _TIME.fullmatch(token)
    return time is not None and int(time["h"]) < 24 and int(time["min"]) < 60


def _dates_and_times(number: str) -> bool:
    return all(_date_or_time(token) for token in number.split(" "))


def _mask_run(match: re.Match[str]) -> str:
    currency, number = match["currency"] or "", match["number"]
    if sum(ch.isdigit() for ch in number) < _MIN_PHONE_DIGITS:
        return match.group()
    if (currency or match["unit"]) and _AMOUNT.fullmatch(number):
        return match.group()  # "₦ 5 000 000 000", "N12500000000.50", "5 000 000 000 naira"
    if _dates_and_times(number):
        return match.group()  # "12-05-2024 10.30", "2026-10-08 2026-10-09"
    return f"{currency}[number]"


def _mask_phone(match: re.Match[str]) -> str:
    digits = sum(ch.isdigit() for ch in match.group())
    keep = digits < _MIN_PHONE_DIGITS or _dates_and_times(match.group())
    return match.group() if keep else "[number]"


def redact(text: str) -> str:
    """Mask personal identifiers before text leaves our system (emails, phone, account and card numbers).

    Amounts (N50,000 / ₦5000000 / ₦ 5 000 000 000), years, dates, times and USSD codes are kept: they carry meaning,
    not identity. Idempotent: redacting redacted text changes nothing.
    """
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub(_mask_phone, text)
    text = _RUN.sub(_mask_run, text)
    return _LONG_NUMBER.sub("[number]", text)
