"""Text normalisation shared by rule matching and knowledge search."""

from __future__ import annotations

import logging
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


# --- redaction --------------------------------------------------------------------------------------------------
#
# ``redact`` masks emails, phone/account/card/BVN numbers and other long digit runs before text is stored, logged or
# exported. Numbers are found on a *detection view* of the text (``_view``): digits of every script as ASCII,
# every kind of space (NBSP, thin, tab, newline, zero-width) as " ", every dash as "-", the rest NFKC-folded and lower
# case (as ``fold``/``deidentify``), with O/o/l/I inside digit tokens read as 0/1. Each view character remembers its
# position in the original, so the original text is masked span by span.

REDACTION_VERSION = (
    2  # bump when redact masks more: stores re-redact rows written by an older version on open
)

_EMAIL = re.compile(
    r"(?<![\w.+-])(?:[\w.+-]+|\"[^\"\n]+\")@(?:[\w-]+(?:\.[\w-]+)+|\[[\d.:a-fA-F]+\])"
)  # anchored: linear on long inputs
# Written-out addresses: "john(at)gmail.com", "j @ x.com", "john [at] gmail [dot] com", "john doe at gmail dot com".
_EMAIL_SPELLED = re.compile(
    r"(?<![\w.+-])[\w.+-]+\s*(?:@|\(at\)|\[at\]|\{at\}|<at>)\s*[\w-]+"
    r"(?:\s*(?:\.|\(dot\)|\[dot\]|\{dot\}|<dot>)\s*[\w-]+)+"
    r"|(?<![\w.+-])[\w.+-]+\s+at\s+[\w-]+(?:\s+dot\s+[\w-]+)*\s+dot\s+(?:com|net|org|ng|co|io|edu|gov|info|uk)\b",
    re.IGNORECASE,
)
# Short secrets named as such: "cvv 123", "PIN: 1234", "OTP is 482913", "exp 09/28".
_SECRET = re.compile(
    r"\b(?:cvv2?|cvc2?|pin|otp|password|passcode|pass code|security code)[\s:=#-]*(?:(?:is|was)\s+)?"
    r"(?P<value>\d{3,})"
    r"|\b(?:exp|expiry|expires|expiry date)[\s:.-]*(?P<expiry>\d\d?\s?/\s?\d\d(?:\d\d)?)(?!\d)",
    re.IGNORECASE,
)
_LOOKALIKES = str.maketrans("oli", "011")
_LOOKALIKE_TOKEN = re.compile(r"(?<![a-z0-9])[0-9oli]*[oli][0-9oli]*(?![a-z0-9])")
# Spelled-out digits (an ASR transcript of a number read aloud). Hausa, Yoruba and Igbo lists are DRAFT (pending native
# review); written without tone marks because they are matched on the folded view.
_DIGIT_WORDS = (
    "zero oh one two three four five six seven eight nine",  # English, Pidgin
    "sifili daya biyu uku hudu biyar shida bakwai takwas tara",  # Hausa (DRAFT)
    "odo okan eji eta erin arun efa eje ejo esan",  # Yoruba (DRAFT)
    "efu otu abuo ato ano ise isii asaa asato itoolu",  # Igbo (DRAFT)
)
_WORD = "|".join(
    sorted({word for words in _DIGIT_WORDS for word in words.split()})
)  # none is a prefix of another
_SPELLED_DIGITS = re.compile(rf"(?<![a-z0-9])(?:{_WORD}|\d+)(?:[ ,.\-]+(?:{_WORD}|\d+))+(?![a-z0-9])")
_SPELLED_TOKEN = re.compile(rf"{_WORD}|\d+")
_MIN_SPELLED_DIGITS = 7
_USSD = re.compile(r"(?<!\d)(?<!\d )\*\d[\d ]*+(?:\*\d[\d ]*+)*+#")  # starts only once per code: linear
# A run: digit groups joined by spaces (any number) and at most three of . - / _ * ~ · , ( ) (possessive: linear).
_RUN = re.compile(r"\d+(?:(?:[ ]++[.\-/_*~·,()]{0,3}+|[.\-/_*~·,()]{1,3}+)[ ]*+\d+)*")
_GROUP = re.compile(r"\d+")
_MIN_PHONE_DIGITS = 10  # every phone, account, BVN/NIN and card number has at least 10 digits
_MIN_LONG_DIGITS = 6  # a single unbroken group this long is masked too (OTPs, short account numbers)
_CURRENCY = re.compile(r"(?:₦|\$|£|€|(?<![a-z])(?:ngn|usd|n))[ ]?$")  # just before a run
_UNIT = re.compile(r"[ ]?(?:naira|kobo|k|m|bn|million|billion)(?![a-z])")
_AMOUNT = re.compile(r"[1-9]\d{0,2}(?:(,|[ ])\d{3})(?:\1\d{3})*(?:\.\d{1,2})?|(?:[1-9]\d*|0)(?:\.\d{1,2})?")
_COMMA_AMOUNT = re.compile(r"[1-9]\d{0,2}(?:,\d{3})+(?:\.\d{1,2})?")
_PLAIN = re.compile(r"[1-9]\d*")
_DATE = re.compile(
    r"(?P<y>(?:19|20)\d\d)([.\-/])(?P<m>\d\d?)\2(?P<d>\d\d?)|(?P<d2>\d\d?)([.\-/])(?P<m2>\d\d?)\6(?:19|20)\d\d"
)
_TIME = re.compile(r"(?P<h>\d\d?)\.(?P<min>\d\d)")
# Only a fully written date is set apart in front of a number ("10-10-2026 0803…"), never "1976.03.6 3…".
_STRICT_DATE = re.compile(r"(?:19|20)\d\d([.\-/])\d\d\1\d\d|\d\d([.\-/])\d\d\2(?:19|20)\d\d")
_YEAR = re.compile(r"(?:19|20)\d\d")
_TOKEN_BREAK = re.compile(r"[ ,]+")
_CLOCK_BEFORE = re.compile(r"\d:$")  # "10:|30 0803…"
_CLOCK_AFTER = re.compile(r":\d\d")  # "…4567 10|:30"
_MINUS = chr(0x2212)


def _detection_chars(ch: str) -> str:
    digit = unicodedata.decimal(ch, None)
    if digit is not None:
        return str(digit)
    category = unicodedata.category(ch)
    if ch.isspace() or category in {
        "Cc",
        "Cf",
    }:  # every space; control, zero-width and other invisible characters
        return " "
    if category == "Pd" or ch == _MINUS:  # every dash and hyphen, the minus sign
        return "-"
    return fold(unicodedata.normalize("NFKC", ch))


def _view(text: str) -> tuple[str, list[int]]:
    """The detection view and, for each of its characters, the index of the original character it came from."""
    chars: list[str] = []
    index: list[int] = []
    for position, ch in enumerate(text):
        for out in _detection_chars(ch):
            chars.append(out)
            index.append(position)
    view = "".join(chars)

    def lookalikes(match: re.Match[str]) -> str:
        token = match.group()
        return token.translate(_LOOKALIKES) if any(c.isdigit() for c in token) else token

    return _LOOKALIKE_TOKEN.sub(lookalikes, view), index


def _date_or_time(token: str) -> bool:
    if date := _DATE.fullmatch(token):
        month, day = int(date["m"] or date["m2"]), int(date["d"] or date["d2"])
        return 1 <= month <= 12 and 1 <= day <= 31
    if _YEAR.fullmatch(token):
        return True
    time = _TIME.fullmatch(token)
    return time is not None and int(time["h"]) < 24 and int(time["min"]) < 60


def _dates_and_times(text: str) -> bool:
    return all(_date_or_time(token) for token in _TOKEN_BREAK.split(text) if token)


class _Run:
    """A maximal run of digit groups in the view, judged as a whole and carved into kept and masked parts."""

    def __init__(self, view: str, start: int, end: int) -> None:
        self.view = view
        self.groups = [(g.start(), g.end()) for g in _GROUP.finditer(view, start, end)]

    def text(self, i: int, j: int) -> str:
        return self.view[self.groups[i][0] : self.groups[j - 1][1]]

    def digits(self, i: int, j: int) -> int:
        return sum(e - s for s, e in self.groups[i:j])

    def sep(self, i: int) -> str:
        """The separator after group ``i``."""
        return self.view[self.groups[i][1] : self.groups[i + 1][0]]

    def spans(self) -> list[tuple[int, int]]:
        view, (start, _), (_, end) = self.view, self.groups[0], self.groups[-1]
        currency = _CURRENCY.search(view, max(0, start - 4), start) is not None
        digits = self.digits(0, len(self.groups))
        if len(self.groups) == 1 and _MIN_LONG_DIGITS <= digits < _MIN_PHONE_DIGITS:
            # one unbroken group of 6-9 digits (an OTP, a short account number), unless an amount: N500000, 500000k
            return [] if currency or _UNIT.match(view, end) else [(start, end)]
        return self._judge(
            0,
            len(self.groups),
            currency=currency,
            colon_before=_CLOCK_BEFORE.search(view, max(0, start - 2), start) is not None,
            hours=_CLOCK_AFTER.match(view, end) is not None and end - self.groups[-1][0] <= 2,
        )

    def _judge(
        self, i: int, j: int, *, currency: bool, colon_before: bool, hours: bool
    ) -> list[tuple[int, int]]:
        """Spans to mask in groups ``i`` to ``j`` (``hours``: the last group is the hour of a time "…10:30")."""
        text = self.text(i, j)
        if self.digits(i, j) < _MIN_PHONE_DIGITS:
            return []
        if currency and _AMOUNT.fullmatch(text):
            return []  # "₦ 5 000 000 000", "N12500000000.50"
        if (
            _COMMA_AMOUNT.fullmatch(text)
            or _dates_and_times(text)
            or (hours and _dates_and_times(self.text(i, j - 1)))
        ):
            return []  # "1,234,567,890", "12-05-2024 10.30", "in 2026 2027 2028", "12-05-2024 10:30"
        if (carved := self._prefix(i, j, currency=currency, colon_before=colon_before)) > i:
            return self._judge(carved, j, currency=False, colon_before=False, hours=hours)
        return [(self.groups[i][0], self.groups[j - 1][1])]

    def _prefix(self, i: int, j: int, *, currency: bool, colon_before: bool) -> int:
        """End of a leading part that is not the number: the minutes of a time ("10:30 0803…"), a date
        ("10-10-2026 0803…"), a comma-grouped amount ("5,000 0803…") or an amount after a currency sign when a whole
        number follows ("N5000 08031234567"). ``i`` when there is none."""
        if colon_before and self.groups[i][1] - self.groups[i][0] == 2 and self._spaced(i):
            return i + 1
        token = i + len(_GROUP.findall(_TOKEN_BREAK.split(self.text(i, j), maxsplit=1)[0]))
        if (
            self._whole(token, j)
            and _STRICT_DATE.fullmatch(self.text(i, token))
            and _date_or_time(self.text(i, token))
        ):
            return token
        if comma := _COMMA_AMOUNT.match(self.text(i, j)):
            amount = i + len(_GROUP.findall(comma.group()))
            if self.text(i, amount) == comma.group() and self._whole(amount, j):
                # "5,000 0803…": the last thousands group (with any decimals) could be the number's first group
                # written after a comma ("N8,015,369,511.9 889…"), so it is masked with the number: "₦5,[number]".
                return amount - 2 if "." in comma.group() else amount - 1
        if (
            currency
            and _PLAIN.fullmatch(self.text(i, i + 1))
            and self._whole(end := self._decimals(i + 1, j), j)
        ):
            return end
        return i

    def _whole(self, end: int, j: int) -> bool:
        """Groups ``end`` to ``j`` are a separate whole number: a date or an amount is set apart only in front of one."""
        return end < j and self._spaced(end - 1) and self.digits(end, j) >= _MIN_PHONE_DIGITS

    def _spaced(self, k: int) -> bool:
        """The separator after group ``k`` ends a token (a space or a comma)."""
        return " " in self.sep(k) or "," in self.sep(k)

    def _decimals(self, k: int, j: int) -> int:
        """``k``, or one more when group ``k`` is the 1-2 digit decimal part of the group before it."""
        if k < j and self.sep(k - 1) == "." and self.groups[k][1] - self.groups[k][0] <= 2:
            return k + 1
        return k


def _number_spans(view: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    runs_view = view
    for ussd in _USSD.finditer(view):  # "*737*1*500000#": fields are kept unless they hold a whole number
        spans += [
            (ussd.start() + f.start(), ussd.start() + f.end())
            for f in re.finditer(r"\d[\d ]*\d|\d", ussd.group())
            if sum(c.isdigit() for c in f.group()) >= _MIN_PHONE_DIGITS
        ]
        runs_view = runs_view[: ussd.start()] + "#" * (ussd.end() - ussd.start()) + runs_view[ussd.end() :]
    for run in _RUN.finditer(runs_view):
        spans += _Run(runs_view, run.start(), run.end()).spans()
    for spelled in _SPELLED_DIGITS.finditer(view):
        tokens = _SPELLED_TOKEN.findall(spelled.group())
        count = sum(len(t) if t.isdigit() else 1 for t in tokens)
        if count >= _MIN_SPELLED_DIGITS and not all(t.isdigit() for t in tokens):
            spans.append((spelled.start(), spelled.end()))
    return spans


def _widen(view: str, start: int, end: int) -> tuple[int, int]:
    """A masked number takes its "+", "(" and ")" along: "+[number]" and "([number]" become "[number]"."""
    while start > 0 and view[start - 1] in "+(":
        start -= 1
    if view[end : end + 1] == ")" and view.count("(", start, end) > view.count(")", start, end):
        end += 1
    return start, end


def redact(text: str) -> str:
    """Mask personal identifiers before text leaves our system (emails, phone, account and card numbers).

    Amounts (N50,000 / ₦5000000 / ₦ 5 000 000 000), years, dates, times and USSD codes are kept: they carry meaning,
    not identity. Idempotent: redacting redacted text changes nothing.
    """
    text = _EMAIL.sub("[email]", text)
    text = _EMAIL_SPELLED.sub("[email]", text)
    text = _SECRET.sub(_mask_secret, text)
    view, index = _view(text)
    merged: list[list[int]] = []
    for start, end in sorted(_widen(view, s, e) for s, e in _number_spans(view)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: list[str] = []
    cursor = 0
    for start, end in merged:
        out += [text[cursor : index[start]], "[number]"]
        cursor = index[end - 1] + 1
    out.append(text[cursor:])
    return "".join(out)


def _mask_secret(match: re.Match[str]) -> str:
    """The value ends the match: everything before it is kept."""
    name = "value" if match["value"] else "expiry"
    return match.group()[: match.start(name) - match.start()] + "[number]"


class RedactingFilter(logging.Filter):
    """Log handler filter: the message, its arguments and any exception text are redacted before they are written
    (an exception raised while handling a turn can quote the user's text). Tracebacks keep their shape."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = redact(record.getMessage()), ()
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True
