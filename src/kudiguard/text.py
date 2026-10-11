"""Text normalisation shared by rule matching and knowledge search."""

from __future__ import annotations

import itertools
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
# ``redact`` masks emails, phone/account/card/BVN numbers, short secrets and other long digit runs before text is
# stored, logged or exported. Everything is found on a *detection view* of the text (``_view``): digits of every
# script as ASCII, every space, control and zero-width character as " ", every dash and the minus sign as "-",
# combining marks (keycaps, variation selectors) dropped, the rest NFKC-folded and lower case (as ``fold`` and
# ``deidentify`` read it), with O/o/l/I/B/S/Z inside digit tokens read as digits. Each view character remembers its
# position in the original, so the original is masked span by span. Every step is linear in the length of the text.

REDACTION_VERSION = (
    2  # bump when redact masks more: stores re-redact rows written by an older version on open
)

_TLD = r"(?:com|net|org|ng|co|io|edu|gov|info|uk|me|biz)"
_PROVIDERS = r"(?:gmail|yahoo|ymail|hotmail|outlook|icloud|live|aol|protonmail|proton)"
_AT = r"(?:@|\(at\)|\[at\]|\{at\}|<at>)"
_DOT = r"(?:\.|,|\(dot\)|\[dot\]|\{dot\}|<dot>)"
_EMAIL = re.compile(
    rf"(?<![\w.+-])(?:[\w.+-]+|\"[^\"\n]+\")\s*{_AT}\s*(?:[\w-]+(?:\s*{_DOT}\s*[\w-]+)+|\[[\d.:a-f]+\]|{_PROVIDERS}\b)"
    rf"|(?<![\w.+-])[\w.+-]+\s+at\s+[\w-]+(?:(?:\s+dot\s+|\.)[\w-]+)*(?:\s+dot\s+|\.){_TLD}(?![\w-])"
)  # on the view: "john(at)gmail.com", "j @ x.com", "ade@gmail.com" in full width, "ade@gmail,com", "… at gmail dot com", "ade@gmail"
_ZERO_WIDTH_CATEGORIES = frozenset({"Cc", "Cf"})
_MINUS = chr(0x2212)
_LOOKALIKES = str.maketrans("olibsz", "011852")
_LOOKALIKE_TOKEN = re.compile(r"(?<![a-z0-9])[0-9olibsz]*[olibsz][0-9olibsz]*(?![a-z0-9])")
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
# Short secrets named as such: "cvv 123", "PIN: 1234", "pin na 1 2 3 4", "OTP is 482-913", "token 4829", "exp 09/28".
_SECRET = re.compile(
    r"(?<![a-z0-9\]])(?:cvv2?|cvc2?|pin|otp|password|passcode|pass code|security code|token)[\s:=#-]*"
    r"(?:(?:is|was|na|be|dey|e)\s+)?(?P<value>\d(?:[ .\-]?\d){2,}|(?:"
    + _WORD
    + r")(?:[ ,\-]+(?:"
    + _WORD
    + r")){2,})"
    r"|(?<![a-z0-9\]])(?:exp|expiry|expires|expiry date)[\s:.-]*(?P<expiry>\d\d?\s?/\s?\d\d(?:\d\d)?)(?!\d)"
)
# USSD codes ("*737*1*500000#"): one start per code (the look-behind), fields without trailing spaces: linear.
_USSD = re.compile(r"(?<![\d*])\*\d++(?: \d++)*+(?:\*\d++(?: \d++)*+)*+#")
_USSD_FIELD = re.compile(r"\d+(?: \d+)*")
_SEPARATORS = r"[.\-/_*~·,()\[\]\\|;'+]"
# A run: digit groups joined by spaces (any number) and at most three separator characters (possessive: linear).
_RUN = re.compile(rf"\d+(?:(?:[ ]++{_SEPARATORS}{{0,3}}+|{_SEPARATORS}{{1,3}}+)[ ]*+\d+)*")
_GROUP = re.compile(r"\d+")
_COLON_SEPARATOR = re.compile(r"(?<=\d{3}):(?=\d{3})")
_MIN_PHONE_DIGITS = 10  # every phone, account, BVN/NIN and card number has at least 10 digits
_MIN_LONG_DIGITS = 6  # a group this long is masked even alone (OTPs, short account numbers), unless an amount
_CURRENCY = re.compile(r"(?:₦|\$|£|€|(?<![a-z])(?:ngn|usd|n))[ ]?$")  # just before a run
_UNIT = re.compile(r"[ ]?(?:naira|kobo|k|m|bn|million|billion)(?![a-z])")
_AMOUNT = re.compile(r"[1-9]\d{0,2}(?:(,|[ ])\d{3})(?:\1\d{3})*(?:\.\d{1,2})?|(?:[1-9]\d*|0)(?:\.\d{1,2})?")
_PLAIN = re.compile(r"[1-9]\d*")
_DATE = re.compile(
    r"(?P<y>(?:19|20)\d\d)([.\-/])(?P<m>\d\d?)\2(?P<d>\d\d?)|(?P<d2>\d\d?)([.\-/])(?P<m2>\d\d?)\6(?:19|20)\d\d"
)
# Only a fully written date is a unit of its own inside a run ("10-10-2026 0803…"), never "1976.03.6 3…".
_STRICT_DATE = re.compile(
    r"(?:19|20)\d\d([.\-/])\d\d\1\d\d|\d\d([.\-/])\d\d\2(?:19|20)\d\d|\d\d?/\d\d?/(?:19|20)\d\d"
)  # one-digit days and months only with "/" ("1/10/2026"): "3-2-2078 1234…" may be one number
_TIME = re.compile(r"(?P<h>\d\d?)\.(?P<min>\d\d)")
_YEAR = re.compile(r"(?:19|20)\d\d")
_DECIMAL = re.compile(r"(?:[1-9]\d{0,2}|0)[.,]\d{1,2}")
_SMALL = re.compile(r"[1-9]\d{0,3}")
_SHORT_DECIMAL = re.compile(r"\d\.\d")
_MOBILE = re.compile(r"(?:0|234)[789][01]\d{8}")
_MOBILE_START = re.compile(r"0[789][01]")
_NO_SEPARATORS = str.maketrans("", "", r" .-/_*~·,()[]\|;'+")
_IP_BEFORE = re.compile(
    r"\bip(?: address)?[ :=]*$"
)  # "IP 192.168.100.200": elsewhere a dotted quad may be a number
_TOKEN_BREAK = re.compile(r"[ ]*,[ ]+|[ ]+")
_CLOCK_BEFORE = re.compile(r"\d:$")  # "10:|30 0803…"
_CLOCK_AFTER = re.compile(r":\d\d")  # "…4567 10|:30"
_ID_FIELD = re.compile(r"\b(?:turn|user|msg|message|interaction|review|sample|id|key|hash)=[0-9a-f]{10,64}\b")


def _detection_chars(ch: str) -> str:
    digit = unicodedata.decimal(ch, None)
    if digit is not None:
        return str(digit)
    category = unicodedata.category(ch)
    if (
        ch.isspace() or category in _ZERO_WIDTH_CATEGORIES
    ):  # every space; control, zero-width and format characters
        return " "
    if category == "Pd" or ch == _MINUS:  # every dash and hyphen, the minus sign
        return "-"
    if category in {"Mn", "Me"}:  # combining marks: keycaps (0️⃣), variation selectors
        return ""
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

    # A colon between groups of 3+ digits separates a number ("0803:123:4567"), never a time ("10:30").
    return _COLON_SEPARATOR.sub("-", _LOOKALIKE_TOKEN.sub(lookalikes, view)), index


def _date_or_time(token: str) -> bool:
    if date := _DATE.fullmatch(token):
        month, day = int(date["m"] or date["m2"]), int(date["d"] or date["d2"])
        return 1 <= month <= 12 and 1 <= day <= 31
    if _YEAR.fullmatch(token):
        return True
    time = _TIME.fullmatch(token)
    return time is not None and int(time["h"]) < 24 and int(time["min"]) < 60


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_BREAK.split(text) if t]


def _not_a_number(text: str, groups: list[str], separators: list[str], *, ip: bool) -> bool:
    """A whole run that is a list, not one number split up: dates/times/years, decimals of one precision, a comma list
    of small numbers, counting ("1 2 3 … 10") or an IPv4 address."""
    tokens = _tokens(text)
    if all(
        _date_or_time(t) for t in tokens
    ):  # in a list, dates fully written: "3.32 7-09-1998" may be one number
        return len(tokens) == 1 or all(_STRICT_DATE.fullmatch(t) or not _DATE.fullmatch(t) for t in tokens)
    listed = all(
        s.strip() == "," and s != "," for s in separators
    )  # "2.5, 3.5, 4.5": items set apart by ", "
    if len(tokens) >= 3 and all(_DECIMAL.fullmatch(t) for t in tokens):
        one_precision = len({len(re.split(r"[.,]", t)[-1]) for t in tokens}) == 1
        return one_precision and (listed or all(_SHORT_DECIMAL.fullmatch(t) for t in tokens))  # "1.5 0.5 2.5"
    if listed:
        return all(_SMALL.fullmatch(t) for t in tokens)  # "10, 20, 30, 40, 50", never "0803, 123, 4567"
    if all(not s.strip() for s in separators):
        numbers = [int(g) for g in groups]
        steps = {b - a for a, b in itertools.pairwise(numbers)}
        # counting ("1 2 … 10"), not digits read out
        return len(numbers) >= 5 and steps in ({1}, {-1}) and max(numbers) >= 10
    # an IPv4 address after "IP"
    return ip and separators == ["."] * 3 and all(g == str(int(g)) and int(g) <= 255 for g in groups)


class _Run:
    """A maximal run of digit groups in the view, split into units that are kept (amounts, dates, times) and the rest.

    Linear: one pass over the groups, every look-ahead bounded or precomputed."""

    def __init__(self, view: str, start: int, end: int) -> None:
        self.view = view
        self.groups = [(g.start(), g.end()) for g in _GROUP.finditer(view, start, end)]

    def group(self, k: int) -> str:
        return self.view[self.groups[k][0] : self.groups[k][1]]

    def sep(self, k: int) -> str:
        """The separator after group ``k``."""
        return self.view[self.groups[k][1] : self.groups[k + 1][0]]

    def text(self, i: int, j: int) -> str:
        return self.view[self.groups[i][0] : self.groups[j - 1][1]]

    def boundary(self, k: int) -> bool:
        """Group ``k`` ends a token: the run ends there, or a space follows it."""
        return k == len(self.groups) - 1 or " " in self.sep(k)

    def spans(self) -> list[tuple[int, int]]:
        view, n = self.view, len(self.groups)
        start, end = self.groups[0][0], self.groups[-1][1]
        currency = _CURRENCY.search(view, max(0, start - 4), start) is not None
        unit = _UNIT.match(view, end) is not None
        text = self.text(0, n)
        if currency and _AMOUNT.fullmatch(text):
            return []  # "₦ 5 000 000 000", "N12500000000.50"
        ip = _IP_BEFORE.search(view, max(0, start - 16), start) is not None
        if _not_a_number(text, [self.group(k) for k in range(n)], [self.sep(k) for k in range(n - 1)], ip=ip):
            return []
        minutes = _CLOCK_BEFORE.search(view, max(0, start - 2), start) is not None
        hour = _CLOCK_AFTER.match(view, end) is not None and n > 1 and " " in self.sep(n - 2)
        label = self._label(start > 0 and view[start - 1].isalpha(), hour=hour)
        kept, comma = self._units(currency, minutes, label, hour=hour)
        return self._mask(kept, comma, currency=currency, unit=unit)

    def _label(self, glued: bool, *, hour: bool) -> int:
        """Groups at the start that label the number rather than belong to it: a version glued to a word ("v2.1
        0803…", at most 6 digits) or a time written with a dot ("10.30 0803…"), but only in front of exactly one
        whole Nigerian mobile number (0803…, 234803…), so a number cannot hide its first digits there."""
        n, end = len(self.groups), 1
        while end < n and self.sep(end - 1) == "." and not self.boundary(end - 1):
            end += 1
        if end >= n or not self.boundary(end - 1):
            return 0
        digits = [len(self.group(g)) for g in range(n)]
        if glued and end >= 2 and all(d <= 3 for d in digits[:end]) and sum(digits[:end]) <= 6:
            return end
        rest = self.text(end, n).replace(" ", "")
        whole = _MOBILE.fullmatch(rest.translate(_NO_SEPARATORS)) is not None
        time = end == 2 and "." in self.sep(0) and _date_or_time(self.text(0, 2))
        return end if time and whole and not hour else 0

    def _units(
        self, currency: bool, minutes: bool, label: int, *, hour: bool
    ) -> tuple[list[bool], list[bool]]:
        """Which groups belong to an amount, a date or a time, and which to a comma-grouped amount (one pass)."""
        n = len(self.groups)
        kept, comma = [False] * n, [False] * n
        chain = [0] * n  # end of the comma-thousands chain starting at each group (right to left: linear)
        for k in range(n - 1, -1, -1):
            follows = k + 1 < n and self.sep(k) == "," and len(self.group(k + 1)) == 3
            chain[k] = chain[k + 1] if follows else k + 1
        for g in range(label):
            kept[g] = True
        k = label
        dates = []
        while k < n:
            end, chained = None, False
            if k == 0 or self.boundary(k - 1) or self.sep(k - 1).strip() == "-":  # a unit starts a token
                end = self._date(k)
                # Glued with "-" after a number, only a date written with "/": "08031234567-10/10/2026", never
                # "…0.22-13-11-1913" (the end of a number written in pieces).
                glued = k > 0 and not self.boundary(k - 1) and not kept[k - 1]
                if end is not None and glued and self.sep(k) != "/":
                    end = None
                if end is not None:
                    dates.append((k, end))
                if (
                    end is None
                    and chain[k] > k + 1
                    and _SMALL.fullmatch(self.group(k))
                    and len(self.group(k)) <= 3
                ):
                    end = self._decimals(chain[k])
                    chained = self.boundary(end - 1)
                    end = end if chained else None
            if end is None and k == 0 and currency and _PLAIN.fullmatch(self.group(0)):
                end = self._decimals(1)
                whole = sum(len(self.group(g)) for g in range(end, n)) >= _MIN_PHONE_DIGITS
                end = end if self.boundary(end - 1) and whole else None  # "N5000 08031234567"
            if end is None:
                k += 1
                continue
            for g in range(k, end):
                kept[g], comma[g] = True, chained
            k = end
        # The minutes ("10:|30 0803…") and the hour ("…4567 10|:30") of a time, unless they would take digits from a
        # number next to them ("10:30 803123456", "07-065-4443 04:19").
        if minutes and len(self.group(0)) == 2 and not kept[0] and (self.boundary(0) or self.sep(0) == ","):
            after = self._digits_until(kept, 1, 1)
            kept[0] = after >= _MIN_PHONE_DIGITS or after + 2 < _MIN_PHONE_DIGITS
        if hour and len(self.group(n - 1)) <= 2 and not kept[n - 1]:
            before = self._digits_until(kept, n - 2, -1)
            kept[n - 1] = before >= _MIN_PHONE_DIGITS or before + len(self.group(n - 1)) < _MIN_PHONE_DIGITS
        # Dates next to a number (``_joined``): decided on the units as found, so the order of the dates is irrelevant.
        units = kept.copy()
        for first, end in dates:
            size = sum(len(self.group(g)) for g in range(first, end))
            if self._joined(units, first - 1, -1, size) or self._joined(units, end, 1, size):
                for g in range(first, end):
                    kept[g] = False
        return kept, comma

    def _joined(self, units: list[bool], g: int, step: int, size: int) -> bool:
        """A date of ``size`` digits next to group ``g`` (``step`` -1: on its left) more likely belongs to the number
        there: it would take digits from it ("54 2075.06.06", "080-2061.07.24"), or that number starts or ends next to
        it, after a plain space, with a group of one or two digits ("2001.03.05 02 3 539…"; not a list "…2006, 34.9…",
        not a mobile number "10-10-2026 09 031…")."""
        side = self._digits_until(units, g, step)
        if side == 0:
            return False
        if side + size >= _MIN_PHONE_DIGITS > side:
            return True
        if self.sep(g if step < 0 else g - 1).strip() or len(self.group(g)) >= 3:
            return False
        start = "".join(self.group(k) for k in range(g, min(g + 3, len(self.groups))))  # its first digits
        return not (step > 0 and _MOBILE_START.match(start))

    def _digits_until(self, kept: list[bool], g: int, step: int) -> int:
        """Digits of the groups not kept from ``g`` on (``step`` -1: leftwards), counted up to 10 (bounded)."""
        count = 0
        while 0 <= g < len(self.groups) and not kept[g] and count < _MIN_PHONE_DIGITS:
            count += len(self.group(g))
            g += step
        return count

    def _date(self, k: int) -> int | None:
        """End of a fully written date starting at group ``k``, if it ends a token or a range follows ("…-12/10/2026").
        ``_units`` then drops a date that would take digits from a number next to it ("07-06-2026 301")."""
        if not self._is_date(k):
            return None
        if self.boundary(k + 2) or (self.sep(k + 2).strip() == "-" and self._is_date(k + 3)):
            return k + 3
        return None

    def _is_date(self, k: int) -> bool:
        if k + 3 > len(self.groups):
            return False
        text = self.text(k, k + 3)
        return _STRICT_DATE.fullmatch(text) is not None and _date_or_time(text)

    def _decimals(self, k: int) -> int:
        """``k``, or one more when group ``k`` is the 1-2 digit decimal part of the group before it."""
        if k < len(self.groups) and self.sep(k - 1) == "." and len(self.group(k)) <= 2:
            return k + 1
        return k

    def _mask(
        self, kept: list[bool], comma: list[bool], *, currency: bool, unit: bool
    ) -> list[tuple[int, int]]:
        """Stretches of groups outside kept units: >= 10 digits masked whole; otherwise each group of >= 6 digits that
        is not an amount (a currency sign before, a unit after, a decimal part) is masked on its own."""
        n, spans = len(self.groups), []
        k = 0
        while k < n:
            if kept[k]:
                k += 1
                continue
            j = k
            while j < n and not kept[j]:
                j += 1
            # A comma amount next to the stretch may hold the number's edge groups, written with a comma: its groups
            # after the first ("7,187,450" + "," + "090 682…") or its first group ("…4188-4 8" + "," + "307,172 naira").
            first, last = k, j + 1 if j < n and comma[j] else j
            while first > 1 and comma[first - 1] and comma[first - 2] and not self.boundary(first - 2):
                first -= 1
            if sum(len(self.group(g)) for g in range(first, last)) >= _MIN_PHONE_DIGITS:
                spans.append((self.groups[first][0], self.groups[last - 1][1]))  # "₦5,[number]"
            else:
                for g in range(k, j):
                    amount = (g == 0 and currency) or (g == n - 1 and unit)
                    fraction = g > 0 and self.sep(g - 1) in {".", ","}
                    decimal = g + 1 < n and self.sep(g) in {".", ","} and len(self.group(g + 1)) <= 2
                    if len(self.group(g)) >= _MIN_LONG_DIGITS and not (amount or fraction or decimal):
                        spans.append(self.groups[g])
            k = j
        return spans


def _number_spans(view: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pieces, cursor = [], 0
    for ussd in _USSD.finditer(view):  # "*737*1*500000#": kept unless its fields hold a whole number
        fields = list(_USSD_FIELD.finditer(view, ussd.start(), ussd.end()))
        if len(fields[0].group()) <= 4:  # the service code ("*737", "*894")
            fields = fields[1:]
        long = [f for f in fields if len(f.group().replace(" ", "")) >= 4]  # never "*894*500*…"
        if sum(len(f.group().replace(" ", "")) for f in long) >= _MIN_PHONE_DIGITS:
            spans += [(f.start(), f.end()) for f in long]  # "*737*0803*1234567#" too
        pieces += [view[cursor : ussd.start()], "#" * (ussd.end() - ussd.start())]
        cursor = ussd.end()
    runs_view = "".join([*pieces, view[cursor:]])
    for run in _RUN.finditer(runs_view):
        spans += _Run(runs_view, run.start(), run.end()).spans()
    for spelled in _SPELLED_DIGITS.finditer(view):
        tokens = _SPELLED_TOKEN.findall(spelled.group())
        count = sum(len(t) if t.isdigit() else 1 for t in tokens)
        if count >= _MIN_SPELLED_DIGITS and not all(t.isdigit() for t in tokens):
            spans.append((spelled.start(), spelled.end()))
    for secret in _SECRET.finditer(view):
        name = "value" if secret["value"] else "expiry"
        spans.append((secret.start(name), secret.end(name)))
    return spans


def _widen(view: str, start: int, end: int) -> tuple[int, int]:
    """A masked number takes its "+" and brackets along: "+[number]" and "([number]" become "[number]"."""
    while start > 0 and view[start - 1] in "+([":
        start -= 1
    opened = view.count("(", start, end) - view.count(")", start, end)
    squared = view.count("[", start, end) - view.count("]", start, end)
    if (view[end : end + 1] == ")" and opened > 0) or (view[end : end + 1] == "]" and squared > 0):
        end += 1
    return start, end


def redact(text: str) -> str:
    """Mask personal identifiers before text leaves our system (emails, phone, account and card numbers, secrets).

    Amounts (N50,000 / ₦5000000 / ₦ 5 000 000 000), years, dates, times, lists and USSD codes are kept: they carry
    meaning, not identity. Idempotent: redacting redacted text changes nothing.
    """
    view, index = _view(text)
    labelled = [(m.start(), m.end(), "[email]") for m in _EMAIL.finditer(view)]
    blank = list(view)
    for start, end, _ in labelled:
        blank[start:end] = "x" * (end - start)  # numbers inside an address belong to the address
    view = "".join(blank)
    labelled += [(*_widen(view, s, e), "[number]") for s, e in _number_spans(view)]
    out: list[str] = []
    cursor = -1
    for start, end, label in sorted(labelled):
        first, last = index[start], index[end - 1] + 1
        while last < len(text) and not _detection_chars(text[last]):  # its keycaps and variation selectors
            last += 1
        if first < cursor:  # overlaps the previous mask: extend it
            cursor = max(cursor, last)
            continue
        out += [text[max(cursor, 0) : first], label]
        cursor = last
    out.append(text[max(cursor, 0) :])
    return "".join(out)


class RedactingFilter(logging.Filter):
    """Log handler filter: the message and any exception text are redacted before they are written (an exception
    raised while handling a turn can quote the user's text). The message is formatted first and redacted whole, so
    hashed ids logged as ``turn=<hex>`` (and similar fields) are recognised and left alone; uvicorn's access lines keep
    their five arguments (its formatter reads them), each redacted. Tracebacks keep their shape. Never raises: a record
    that cannot be redacted is replaced by a fixed line."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.name == "uvicorn.access" and isinstance(record.args, tuple):
                record.args = tuple(_redact_log(a) if isinstance(a, str) else a for a in record.args)
                record.msg = _redact_log(str(record.msg))
                record.getMessage()  # fails here, not in the handler, if the arguments do not fit
            else:
                record.msg, record.args = _redact_log(record.getMessage()), ()
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = _redact_log(record.exc_text)
            if record.stack_info:
                record.stack_info = _redact_log(record.stack_info)
        except Exception:  # a log call must never raise
            record.msg, record.args, record.exc_info, record.exc_text = (
                "(log record withheld: not redactable)",
                (),
                None,
                None,
            )
        return True


def _redact_log(text: str) -> str:
    """``redact``, leaving hashed ids (``turn=3fa9…``) intact."""
    out, cursor = [], 0
    for field in _ID_FIELD.finditer(text):
        out += [redact(text[cursor : field.start()]), field.group()]
        cursor = field.end()
    out.append(redact(text[cursor:]))
    return "".join(out)
