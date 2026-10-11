"""Random identifiers, amounts, dates and times for redaction property tests and scripts/redaction_compare.py.

``case(rng)`` returns a text and the span of the identifier in it: the identifier must always be masked whole, and the
context around it (an amount, a date, a time or plain words, joined in either order) must survive.
"""

from __future__ import annotations

import difflib
import random

SEPARATORS = (" ", "-", ".")
JOINS = (", ", ",", ": ", " ", " and ", "; ")


def split_randomly(rng: random.Random, digits: str) -> str:
    groups, rest = [], digits
    while rest:
        size = rng.randint(1, 4)
        groups.append(rest[:size])
        rest = rest[size:]
    text = groups[0]
    for group in groups[1:]:
        text += rng.choice(SEPARATORS) + group
    return text


def random_identifier(rng: random.Random) -> str:
    def digits(n: int) -> str:
        return "".join(rng.choice("0123456789") for _ in range(n))

    kind = rng.randrange(6)
    if kind == 0:  # Nigerian mobile, local
        return split_randomly(rng, "0" + rng.choice(["70", "80", "81", "90", "91"]) + digits(8))
    if kind == 1:  # international
        return "+" + split_randomly(rng, "234" + rng.choice("789") + rng.choice("01") + digits(8))
    if kind == 2:  # NUBAN account number
        return split_randomly(rng, digits(10))
    if kind == 3:  # BVN / NIN
        return split_randomly(rng, rng.choice("12") + digits(10))
    if kind == 4:  # card, 13-19 digits
        return split_randomly(rng, rng.choice("3456") + digits(rng.randint(12, 18)))
    return split_randomly(rng, rng.choice("123456789") + digits(rng.randint(19, 30)))  # long runs


def random_amount(rng: random.Random) -> str:
    whole = rng.randint(1, 10**12)
    sign = rng.choice(["₦", "N", "$", "₦ ", "N ", "$ ", "NGN "])
    decimals = rng.choice(["", f".{rng.randint(0, 99):02d}"])
    style = rng.randrange(4)
    if style == 0:
        return f"{sign}{whole:,}{decimals}"
    if style == 1:
        return f"{sign}{whole:,}{decimals}".replace(",", " ")
    if style == 2:
        return f"{sign}{whole}{decimals}"
    return f"{whole:,} naira"


def random_date_or_time(rng: random.Random) -> str:
    day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1990, 2030)
    sep = rng.choice("-./")
    date = rng.choice([f"{day:02d}{sep}{month:02d}{sep}{year}", f"{year}{sep}{month:02d}{sep}{day:02d}"])
    time_sep = rng.choice(":.")
    clock = f"{rng.randint(0, 23):02d}{time_sep}{rng.randint(0, 59):02d}"
    return rng.choice([date, f"{date} {clock}", f"{date} at {clock}", clock])


def random_context(rng: random.Random) -> str:
    """Something next to an identifier that must be kept: an amount, a comma amount, a time, a date or words."""
    kind = rng.randrange(5)
    if kind == 0:
        return random_amount(rng)
    if kind == 1:
        return f"{rng.choice(['₦', 'N', ''])}{rng.randint(1, 10**7):,}"
    if kind == 2:
        return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    if kind == 3:
        return f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-{rng.randint(1990, 2030)}"
    return rng.choice(["call", "my acct is", "Card:", "BVN:", "pls", "now"])


def random_code(rng: random.Random) -> str:
    """A one-time code or short account number: 6-9 digits in one piece ("482913")."""
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(6, 9)))


def random_small_context(rng: random.Random) -> str:
    """What follows a code in a message: "10 minutes", "5 mins", "(2)", "2x", "3 times"."""
    return rng.choice(["10 minutes", "5 mins", "(2)", "2x", "3 times", "1", "10", "2 tries", "15 min"])


def random_dates(rng: random.Random) -> str:
    """A run of dates, as in a statement ("10/10/2026 11/10/2026 …") or a range ("10/10/2026 - 12/10/2026")."""
    dates = [
        f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2030)}"
        for _ in range(rng.randint(1, 6))
    ]
    return rng.choice([" ", " - ", "-"]).join(dates)


def case(rng: random.Random) -> tuple[str, tuple[int, int], str]:
    """(text, identifier span, context): an identifier joined to a context, in either order."""
    kind = rng.random()
    if kind < 0.2:  # a code followed by a small number: "482913. 10 minutes", "482913-1", "482913 (2)"
        identifier, context, join = (
            random_code(rng),
            random_small_context(rng),
            rng.choice([". ", ", ", " - ", "-", "/", " "]),
        )
        return f"{identifier}{join}{context}", (0, len(identifier)), context
    identifier, join = random_identifier(rng), rng.choice(JOINS)
    context = random_dates(rng) if kind < 0.3 else random_context(rng)
    if rng.random() < 0.5:
        text = f"{context}{join}{identifier}"
        start = len(context) + len(join)
    else:
        text = f"{identifier}{join}{context}"
        start = 0
    return text, (start, start + len(identifier)), context


def kept_digits(text: str, output: str) -> set[int]:
    """Indices of the input's digits that appear unchanged in the redacted output (aligned with difflib)."""
    matcher = difflib.SequenceMatcher(None, text, output, autojunk=False)
    return {
        i
        for block in matcher.get_matching_blocks()
        for i in range(block.a, block.a + block.size)
        if text[i].isdigit()
    }


def random_text(rng: random.Random, length: int) -> tuple[str, list[tuple[int, int]]]:
    """About ``length`` characters of short runs, amounts, dates, times, codes, identifiers and words (long inputs),
    with the spans of the identifiers and codes in it (they must be masked whole)."""
    makers = [
        lambda: " ".join(str(rng.randint(0, 999)) for _ in range(rng.randint(1, 4))),
        lambda: random_amount(rng), lambda: random_date_or_time(rng), lambda: random_dates(rng),
        lambda: rng.choice(["call", "pay", "now", "and", "to"]),
    ]  # fmt: skip
    text, spans = "", []
    while len(text) < length:
        if rng.random() < 0.25:
            piece = random_code(rng) if rng.random() < 0.3 else random_identifier(rng)
            spans.append((len(text), len(text) + len(piece)))
        else:
            piece = rng.choice(makers)()
        text += piece + rng.choice([" ", " ", " and ", ", ", ". "])
    return text, spans
