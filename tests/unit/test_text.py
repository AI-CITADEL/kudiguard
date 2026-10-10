import random
import time

import pytest

from kudiguard.text import normalize, redact, tokens


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("my account 0123456789 was debited", "my account [number] was debited"),
        ("call +234 803 123 4567 now", "call [number] now"),
        ("call 0803-123-4567 now", "call [number] now"),
        ("email me at ada.obi+x@mail.example.com", "email me at [email]"),
        ("BVN 22212345678 and card 5399 8312 3456 7890", "BVN [number] and card [number]"),
        ("I paid N5,000 for 2 items in 2024", "I paid N5,000 for 2 items in 2024"),
        ("dial *565*0# to check", "dial *565*0# to check"),
        ("my account is 0123456789.", "my account is [number]."),
        ("pay into 0123456789, then call", "pay into [number], then call"),
        ("on 12-05-2024 and 01-05-2024", "on 12-05-2024 and 01-05-2024"),
        ("amounts N5000000 and ₦5000000 and 1 500 000", "amounts N5000000 and ₦5000000 and 1 500 000"),
        ("rate 2.5000000 or 1,500000", "rate 2.5000000 or 1,500000"),
        ("call 0700-CALL-CBN", "call 0700-CALL-CBN"),
    ],
)
def test_redact(raw: str, expected: str) -> None:
    assert redact(raw) == expected


def test_tokens_drop_stopwords_and_normalise() -> None:
    assert tokens("How do I check my O.T.P?", frozenset({"how", "do", "i", "my"})) == ["check", "otp"]


@pytest.mark.parametrize(
    "amount",
    ["Pay 500000 naira now", "N 500000", "₦ 500000", "they want 1000000 Naira", "send 500000k"],
)
def test_redact_keeps_amounts_written_without_commas(amount: str) -> None:
    assert redact(amount) == amount


def test_redact_keeps_ussd_amounts() -> None:
    assert redact("Dial *737*1*500000# to send") == "Dial *737*1*500000# to send"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("call 803 123 4567 now", "call [number] now"),
        ("my number is 123-456-7890", "my number is [number]"),
        ("acct 0123 456 789", "acct [number]"),
        ("on 2026 10 08 at noon", "on 2026 10 08 at noon"),  # a date: kept
        ("pay 5 000 000 naira", "pay 5 000 000 naira"),  # an amount: kept
        ("dial *737*1*500 000#", "dial *737*1*500 000#"),
        ("meet 10:30 12 45 2026", "meet 10:30 12 45 2026"),  # a time, not a number
        ("pay N 50 000 000 000 now", "pay N 50 000 000 000 now"),  # an amount
        ("BVN is 2 2 3 1 5 6 7 8 9 0 1", "BVN is [number]"),  # read out digit by digit
        ("the 1 2 3 steps", "the 1 2 3 steps"),
    ],
)
def test_grouped_numbers_without_prefix_are_masked(raw: str, expected: str) -> None:
    assert redact(raw) == expected


# --- found by mutation testing (docs/SPIKES.md, G): each case pins one boundary of a redaction pattern ------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Card:5399 8312 3456 7890", "Card:[number]"),  # after ':' only the card pattern applies
        ("ref:5399 8312 3456", "ref:[number]"),  # 12 digits in fours
        ("ref:5399 8312", "ref:5399 8312"),  # 8 digits: not a card
        ("call 80 312 345 67 now", "call [number] now"),  # 2-digit first group
        ("call 8031 2345 67 now", "call [number] now"),  # 4-digit first group
        ("my number is 803 12345 67", "my number is [number]"),  # 5-digit inner group
        ("pairs 12 34 56 78 90", "pairs [number]"),  # five groups
        ("code 1 23 45 67 89 01", "code [number]"),  # a single-digit first group is part of the number
        ("in 12 345 678 90", "in [number]"),  # "n " ending a word is not the naira sign
        ("₦ 12 345 678 900 and $ 1 234 567 890", "₦ 12 345 678 900 and $ 1 234 567 890"),  # amounts
        ("₦ 12 345 678 90", "₦ [number]"),  # not grouped in thousands: a number after a currency sign
        ("balance ₦1 234 567 890.50", "balance ₦1 234 567 890.50"),  # a decimal amount
        ("balance 12 345 678 90.50", "balance [number]"),  # without a currency sign: a number
        ("BVN 2 2 3 1 5 6 7 8 9 0", "BVN [number]"),  # ten digits read out
        ("PIN 2 2 3 1 5 6 7 8 9", "PIN 2 2 3 1 5 6 7 8 9"),  # nine: kept
        ("acct 123456 ok", "acct [number] ok"),  # six digits: masked
        ("code 12345 ok", "code 12345 ok"),  # five: kept
        ("acct 1234567890 kindly", "acct [number] kindly"),  # "k" starting a word is no amount
        ("N 0803123456", "N [number]"),  # amounts never start with 0: a phone number
    ],
)
def test_redaction_boundaries(raw: str, expected: str) -> None:
    assert redact(raw) == expected


def test_email_pattern_stays_linear_on_long_words() -> None:
    start = time.perf_counter()
    assert redact("a" * 20_000 + "@") == "a" * 20_000 + "@"
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ɓaƙo ƴa ‘x’ ʼy", "bako ya 'x' 'y"),  # noqa: RUF001 - Hausa hooked letters and curly quotes
        ("e.g. i.e", "e.g. i.e"),  # one separator is not a spelled-out acronym
        ("xo.t.p and o.t.pin", "xo.t.p and o.t.pin"),  # letters must stand alone
        ("plan a b", "plan a b"),
        ("xo t p and o t pin", "xo t p and o t pin"),
    ],
)
def test_normalize_edges(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


# --- numbers however they are split (privacy fix after mutation testing) ------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "call 8 031 234 567 now",
        "call +234 8 031 234 567 now",
        "call 0 803 123 4567 now",
        "call 0803-123-4567 now",
        "call 0803.123.4567 now",
        "call +234.803.123.4567 now",
        "call 0 8 0 3 123 45 67 now",  # read out, partly digit by digit
        "call 0803 123-45.67 now",  # mixed separators
        "pay to 2 2 3 1 5 6 7 8 9 0 now",
        "card 5399 8312 3456 7890 1234 now",  # 20 digits: masked whole, not after 16
        "card 5399-8312-3456-7890-123 now",  # 19 digits
        "card 3782 822463 10005 now",  # 15 digits, 4-6-5
        "card 53998312345678901234 now",
        "Card:5399 8312 3456 7890 1234 now",
        "card no.5399 8312 3456 7890 now",
        "call 0803 123 4567, then 0803 765 4321 now",
        "call (0803) 123 4567 now",
    ],
)
def test_numbers_are_masked_whole_however_they_are_split(raw: str) -> None:
    masked = redact(raw)
    assert not any(ch.isdigit() for ch in masked), masked
    assert masked.endswith(" now") and "[number]" in masked


@pytest.mark.parametrize(
    "raw",
    [
        "N5,000",
        "₦1,250,000.50",
        "pay ₦ 5 000 000 000 now",
        "pay N12500000000.50 now",
        "pay $ 1 234 567 890 now",
        "5 000 000 000 naira",
        "on 12-05-2024 10.30 and 2026-10-08 14:30",
        "from 01-05-2024 10:30 to 2026-10-08 2026-10-09",
        "times 10.30 12.45 14.00 16.15",
        "meet 08.10.2026 10.30",
    ],
)
def test_amounts_dates_and_times_are_kept(raw: str) -> None:
    assert redact(raw) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("on 31-12-2024 23.59", "on 31-12-2024 23.59"),  # the last valid day, month, hour and minute
        ("on 01-01-2024 00.00", "on 01-01-2024 00.00"),
        ("on 32-12-2024 23.59", "on [number]"),  # not a date: a number
        ("on 00-12-2024 23.59", "on [number]"),
        ("on 31-13-2024 23.59", "on [number]"),
        ("on 31-00-2024 23.59", "on [number]"),
        ("on 31-12-2024 24.00", "on [number]"),  # not a time
        ("on 31-12-2024 23.60", "on [number]"),
        ("on 31-12-2024 25.00", "on [number]"),
        ("on 31-12-2024 23.61", "on [number]"),
        ("in 1 234 567 890", "in [number]"),  # thousands, but no currency sign
        ("acct 1 234 567 890 kg", "acct [number] kg"),
        ("on 31-12-2724 23.59", "on [number]"),  # not a year
        ("pay ₦ 1234 567 890", "pay ₦ [number]"),  # not grouped in thousands
        ("pay ₦1 234 567 890.5", "pay ₦1 234 567 890.5"),
        ("pay ₦1 234 567 890.505", "pay ₦[number]"),  # three decimals: not an amount
        ("pay N12345678901.5", "pay N12345678901.5"),
        ("pay N12345678901.505", "pay N[number]"),
        ("pay x₦ 12 345 678 90", "pay x₦ [number]"),  # a currency sign glued to a word still counts
        ("call (080) 312 3456", "call ([number]"),  # ten digits with brackets
        ("call 0700 12345", "call 0700 12345"),  # nine digits: kept
        ("login 123456", "login [number]"),  # "n " ending a word is not the naira sign
        ("code 123456 kindly", "code [number] kindly"),  # "k" starting a word is not thousands
        ("acct 1234567890 kg", "acct [number] kg"),
    ],
)
def test_amount_date_and_time_boundaries(raw: str, expected: str) -> None:
    assert redact(raw) == expected


def split_randomly(rng: random.Random, digits: str) -> str:
    groups, rest = [], digits
    while rest:
        size = rng.randint(1, 4)
        groups.append(rest[:size])
        rest = rest[size:]
    text = groups[0]
    for group in groups[1:]:
        text += rng.choice(" -.") + group
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


def test_random_identifiers_are_always_masked_whole() -> None:
    rng = random.Random(2026)
    for _ in range(3000):
        identifier = random_identifier(rng)
        lead = rng.choice(["call", "my acct is", "send to", "Card:", "BVN:", "(", "number", ""])
        tail = rng.choice([" now", ".", ", thanks", "", "!", " please"])
        masked = redact(f"{lead} {identifier}{tail}")
        assert not any(ch.isdigit() for ch in masked), (identifier, masked)


def random_amount(rng: random.Random) -> str:
    whole = rng.randint(1, 10**12)
    sign = rng.choice(["₦", "N", "$", "₦ ", "N ", "$ "])
    decimals = rng.choice(["", f".{rng.randint(0, 99):02d}"])
    style = rng.randrange(4)
    if style == 0:
        return f"{sign}{whole:,}{decimals}"
    if style == 1:
        return f"{sign}{whole:,}{decimals}".replace(",", " ")
    if style == 2:
        return f"{sign}{whole}{decimals}"
    return f"{whole:,} naira".replace(",", rng.choice([",", " "]))


def random_date_or_time(rng: random.Random) -> str:
    day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1990, 2030)
    sep = rng.choice("-./")
    date = rng.choice([f"{day:02d}{sep}{month:02d}{sep}{year}", f"{year}{sep}{month:02d}{sep}{day:02d}"])
    time_sep = rng.choice(":.")
    clock = f"{rng.randint(0, 23):02d}{time_sep}{rng.randint(0, 59):02d}"
    return rng.choice([date, f"{date} {clock}", f"{date} at {clock}", clock])


def test_random_amounts_dates_and_times_are_kept() -> None:
    rng = random.Random(2026)
    for _ in range(3000):
        text = f"pay {random_amount(rng)} on {random_date_or_time(rng)} please"
        assert redact(text) == text, text


def test_redaction_is_idempotent_and_fast() -> None:
    rng = random.Random(7)
    for _ in range(500):
        once = redact(f"{random_identifier(rng)} {random_amount(rng)} {random_date_or_time(rng)}")
        assert redact(once) == once
    start = time.perf_counter()
    for text in ("1 " * 5000, "1-" * 5000 + "x", "N 1" + " 000" * 3000, "0." * 5000):
        redact(text)
    assert time.perf_counter() - start < 1.0
