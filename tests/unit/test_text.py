import logging
import random
import re
import sys
import time

import pytest

from kudiguard.text import RedactingFilter, normalize, redact, tokens
from redaction_fuzz import case, kept_digits, random_amount, random_date_or_time, random_identifier


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
        "5,000,000,000 naira",
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
        ("pay ₦1 234 567 890.505", "pay ₦1 [number]"),  # three decimals: not an amount
        ("pay N12345678901.5", "pay N12345678901.5"),
        ("pay N12345678901.505", "pay N[number]"),
        ("pay x₦ 12 345 678 90", "pay x₦ [number]"),  # a currency sign glued to a word still counts
        ("call (080) 312 3456", "call [number]"),  # ten digits with brackets, brackets included
        ("call 0700 12345", "call 0700 12345"),  # nine digits: kept
        ("login 123456", "login [number]"),  # "n " ending a word is not the naira sign
        ("code 123456 kindly", "code [number] kindly"),  # "k" starting a word is not thousands
        ("acct 1234567890 kg", "acct [number] kg"),
    ],
)
def test_amount_date_and_time_boundaries(raw: str, expected: str) -> None:
    assert redact(raw) == expected


def test_random_identifiers_are_always_masked_whole() -> None:
    rng = random.Random(2026)
    for _ in range(3000):
        identifier = random_identifier(rng)
        lead = rng.choice(["call", "my acct is", "send to", "Card:", "BVN:", "(", "number", ""])
        tail = rng.choice([" now", ".", ", thanks", "", "!", " please"])
        masked = redact(f"{lead} {identifier}{tail}")
        assert not any(ch.isdigit() for ch in masked), (identifier, masked)


def test_random_identifiers_next_to_amounts_dates_and_times() -> None:
    """Joined with ", ", ",", ": ", " " (and more) in both orders: the identifier is always masked whole; a plain
    amount after a currency sign, a date or a time in front of it, with a space between, is kept."""
    rng = random.Random(2027)
    for _ in range(5000):
        text, (start, end), context = case(rng)
        masked = redact(text)
        assert not [i for i in kept_digits(text, masked) if start <= i < end], (text, masked)
        spaced = re.search(r"\d[ ,]\d", context)  # "₦5,000 0803…": where the amount ends is ambiguous
        if text.startswith(context) and not spaced and " " in text[len(context) : start]:
            assert masked.startswith(context), (text, masked)


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
    for text in (
        "1 " * 5000,
        "1-" * 5000 + "x",
        "N 1" + " 000" * 3000,
        "0." * 5000,
        "1," * 5000,
        "*1" * 5000,  # USSD-like: one start per code
        "o8 " * 3000,
        "one " * 3000,
        "1:" * 5000,
    ):
        redact(text)
    assert time.perf_counter() - start < 2.0


def test_log_records_are_redacted(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("kudiguard.test_redaction")
    caplog.handler.addFilter(RedactingFilter())
    try:
        raise ValueError("bad input: call 0803 123 4567")
    except ValueError:
        logger.exception("turn failed for %s", "acct 0123456789", stack_info=True)
    caplog.handler.filters.clear()
    text = caplog.text
    assert "turn failed for acct [number]" in text and "ValueError: bad input: call [number]" in text
    assert "Traceback" in text and "0803" not in text and "0123456789" not in text


def test_the_log_filter_redacts_an_unformatted_exception() -> None:
    try:
        raise ValueError("call 0803 123 4567")
    except ValueError:
        record = logging.LogRecord(
            "kudiguard", logging.ERROR, __file__, 1, "failed %s", ("0123456789",), sys.exc_info()
        )
    assert RedactingFilter().filter(record)
    assert record.getMessage() == "failed [number]" and record.exc_text and "0803" not in record.exc_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("call (08031234567) now", "call [number] now"),  # brackets around the number go with it
        ("call zero 0803 123 4567 now", "call [number] now"),  # spelled and written digits overlap
    ],
)
def test_masks_take_brackets_and_merge(raw: str, expected: str) -> None:
    assert redact(raw) == expected


# --- boundaries of the rebuilt redaction, found by mutation testing ------------------------------------------------


@pytest.mark.parametrize(
    "separator",
    [
        *map(chr, (8203, 8204, 8205, 8206, 8207, 8288, 65279, 173)),
        "\xa0",
        "\u2009",
        "\u202f",
        "\t",
        *map(chr, (*range(8208, 8214), 8722, 65112, 65123, 65293)),
        "-",
    ],
)
def test_every_space_and_dash_separates_groups(separator: str) -> None:
    assert redact(f"call 0803{separator}123{separator}4567 now") == "call [number] now"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # spelled-out digits: the first and the last word of every list, seven in a row
        ("call zero nine zero nine zero nine zero now", "call [number] now"),
        ("call zero nine zero nine zero nine now", "call zero nine zero nine zero nine now"),  # six: kept
        ("kira sifili tara sifili tara sifili tara sifili yanzu", "kira [number] yanzu"),
        ("pe odo esan odo esan odo esan odo bayi", "pe [number] bayi"),
        ("kpọọ efu itoolu efu itoolu efu itoolu efu ugbu a", "kpọọ [number] ugbu a"),
        ("call zero , nine - zero . nine  zero nine zero now", "call [number] now"),
        ("call zero 0803 123 4567 nine now", "call [number] now"),  # one span holds the other
        ("call 0803 123 4567 and zero 0805 123 4567 now", "call [number] and [number] now"),
        # look-alikes only inside digit tokens
        ("hello8031234567", "hello[number]"),
        ("call 08031234567olx", "call [number]olx"),
        ("call 0803 123 456 o7 now", "call [number] now"),
        ("i 803 123 456 now", "i 803 123 456 now"),
        # times and dates next to numbers
        ("12-05-2024 9:30 ok", "12-05-2024 9:30 ok"),
        ("on 12-05-2024 100:30 ok", "on [number]:30 ok"),
        ("at 1:5 08031234567", "at 1:[number]"),
        ("at 10:300 0803 123 4567", "at 10:[number]"),
        ("10:30,0803 123 4567", "10:30,[number]"),
        # comma and currency amounts next to numbers
        ("₦1,234,567,890.00 0803 123 4567", "₦1,234,567,[number]"),
        ("₦5,000,000 0803 123 4567", "₦5,000,[number]"),
        ("N5000.5 08031234567", "N5000.5 [number]"),
        ("N5000.50 08031234567", "N5000.50 [number]"),
        ("N5000.505 08031234567", "N[number]"),
        ("acct 1,234,567,890.5", "acct 1,234,567,890.5"),
        ("acct 1,234,567,890.505", "acct [number]"),
        ("acct 1234,567,890", "acct [number]"),
        ("call 0803    123    4567 now", "call [number] now"),  # any number of spaces
        ("call 0803 ... 123 ... 4567 now", "call [number] now"),  # up to three separators
        ("call 0803...123...4567 now", "call [number] now"),
        ("call 0803 -- 123 -- 4567 now", "call [number] now"),
        ("call 0803....123....4567 now", "call 0803....123....4567 now"),  # four: not one number
        ("call 0803\x01123\x014567 now", "call [number] now"),  # control characters separate too
        ("call 0803\u221e123\u221e4567 now", "call 0803\u221e123\u221e4567 now"),  # other symbols do not
        ("acct 1234567 ok", "acct [number] ok"),  # seven unbroken digits
        ("1:30 0803 123 4567", "1:30 [number]"),
        ("valid 2026-10-08 2026-10-09 9:30", "valid 2026-10-08 2026-10-09 9:30"),
        ("1,2345 0803 123 4567", "[number]"),  # not a comma amount
        ("N5000/50 08031234567", "N[number]"),  # only "." starts decimals
        ("call 0803 123 4567 and zero 0805 123 4567 nine now", "call [number] and [number] now"),
        ("codexp 09/28 ok", "codexp 09/28 ok"),
        ("exp 09/281 ok", "exp 09/281 ok"),  # not an expiry date
        ("write to john at gmail dot community", "write to john at gmail dot community"),
        ("callzero nine zero nine zero nine zero", "callzero nine zero nine zero nine zero"),  # whole words
        ("zero nine zero nine zero nine zerox", "zero nine zero nine zero nine zerox"),
        ("call 0803 .... 123 .... 4567 now", "call 0803 .... 123 .... 4567 now"),
        # USSD codes: kept field by field, a whole number in a field masked, the text after untouched
        ("dial *737*2*08031234567# now", "dial *737*2*[number]# now"),
        ("dial *737*0123456789#", "dial *737*[number]#"),
        ("dial *737*1*500000# then call 0803 123 4567 now", "dial *737*1*500000# then call [number] now"),
        # brackets and plus signs
        ("+2348031234567 is mine", "[number] is mine"),
        ("08031234567 call me (", "[number] call me ("),
        ("call 08031234567) now", "call [number]) now"),
        ("call (08031234567x", "call [number]x"),
        ("call (08031234567 now", "call [number] now"),
        ("call 0803)123)4567) now", "call [number]) now"),
        ("call (0803)1234567) now", "call [number]) now"),
        # secrets and spelled emails, exactly
        ("my pin 1234 ok", "my pin [number] ok"),
        ("my pin1234 ok", "my pin[number] ok"),
        ("spin 1234 ok", "spin 1234 ok"),
        ("pin 12 ok", "pin 12 ok"),
        ("otp::  482913", "otp::  [number]"),
        ("card exp 9/28 ok", "card exp [number] ok"),
        ("card expiry: 09 / 2028 ok", "card expiry: [number] ok"),
        ("mail john(at)gmail.com now", "mail [email] now"),
        ('mail "a"@x.com now', "mail [email] now"),
    ],
)
def test_rebuilt_redaction(raw: str, expected: str) -> None:
    assert redact(raw) == expected
