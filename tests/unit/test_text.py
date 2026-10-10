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
        ("code 1 23 45 67 89 01", "code 1 [number]"),  # a stray digit does not hide the number
        ("in 12 345 678 90", "in [number]"),  # "n " ending a word is not the naira sign
        ("₦ 12 345 678 90 and $ 12 345 678 90", "₦ 12 345 678 90 and $ 12 345 678 90"),  # amounts
        ("balance 12 345 678 90.50", "balance 12 345 678 90.50"),  # a decimal amount
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
