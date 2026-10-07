import pytest

from kudiguard.text import redact, tokens


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
