import pytest

from kudiguard.speech.metrics import word_error_rate


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("send the OTP", "send the OTP", 0.0),
        ("Send the OTP.", "send the otp", 0.0),
        ("send the OTP now", "send OTP now", 0.25),
        ("send the OTP", "send the code OTP", 1 / 3),
        ("Ẹ jọ̀ọ́ ẹ fi ránṣẹ́", "e joo e fi ranse", 0.0),
        ("a b", "", 1.0),
        ("", "", 0.0),
    ],
)
def test_word_error_rate(reference: str, hypothesis: str, expected: float) -> None:
    assert word_error_rate(reference, hypothesis) == pytest.approx(expected)


def test_tone_sensitive_wer_counts_tone_errors() -> None:
    assert word_error_rate("ọkọ́ ọkọ̀", "oko oko") == 0.0
    assert word_error_rate("ọkọ́ ọkọ̀", "ọkọ́ ọkọ́", tone_sensitive=True) == 0.5  # husband vs vehicle


def test_character_error_rate() -> None:
    from kudiguard.speech.metrics import character_error_rate

    assert character_error_rate("send otp", "send otp") == 0.0
    assert character_error_rate("send otp", "sendotp") == pytest.approx(1 / 8)
    assert character_error_rate("", "") == 0.0
