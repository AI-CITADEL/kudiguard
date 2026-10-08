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


def test_pooled_counts_match_rates_and_weight_long_clips() -> None:
    import unicodedata

    from kudiguard.speech.metrics import char_errors, character_error_rate, word_errors

    assert word_errors("Send the OTP now!", "send OTP now") == (1, 4)  # one deletion, punctuation ignored
    assert char_errors("abc", "abd") == (1, 3)
    assert word_errors("", "") == (0, 0)
    assert word_errors("", "noise") == (
        1,
        0,
    )  # an insertion against nothing: the caller pools, never divides by 0
    # Tone marks: folded by default, counted when asked.
    assert word_errors("ọ̀kọ̀", "oko") == (0, 1)
    assert word_errors("ọ̀kọ̀", "oko", tone_sensitive=True) == (1, 1)
    assert char_errors("ọ̀kọ̀", "oko", tone_sensitive=True)[1] == len(unicodedata.normalize("NFC", "ọ̀kọ̀"))
    # Pooling: a 1-word clip with an error and a 9-word perfect clip pool to 1/10, not the per-clip mean of 0.5.
    clips = [("yes", "no"), ("one two three four five six seven eight nine",) * 2]
    edits, words = map(sum, zip(*(word_errors(r, h) for r, h in clips), strict=True))
    assert (edits, words) == (1, 10)
    for ref, hyp in clips:
        e, n = word_errors(ref, hyp)
        assert e / n == word_error_rate(ref, hyp)
        ce, cn = char_errors(ref, hyp)
        assert ce / cn == character_error_rate(ref, hyp)
