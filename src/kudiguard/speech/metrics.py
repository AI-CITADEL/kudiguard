"""Speech-recognition accuracy metrics.

``tone_sensitive=False`` folds *all* diacritics (tone marks and under-dots: ẹ→e, ọ→o, ṣ→s, ụ→u), useful for
Hausa/English; Yoruba and Igbo diacritics change meaning, so report both views for them. For a data set, pool the
counts (``word_errors``/``char_errors``: total edits / total reference length) rather than averaging per-clip rates,
which over-weights short clips.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence


def _normalise(text: str, *, tone_sensitive: bool) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    if not tone_sensitive:
        text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    # Remove punctuation/symbols by Unicode category: a regex like [^\w\s] would also delete combining tone marks.
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return " ".join(text.split())


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, start=1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1]


def word_error_rate(reference: str, hypothesis: str, *, tone_sensitive: bool = False) -> float:
    """Word-level edit distance / reference length, ignoring case and punctuation (and tones unless asked)."""
    ref = _normalise(reference, tone_sensitive=tone_sensitive).split()
    hyp = _normalise(hypothesis, tone_sensitive=tone_sensitive).split()
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def character_error_rate(reference: str, hypothesis: str, *, tone_sensitive: bool = False) -> float:
    """Character-level edit distance / reference length (spaces collapsed); robust to word-segmentation noise."""
    ref = _normalise(reference, tone_sensitive=tone_sensitive)
    hyp = _normalise(hypothesis, tone_sensitive=tone_sensitive)
    return _edit_distance(list(ref), list(hyp)) / max(len(ref), 1)


def word_errors(reference: str, hypothesis: str, *, tone_sensitive: bool = False) -> tuple[int, int]:
    """(edits, reference words), for pooling over a data set."""
    ref = _normalise(reference, tone_sensitive=tone_sensitive).split()
    hyp = _normalise(hypothesis, tone_sensitive=tone_sensitive).split()
    return _edit_distance(ref, hyp), len(ref)


def char_errors(reference: str, hypothesis: str, *, tone_sensitive: bool = False) -> tuple[int, int]:
    """(edits, reference characters), for pooling over a data set."""
    ref = _normalise(reference, tone_sensitive=tone_sensitive)
    hyp = _normalise(hypothesis, tone_sensitive=tone_sensitive)
    return _edit_distance(list(ref), list(hyp)), len(ref)


# The reported metrics, shared by scripts/eval_asr.py and ``kudiguard voice wer``: name -> (counter, tone-sensitive).
POOLED_METRICS: dict[str, tuple[Callable[..., tuple[int, int]], bool]] = {
    "wer": (word_errors, False),  # diacritics folded
    "wer_diacritics": (word_errors, True),
    "cer": (char_errors, False),
}
