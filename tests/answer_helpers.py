"""Shared test helpers for the answer-quality evaluation (kudiguard.evaluation)."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from kudiguard.evaluation.answers import AnswerCase
from kudiguard.knowledge.embeddings import l2_normalize

STOP = frozenset({
    "the", "a", "an", "to", "your", "you", "is", "it", "and", "of", "for", "by", "or", "this", "that", "be", "are",
    "with", "not", "do", "on", "in", "if", "any", "as", "so", "from", "can", "at", "what", "who", "please",
})  # fmt: skip


class WordEmbedder:
    """Hashed bag of content words: shared words -> similar; blind to negation, like the real model."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in STOP:
                v[zlib.crc32(word.encode()) % 256] += 1
        return l2_normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, 0), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def case(**overrides: Any) -> AnswerCase:
    data: dict[str, Any] = {
        "id": "c1",
        "lang": "en",
        "kind": "scam",
        "text": "they want my otp",
        "verdicts": ["likely_scam", "caution"],
        "must": [
            {"point": "Bank never asks OTP", "from": "flag-asks-credentials"},
            {"point": "End call now", "from": "flag-asks-credentials"},
        ],
        "must_not": ["Share OTP with caller"],
    }
    return AnswerCase.model_validate(data | overrides)
