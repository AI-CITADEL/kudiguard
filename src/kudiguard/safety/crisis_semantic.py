"""A second crisis layer: meaning similarity to a few crisis exemplars (docs/ABUSE.md, threat 6).

The phrase screen (``safety.abuse``) misses crisis messages worded in ways no phrase covers. This layer embeds the
message with the bge-small model the knowledge search already loads and compares it with ``abuse.crisis_exemplars``
of every language (cosine similarity). Shadow by default: it logs ``crisis_semantic`` with the score and whether the
phrase screen also fired (never the text), so recall and false positives can be read off real traffic before it is
promoted to enforce, where a hit the phrases missed is answered like a phrase hit (crisis reply first). Fails
open: an embedding error is logged and the message is answered as usual.

bge-small is English-centred: the Hausa/Yoruba/Igbo exemplars help little, so the phrase lists remain the primary
layer in those languages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from kudiguard.knowledge.embeddings import Embedder
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.abuse import MAX_SCREEN_CHARS, ScreenMode

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.82  # dev sets only: 0/494 genuine above it (top 0.813), 27/50 crisis (docs/ABUSE.md)


@dataclass(frozen=True, slots=True)
class CrisisScore:
    score: float  # best cosine similarity to an exemplar
    hit: bool  # score >= threshold
    ms: int


class CrisisSimilarity:
    def __init__(
        self,
        registry: LanguageRegistry,
        embedder: Embedder,
        *,
        mode: ScreenMode = ScreenMode.SHADOW,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("the crisis similarity threshold must be between 0 and 1")
        exemplars = [s for lang in registry.enabled for s in lang.abuse.crisis_exemplars]
        if not exemplars:
            raise ValueError("no crisis_exemplars configured in data/languages/*.yaml")
        self.mode = mode
        self.threshold = threshold
        self._embedder = embedder
        self._exemplars = np.asarray(embedder.embed_documents(exemplars), dtype=np.float32)

    def score(self, text: str) -> CrisisScore:
        """Blocking (embedding on CPU): call from a worker thread."""
        started = time.perf_counter()
        vector = np.asarray(self._embedder.embed_query(text[:MAX_SCREEN_CHARS]), dtype=np.float32)
        best = float(np.max(self._exemplars @ vector))
        ms = int((time.perf_counter() - started) * 1000)
        return CrisisScore(best, best >= self.threshold, ms)
