"""Second guard layer: a small learned classifier for unsafe advice in English (and Nigerian Pidgin) replies.

The regex guard (``safety.output_guard``) is precise but only knows the phrasings it was written for. This layer
scores each reply sentence and clause (``OutputGuard.units``) with logistic regression over the bge-small embedding
(the knowledge-search model, already loaded) plus a few cheap lexical features, and flags a unit whose probability
reaches the threshold, unless the guard marks it as a question, negated or reported speech.

Weights are a versioned artefact (``data/guard/en_classifier.npz`` + ``.json`` metadata) trained by
``scripts/train_guard_classifier.py`` (deterministic, seeded; threshold chosen on a dev split). Reply text is never
cached. Scoring runs on one worker thread with a time budget: busy or late means regex-only (fail open, logged).
Hausa, Yoruba and Igbo are not classified (bge-small is English; see docs/SPIKES.md S8).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kudiguard.domain import LanguageCode
from kudiguard.knowledge.embeddings import CachedEmbedder, Embedder
from kudiguard.languages import PIVOT
from kudiguard.safety.output_guard import OutputGuard
from kudiguard.text import fold

log = logging.getLogger(__name__)

DEFAULT_BUDGET_S = 0.1

Vector = NDArray[np.float64]


class ClassifierMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"  # score after the reply is sent, log only
    ENFORCE = "enforce"  # score before sending; a flagged reply is retried, then replaced by fixed text


# Cheap lexical cues (on folded text) that help a linear model separate "send the OTP to him" from "never send it".
LEXICAL: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        (
            "credential",
            r"\b(?:otp|pin|password|passcode|cvv|code|bvn|nin|card (?:details|number)|token|login)\b",
        ),
        (
            "payment",
            r"\b(?:pay|transfer|send money|fee|charge|deposit|n\d|\d+k|naira|usdt|bitcoin|gift card|voucher)\b",
        ),
        (
            "untrusted",
            r"\b(?:him|her|them|agent|caller|officer|link|whatsapp|account he|number he|he sent|they sent|"
            r"his|their|wallet|safe account|anydesk|teamviewer)\b",
        ),
        (
            "trusted",
            r"\b(?:your bank app|bank app|branch|atm|your own|official app|play store|app store|your bank|"
            r"police|efcc|ncc|fccpc|cbn|fraud desk|fraud line)\b",
        ),
        ("negation", r"\b(?:not|no|never|don't|dont|do not|avoid|refuse|stop|ignore)\b"),
        ("reported", r"\b(?:said|says|told|asked|wants?|claims?|will ask|will tell|scammers?|fraudsters?)\b"),
        (
            "imperative",
            r"^(?:please |kindly |just |then |now |abeg |oya |make you )?(?:send|give|share|pay|transfer|"
            r"tell|enter|type|click|install|download|buy|move|forward|read|drop|hand|scan|approve|confirm)\b",
        ),
    )
)


def lexical_features(text: str) -> Vector:
    folded = fold(text)
    return np.array([1.0 if pattern.search(folded) else 0.0 for _, pattern in LEXICAL], dtype=np.float64)


_CUES = {"credential", "payment", "untrusted", "imperative"}
# sent to the bank itself or the authorities, or typed into the user's own app: the classifier leaves these to the
# regex guard ("Forward his message to your bank", "Type the code into your own app")
_TRUSTED_TAIL = re.compile(
    r"\b(?:to|into|in|with|at)\s+(?:your\s+(?:own\s+)?(?:bank(?:'s)?(?:\s+(?:app|branch|fraud\s+(?:desk|line)))?"
    r"(?!\s*(?:'s\s+)?(?:staff|agent|officer|official|manager|man|woman|worker|customer|rep))|app|phone)"
    r"|the\s+(?:police|efcc|ncc|fccpc)|efcc|police)\b(?!.*\b(?:to|into)\s+(?:him|her|them|the\s+(?:agent|caller|man)|his|their)\b)"
)


def _has_cue(text: str) -> bool:
    folded = fold(text)
    return any(
        pattern.search(folded) for name, pattern in LEXICAL if name in _CUES
    ) and not _TRUSTED_TAIL.search(folded)


def feature_matrix(texts: Sequence[str], vectors: NDArray[np.float32]) -> NDArray[np.float64]:
    """Embedding (unit vector) followed by the lexical cues, one row per text."""
    lexical = np.stack([lexical_features(t) for t in texts]) if texts else np.zeros((0, len(LEXICAL)))
    return np.hstack([vectors.astype(np.float64), lexical])


@dataclass(frozen=True, slots=True)
class ClassifierModel:
    weights: Vector
    bias: float
    threshold: float
    version: str
    metadata: dict[str, object]

    def probabilities(self, features: NDArray[np.float64]) -> Vector:
        if features.shape[1] != self.weights.shape[0]:
            raise ValueError(f"model expects {self.weights.shape[0]} features, got {features.shape[1]}")
        result: Vector = 1.0 / (1.0 + np.exp(-(features @ self.weights + self.bias)))
        return result

    @classmethod
    def load(cls, path: Path) -> ClassifierModel:
        """``path``: the ``.npz`` weights; metadata (version, threshold, data hash, dev metrics) beside it."""
        with np.load(path) as data:
            weights, bias = np.asarray(data["weights"], dtype=np.float64), float(data["bias"])
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(weights, bias, float(metadata["threshold"]), str(metadata["version"]), metadata)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, bias=np.array(self.bias))
        meta = {**self.metadata, "version": self.version, "threshold": self.threshold}
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@dataclass(frozen=True, slots=True)
class ClassifierVerdict:
    flagged: bool
    probability: float  # highest among the units that may block
    unit: str  # that unit (folded text): for review, never logged
    ms: int


class ClassifierError(RuntimeError):
    """The classifier could not score in time (busy, budget) or failed. Callers fall back to the regex guard."""


class UnsafeAdviceClassifier:
    def __init__(
        self,
        model: ClassifierModel,
        embedder: Embedder,
        guard: OutputGuard,
        *,
        threshold: float | None = None,
        budget_s: float | None = DEFAULT_BUDGET_S,
    ) -> None:
        if budget_s is not None and budget_s <= 0:
            raise ValueError(f"budget_s must be positive, got {budget_s}")
        self.model = model
        self.threshold = model.threshold if threshold is None else threshold
        if not 0 < self.threshold < 1:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")
        self.budget_s = budget_s
        # reply text is user-specific: always embedded directly, never cached
        self._embedder = embedder.inner if isinstance(embedder, CachedEmbedder) else embedder
        self._guard = guard
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guard-classifier")
        self._running: Future[ClassifierVerdict] | None = None

    def supports(self, language: LanguageCode) -> bool:
        return language == PIVOT

    def score(self, text: str, language: LanguageCode) -> ClassifierVerdict:
        """Blocking CPU work: run it off the event loop (``check``)."""
        start = time.perf_counter()
        # only units that may block and carry a cue (a secret, a payment, a stranger, an imperative): fewer embeddings
        texts = [u.text for u in self._guard.units(text, language) if not u.gated and _has_cue(u.text)]
        best, best_unit = 0.0, ""
        if texts:
            probabilities = self.model.probabilities(
                feature_matrix(texts, self._embedder.embed_documents(texts))
            )
            index = int(np.argmax(probabilities))
            best, best_unit = float(probabilities[index]), texts[index]
        ms = int((time.perf_counter() - start) * 1000)
        return ClassifierVerdict(best >= self.threshold, round(best, 4), best_unit, ms)

    async def check(self, text: str, language: LanguageCode) -> ClassifierVerdict:
        """Raises ``ClassifierError`` when busy or past the budget (the caller keeps the regex verdict)."""
        if self._running is not None and not self._running.done():
            raise ClassifierError("busy: a previous classification is still running")
        self._running = self._worker.submit(self.score, text, language)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(self._running), self.budget_s)
        except TimeoutError as exc:
            raise ClassifierError("out of time for the guard classifier") from exc

    async def aclose(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)
