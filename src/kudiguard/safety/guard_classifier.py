"""Second guard layer: a small learned classifier for unsafe advice in English replies.

The regex guard (``safety.output_guard``) is precise but only knows the phrasings it was written for. This layer
scores reply sentences and clauses (``OutputGuard.units``) with logistic regression over the bge-small embedding
(the knowledge-search model, already loaded) plus a few cheap lexical features, and flags a unit whose probability
reaches the threshold, unless the guard marks it as a question, negated, quoted or reported speech.

Every ungated unit of three words or more is a candidate (no keyword pre-filter: unsafe advice comes in phrasings no
list anticipates), except a unit whose verb's own recipient is the user's bank, the authorities or their own app
(``left_to_the_regex``; a person before that destination, "the man at your bank", keeps it a candidate). The most suspicious ``MAX_UNITS`` (by lexical cues) are embedded in one batch; the rest are
counted as unscored. Nigerian Pidgin replies arrive as English (``en``) and are scored by the same English model;
``is_pidgin`` lets the engine run them in a different mode. Hausa, Yoruba and Igbo are not classified (bge-small is
English; see docs/SPIKES.md S8).

Weights are a versioned artefact (``data/guard/en_classifier.npz`` + ``.json`` metadata) trained by
``scripts/train_guard_classifier.py`` from a frozen training set; ``ClassifierModel.load`` checks the weights
checksum and the feature list, and the classifier checks the embedding model and dimension, so a stale or mismatched
artefact disables the layer at startup instead of scoring garbage. Reply text is never cached or logged. Scoring
runs on one worker thread: a call waits in a short bounded queue within its budget; a full queue or a late result
means the regex verdict alone (fail open), counted by reason in ``stats``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import Counter
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

DEFAULT_BUDGET_S = 0.25  # measured: p95 ~110 ms for a dense 8-unit reply on the pilot CPU (docs/SPIKES.md S8)
MAX_UNITS = 8  # units embedded per reply (most suspicious first); the rest are counted as unscored
MIN_WORDS = 3
MAX_PENDING = 3  # running + queued calls; beyond that a call fails open at once ("busy")
SHADOW_PENDING = 2  # shadow calls (nobody waits for them) queue behind at most one other call

Vector = NDArray[np.float64]


class ClassifierMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"  # score after the reply is sent, log only
    ENFORCE = "enforce"  # score before sending; a flagged reply is retried, then replaced by fixed text


# Cheap lexical cues (on folded text) that help a linear model separate "send the OTP to him" from "never send it".
# Changing this list changes the feature layout: retrain (the artefact records the names and is checked at load).
LEXICAL: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        (
            "credential",
            r"\b(?:otp|pin|password|passcode|cvv|code|bvn|nin|card|token|login|log ?in details|account details)\b",
        ),
        (
            "payment",
            r"\b(?:pay|transfer|send money|fee|charge|deposit|n\d|\d+k|naira|usdt|bitcoin|crypto|gift card|voucher"
            r"|refund|reverse|reversal|return (?:the|his|her|their) money|send (?:it|the money) back)\b",
        ),
        (
            "untrusted",
            r"\b(?:him|her|them|agent|caller|officer|link|account he|number he|he sent|they sent|his|their|wallet"
            r"|safe account|dm|inbox|stranger|coordinator|the man|the woman|the lady)\b",
        ),
        (
            "access",
            r"\b(?:anydesk|teamviewer|quicksupport|remote|share (?:your )?screen|screen ?shar\w*|permission|grant"
            r"|allow|access|attachment|apk|website|web ?site|site|portal|form)\b",
        ),
        (
            "endorsement",
            r"\b(?:trustworthy|genuine|legit|legitimate|verified|real deal|go ahead|proceed|is safe|are safe"
            r"|safe to|nothing to worry|e dey safe|no wahala)\b",
        ),
        (
            "trusted",
            r"\b(?:your bank app|bank app|branch|atm|your own|official app|play store|app store|your bank|"
            r"police|efcc|ncc|fccpc|cbn|fraud desk|fraud line|counter|teller|customer care line on your card)\b",
        ),
        ("negation", r"\b(?:not|no|never|don't|dont|do not|avoid|refuse|stop|ignore|nobody|no one)\b"),
        ("reported", r"\b(?:said|says|told|asked|wants?|claims?|will ask|will tell|scammers?|fraudsters?)\b"),
        (
            "imperative",
            r"^(?:please |kindly |just |then |now |abeg |oya |make you |quickly |simply )?(?:send|give|share|pay"
            r"|transfer|tell|enter|type|click|tap|install|download|buy|move|forward|read|drop|hand|scan|approve"
            r"|confirm|allow|grant|open|go|visit|log|sign|accept|return|refund|reverse|dm|snap|proceed|follow|let"
            r"|reply|text|call|fill|verify|update|submit|use|put|load|top|activate)\b",
        ),
    )
)
_CUES = frozenset({"credential", "payment", "untrusted", "access", "endorsement", "imperative"})

# Nigerian Pidgin markers (folded text). A reply is Pidgin when it has at least PIDGIN_MARKERS of them: one word ("oga",
# "wahala", a quoted "abeg") turns up in English replies too.
_PIDGIN = re.compile(
    r"(?<![\w'])(?:dey|una|wetin|abeg|oya|wey|sabi|wahala|comot|abi|sef|shey|pikin|dem|na|no be|make you|e don"
    r"|don do|wan|oga|am)(?![\w'])"
)
PIDGIN_MARKERS = 2
# one of these is enough in a one-sentence reply ("Make una no send code give anybody.")
_STRONG_PIDGIN = re.compile(r"(?<![\w'])(?:una|wetin|abeg|dey|wey|sabi|comot|pikin|oya)(?![\w'])")
_ENGLISH_AM = re.compile(r"(?<![\w'])(?:i|\d+(?::\d+)?)\s*am(?![\w'])")  # "I am", "8 am", "8:30am"


def is_pidgin(text: str) -> bool:
    folded = _ENGLISH_AM.sub(" ", fold(text))  # Pidgin "am" is an object ("pay am")
    if len(_PIDGIN.findall(folded)) >= PIDGIN_MARKERS:
        return True
    one_sentence = len([s for s in re.split(r"[.!?\n]+", folded) if s.strip()]) == 1
    return one_sentence and _STRONG_PIDGIN.search(folded) is not None


def lexical_features(text: str) -> Vector:
    folded = fold(text)
    return np.array([1.0 if pattern.search(folded) else 0.0 for _, pattern in LEXICAL], dtype=np.float64)


# The verb's own recipient is the bank itself, the authorities or the user's own app/phone ("Forward his message to
# your bank", "Enter your PIN in your banking app"): left to the regex guard, which knows these destinations. Only
# when no person comes before that destination ("...to the man at your bank" is a handover to a person) and no
# person or supplied page comes after it.
_TRUSTED_RECIPIENT = re.compile(
    r"\b(?:to|into|in|with|at|on)\s+(?:your\s+(?:own\s+)?(?:bank(?:'s)?(?:\s+(?:app|branch|fraud\s+(?:desk|line)))?"
    r"(?!\s*(?:'s\s+)?(?:staff|agent|officer|official|manager|man|woman|worker|customer|rep))|banking\s+app|app|phone)"
    r"|the\s+(?:police|efcc|ncc|fccpc)|efcc|police)\b"
)
_PERSON = re.compile(
    r"\b(?:him|her|them|someone|anyone|man|men|woman|lady|gentleman|guy|person|people|coordinator|agent|staff"
    r"|officer|official|manager|caller|worker|rep\w*)\b"
)
_LATER_UNTRUSTED = re.compile(
    r"\b(?:to|into|with|for)\s+(?:him|her|them|the\s+(?:agent|caller|man|woman|person|officer))\b"
    r"|\b(?:he|she|they|someone)\s+(?:sent|gave|shared)\b|\b(?:link|page|form)\b"
)
_PERSON_AFTER = re.compile(rf"{_PERSON.pattern}|\bsafe\s+account\b|\bwhile\b|\b(?:he|she|they|someone)\b")
# BVN and NIN (identity numbers, not secrets) handed to staff inside the branch: the regex guard allows these too
_IDENTITY_AT_BRANCH = re.compile(
    r"^(?!.*\b(?:otp|pin|password|passcode|cvv|code|token)\b).*\b(?:bvn|nin)\b.*"
    r"(?:\b(?:at|in|inside|behind)\s+(?:the\s+|your\s+)?(?:bank\s+|branch\s+)?(?:branch|counter|banking\s+hall)\b"
    r"|\bto\s+the\s+(?:bank\s+)?teller\b)"
)


def left_to_the_regex(text: str) -> bool:
    """A unit whose recipient is trusted (see ``_TRUSTED_RECIPIENT``) or an ID number given at the branch counter."""
    folded = fold(text)
    if _LATER_UNTRUSTED.search(folded):
        return False
    if _IDENTITY_AT_BRANCH.search(folded):
        return True
    match = _TRUSTED_RECIPIENT.search(folded)
    if match is None or _PERSON.search(folded[: match.start()]):
        return False
    # a person, a "safe account" or "while he ..." after the trusted place: "with your bank app to the account the
    # caller gave you", "in your bank app while the agent watches your screen"
    return not _PERSON_AFTER.search(folded[match.end() :])


def _suspicion(text: str) -> int:
    folded = fold(text)
    return sum(1 for name, pattern in LEXICAL if name in _CUES and pattern.search(folded))


def feature_matrix(texts: Sequence[str], vectors: NDArray[np.float32]) -> NDArray[np.float64]:
    """Embedding (unit vector) followed by the lexical cues, one row per text."""
    lexical = np.stack([lexical_features(t) for t in texts]) if texts else np.zeros((0, len(LEXICAL)))
    return np.hstack([vectors.astype(np.float64), lexical])


def weights_checksum(weights: Vector, bias: float) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(weights, dtype="<f8").tobytes() + repr(float(bias)).encode()
    ).hexdigest()


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
        """``path``: the ``.npz`` weights; metadata (version, threshold, checksum, features, dev metrics) beside it.
        Raises ``ValueError`` when the weights do not match their checksum or the features differ from ``LEXICAL``."""
        with np.load(path) as data:
            weights, bias = np.asarray(data["weights"], dtype=np.float64), float(data["bias"])
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        names = [name for name, _ in LEXICAL]
        if metadata.get("lexical_features") != names:
            raise ValueError(f"artefact features {metadata.get('lexical_features')} != code {names}: retrain")
        if metadata.get("weights_sha256") != weights_checksum(weights, bias):
            raise ValueError("artefact weights do not match their checksum (corrupt or edited): retrain")
        return cls(weights, bias, float(metadata["threshold"]), str(metadata["version"]), metadata)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, bias=np.array(self.bias))
        meta = {
            **self.metadata,
            "version": self.version,
            "threshold": self.threshold,
            "lexical_features": [name for name, _ in LEXICAL],
            "weights_sha256": weights_checksum(self.weights, self.bias),
        }
        path.with_suffix(".json").write_bytes(
            (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )


@dataclass(frozen=True, slots=True)
class ClassifierVerdict:
    flagged: bool
    probability: float  # highest among the scored units
    unit: str  # that unit (folded text): for review, never logged
    ms: int  # scoring time (without queueing)
    scored: int = 0  # units embedded
    unscored: int = 0  # ungated units left out by MAX_UNITS


class ClassifierError(RuntimeError):
    """The classifier could not score in time (busy, budget) or failed. Callers fall back to the regex guard.
    ``reason``: "busy", "timeout" or "error" (the ``stats`` key)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class UnsafeAdviceClassifier:
    def __init__(
        self,
        model: ClassifierModel,
        embedder: Embedder,
        guard: OutputGuard,
        *,
        threshold: float | None = None,
        budget_s: float | None = DEFAULT_BUDGET_S,
        max_units: int = MAX_UNITS,
    ) -> None:
        if budget_s is not None and budget_s <= 0:
            raise ValueError(f"budget_s must be positive, got {budget_s}")
        if max_units < 1:
            raise ValueError(f"max_units must be at least 1, got {max_units}")
        self.model = model
        self.threshold = model.threshold if threshold is None else threshold
        if not 0 < self.threshold < 1:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")
        self.budget_s = budget_s
        self.max_units = max_units
        # reply text is user-specific: always embedded directly, never cached
        self._embedder = embedder.inner if isinstance(embedder, CachedEmbedder) else embedder
        expected = model.metadata.get("embedding_model")
        name = getattr(self._embedder, "model_name", None)
        if expected is not None and name is not None and name != expected:
            raise ValueError(f"artefact was trained on {expected}, the embedder is {name}")
        dim = self._embedder.embed_documents(["send the code"]).shape[1]
        if dim + len(LEXICAL) != model.weights.shape[0]:
            raise ValueError(
                f"artefact expects {model.weights.shape[0] - len(LEXICAL)}-dim embeddings, the embedder gives {dim}"
            )
        self._guard = guard
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guard-classifier")
        self._inflight: set[Future[ClassifierVerdict]] = set()
        # fail-open and coverage counters since startup (by reason): logged with every failure, readable by ops
        self.stats: Counter[str] = Counter()

    def supports(self, language: LanguageCode) -> bool:
        return language == PIVOT

    def candidates(self, text: str, language: LanguageCode) -> tuple[list[str], int]:
        """The units to score (most suspicious first, at most ``max_units``) and how many were left out."""
        units = [
            u.text
            for u in self._guard.units(text, language)
            if not u.gated and len(u.text.split()) >= MIN_WORDS and not left_to_the_regex(u.text)
        ]
        units = list(dict.fromkeys(units))  # a one-clause sentence is also its own clause
        ranked = sorted(range(len(units)), key=lambda i: (-_suspicion(units[i]), i))
        chosen = sorted(ranked[: self.max_units])
        return [units[i] for i in chosen], max(len(units) - self.max_units, 0)

    def score(self, text: str, language: LanguageCode) -> ClassifierVerdict:
        """Blocking CPU work (one embedding batch): run it off the event loop (``check``)."""
        start = time.perf_counter()
        texts, unscored = self.candidates(text, language)
        best, best_unit = 0.0, ""
        if texts:
            probabilities = self.model.probabilities(
                feature_matrix(texts, self._embedder.embed_documents(texts))
            )
            index = int(np.argmax(probabilities))
            best, best_unit = float(probabilities[index]), texts[index]
        ms = int((time.perf_counter() - start) * 1000)
        return ClassifierVerdict(best >= self.threshold, round(best, 4), best_unit, ms, len(texts), unscored)

    async def check(self, text: str, language: LanguageCode, *, wait: bool = True) -> ClassifierVerdict:
        """Score within the budget, queueing behind running calls. ``wait=False`` (shadow: nobody waits for the
        result) queues behind at most one call, so it never delays an enforce check by more than one scoring.
        Raises ``ClassifierError`` when the queue is full, the budget runs out or scoring fails."""
        self._inflight = {f for f in self._inflight if not f.done()}
        if len(self._inflight) >= (MAX_PENDING if wait else SHADOW_PENDING):
            raise self._fail("busy", f"busy: {len(self._inflight)} classifications running or queued")
        future = self._worker.submit(self.score, text, language)
        self._inflight.add(future)
        try:
            verdict = await asyncio.wait_for(asyncio.wrap_future(future), self.budget_s)
        except TimeoutError as exc:
            future.cancel()  # still queued: it never runs
            raise self._fail("timeout", "out of time for the guard classifier") from exc
        except Exception as exc:
            raise self._fail("error", f"guard classifier failed: {type(exc).__name__}") from exc
        self.stats["scored"] += 1
        self.stats["unscored_units"] += verdict.unscored
        return verdict

    def _fail(self, reason: str, message: str) -> ClassifierError:
        self.stats[f"fail_open_{reason}"] += 1
        counts = " ".join(f"{k}={v}" for k, v in sorted(self.stats.items()))
        return ClassifierError(reason, f"{message} ({counts})")

    async def aclose(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)
