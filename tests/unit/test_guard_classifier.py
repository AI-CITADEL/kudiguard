"""The learned guard layer with a fake embedder and a hand-set model (the real artefact: -m live)."""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fakes import BagOfWordsEmbedder
from kudiguard.knowledge.embeddings import CachedEmbedder
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.guard_classifier import (
    LEXICAL,
    ClassifierError,
    ClassifierModel,
    UnsafeAdviceClassifier,
    feature_matrix,
    lexical_features,
)
from kudiguard.safety.output_guard import OutputGuard

ROOT = Path(__file__).resolve().parents[2]
NAMES = [name for name, _ in LEXICAL]


def hand_model(threshold: float = 0.5) -> ClassifierModel:
    """Embedding weights zero; "credential + imperative" pushes the score up, a negation pulls it down."""
    weights = np.zeros(BagOfWordsEmbedder.DIM + len(LEXICAL))
    lexical = dict.fromkeys(NAMES, 0.0) | {"credential": 3.0, "imperative": 3.0, "negation": -6.0}
    weights[BagOfWordsEmbedder.DIM :] = [lexical[n] for n in NAMES]
    return ClassifierModel(weights, -4.0, threshold, "test", {})


def classifier(registry: LanguageRegistry, **kwargs: object) -> UnsafeAdviceClassifier:
    return UnsafeAdviceClassifier(hand_model(), BagOfWordsEmbedder(), OutputGuard(registry), **kwargs)  # type: ignore[arg-type]


def test_lexical_features_and_matrix() -> None:
    features = dict(zip(NAMES, lexical_features("Send the OTP to the agent on WhatsApp."), strict=True))
    assert features["credential"] == features["imperative"] == features["untrusted"] == 1.0
    assert features["negation"] == features["trusted"] == 0.0
    vectors = BagOfWordsEmbedder().embed_documents(["a b", "c d"])
    assert feature_matrix(["a b", "c d"], vectors).shape == (2, BagOfWordsEmbedder.DIM + len(LEXICAL))
    assert feature_matrix([], np.zeros((0, 0), dtype=np.float32)).shape == (0, len(LEXICAL))


def test_model_round_trip_and_shape_check(tmp_path: Path) -> None:
    model = ClassifierModel(np.array([1.0, -1.0]), 0.5, 0.7, "v1", {"note": "x"})
    path = tmp_path / "guard" / "m.npz"
    model.save(path)
    loaded = ClassifierModel.load(path)
    assert loaded.version == "v1" and loaded.threshold == 0.7 and loaded.metadata["note"] == "x"
    np.testing.assert_allclose(loaded.weights, model.weights)
    assert loaded.probabilities(np.array([[2.0, 0.0]]))[0] == pytest.approx(1 / (1 + np.exp(-2.5)))
    with pytest.raises(ValueError, match="expects 2 features"):
        loaded.probabilities(np.zeros((1, 3)))


def test_units_split_sentences_and_clauses_and_gate_them(registry: LanguageRegistry) -> None:
    guard = OutputGuard(registry)
    units = guard.units("Do not panic, send the OTP to him. Is it real? They said you should pay.", "en")
    assert [(u.text, u.gated) for u in units] == [
        ("do not panic, send the otp to him", True),  # the sentence opens with a negation
        ("do not panic", True),
        ("send the otp to him", False),
        ("is it real", True),  # a question
        ("they said you should pay", True),  # reported speech
    ]
    condition = guard.units("They said your account will be blocked unless you pay the fee.", "en")
    assert [u.gated for u in condition] == [True, True, True]
    assert guard.units("", "en") == []
    gated = {u.text: u.gated for u in guard.units(
        "If they ask you to share the code, end the call. You send them the code and they empty your account. "
        "Scammers often say: send the code now. Do as he says: send him the OTP. Tell him: no, I won't share it. "
        "You entered your PIN on his phone, so change it. Send it.", "en")}  # fmt: skip
    assert gated["if they ask you to share the code"] and not gated["end the call"]
    assert gated["you send them the code and they empty your account"]
    assert gated["send the code now"]  # after a reported intro
    assert gated["do as he says"] and not gated["send him the otp"]  # an intro, then the instruction
    assert gated["i won't share it"] and gated["you entered your pin on his phone"] and gated["send it"]


def test_trusted_recipients_are_left_to_the_regex(registry: LanguageRegistry) -> None:
    model = classifier(registry)
    assert not model.score("Forward the code message to your bank.", "en").flagged
    assert not model.score("Send the code to the police.", "en").flagged
    assert model.score("Send the code to your bank's agent.", "en").flagged  # bank staff are not the bank


def test_score_flags_ungated_units_only(registry: LanguageRegistry) -> None:
    model = classifier(registry)
    flagged = model.score("Stay calm. Send the OTP to the agent.", "en")
    assert flagged.flagged and flagged.unit == "send the otp to the agent" and flagged.probability > 0.8
    assert not model.score("Never send the OTP to anyone.", "en").flagged
    assert not model.score("They told you to send the OTP.", "en").flagged  # reported
    empty = model.score("", "en")
    assert (empty.flagged, empty.probability, empty.unit) == (False, 0.0, "")


def test_threshold_override_and_validation(registry: LanguageRegistry) -> None:
    assert classifier(registry, threshold=0.99).threshold == 0.99
    assert not classifier(registry, threshold=0.99).score("Send the OTP to him.", "en").flagged
    for bad in ({"threshold": 1.0}, {"threshold": 0.0}, {"budget_s": 0}):
        with pytest.raises(ValueError):
            classifier(registry, **bad)
    model = classifier(registry)
    assert model.supports("en") and not model.supports("ha")


def test_reply_text_is_never_cached(registry: LanguageRegistry) -> None:
    inner = BagOfWordsEmbedder()
    cached = CachedEmbedder(inner)
    model = UnsafeAdviceClassifier(hand_model(), cached, OutputGuard(registry))
    model.score("Send the OTP to him.", "en")
    assert inner.batches == [["send the otp to him"]] and cached._vectors == {}


async def test_check_runs_off_the_loop(registry: LanguageRegistry) -> None:
    model = classifier(registry)
    verdict = await model.check("Send the OTP to him.", "en")
    assert verdict.flagged and verdict.ms >= 0
    await model.aclose()


class Gated(BagOfWordsEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        self.release.wait(5)
        return super().embed_documents(texts)


async def test_budget_and_busy_fail_to_the_regex(registry: LanguageRegistry) -> None:
    embedder = Gated()
    model = UnsafeAdviceClassifier(hand_model(), embedder, OutputGuard(registry), budget_s=0.05)
    start = time.perf_counter()
    with pytest.raises(ClassifierError, match="out of time"):
        await model.check("Send the OTP to him.", "en")
    assert time.perf_counter() - start < 1
    with pytest.raises(ClassifierError, match="busy"):
        await model.check("Send the OTP to him.", "en")
    embedder.release.set()
    assert model._running is not None
    await asyncio.wrap_future(model._running)
    assert (await model.check("Send the OTP to him.", "en")).flagged
    await model.aclose()


def test_shipped_artefact_is_consistent() -> None:
    path = ROOT / "data" / "guard" / "en_classifier.npz"
    model = ClassifierModel.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert model.weights.shape == (384 + len(LEXICAL),) and 0 < model.threshold < 1
    assert meta["lexical_features"] == NAMES and meta["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert meta["dev_combined"]["false_positive_rate"] <= meta["max_dev_false_positive_rate"]
    assert model.version.endswith(meta["training_data_sha256_16"][:8])


def test_check_logs_nothing_of_the_text(registry: LanguageRegistry, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="kudiguard.safety.guard_classifier"):
        classifier(registry).score("Send the OTP to Musa.", "en")
    assert "musa" not in caplog.text.lower()
