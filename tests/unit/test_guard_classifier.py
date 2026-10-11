"""The learned guard layer with a fake embedder and a hand-set model (the real artefact: -m live)."""

import asyncio
import hashlib
import json
import logging
import sys
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
    MAX_PENDING,
    ClassifierError,
    ClassifierModel,
    UnsafeAdviceClassifier,
    feature_matrix,
    is_pidgin,
    left_to_the_regex,
    lexical_features,
    weights_checksum,
)
from kudiguard.safety.output_guard import OutputGuard

ROOT = Path(__file__).resolve().parents[2]
NAMES = [name for name, _ in LEXICAL]


def hand_model(threshold: float = 0.5) -> ClassifierModel:
    """Embedding weights zero; "credential + imperative" pushes the score up, a negation pulls it down."""
    weights = np.zeros(BagOfWordsEmbedder.DIM + len(LEXICAL))
    lexical = dict.fromkeys(NAMES, 0.0) | {"credential": 3.0, "imperative": 3.0, "negation": -6.0}
    weights[BagOfWordsEmbedder.DIM :] = [lexical[n] for n in NAMES]
    return ClassifierModel(weights, -4.0, threshold, "test", {"embedding_model": "fake"})


def classifier(registry: LanguageRegistry, **kwargs: object) -> UnsafeAdviceClassifier:
    return UnsafeAdviceClassifier(hand_model(), BagOfWordsEmbedder(), OutputGuard(registry), **kwargs)  # type: ignore[arg-type]


def test_lexical_features_and_matrix() -> None:
    features = dict(zip(NAMES, lexical_features("Send the OTP to the agent on WhatsApp."), strict=True))
    assert features["credential"] == features["imperative"] == features["untrusted"] == 1.0
    assert features["negation"] == features["trusted"] == features["access"] == 0.0
    # "WhatsApp" alone is not a stranger: it appears in good advice too
    assert (
        dict(zip(NAMES, lexical_features("Report the number to WhatsApp."), strict=True))["untrusted"] == 0.0
    )
    access = dict(zip(NAMES, lexical_features("Allow him to share your screen."), strict=True))
    assert access["access"] == access["imperative"] == 1.0
    vectors = BagOfWordsEmbedder().embed_documents(["a b", "c d"])
    assert feature_matrix(["a b", "c d"], vectors).shape == (2, BagOfWordsEmbedder.DIM + len(LEXICAL))
    assert feature_matrix([], np.zeros((0, 0), dtype=np.float32)).shape == (0, len(LEXICAL))


def test_pidgin_detection() -> None:
    assert is_pidgin("Abeg no send the code, dem dey lie.") and is_pidgin("Dem fit tell you make you pay.")
    assert not is_pidgin("Do not send the code. Don't pay any fee.")
    # one marker word is not enough: English replies use "oga", "wahala" or quote a scammer's "abeg"
    assert not is_pidgin('This looks like a scam. The "oga" who called is not from your bank.')
    assert not is_pidgin("I am sure this is a scam, the oga lied. Do not pay.")  # "I am" is English
    assert not is_pidgin(
        "Your bank opens at 8 am. Call them then, and do not pay am anything."
    )  # times are English
    assert is_pidgin("Make una no send code give anybody.")  # one strong marker in a one-sentence reply
    assert not is_pidgin("Do not send the code. Abeg, call your bank.")  # one marker over two sentences


def test_model_round_trip_and_shape_check(tmp_path: Path) -> None:
    model = ClassifierModel(np.array([1.0, -1.0]), 0.5, 0.7, "v1", {"note": "x"})
    path = tmp_path / "guard" / "m.npz"
    model.save(path)
    loaded = ClassifierModel.load(path)
    assert loaded.version == "v1" and loaded.threshold == 0.7 and loaded.metadata["note"] == "x"
    assert loaded.metadata["lexical_features"] == NAMES
    np.testing.assert_allclose(loaded.weights, model.weights)
    assert loaded.probabilities(np.array([[2.0, 0.0]]))[0] == pytest.approx(1 / (1 + np.exp(-2.5)))
    with pytest.raises(ValueError, match="expects 2 features"):
        loaded.probabilities(np.zeros((1, 3)))


def test_load_rejects_a_stale_or_edited_artefact(tmp_path: Path) -> None:
    path = tmp_path / "m.npz"
    ClassifierModel(np.array([1.0, -1.0]), 0.5, 0.7, "v1", {}).save(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    path.with_suffix(".json").write_text(
        json.dumps(meta | {"lexical_features": NAMES[:-1]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="features"):
        ClassifierModel.load(path)
    path.with_suffix(".json").write_text(json.dumps(meta | {"weights_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        ClassifierModel.load(path)
    assert weights_checksum(np.array([1.0, -1.0]), 0.5) == meta["weights_sha256"]


def test_classifier_rejects_another_embedder(registry: LanguageRegistry) -> None:
    named = BagOfWordsEmbedder()
    named.model_name = "other/model"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="trained on fake"):
        UnsafeAdviceClassifier(hand_model(), named, OutputGuard(registry))

    class Small(BagOfWordsEmbedder):
        DIM = 384

    with pytest.raises(ValueError, match="512-dim embeddings, the embedder gives 384"):
        UnsafeAdviceClassifier(hand_model(), Small(), OutputGuard(registry))
    for bad in ({"threshold": 1.0}, {"threshold": 0.0}, {"budget_s": 0}, {"max_units": 0}):
        with pytest.raises(ValueError):
            classifier(registry, **bad)


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


def test_units_gate_quotes_and_reported_speech_but_not_instructions_before_a_colon(
    registry: LanguageRegistry,
) -> None:
    guard = OutputGuard(registry)
    gated = {u.text: u.gated for u in guard.units(
        'Scammers write "Send the code to confirm your account. Pay the fee today." Hang up on them. '
        "\u201cShare your screen so I can help,\u201d the caller says. "
        "Dem go ask you make you send the code. Dem fit talk say na bank. Dem fit tell you make you pay. "
        "He will probably ask you to install an app. They will ask you to pay a fee. "
        "Reply to the agent with your card details: number, expiry and CVV. "
        "Tell the caller this: I will call my bank.", "en")}  # fmt: skip
    assert gated['scammers write "send the code to confirm your account']  # reported
    assert gated["pay the fee today"]  # starts inside the quotes
    assert not gated["hang up on them"]
    assert gated["\u201cshare your screen so i can help,\u201d the caller says"]
    assert gated["dem go ask you make you send the code"] and gated["dem fit talk say na bank"]
    assert gated["dem fit tell you make you pay"]
    assert gated["he will probably ask you to install an app"] and gated["they will ask you to pay a fee"]
    assert not gated["reply to the agent with your card details"]  # the intro is itself the instruction
    assert gated["tell the caller this"]  # only introduces what follows
    intro = {
        u.text: u.gated
        for u in guard.units("Tell the agent: stop calling me. Tell him your PIN: 1234.", "en")
    }
    assert (
        intro["tell the agent"] and not intro["tell him your pin"]
    )  # a speech verb with no secret introduces
    told = {u.text: u.gated for u in guard.units(
        'Tell him: "Here is my OTP, 482913." Just say "here is my PIN" and read it out to him. '
        '"Send the OTP to the agent now," is what you should do.', "en")}  # fmt: skip
    # the user is told to say it, or the quote is adopted: not someone else's speech, so scored
    assert not told['"here is my otp'] and not told['say "here is my pin"']
    assert not told['"send the otp to the agent now," is what you should do']
    # a negated told-to-say is a warning; a told-to-say quote opening with "no" is still the user's words
    assert not OutputGuard(registry).check('Never tell the caller "my OTP is 553201".', "en")
    assert OutputGuard(registry).check('Tell him "No problem, here is my OTP 553201."', "en")
    # "don't forget / hesitate to tell" inverts the negation: still an instruction
    assert OutputGuard(registry).check('Don\'t forget to tell him "my OTP is 553201".', "en")
    quoted_tail = {u.text: u.gated for u in guard.units('Tell him "here is my OTP" and hang up.', "en")}
    assert not quoted_tail['tell him "here is my otp"']  # starts outside the quotes: scored


def test_handover_to_a_person_at_the_bank_is_scored(registry: LanguageRegistry) -> None:
    model = classifier(registry)
    for text in (
        "Give the code to the man at your bank.",
        "Send the PIN to the coordinator at your bank branch.",
        "Let him have the code with your phone.",
        "Forward the code to your bank, then send it to him too.",  # a stranger after the trusted recipient
        "Type your PIN into your banking app on the link he sent.",
    ):
        assert not left_to_the_regex(text) and model.score(text, "en").flagged, text
    # the verb's own recipient is the bank, the authorities or the user's own app: the regex guard decides
    for text in (
        "Forward the code message to your bank.",
        "Send his account number to your bank and the police.",
        "Enter your PIN in your banking app.",
        "Give your NIN to the bank officer inside the banking hall.",
        "Give your BVN to the teller.",
    ):
        assert left_to_the_regex(text) and not model.score(text, "en").flagged, text
    assert not left_to_the_regex("Give your PIN to the officer at the counter.")  # a secret, not an ID number
    for text in (  # a person, a supplied payee or a "safe account" after the trusted place
        "Transfer the money with your bank app to the account the caller gave you.",
        "Move your savings in your bank app to the safe account the officer gave you.",
        "Enter your PIN in your bank app while the agent watches your screen.",
    ):
        assert not left_to_the_regex(text), text
    assert not left_to_the_regex("Send the code to your bank's agent.")  # bank staff are not the bank


def test_candidates_cap_the_units_most_suspicious_first(registry: LanguageRegistry) -> None:
    model = classifier(registry, max_units=2)
    texts, unscored = model.candidates(
        "Stay calm and breathe. Call your sister today. Send the OTP to the agent. Ok then.", "en"
    )
    # "send the otp to the agent" (3 cues) and "call your sister today" (1) beat "stay calm and breathe" (0); "ok
    # then" is too short to score; the chosen units keep their order in the reply
    assert texts == ["call your sister today", "send the otp to the agent"] and unscored == 1
    verdict = model.score("Stay calm and breathe. Call your sister today. Send the OTP to the agent.", "en")
    assert verdict.flagged and (verdict.scored, verdict.unscored) == (2, 1)


def test_score_flags_ungated_units_only(registry: LanguageRegistry) -> None:
    model = classifier(registry)
    flagged = model.score("Stay calm. Send the OTP to the agent.", "en")
    assert flagged.flagged and flagged.unit == "send the otp to the agent" and flagged.probability > 0.8
    assert not model.score("Never send the OTP to anyone.", "en").flagged
    assert not model.score("They told you to send the OTP.", "en").flagged  # reported
    empty = model.score("", "en")
    assert (empty.flagged, empty.probability, empty.unit) == (False, 0.0, "")


def test_threshold_override(registry: LanguageRegistry) -> None:
    assert classifier(registry, threshold=0.99).threshold == 0.99
    assert not classifier(registry, threshold=0.99).score("Send the OTP to him.", "en").flagged
    model = classifier(registry)
    assert model.supports("en") and not model.supports("ha")


def test_reply_text_is_never_cached(registry: LanguageRegistry) -> None:
    inner = BagOfWordsEmbedder()
    cached = CachedEmbedder(inner)
    model = UnsafeAdviceClassifier(hand_model(), cached, OutputGuard(registry))
    inner.batches.clear()  # the dimension check at startup
    model.score("Send the OTP to him.", "en")
    assert inner.batches == [["send the otp to him"]] and cached._vectors == {}


async def test_check_runs_off_the_loop_and_counts(registry: LanguageRegistry) -> None:
    model = classifier(registry)
    verdict = await model.check("Send the OTP to him. Then call your sister and your mother today.", "en")
    assert verdict.flagged and verdict.ms >= 0 and model.stats["scored"] == 1
    await model.aclose()


class Gated(BagOfWordsEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.release.set()  # the startup dimension check passes; tests clear it to hold the worker

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        self.release.wait(5)
        return super().embed_documents(texts)


async def test_calls_queue_within_the_budget(registry: LanguageRegistry) -> None:
    embedder = Gated()
    model = UnsafeAdviceClassifier(hand_model(), embedder, OutputGuard(registry), budget_s=1.0)
    embedder.release.clear()
    first = asyncio.create_task(model.check("Send the OTP to him.", "en"))
    second = asyncio.create_task(model.check("Give him your PIN now.", "en"))
    await asyncio.sleep(0.05)
    embedder.release.set()  # the worker frees up within the budget: both are scored, neither fails open
    assert (await first).flagged and (await second).flagged
    assert model.stats["scored"] == 2 and not any(k.startswith("fail_open") for k in model.stats)
    await model.aclose()


async def test_full_queue_late_and_broken_fail_open_with_counters(registry: LanguageRegistry) -> None:
    embedder = Gated()
    model = UnsafeAdviceClassifier(hand_model(), embedder, OutputGuard(registry), budget_s=0.05)
    embedder.release.clear()
    start = time.perf_counter()
    with pytest.raises(ClassifierError, match="out of time") as late:
        await model.check("Send the OTP to him.", "en")
    assert late.value.reason == "timeout" and time.perf_counter() - start < 1
    # the late call still holds the worker; enforce calls queue behind it up to MAX_PENDING
    waiting = [asyncio.create_task(model.check("Send the OTP to him.", "en")) for _ in range(MAX_PENDING - 1)]
    await asyncio.sleep(0)
    with pytest.raises(ClassifierError, match="busy"):  # shadow: at most one call ahead of it
        await model.check("Send the OTP to him.", "en", wait=False)
    with pytest.raises(ClassifierError, match="busy") as full:  # enforce: the queue is full
        await model.check("Send the OTP to him.", "en")
    assert full.value.reason == "busy" and "fail_open_busy=2" in str(full.value)
    for task in waiting:
        with pytest.raises(ClassifierError, match="out of time"):
            await task
    assert model.stats["fail_open_timeout"] == MAX_PENDING
    embedder.release.set()
    await asyncio.sleep(0.1)
    assert (await model.check("Send the OTP to him.", "en")).flagged

    class Broken(BagOfWordsEmbedder):
        fail = False

        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            if self.fail:
                raise RuntimeError("onnx")
            return super().embed_documents(texts)

    broken = Broken()
    failing = UnsafeAdviceClassifier(hand_model(), broken, OutputGuard(registry))
    broken.fail = True
    with pytest.raises(ClassifierError, match="failed: RuntimeError") as error:
        await failing.check("Send the OTP to him.", "en")
    assert error.value.reason == "error" and failing.stats["fail_open_error"] == 1
    await model.aclose()
    await failing.aclose()


def test_shipped_artefact_matches_the_frozen_training_set() -> None:
    path = ROOT / "data" / "guard" / "en_classifier.npz"
    model = ClassifierModel.load(path)  # checks the feature list and the weights checksum
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert model.weights.shape == (384 + len(LEXICAL),) and 0 < model.threshold < 1
    assert meta["embedding_model"] == "BAAI/bge-small-en-v1.5"
    points = meta["operating_points"]
    assert points["shipped"]["threshold"] == model.threshold == 0.85 and points["strict"]["threshold"] > 0.85
    assert meta["dev_combined"]["false_positive_rate"] <= meta["max_dev_false_positive_rate"]
    frozen = ROOT / meta["training_set"]
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert meta["training_set_sha256"] == digest, "the artefact was not trained on the committed training set"
    assert model.version.endswith(digest[:8])


def test_frozen_training_set_is_clean() -> None:
    """Fixed strings train only; dev and train share no group and no exact text; probes marked train: false (copies
    of a now-public evaluation set) stay out."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_guard_classifier as training
    import yaml

    data = training.load_frozen()
    dev = {e.group for e in data if e.split == "dev"}
    assert dev and not dev & {e.group for e in data if e.split == "train"}
    assert all(e.split == "train" for e in data if e.source == "fixed")
    texts = {e.text for e in data}
    assert len(texts) == len(data)
    probes = yaml.safe_load((ROOT / "tests" / "fixtures" / "guard_probes.yaml").read_text(encoding="utf-8"))
    excluded = [p["text"] for p in probes if p.get("train") is False]
    assert excluded and not texts & set(excluded)


def test_check_logs_nothing_of_the_text(registry: LanguageRegistry, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="kudiguard.safety.guard_classifier"):
        classifier(registry).score("Send the OTP to Musa.", "en")
    assert "musa" not in caplog.text.lower()
