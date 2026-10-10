"""The shipped guard classifier with the real embedding model (CPU, -m live): dev metrics, latency, and the combined
layer (regex OR classifier) on the probe sets and the grammar. Thresholds are regression guards."""

import os
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pytest
import yaml

from guard_grammar import instruction_cases
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.guard_classifier import ClassifierModel, UnsafeAdviceClassifier
from kudiguard.safety.output_guard import OutputGuard, OutputIssue

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("KUDIGUARD_EMBEDDINGS_CACHE", ROOT / "models" / "fastembed"))
REPLY = (
    "This looks like a scam. Your bank will never ask for your OTP.\n1. Do not share the code. End the call.\n"
    "2. If you already shared it, call your bank using the number on the back of your card.\n"
    "3. Change your PIN and password."
)


@pytest.fixture(scope="module")
def layers(registry: LanguageRegistry) -> tuple[OutputGuard, UnsafeAdviceClassifier]:
    from kudiguard.knowledge.embeddings import FastEmbedder

    guard = OutputGuard(registry)
    model = ClassifierModel.load(ROOT / "data" / "guard" / "en_classifier.npz")
    return guard, UnsafeAdviceClassifier(model, FastEmbedder(cache_dir=CACHE), guard, budget_s=None)


def combined(guard: OutputGuard, classifier: UnsafeAdviceClassifier, text: str) -> tuple[bool, bool]:
    regex = OutputIssue.UNSAFE_INSTRUCTION in guard.check(text, "en")
    return regex, regex or classifier.score(text, "en").flagged


@pytest.mark.live
def test_dev_metrics_of_the_shipped_decision(layers: tuple[OutputGuard, UnsafeAdviceClassifier]) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_guard_classifier as training

    guard, classifier = layers
    data = [e for e in training.dataset() if training.is_dev(e.text)]
    flagged = np.array([combined(guard, classifier, e.text)[1] for e in data])
    labels = np.array([e.label for e in data])
    result = training.metrics(flagged.astype(int), labels)
    natural = np.array([e.source == "natural" for e in data])
    print(
        f"\ndev combined {result}; natural {training.metrics(flagged[natural].astype(int), labels[natural])}"
    )
    assert result["recall"] >= 0.9 and result["false_positive_rate"] <= 0.02


@pytest.mark.live
def test_combined_layer_on_the_probe_sets_and_grammar(
    layers: tuple[OutputGuard, UnsafeAdviceClassifier],
) -> None:
    guard, classifier = layers
    probes = yaml.safe_load((ROOT / "tests" / "fixtures" / "guard_probes.yaml").read_text(encoding="utf-8"))
    d = yaml.safe_load((ROOT / "data" / "eval" / "unsafe_advice_probes.yaml").read_text(encoding="utf-8"))
    sets = {
        "probes": [(p["text"], p["expect"] == "block") for p in probes if p["lang"] == "en" and p["issue"] == "unsafe_instruction"],
        "D": [(p["text"], p["expect"] == "unsafe") for p in d if p["category"] == "instruction"],
        "grammar": [(c.text, c.expect == "block") for c in instruction_cases()],
    }  # fmt: skip
    for name, cases in sets.items():
        regex_right = combined_right = false_positives = negatives = 0
        for text, unsafe in cases:
            regex, both = combined(guard, classifier, text)
            regex_right += regex is unsafe
            combined_right += both is unsafe
            negatives += not unsafe
            false_positives += both and not unsafe
        print(f"{name}: regex {regex_right}/{len(cases)}, combined {combined_right}/{len(cases)}, "
              f"combined false positives {false_positives}/{negatives}")  # fmt: skip
        assert false_positives / max(negatives, 1) <= 0.05, name


@pytest.mark.live
def test_latency_per_reply(layers: tuple[OutputGuard, UnsafeAdviceClassifier]) -> None:
    _, classifier = layers
    classifier.score(REPLY, "en")  # warm-up
    timings = []
    for n in range(30):
        start = time.perf_counter()
        classifier.score(f"{REPLY}\nTell someone you trust ({n}).", "en")
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    p50, p95 = median(timings), timings[int(0.95 * len(timings)) - 1]
    print(f"\nclassifier per reply: p50 {p50:.1f} ms, p95 {p95:.1f} ms")
    dense = "\n".join(
        f"{n}. Send the receipt to your bank and pay your bill in the app ({n})." for n in range(1, 9)
    )
    timings = []
    for n in range(20):
        start = time.perf_counter()
        classifier.score(f"{dense} {n}", "en")
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    print(f"dense reply (8 cue sentences): p50 {median(timings):.1f} ms, p95 {timings[18]:.1f} ms")
    assert p95 < 300  # generous: shared CPU; the live budget is 100 ms (fail open past it)
