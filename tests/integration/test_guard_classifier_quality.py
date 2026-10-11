"""The shipped guard classifier with the real embedding model (CPU, -m live): dev metrics, latency, and the combined
layer (regex OR classifier) on the probe sets and the grammar. Thresholds are regression guards.

The dev gate runs on the FROZEN dev split (data/guard/train_set.jsonl), not on the live sources, so merging new fixed
strings or probes cannot move it; it allows a small band around the metrics recorded in the artefact because the
regex half of the decision follows data/languages/en.yaml, which merges do change."""

import os
import sys
import time
from pathlib import Path
from statistics import median, quantiles

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


RECALL_BAND = 0.03  # allowed drop in dev recall vs the artefact's record
FPR_BAND = 0.01  # allowed rise in dev false-positive rate


@pytest.mark.live
def test_dev_metrics_of_the_shipped_decision(layers: tuple[OutputGuard, UnsafeAdviceClassifier]) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_guard_classifier as training

    guard, classifier = layers
    data = [e for e in training.load_frozen() if e.split == "dev"]
    flagged = np.array([combined(guard, classifier, e.text)[1] for e in data])
    labels = np.array([e.label for e in data])
    result = training.metrics(flagged.astype(int), labels)
    recorded = classifier.model.metadata["dev_combined"]
    assert isinstance(recorded, dict)
    natural = np.array([e.source == "natural" for e in data])
    print(
        f"\ndev combined {result} (recorded {recorded}); "
        f"natural {training.metrics(flagged[natural].astype(int), labels[natural])}"
    )
    assert result["recall"] >= recorded["recall"] - RECALL_BAND, (
        f"dev recall {result['recall']} fell more than {RECALL_BAND} below the artefact's {recorded['recall']}: "
        "the regex guard or gating changed under the classifier; check en.yaml output_guard or retrain"
    )
    assert result["false_positive_rate"] <= recorded["false_positive_rate"] + FPR_BAND, (
        f"dev false-positive rate {result['false_positive_rate']} rose more than {FPR_BAND} above the artefact's "
        f"{recorded['false_positive_rate']}: a regex or gating change now flags safe dev sentences"
    )


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


def _timings(
    classifier: UnsafeAdviceClassifier, make: object, runs: int = 40
) -> tuple[float, float, int, int]:
    timings, verdict = [], None
    for n in range(runs):
        start = time.perf_counter()
        verdict = classifier.score(make(n), "en")  # type: ignore[operator]
        timings.append((time.perf_counter() - start) * 1000)
    assert verdict is not None
    return median(timings), quantiles(timings, n=20)[-1], verdict.scored, verdict.unscored


@pytest.mark.live
def test_latency_per_reply(layers: tuple[OutputGuard, UnsafeAdviceClassifier]) -> None:
    """One embedding batch per reply, at most MAX_UNITS units: typical, 8 and 20 cue sentences."""
    _, classifier = layers
    classifier.score(REPLY, "en")  # warm-up

    def dense(count: int, n: int) -> str:
        return "\n".join(
            f"{i}. Send the receipt to your landlord and pay the electricity bill today, then check it ({i}{n})."
            for i in range(1, count + 1)
        )

    worst = 0.0
    for name, make in (
        ("typical reply", lambda n: f"{REPLY}\nTell someone you trust ({n})."),
        ("8 cue sentences", lambda n: dense(8, n)),
        ("20 cue sentences", lambda n: dense(20, n)),
    ):
        p50, p95, scored, unscored = _timings(classifier, make)
        worst = max(worst, p95)
        print(f"\n{name}: p50 {p50:.0f} ms, p95 {p95:.0f} ms ({scored} units scored, {unscored} unscored)")
    # generous: shared CPU; the live budget (DEFAULT_BUDGET_S) covers one queued call ahead (fail open past it)
    assert worst < 300
