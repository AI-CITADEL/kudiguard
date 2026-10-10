"""Train the guard's second layer (safety/guard_classifier.py): logistic regression on bge-small + lexical cues.

    python scripts/train_guard_classifier.py                   # writes data/guard/en_classifier.npz + .json
    python scripts/train_guard_classifier.py --report          # train and print metrics only

Deterministic: fixed seed, fixed data order, full-batch gradient descent. Data (labels for "unsafe advice"):
tests/fixtures/guard_probes.yaml (English, unsafe_instruction), D's data/eval/unsafe_advice_probes.yaml
(instruction), a stratified sample of the generated grammar (tests/guard_grammar.py, down-weighted so templates do not
dominate), data/guard/natural_en.yaml (hand-written English and Pidgin), and negatives from every pack fact and action,
tests/fixtures/safe_advice.yaml and the English fixed strings. A hash of the text puts 20% in a dev split, used only
to choose the threshold (false positives on dev negatives <= 2%) and to report metrics. Never add held-out sets.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from guard_grammar import _all_instruction_cases, _sample  # noqa: E402
from kudiguard.knowledge.embeddings import FastEmbedder  # noqa: E402
from kudiguard.knowledge.file_pack import FilePackSource  # noqa: E402
from kudiguard.languages import LanguageRegistry  # noqa: E402
from kudiguard.safety.guard_classifier import (  # noqa: E402
    LEXICAL,
    ClassifierModel,
    UnsafeAdviceClassifier,
    feature_matrix,
)
from kudiguard.safety.output_guard import OutputGuard, OutputIssue  # noqa: E402
from kudiguard.text import normalize  # noqa: E402

SEED = 20261010
MAX_DEV_FALSE_POSITIVE_RATE = 0.02
L2 = 1e-4  # chosen on dev among 1e-2..0 (docs/SPIKES.md S8); lower barely changes dev, higher loses recall
STEPS = 3000
LEARNING_RATE = 2.0
OUT = ROOT / "data" / "guard" / "en_classifier.npz"


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    label: int  # 1 = unsafe advice
    source: str
    weight: float


def _examples() -> Iterator[Example]:
    natural = yaml.safe_load((ROOT / "data" / "guard" / "natural_en.yaml").read_text(encoding="utf-8"))
    for text in natural["unsafe"]:
        yield Example(text, 1, "natural", 4.0)
    for text in natural["safe"]:
        yield Example(text, 0, "natural", 4.0)
    for probe in yaml.safe_load(
        (ROOT / "tests" / "fixtures" / "guard_probes.yaml").read_text(encoding="utf-8")
    ):
        if probe["lang"] == "en" and probe["issue"] == "unsafe_instruction":
            yield Example(probe["text"], int(probe["expect"] == "block"), "probes", 1.0)
    for probe in yaml.safe_load(
        (ROOT / "data" / "eval" / "unsafe_advice_probes.yaml").read_text(encoding="utf-8")
    ):
        if probe["category"] == "instruction":
            yield Example(probe["text"], int(probe["expect"] == "unsafe"), "d_probes", 1.0)
    for case in _sample(_all_instruction_cases(), 2, SEED):
        yield Example(case.text, int(case.expect == "block"), "grammar", 0.5)
    for pack in sorted((ROOT / "data" / "packs").iterdir()):
        manifest = pack / "pack.yaml"
        if manifest.is_file() and "type: http" not in manifest.read_text(encoding="utf-8"):
            for entry in FilePackSource.load(pack).entries:
                for text in (*entry.facts, *entry.actions):
                    yield Example(text, 0, "packs", 1.0)
    advice = yaml.safe_load((ROOT / "tests" / "fixtures" / "safe_advice.yaml").read_text(encoding="utf-8"))
    for text in advice["en"]:
        yield Example(text, 0, "safe_advice", 1.0)
    english = yaml.safe_load((ROOT / "data" / "languages" / "en.yaml").read_text(encoding="utf-8"))
    for value in english["strings"].values():
        for line in str(value).splitlines():
            if len(line.split()) >= 3:
                yield Example(line, 0, "fixed", 1.0)


def dataset() -> list[Example]:
    seen: dict[str, Example] = {}
    for example in _examples():
        key = normalize(example.text)
        if key and key not in seen:  # first source wins (natural > probes > D > grammar > negatives)
            seen[key] = example
    return list(seen.values())


def is_dev(text: str) -> bool:
    return int(hashlib.sha256(normalize(text).encode()).hexdigest(), 16) % 5 == 0


def train(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Weighted, class-balanced logistic regression by full-batch gradient descent (deterministic)."""
    balance = np.where(labels == 1, 0.5 / labels.mean(), 0.5 / (1 - labels.mean()))
    sample = weights * balance
    sample = sample / sample.sum()
    w = np.zeros(features.shape[1])
    b = 0.0
    for _ in range(STEPS):
        p = 1.0 / (1.0 + np.exp(-(features @ w + b)))
        error = (p - labels) * sample
        w -= LEARNING_RATE * (features.T @ error + L2 * w)
        b -= LEARNING_RATE * error.sum()
    return w, b


def choose_threshold(unit_best: np.ndarray, regex: np.ndarray, labels: np.ndarray) -> float:
    """The lowest threshold at which the shipped decision (regex OR classifier) keeps the false-positive rate on dev
    negatives <= MAX_DEV_FALSE_POSITIVE_RATE. ``unit_best``: per example, the best ungated unit probability."""
    negatives = labels == 0
    for threshold in np.arange(0.5, 0.995, 0.005):
        flagged = regex | (unit_best >= threshold)
        if flagged[negatives].mean() <= MAX_DEV_FALSE_POSITIVE_RATE:
            return float(round(threshold, 3))
    return 0.99


def metrics(predicted: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    negatives = int((labels == 0).sum())
    return {
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "false_positive_rate": round(fp / max(negatives, 1), 4),
        "n": len(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "models" / "fastembed")
    parser.add_argument("--report", action="store_true", help="print metrics, do not write the artefact")
    args = parser.parse_args()
    data = dataset()
    texts = [normalize(e.text) for e in data]
    labels = np.array([e.label for e in data], dtype=np.float64)
    weights = np.array([e.weight for e in data])
    dev = np.array([is_dev(t) for t in texts])
    embedder = FastEmbedder(cache_dir=args.cache)
    features = feature_matrix(texts, embedder.embed_documents(texts))
    w, b = train(features[~dev], labels[~dev], weights[~dev])
    probabilities = 1.0 / (1.0 + np.exp(-(features @ w + b)))
    # the full decision as shipped: regex guard OR classifier (units, gated), on the dev sentences
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    guard = OutputGuard(registry)
    classifier = UnsafeAdviceClassifier(ClassifierModel(w, b, 0.5, "", {}), embedder, guard, budget_s=None)
    regex = np.array([OutputIssue.UNSAFE_INSTRUCTION in guard.check(e.text, "en") for e in data])
    unit_best = np.array(
        [classifier.score(e.text, "en").probability if d else 0.0 for e, d in zip(data, dev, strict=True)]
    )
    threshold = choose_threshold(unit_best[dev], regex[dev], labels[dev])
    learned = unit_best >= threshold
    report = {
        "train": {"n": int((~dev).sum()), "positives": int(labels[~dev].sum())},
        "dev_classifier_alone": metrics((probabilities[dev] >= threshold).astype(int), labels[dev]),
        "dev_regex_alone": metrics(regex[dev].astype(int), labels[dev]),
        "dev_combined": metrics((regex[dev] | learned[dev]).astype(int), labels[dev]),
        "dev_by_source": {
            source: metrics(((regex | learned)[dev & mask]).astype(int), labels[dev & mask])
            for source in sorted({e.source for e in data})
            if (mask := np.array([e.source == source for e in data])).any() and (dev & mask).any()
        },
    }
    data_hash = hashlib.sha256(
        "\n".join(f"{e.label}\t{normalize(e.text)}" for e in data).encode()
    ).hexdigest()[:16]
    version = f"{datetime.date.today().isoformat()}-{data_hash[:8]}"
    metadata = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "lexical_features": [name for name, _ in LEXICAL],
        "seed": SEED,
        "training_data_sha256_16": data_hash,
        "max_dev_false_positive_rate": MAX_DEV_FALSE_POSITIVE_RATE,
        "trained": datetime.date.today().isoformat(),
        **report,
    }
    print(yaml.safe_dump({"threshold": round(threshold, 4), **report}, sort_keys=False))
    if not args.report:
        ClassifierModel(w, b, round(threshold, 4), version, metadata).save(OUT)
        print(f"wrote {OUT} (version {version})")


if __name__ == "__main__":
    main()
