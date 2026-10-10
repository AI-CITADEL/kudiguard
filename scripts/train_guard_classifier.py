"""Train the guard's second layer (safety/guard_classifier.py): logistic regression on bge-small + lexical cues.

    python scripts/train_guard_classifier.py --freeze          # rebuild data/guard/train_set.jsonl from the sources
    python scripts/train_guard_classifier.py                   # train on the frozen set, write en_classifier.npz/.json
    python scripts/train_guard_classifier.py --report          # train and print metrics only

The training set is FROZEN: ``data/guard/train_set.jsonl`` is committed and the artefact records its SHA-256, so a
merge that changes a source file (new fixed strings, new probes) cannot silently change the model or the dev gate.
``--freeze`` rebuilds it from the sources: tests/fixtures/guard_probes.yaml (English, unsafe_instruction; items
marked ``train: false`` are left out), D's data/eval/unsafe_advice_probes.yaml (instruction), a stratified sample of
the generated grammar (tests/guard_grammar.py, down-weighted), data/guard/natural_en.yaml (hand-written English and
Pidgin), and negatives from every pack fact and action, tests/fixtures/safe_advice.yaml and the English fixed strings
(always in train). Never add held-out sets.

Dev split (20%, threshold choice and reporting only) is by GROUP, not by sentence, so template near-duplicates do not
leak: generated sentences are grouped by destination (every template around one destination together), and all
items are merged with their near-duplicates (bge-small cosine >= ``CLUSTER_COSINE`` or character-trigram Jaccard
>= ``CLUSTER_JACCARD``). The report gives the near-duplicate leak that remains under several measures.
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from guard_grammar import _all_instruction_cases, _sample  # noqa: E402
from kudiguard.knowledge.embeddings import DEFAULT_MODEL, FastEmbedder  # noqa: E402
from kudiguard.knowledge.file_pack import FilePackSource  # noqa: E402
from kudiguard.languages import LanguageRegistry  # noqa: E402
from kudiguard.safety.guard_classifier import (  # noqa: E402
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
CLUSTER_COSINE = (
    0.9  # near-duplicates kept on one side of the split (0.8 chains the whole domain into one cluster)
)
CLUSTER_JACCARD = 0.6  # character trigrams: template variants (0.5 chains a third of the data together)
DEV_SHARE = 0.2
FOLDS = 5
CURVE = (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)  # reported: the recall/false-positive trade-off
OUT = ROOT / "data" / "guard" / "en_classifier.npz"
FROZEN = ROOT / "data" / "guard" / "train_set.jsonl"


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    label: int  # 1 = unsafe advice
    source: str
    weight: float
    group: str = (
        ""  # split unit: grammar destination, else the text (merged with near-duplicates on --freeze)
    )
    split: str = ""  # "train" | "dev" (set on --freeze)


def _examples() -> Iterator[Example]:
    natural = yaml.safe_load((ROOT / "data" / "guard" / "natural_en.yaml").read_text(encoding="utf-8"))
    for text in natural["unsafe"]:
        yield Example(text, 1, "natural", 4.0)
    for text in natural["safe"]:
        yield Example(text, 0, "natural", 4.0)
    for probe in yaml.safe_load(
        (ROOT / "tests" / "fixtures" / "guard_probes.yaml").read_text(encoding="utf-8")
    ):
        if probe["lang"] == "en" and probe["issue"] == "unsafe_instruction" and probe.get("train", True):
            yield Example(probe["text"], int(probe["expect"] == "block"), "probes", 1.0)
    for probe in yaml.safe_load(
        (ROOT / "data" / "eval" / "unsafe_advice_probes.yaml").read_text(encoding="utf-8")
    ):
        if probe["category"] == "instruction":
            yield Example(probe["text"], int(probe["expect"] == "unsafe"), "d_probes", 1.0)
    for case in _sample(_all_instruction_cases(), 2, SEED):
        destination = case.cell.rsplit("/", 1)[-1]
        yield Example(case.text, int(case.expect == "block"), "grammar", 0.5, f"grammar:{destination}")
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
                yield Example(line, 0, "fixed", 1.0, "fixed")


def collect() -> list[Example]:
    """The examples from the live sources, deduplicated (first source wins: natural > probes > D > grammar > rest)."""
    seen: dict[str, Example] = {}
    for example in _examples():
        key = normalize(example.text)
        if key and key not in seen:
            seen[key] = example
    return list(seen.values())


def _hash(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16)


def trigram_jaccard(texts: list[str]) -> np.ndarray:
    """Pairwise Jaccard similarity of character trigrams (surface near-duplicates: templates, small edits)."""
    shingles = [{f" {t} "[i : i + 3] for i in range(len(t))} for t in (" ".join(x.split()) for x in texts)]
    vocabulary = {g: i for i, g in enumerate(sorted(set().union(*shingles)))}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for row, grams in enumerate(shingles):
        matrix[row, [vocabulary[g] for g in grams]] = 1.0
    inter = matrix @ matrix.T
    sizes = matrix.sum(axis=1)
    result: np.ndarray = inter / (sizes[:, None] + sizes[None, :] - inter)
    return result


def assign_splits(data: list[Example], vectors: np.ndarray) -> list[Example]:
    """Group each grammar destination and every pair of near-duplicates (bge cosine >= ``CLUSTER_COSINE`` or
    character-trigram Jaccard >= ``CLUSTER_JACCARD``) together, then fill dev with whole groups (in hash order) up to
    ``DEV_SHARE`` of each source. Fixed strings always train (they are never model replies)."""
    parent = list(range(len(data)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, example in enumerate(data):
        key = example.group or normalize(example.text)
        parent[find(i)] = find(first.setdefault(key, i))
    texts = [normalize(e.text) for e in data]
    near = (vectors @ vectors.T >= CLUSTER_COSINE) | (trigram_jaccard(texts) >= CLUSTER_JACCARD)
    rows, cols = np.where(np.triu(near, 1))
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        parent[find(i)] = find(j)
    members: dict[int, list[int]] = {}
    for i in range(len(data)):
        members.setdefault(find(i), []).append(i)
    groups = sorted(
        ((min(data[i].group or texts[i] for i in indices), indices) for indices in members.values()),
        key=lambda item: _hash(item[0]),
    )
    totals = Counter(e.source for e in data)
    in_dev: Counter[str] = Counter()
    result = list(data)
    for name, indices in groups:
        source = Counter(data[i].source for i in indices).most_common(1)[0][0]
        fits = in_dev[source] + len(indices) <= DEV_SHARE * totals[source]
        split = "dev" if fits and source != "fixed" else "train"
        if split == "dev":
            in_dev[source] += len(indices)
        for i in indices:
            result[i] = Example(data[i].text, data[i].label, data[i].source, data[i].weight, name, split)
    return result


def leak(texts: list[str], vectors: np.ndarray, dev: np.ndarray) -> dict[str, float]:
    """Share of dev items with a training item that is a near-duplicate by each measure."""
    if not dev.any() or dev.all():
        return {}
    cosine = (vectors[dev] @ vectors[~dev].T).max(axis=1)
    jaccard = trigram_jaccard(texts)[np.ix_(dev, ~dev)].max(axis=1)
    train = [t for t, d in zip(texts, dev, strict=True) if not d]
    close = 0
    for text in (t for t, d in zip(texts, dev, strict=True) if d):
        matcher = difflib.SequenceMatcher(None, text)
        for other in train:
            matcher.set_seq1(other)
            if matcher.real_quick_ratio() >= 0.8 and matcher.quick_ratio() >= 0.8 and matcher.ratio() >= 0.8:
                close += 1
                break
    return {
        "difflib_ratio_0.8": round(close / int(dev.sum()), 4),
        "trigram_jaccard_0.6": round(float((jaccard >= 0.6).mean()), 4),
        "bge_cosine_0.9": round(float((cosine >= 0.9).mean()), 4),
        "bge_cosine_0.8": round(float((cosine >= 0.8).mean()), 4),  # same topic, not duplicates (bge-small)
    }


def save_frozen(data: list[Example], path: Path = FROZEN) -> None:
    lines = [json.dumps(asdict(e), ensure_ascii=False, sort_keys=True) for e in data]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def load_frozen(path: Path = FROZEN) -> list[Example]:
    return [Example(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def frozen_sha256(path: Path = FROZEN) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Memo:
    """Embeds each unit once across the fold models (training only; the service never caches reply text)."""

    def __init__(self, embedder: FastEmbedder) -> None:
        self._embedder = embedder
        self._vectors: dict[str, np.ndarray] = {}

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if missing:
            self._vectors.update(zip(missing, self._embedder.embed_documents(missing), strict=True))
        return np.stack([self._vectors[t] for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embedder.embed_query(text)


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
    """The lowest threshold at which the shipped decision (regex OR classifier) keeps the false-positive rate on the
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
    parser.add_argument(
        "--freeze", action="store_true", help="rebuild the frozen training set from the sources"
    )
    parser.add_argument("--report", action="store_true", help="print metrics, do not write the artefact")
    args = parser.parse_args()
    embedder = FastEmbedder(cache_dir=args.cache)
    if args.freeze:
        collected = collect()
        save_frozen(
            assign_splits(collected, embedder.embed_documents([normalize(e.text) for e in collected]))
        )
        print(f"froze {len(collected)} examples into {FROZEN}")
    data = load_frozen()
    texts = [normalize(e.text) for e in data]
    labels = np.array([e.label for e in data], dtype=np.float64)
    weights = np.array([e.weight for e in data])
    dev = np.array([e.split == "dev" for e in data])
    vectors = embedder.embed_documents(texts)
    features = feature_matrix(texts, vectors)
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    guard = OutputGuard(registry)
    memo = _Memo(embedder)
    regex = np.array([OutputIssue.UNSAFE_INSTRUCTION in guard.check(e.text, "en") for e in data])
    # threshold: out-of-fold over the training split (5 folds by group), on the hand-written and probe items
    # (generated grammar and fixed strings are easy negatives that would flatter the false-positive rate); the dev
    # split stays untouched for the report
    fold = np.array([_hash(e.group) % FOLDS for e in data])
    pick = ~dev & np.array([e.source not in ("grammar", "fixed") for e in data])
    out_of_fold = np.zeros(len(data))
    for k in range(FOLDS):
        inside = ~dev & (fold != k)
        w, b = train(features[inside], labels[inside], weights[inside])
        scorer = UnsafeAdviceClassifier(ClassifierModel(w, b, 0.5, "", {}), memo, guard, budget_s=None)
        for i in np.flatnonzero(pick & (fold == k)):
            out_of_fold[i] = scorer.score(data[i].text, "en").probability
    threshold = choose_threshold(out_of_fold[pick], regex[pick], labels[pick])
    w, b = train(features[~dev], labels[~dev], weights[~dev])
    whole_sentence = 1.0 / (1.0 + np.exp(-(features @ w + b)))
    # the full decision as shipped: regex guard OR classifier (gated units), on the dev items
    classifier = UnsafeAdviceClassifier(ClassifierModel(w, b, 0.5, "", {}), memo, guard, budget_s=None)
    unit_best = np.array(
        [classifier.score(e.text, "en").probability if d else 0.0 for e, d in zip(data, dev, strict=True)]
    )
    learned = unit_best >= threshold
    curve = {
        str(t): {
            "out_of_fold": metrics((regex[pick] | (out_of_fold[pick] >= t)).astype(int), labels[pick]),
            "dev": metrics((regex[dev] | (unit_best[dev] >= t)).astype(int), labels[dev]),
        }
        for t in CURVE
    }
    report = {
        "train": {"n": int((~dev).sum()), "positives": int(labels[~dev].sum())},
        "threshold_chosen_on": {
            "how": f"out-of-fold, {FOLDS} group folds of the training split, hand-written and probe items",
            **metrics((regex[pick] | (out_of_fold[pick] >= threshold)).astype(int), labels[pick]),
        },
        "dev_leak_share": leak(texts, vectors, dev),
        # the model on each whole dev text, no gating or unit split (diagnostic; not the shipped decision)
        "dev_model_whole_text_ungated": metrics((whole_sentence[dev] >= threshold).astype(int), labels[dev]),
        # the classifier as shipped (gated units, MAX_UNITS) without the regex
        "dev_classifier_units_alone": metrics(learned[dev].astype(int), labels[dev]),
        "dev_regex_alone": metrics(regex[dev].astype(int), labels[dev]),
        "dev_combined": metrics((regex[dev] | learned[dev]).astype(int), labels[dev]),
        "threshold_curve": curve,
        "dev_by_source": {
            source: metrics(((regex | learned)[dev & mask]).astype(int), labels[dev & mask])
            for source in sorted({e.source for e in data})
            if (mask := np.array([e.source == source for e in data])).any() and (dev & mask).any()
        },
    }
    data_hash = frozen_sha256()
    version = f"{datetime.date.today().isoformat()}-{data_hash[:8]}"
    metadata = {
        "embedding_model": DEFAULT_MODEL,
        "seed": SEED,
        "training_set": FROZEN.relative_to(ROOT).as_posix(),
        "training_set_sha256": data_hash,
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
