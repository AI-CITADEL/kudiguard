"""Compare embedding models for the grounding check on data/eval/grounding_claims.yaml (CPU only, no LLM).

    python scripts/eval_grounding_embedders.py                    # bge-small (the live model)
    python scripts/eval_grounding_embedders.py --model multilingual-minilm --model labse

Claims are compared directly with the English evidence (no back-translation), so for ha/yo/ig this measures whether a
cross-lingual model could check replies live. Prints, per language, the best achievable accuracy and its threshold,
the score range of supported vs invented claims, and embedding latency. Results: docs/SPIKES.md (S8).

LaBSE is not in fastembed: download ``Xenova/LaBSE`` (onnx/model_int8.onnx + tokenizer.json, ~470 MB) and
``sentence-transformers/LaBSE`` 2_Dense/model.safetensors into ``<cache>/Xenova-LaBSE`` (``st/2_Dense``) first.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import yaml

from kudiguard.knowledge.embeddings import FastEmbedder, Matrix, l2_normalize
from kudiguard.knowledge.file_pack import FilePackSource
from kudiguard.knowledge.models import KnowledgeResult, SourcedEntry
from kudiguard.safety.grounding import GENERIC_FRAMING, evidence_for, evidence_units

ROOT = Path(__file__).resolve().parents[1]
CASES = yaml.safe_load((ROOT / "data" / "eval" / "grounding_claims.yaml").read_text(encoding="utf-8"))


class LaBSE:
    """LaBSE int8 ONNX: [CLS] -> dense + tanh (the sentence-transformers head) -> L2 normalise."""

    def __init__(self, directory: Path) -> None:
        import onnxruntime as ort
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.enable_padding(pad_id=0)
        self._tokenizer.enable_truncation(128)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self._session = ort.InferenceSession(str(directory / "onnx" / "model_int8.onnx"), options)
        head = load_file(str(directory / "st" / "2_Dense" / "model.safetensors"))
        self._weight, self._bias = head["linear.weight"], head["linear.bias"]

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        encoded = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        cls = self._session.run(None, feeds)[0][:, 0]
        return l2_normalize(np.tanh(cls @ self._weight.T + self._bias).astype(np.float32))

    def embed_query(self, text: str) -> Matrix:
        return self.embed_documents([text])[0]


MODELS: dict[str, Callable[[Path], object]] = {
    "bge-small": lambda cache: FastEmbedder(cache_dir=cache),
    "multilingual-minilm": lambda cache: FastEmbedder(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", cache_dir=cache
    ),
    "labse": lambda cache: LaBSE(cache / "Xenova-LaBSE"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", action="append", choices=sorted(MODELS), help="repeatable; default bge-small"
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "models" / "fastembed")
    args = parser.parse_args()
    core = {e.id: e for e in FilePackSource.load(ROOT / "data" / "packs" / "kudiguard-core").entries}
    for name in args.model or ["bge-small"]:
        embedder = MODELS[name](args.cache)
        print(f"== {name}")
        by_language: dict[str, list[tuple[float, bool]]] = {}
        for case in CASES:
            entries = tuple(SourcedEntry(core[i], "kudiguard-core", "KudiGuard") for i in case["entries"])
            evidence = evidence_units([*evidence_for(KnowledgeResult(entries=entries)), *GENERIC_FRAMING])
            claim = embedder.embed_documents([case["claim"]])  # type: ignore[attr-defined]
            score = float((embedder.embed_documents(evidence) @ claim.T).max())  # type: ignore[attr-defined]
            by_language.setdefault(case["lang"], []).append((score, case["supported"]))
        for language, rows in by_language.items():
            good = [s for s, ok in rows if ok]
            bad = [s for s, ok in rows if not ok]
            accuracy, threshold = max(
                (sum((s >= t) == ok for s, ok in rows) / len(rows), t) for t in np.arange(0.2, 0.99, 0.01)
            )
            print(
                f"  {language}: n={len(rows)} best accuracy {accuracy:.0%} at {threshold:.2f} | "
                f"supported {min(good):.2f}-{max(good):.2f} | invented {min(bad):.2f}-{max(bad):.2f}"
            )
        sample = [c["claim"] for c in CASES if c["lang"] == "ha"][:4]
        start = time.perf_counter()
        for _ in range(10):
            embedder.embed_documents(sample)  # type: ignore[attr-defined]
        print(f"  latency: {(time.perf_counter() - start) * 100:.1f} ms per 4 claims")


if __name__ == "__main__":
    main()
