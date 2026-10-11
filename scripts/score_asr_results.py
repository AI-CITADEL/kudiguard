"""Re-score stored ASR results (``scripts/eval_asr.py`` JSON lines) without the models: pooled WER/CER per language.

    python scripts/score_asr_results.py data/eval/asr/results/int8.jsonl

Same scoring as ``eval_asr.py`` (``kudiguard.speech.metrics``; a rejected clip counts as every word deleted), so the
published numbers (docs/CLAIMS.md) can be checked from the committed file alone.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from kudiguard.speech.metrics import char_errors, word_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score stored ASR results per language.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    # per language: word edits/words (folded), word edits/words (diacritics kept), char edits/chars, clips, rejected
    totals: dict[str, list[int]] = defaultdict(lambda: [0] * 8)
    for line in args.results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        hypothesis = row.get("hypothesis") or ""  # rejected clips have none: all words deleted
        counts = (
            *word_errors(row["reference"], hypothesis),
            *word_errors(row["reference"], hypothesis, tone_sensitive=True),
            *char_errors(row["reference"], hypothesis),
            1,
            int("rejected" in row),
        )
        totals[row["lang"]] = [a + b for a, b in zip(totals[row["lang"]], counts, strict=True)]
    for language, (we, wn, de, dn, ce, cn, clips, rejected) in sorted(totals.items()):
        print(
            f"{language}: clips={clips} rejected={rejected}  WER {we / wn:.1%}  WER+d {de / dn:.1%}  CER {ce / cn:.1%}"
        )


if __name__ == "__main__":
    main()
