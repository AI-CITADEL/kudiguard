"""Write blind judge batches from an end-to-end results file (scripts/eval_e2e.py).

    python scripts/judge_e2e_prep.py data/eval/results/e2e-live.jsonl --batch-size 40

Each item has only what an answer judge needs: id, lang, the user's message (text), the reply the user was shown,
and the case's must_advise / must_not. No label, accept, verdict field, flags, notes or scores are included (the
reply itself shows its headline, as the user saw it). Error rows are skipped and listed. Batches are written as
JSONL and CSV to data/eval/results/judge/e2e-<stem>-NN.{jsonl,csv}; the item order is shuffled with a fixed seed so
one batch does not hold one language or one label.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from kudiguard.evaluation.e2e import JUDGE_FIELDS, judge_item

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "eval" / "results" / "judge"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = Path(args.results)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {row["id"]: row for row in rows}  # a resumed run may repeat an id: the last row wins
    errors = sorted(i for i, row in latest.items() if row.get("error"))
    items = [judge_item(row) for row in latest.values() if not row.get("error")]
    random.Random(args.seed).shuffle(items)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for number, start in enumerate(range(0, len(items), args.batch_size), 1):
        batch = items[start : start + args.batch_size]
        stem = out_dir / f"e2e-{source.stem}-{number:02d}"
        with stem.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in batch)
        with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JUDGE_FIELDS)
            writer.writeheader()
            for item in batch:
                writer.writerow({**item, "must_advise": " | ".join(item["must_advise"]),
                                 "must_not": " | ".join(item["must_not"])})  # fmt: skip
        print(f"{stem.name}: {len(batch)} items")
    print(f"{len(items)} items judged blind; {len(errors)} error rows skipped: {', '.join(errors) or 'none'}")


if __name__ == "__main__":
    main()
