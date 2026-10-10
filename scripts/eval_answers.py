"""Answer quality on the graded dev set (data/eval/answers.yaml) against a running N-ATLAS server.

    python scripts/eval_answers.py --label t01 --runs 3            # writes data/eval/results/answers-t01.jsonl
    python scripts/eval_answers.py --label t01 --only ha,en-otp-caller
    python scripts/eval_answers.py --compare data/eval/results/answers-t01.jsonl data/eval/results/answers-t0.jsonl

Each case runs through the production Engine (hybrid knowledge search with bge-small), the reply is back-translated
to English and scored (kudiguard.evaluation.answers): verdict in the allowed set, coverage of the `must` points,
`must_not` / pattern violations, and claims unsupported by the facts. Prints per language and overall: verdict
accuracy, mean coverage, % answers with any violation, % fully grounded, fallback rate, latency p50/p95, and the share
of cases whose reply is identical across runs (use --runs > 1 to compare temperature/seed settings).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx

from kudiguard.engine import Engine
from kudiguard.evaluation.answers import AnswerScorer, load_cases, make_grounding, pack_entries, run_case
from kudiguard.evaluation.summary import (
    Record,
    format_comparison,
    format_summary,
    read_records,
    write_records,
)
from kudiguard.knowledge.embeddings import FastEmbedder
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.openai_compat import OpenAICompatLLM

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "eval" / "answers.yaml"
PACKS = ROOT / "data" / "packs"
RESULTS = ROOT / "data" / "eval" / "results"


async def main(url: str, model_cache: Path, runs: int, only: set[str], label: str) -> None:
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    cases = [c for c in load_cases(CASES) if not only or c.lang in only or c.id in only]
    entries = pack_entries(PACKS)
    embedder = FastEmbedder(cache_dir=model_cache, local_files_only=True)
    scorer = AnswerScorer.for_packs(embedder, registry, entries)
    out = RESULTS / f"answers-{label}.jsonl"
    records: list[Record] = []
    async with httpx.AsyncClient(timeout=120) as client:
        llm = OpenAICompatLLM(url, client=client)
        knowledge = knowledge_base_from_packs(PACKS, client=client, embedder=embedder)
        engine = Engine(registry=registry, llm=llm, knowledge=knowledge)
        grounding = make_grounding(llm, embedder, registry)
        print(f"grounding: {type(grounding).__name__}; {len(cases)} cases x {runs} run(s)")
        for run in range(runs):
            for case in cases:
                result = await run_case(engine, grounding, scorer, case, entries)
                record = Record.from_result(case, result, run)
                records.append(record)
                misses = [p.point for p, ok in zip(case.must, record.covered, strict=True) if not ok]
                problems = misses + record.violations + [f"unsupported: {u}" for u in record.unsupported]
                if not record.verdict_ok:
                    problems.insert(0, f"verdict {record.verdict} (allowed {record.expected})")
                status = "ok" if not problems else "; ".join(problems)
                print(f"  [{run}] {case.id}: coverage {record.coverage:.2f}  {status}")
    write_records(out, records)
    print(f"\nAnswer quality ({label}) on {CASES.relative_to(ROOT)}")
    print(format_summary(records))
    print(f"per-item results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "models" / "fastembed")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", default="", help="comma-separated languages and/or case ids")
    parser.add_argument("--label", default="latest", help="results file suffix, e.g. t01 / t0-seed")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"), help="diff two results files")
    args = parser.parse_args()
    if args.compare:
        a, b = args.compare
        print(format_comparison(read_records(a), read_records(b), (a.stem, b.stem)))
    else:
        only = {item.strip() for item in args.only.split(",") if item.strip()}
        asyncio.run(main(args.url, args.model_cache, args.runs, only, args.label))
