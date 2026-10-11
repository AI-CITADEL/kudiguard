"""Answer quality on the graded dev set (data/eval/answers.yaml) against a running N-ATLAS server.

    python scripts/eval_answers.py --runs 3 --temperature 0.1 --no-seed --label t01   # the old sampling
    python scripts/eval_answers.py --runs 3 --label t0seed                             # production sampling
    python scripts/eval_answers.py --label t0seed --only ha,en-otp-caller
    python scripts/eval_answers.py --runs 3 --label direct --compose-strategy direct   # in-language compose
    python scripts/eval_answers.py --compare data/eval/results/answers-t01.jsonl data/eval/results/answers-t0seed.jsonl

Each case runs through the production Engine (hybrid knowledge search with bge-small, the pipeline's turn deadline);
the reply is back-translated to English and scored (kudiguard.evaluation): verdict in the allowed set, coverage of the
`must` points, `must_not`/pattern violations, claims unsupported by the facts the model saw. The sampling options
apply to the engine only; back-translation always uses the plain backend so translator noise stays out of an A/B.
Results: data/eval/results/answers-<label>.jsonl (a meta line, then one record per case and run, appended as they
are produced). Once workstream B merges, build the engine with grounding on, as production does.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from kudiguard.engine import Engine
from kudiguard.evaluation.answers import AnswerScorer, KeyPoint, load_cases, pack_entries
from kudiguard.evaluation.grounding import make_grounding, support_threshold, text_specifics
from kudiguard.evaluation.runner import run_case
from kudiguard.evaluation.summary import (
    Record,
    ResultsWriter,
    format_comparison,
    format_summary,
    read_meta,
    read_records,
)
from kudiguard.knowledge.embeddings import FastEmbedder
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.openai_compat import OpenAICompatLLM
from kudiguard.llm.sampling import add_sampling_arguments, describe, with_sampling
from kudiguard.reply_translation import ComposePolicy

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "eval" / "answers.yaml"
PACKS = ROOT / "data" / "packs"
RESULTS = ROOT / "data" / "eval" / "results"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        )
    except OSError:
        return "unknown"
    return out.stdout.strip() or "unknown"


async def main(args: argparse.Namespace) -> None:
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    every = load_cases(CASES)
    if unknown := sorted(only - {c.lang for c in every} - {c.id for c in every}):
        print(f"warning: --only: unknown languages/case ids {unknown}")
    cases = [c for c in every if not only or c.lang in only or c.id in only]
    if not cases:
        sys.exit(f"--only {args.only!r} matches no case (use languages and/or case ids)")
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    entries = pack_entries(PACKS)
    embedder = FastEmbedder(cache_dir=args.model_cache, local_files_only=True)
    names, numbers = text_specifics()
    scorer = AnswerScorer.for_packs(embedder, registry, entries, names=names, numbers=numbers)
    out = RESULTS / f"answers-{args.label}.jsonl"
    if out.exists():
        print(f"warning: overwriting {out}")
    records: list[Record] = []
    async with httpx.AsyncClient(timeout=120) as client:
        base = OpenAICompatLLM(args.url, client=client, cache_prompt=not args.no_cache_prompt)
        if not await base.ready():
            sys.exit(f"N-ATLAS at {args.url} is not ready: start it (scripts/run_llm.ps1) and retry")
        knowledge = knowledge_base_from_packs(PACKS, client=client, embedder=embedder)
        policy = ComposePolicy.parse(args.compose_strategy)
        engine = Engine(
            registry=registry, llm=with_sampling(base, args), knowledge=knowledge, compose_policy=policy
        )
        grounding = make_grounding(base, embedder, registry)  # unwrapped: the A/B varies the engine only
        meta = {
            "sampling": describe(args),
            "compose_strategy": args.compose_strategy,
            "grounding": type(grounding).__name__,
            "cover_threshold": scorer.cover_threshold,
            "forbid_threshold": scorer.forbid_threshold,
            "support_threshold": support_threshold(),
            "commit": git_commit(),
            "url": args.url,
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "runs": args.runs,
            "cases": len(cases),
        }
        print(f"{meta['sampling']}; grounding {meta['grounding']}; {len(cases)} cases x {args.runs} run(s)")
        writer = ResultsWriter(out, meta)
        try:
            for run in range(args.runs):
                for case in cases:
                    try:
                        result = await run_case(engine, knowledge, grounding, scorer, case)
                    except Exception as exc:  # one broken case must not lose the run
                        record = Record.failed(case, run, repr(exc))
                        print(f"  [{run}] {case.id}: ERROR {exc!r}")
                    else:
                        record = Record.from_result(case, result, run)
                        print(f"  [{run}] {case.id}: {status(case.must, record)}")
                    writer.add(record)
                    records.append(record)
        finally:
            writer.close()
    print(f"\nAnswer quality ({args.label}: {meta['sampling']}) on {CASES.relative_to(ROOT)}")
    print(format_summary(records, meta))
    print(f"per-item results: {out}")


def status(must: Sequence[KeyPoint], record: Record) -> str:
    coverage = "-" if record.coverage is None else f"{record.coverage:.2f}"
    problems = [p.point for p, ok in zip(must, record.covered, strict=False) if not ok]
    problems += record.violations + [f"unsupported: {u}" for u in record.unsupported]
    problems += [f"copied example: {p}" for p in record.example_copied]
    if not record.verdict_ok:
        problems.insert(0, f"verdict {record.verdict} (allowed {record.expected})")
    return f"coverage {coverage}  " + ("; ".join(problems) or "ok")


def compare(a: Path, b: Path) -> None:
    metas = (read_meta(a), read_meta(b))
    print(format_comparison(read_records(a), read_records(b), (a.stem, b.stem), metas))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "models" / "fastembed")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", default="", help="comma-separated languages and/or case ids")
    parser.add_argument(
        "--compose-strategy",
        default="english_first",
        help='kudiguard.reply_translation: "english_first", "direct" or e.g. "english_first,ig=direct"',
    )
    parser.add_argument("--label", default="latest", help="results file suffix, e.g. t01 / t0seed")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"), help="diff two results files")
    add_sampling_arguments(parser)
    parsed = parser.parse_args()
    if parsed.compare:
        compare(*parsed.compare)
    else:
        asyncio.run(main(parsed))
