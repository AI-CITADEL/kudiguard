"""Run the frozen end-to-end test set T (data/eval/e2e/cases.yaml, v2.1) through the real Engine.

    python scripts/eval_e2e.py --tag live                    # full engine against a running N-ATLAS server
    python scripts/eval_e2e.py --rules-only --tag rules      # deterministic rules alone (no model): the safety floor
    python scripts/eval_e2e.py --tag live --resume           # continue an interrupted run (retries error rows)
    python scripts/eval_e2e.py --summary data/eval/results/e2e-live.jsonl   # score a results file, no model calls

Cases run one at a time with the default sampling (temperature 0, seed 42). A case that times out or raises becomes
an error row instead of stopping the run. Voice cases (``voice: true``) run as text: there is no TTS step.
Scoring follows the cases file header (see ``kudiguard.evaluation.e2e``); must_advise / must_not are copied to each
row for an independent judge (scripts/judge_e2e_prep.py) and are not scored here.
The cases file is frozen: never tune rules, prompts, lexicons, retrieval or thresholds on it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from kudiguard.engine import AnswerKind, Engine
from kudiguard.evaluation.e2e import (
    DIAGNOSED_LANGUAGES,
    CountingLLM,
    engine_language,
    format_summary,
    load_cases,
    record,
    resume_ids,
    summarize,
    too_many_errors,
)
from kudiguard.evaluation.runner import TURN_DEADLINE_S, turn_deadline
from kudiguard.i18n import Msg
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import PIVOT, LanguageRegistry
from kudiguard.llm.base import ChatRequest, Completion, LLMError
from kudiguard.llm.openai_compat import OpenAICompatLLM
from kudiguard.render import render_check_reply
from kudiguard.safety.scam_rules import assess

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "eval" / "e2e" / "cases.yaml"


class NoModel:
    """Rules-only baseline: every model call fails, so the engine degrades to the deterministic policy."""

    async def complete(self, request: ChatRequest) -> Completion:
        raise LLMError("rules-only evaluation")

    async def ready(self) -> bool:
        return False


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def shown_reply(engine: Engine, answer: Any, language: str) -> str:
    """The text the user would see (headline + body + report contacts + disclaimer, as the pipeline renders it)."""
    if answer.kind is AnswerKind.CHECK:
        return render_check_reply(
            engine.messages, language, verdict=answer.verdict, body=answer.body, contacts=answer.contacts
        )
    return engine.messages.text(language, Msg.GREETING_REPLY)


async def run(args: argparse.Namespace) -> Path:
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    meta, cases = load_cases(Path(args.cases))
    if args.langs:
        wanted = set(args.langs.split(","))
        cases = [case for case in cases if case["lang"] in wanted]
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out) if args.out else ROOT / "data" / "eval" / "results" / f"e2e-{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    mode, url = ("rules-only", "") if args.rules_only else ("engine", args.url)
    try:  # error rows are retried; a file from another mode or server is refused
        done = resume_ids(read_rows(out), mode=mode, url=url) if args.resume else set()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if not args.resume and out.exists():
        out.unlink()
    english_lexicon = registry.lexicon_for(PIVOT)
    print(f"T v{meta.get('version', '?')}: {len(cases)} cases ({len(done)} already done), {mode} {url} -> {out}"
          f" (turn deadline {args.turn_deadline:.0f}s, hang guard {args.timeout:.0f}s)")  # fmt: skip
    async with httpx.AsyncClient() as client:
        llm = CountingLLM(NoModel() if args.rules_only else OpenAICompatLLM(args.url, client=client))
        engine = Engine(
            registry=registry,
            llm=llm,
            knowledge=knowledge_base_from_packs(ROOT / "data" / "packs", client=client),
        )
        with out.open("a", encoding="utf-8") as log:
            for index, case in enumerate(cases, 1):
                if case["id"] in done:
                    continue
                llm.calls = 0
                language = engine_language(case["lang"])  # Pidgin runs as "en", as in the product
                started = time.perf_counter()
                # The product's turn deadline (pipeline: turn limit minus REPLY_RESERVE_S); wait_for only guards
                # against a hang, well above it.
                deadline = turn_deadline(time.monotonic(), args.turn_deadline)
                try:
                    answer = await asyncio.wait_for(
                        engine.check(case["text"], language, deadline=deadline), args.timeout
                    )
                except Exception as exc:
                    error = "timeout" if isinstance(exc, TimeoutError) else f"{type(exc).__name__}: {exc}"
                    row = record(case, None, latency_s=time.perf_counter() - started, error=error,
                                 model_calls=llm.calls, mode=mode, url=url)  # fmt: skip
                else:
                    latency = time.perf_counter() - started
                    rule_flags = (
                        [f.value for f in assess([answer.english], english_lexicon).flags]
                        if case["lang"] in DIAGNOSED_LANGUAGES and answer.english
                        else []
                    )
                    row = record(case, answer, reply=shown_reply(engine, answer, language),
                                 latency_s=latency, model_calls=llm.calls,
                                 english_rule_flags=rule_flags, mode=mode, url=url)  # fmt: skip
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                status = "ERROR " + row["error"] if row["error"] else ("pass" if row["passed"] else "FAIL")
                print(f"  [{index}/{len(cases)}] {case['id']}: {row['verdict'] or '-'} {status} "
                      f"{row['latency_s']:.1f}s calls={row['model_calls']}")  # fmt: skip
    return out


def report(path: Path, mode: str) -> None:
    summary = summarize(read_rows(path))
    print(format_summary(summary, mode=mode))
    if too_many_errors(summary):
        raise SystemExit(1)


def main() -> None:
    # Windows consoles default to cp1252, which cannot print the summary's "≥".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases (after --langs)")
    parser.add_argument("--langs", default="", help="comma-separated, e.g. yo,ig")
    parser.add_argument("--resume", action="store_true", help="append to --out and skip ids already in it")
    parser.add_argument("--tag", default="", help="results file name: data/eval/results/e2e-<tag>.jsonl")
    parser.add_argument("--out", default="", help="results file (overrides --tag)")
    parser.add_argument(
        "--turn-deadline",
        type=float,
        default=TURN_DEADLINE_S,
        help="product turn limit in seconds (default 60)",
    )
    parser.add_argument("--timeout", type=float, default=240.0, help="per-case hang guard in seconds")
    parser.add_argument("--summary", default="", help="print the scored summary of a results file and exit")
    args = parser.parse_args()
    if args.summary:
        report(Path(args.summary), Path(args.summary).stem)
        return
    if args.timeout < args.turn_deadline + 30:
        parser.error("--timeout is a hang guard: keep it at least 30 s above --turn-deadline")
    if not args.tag and not args.out:
        args.tag = "rules" if args.rules_only else "engine"
    out = asyncio.run(run(args))
    print()
    print(f"per-case results: {out}")
    report(out, "rules-only" if args.rules_only else "engine")


if __name__ == "__main__":
    main()
