"""Live prompt-quality evaluation against a running N-ATLAS server (llama.cpp, OpenAI-compatible).

    python scripts/eval_prompts.py [--url http://127.0.0.1:8081/v1/chat/completions] [--runs 3]

Runs analyse -> (translate fallback) -> rules -> knowledge -> compose for every case in data/eval/prompts.yaml,
mirroring the production policy, and checks objective properties: usable translation, intent, verdict, output
guard (safety claims, refusals), truncation, sign-offs, wrong reply language, worked-example leakage and task echo.
Decoding is greedy (temperature 0, seeded retry), so runs are near-identical; llama.cpp's prompt cache can still flip
near-tied tokens, so use --runs > 1 to check stability. Compare settings with --temperature/--no-seed (FixedSampling)
and --no-cache-prompt for bit-exact A/B runs on a cold server.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import re
import time
from collections import Counter
from pathlib import Path
from statistics import median

import httpx
import yaml

from kudiguard.domain import Verdict
from kudiguard.engine import Engine
from kudiguard.knowledge.models import Query
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import PIVOT, LanguageRegistry
from kudiguard.llm.base import LLM, ChatRequest, LLMError
from kudiguard.llm.openai_compat import OpenAICompatLLM
from kudiguard.llm.sampling import add_sampling_arguments, describe, with_sampling
from kudiguard.prompts import (
    Intent,
    clean_reply,
    compose_request,
)
from kudiguard.safety.decision import decide, is_content_free
from kudiguard.safety.output_guard import OutputGuard

ROOT = Path(__file__).resolve().parents[1]
ENGLISH_STOPWORDS = {
    "the",
    "and",
    "you",
    "your",
    "to",
    "is",
    "of",
    "it",
    "do",
    "not",
    "if",
    "for",
    "with",
    "this",
}
SIGN_OFF = re.compile(r"\b(best regards|regards|dear user|sincerely|zauna lafiya)\b", re.IGNORECASE)
# Phrases unique to the worked examples: seeing one that is in neither the example used nor the facts = a leak.
EXAMPLE_PHRASES = ("code sent to your phone", "real offers do not ask", "*346#")
TASK_ECHO = re.compile(r"\b(task:|answer the user's|money question)", re.IGNORECASE)


@dataclasses.dataclass
class Context:
    kb: KnowledgeBase
    registry: LanguageRegistry
    guard: OutputGuard
    llm: LLM
    engine: Engine
    timings: dict[str, list[float]]


def english_ratio(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    return sum(w in ENGLISH_STOPWORDS for w in words) / max(len(words), 1)


async def chat(ctx: Context, request: ChatRequest, timer: str | None = None) -> tuple[str, str | None]:
    started = time.perf_counter()
    completion = await ctx.llm.complete(request)  # through the LLM port, so --temperature/--no-seed apply
    if timer:
        ctx.timings[timer].append(time.perf_counter() - started)
    return completion.text, completion.finish_reason


async def run_case(ctx: Context, case: dict[str, str], *, verbose: bool) -> list[str]:
    lang, text = case["lang"], case["text"]
    name = ctx.registry.get(lang).name
    tag = f"[{lang}] {text[:48]}"
    problems: list[str] = []

    started = time.perf_counter()
    analysis, _, model_down = await ctx.engine.analyse(text, lang)  # the production analyse/translate policy
    ctx.timings["analyse"].append(time.perf_counter() - started)
    if model_down:
        return [f"{tag}: analyse failed (model unavailable)"]
    if lang != PIVOT and not analysis.english:
        problems.append(f"{tag}: no usable translation (rules on the original only)")
    if analysis.intent != case["expect"]:
        problems.append(f"{tag}: intent {analysis.intent} (expected {case['expect']})")

    result = decide(text, analysis, ctx.registry.lexicon_for(lang))  # the production verdict policy
    if allowed_verdicts := case.get("verdicts"):
        if result.verdict not in allowed_verdicts:
            problems.append(f"{tag}: verdict {result.verdict} (allowed {allowed_verdicts})")
    else:
        is_scam_verdict = result.verdict in (Verdict.LIKELY_SCAM, Verdict.CAUTION)
        if case["expect"] == "scam_check" and not is_scam_verdict:
            problems.append(f"{tag}: verdict {result.verdict} for a scam")
        if case["expect"] != "scam_check" and is_scam_verdict:
            problems.append(f"{tag}: verdict {result.verdict} for a plain question")

    # Production policy: greetings and content-free messages get fixed text; the model never composes them.
    if is_content_free(text) or (analysis.intent is Intent.GREETING and not result.flags):
        if verbose:
            print(f"\n{tag}\n  fixed reply (no compose): intent={analysis.intent} verdict={result.verdict}")
        return problems

    knowledge = await ctx.kb.search(Query(text=analysis.english, flags=result.flags))
    request = compose_request(
        language_name=name,
        english=analysis.english,
        verdict=result.verdict,
        flags=result.flags,
        knowledge=knowledge,
    )
    try:
        content, finish = await chat(ctx, request, "compose")
    except LLMError as exc:  # production budget (20 s): a slow compose is a finding, not a crash
        return [*problems, f"{tag}: compose failed ({exc})"]
    truncated = finish == "length"
    reply = clean_reply(content, truncated=truncated)

    # The example chosen for this request (its question and reply) and the real facts may legitimately be echoed.
    example = request.messages[1]["content"] + " " + request.messages[2]["content"]
    allowed = (example + " ".join(f for e in knowledge.entries for f in e.entry.facts)).lower()
    leaked = [p for p in EXAMPLE_PHRASES if p in reply.lower() and p not in allowed]
    checks = {
        "truncated": truncated,
        "sign-off": bool(SIGN_OFF.search(reply)),
        "wrong language": lang != "en" and english_ratio(reply) > 0.08,
        "example leak": bool(leaked),
        "task echo": bool(TASK_ECHO.search(reply)),
        **{f"guard {issue.value}": True for issue in ctx.guard.check(reply, lang)},
    }
    problems.extend(f"{tag}: {check}" for check, failed in checks.items() if failed)
    if verbose:
        print(f"\n{tag}\n  en={analysis.english!r} intent={analysis.intent} verdict={result.verdict}")
        print(f"  kb={[e.entry.id for e in knowledge.entries][:3]}")
        print("  " + reply.replace("\n", "\n  "))
    return problems


async def main(args: argparse.Namespace) -> None:
    url, runs = args.url, args.runs
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    cases = yaml.safe_load((ROOT / "data" / "eval" / "prompts.yaml").read_text(encoding="utf-8"))
    problems: list[str] = []
    async with httpx.AsyncClient(timeout=120) as client:
        kb = knowledge_base_from_packs(ROOT / "data" / "packs", client=client)
        base = OpenAICompatLLM(  # no transport retry: it would blur the timings
            url.split("/v1/")[0], client=client, retries=0, cache_prompt=not args.no_cache_prompt
        )
        llm = with_sampling(base, args)
        ctx = Context(
            kb=kb,
            registry=registry,
            guard=OutputGuard(registry),
            llm=llm,
            engine=Engine(registry=registry, llm=llm, knowledge=kb),
            timings={"analyse": [], "compose": []},
        )
        for run in range(runs):
            for case in cases:
                problems += await run_case(ctx, case, verbose=run == 0)

    print(f"\n{runs} run(s) x {len(cases)} cases = {runs * len(cases)} conversations ({describe(args)})")
    a, c = ctx.timings["analyse"], ctx.timings["compose"]
    if a and c:
        print(f"analyse p50 {median(a):.1f}s  compose p50 {median(c):.1f}s  max {max(c):.1f}s")
    kinds = Counter(p.rsplit(": ", 1)[1].split(" (")[0] for p in problems)
    print(f"{len(problems)} problems: " + (", ".join(f"{k}={v}" for k, v in kinds.most_common()) or "none"))
    for problem in problems:
        print("  -", problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--runs", type=int, default=1)
    add_sampling_arguments(parser)
    args = parser.parse_args()
    asyncio.run(main(args))
