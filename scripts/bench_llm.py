"""Latency benchmark for the N-ATLAS server: the production analyse -> compose sequence over data/eval/prompts.yaml.

    python scripts/bench_llm.py --label np1                    # server started with -np 1
    python scripts/bench_llm.py --label np2 --slots            # server started with -np 2 (analyse/translate=0, compose/warn=1)

Reports p50/p95 per call type and prompt-processing time from llama.cpp's ``timings`` (cache effectiveness).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from statistics import median, quantiles

import httpx
import yaml

from kudiguard.domain import Verdict
from kudiguard.knowledge.models import KnowledgeResult
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import ChatRequest, Purpose
from kudiguard.llm.openai_compat import OpenAICompatLLM
from kudiguard.prompts import analyse_request, compose_request

ROOT = Path(__file__).resolve().parents[1]
SLOTS = {Purpose.ANALYSE: 0, Purpose.TRANSLATE: 0, Purpose.COMPOSE: 1, Purpose.WARN: 1}


def p95(values: list[float]) -> float:
    return quantiles(values, n=20)[-1] if len(values) >= 2 else values[0]


async def timed(client: httpx.AsyncClient, llm: OpenAICompatLLM, request: ChatRequest) -> tuple[float, float]:
    """Wall time and llama.cpp prompt-processing ms for one call (raw call to read ``timings``)."""
    body = llm.build_body(request)
    started = time.perf_counter()
    response = await client.post(f"{llm.base_url}/v1/chat/completions", json=body, timeout=120)
    wall = (time.perf_counter() - started) * 1000
    timings = response.json().get("timings", {})
    return wall, float(timings.get("prompt_ms", 0.0))


async def main(url: str, label: str, use_slots: bool, runs: int) -> None:
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    cases = yaml.safe_load((ROOT / "data" / "eval" / "prompts.yaml").read_text(encoding="utf-8"))
    results: dict[str, list[tuple[float, float]]] = {"analyse": [], "compose": []}
    async with httpx.AsyncClient() as client:
        llm = OpenAICompatLLM(url, client=client, slots=SLOTS if use_slots else None)
        for _ in range(runs):
            for case in cases:
                name = registry.get(case["lang"]).name
                results["analyse"].append(await timed(client, llm, analyse_request(case["text"], name)))
                compose = compose_request(
                    language_name=name,
                    english=case["text"],
                    verdict=Verdict.NOT_A_CHECK,
                    flags=(),
                    knowledge=KnowledgeResult(),
                )
                results["compose"].append(await timed(client, llm, compose))
    summary = {}
    for kind, rows in results.items():
        walls, prompts = [w for w, _ in rows], [p for _, p in rows]
        summary[kind] = {
            "wall_p50_ms": round(median(walls)),
            "wall_p95_ms": round(p95(walls)),
            "prompt_p50_ms": round(median(prompts)),
        }
    print(
        json.dumps(
            {"label": label, "slots": use_slots, "calls": len(results["analyse"]) * 2, **summary}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--label", default="run")
    parser.add_argument("--slots", action="store_true")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.label, args.slots, args.runs))
