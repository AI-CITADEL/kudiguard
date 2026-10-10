"""Scam detection precision/recall on the frozen blind scenarios (data/eval/scenarios.yaml).

    python scripts/eval_scenarios.py               # full engine against a running N-ATLAS server
    python scripts/eval_scenarios.py --rules-only  # deterministic rules alone (no model): the safety floor

A scenario counts as "flagged" when the verdict is LIKELY_SCAM or CAUTION. Reports precision, recall and the false
positive rate per language and overall, plus every miss, and writes per-item results to data/eval/results/.
The scenario file is frozen: never tune lexicons or prompts on it (that would make these numbers meaningless).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from kudiguard.domain import Verdict
from kudiguard.engine import AnswerKind, Engine
from kudiguard.evaluation.stats import percentile
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import ChatRequest, Completion, LLMError
from kudiguard.llm.openai_compat import OpenAICompatLLM
from kudiguard.render import REPORT_VERDICTS

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "data" / "eval" / "scenarios.yaml"


class NoModel:
    """Rules-only baseline: every model call fails, so the engine degrades to the deterministic policy."""

    async def complete(self, request: ChatRequest) -> Completion:
        raise LLMError("rules-only evaluation")

    async def ready(self) -> bool:
        return False


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, scam: bool, flagged: bool) -> None:
        if scam:
            self.tp += flagged
            self.fn += not flagged
        else:
            self.fp += flagged
            self.tn += not flagged

    def line(self) -> str:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        fpr = self.fp / max(self.fp + self.tn, 1)
        n = self.tp + self.fp + self.tn + self.fn
        return f"n={n:3d}  precision {precision:.0%}  recall {recall:.0%}  false-positive rate {fpr:.0%}"


async def main(url: str, rules_only: bool) -> None:
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    items = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
    by_language: dict[str, Confusion] = defaultdict(Confusion)
    overall = Confusion()
    misses: list[str] = []
    latencies: list[float] = []
    mode = "rules-only" if rules_only else "engine"
    out = ROOT / "data" / "eval" / "results" / f"scenarios-{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        llm = NoModel() if rules_only else OpenAICompatLLM(url, client=client)
        engine = Engine(
            registry=registry,
            llm=llm,
            knowledge=knowledge_base_from_packs(ROOT / "data" / "packs", client=client),
        )
        with out.open("w", encoding="utf-8") as log:
            for item in items:
                started = time.perf_counter()
                answer = await engine.check(item["text"], item["lang"])
                latencies.append(time.perf_counter() - started)
                verdict = answer.verdict if answer.kind is AnswerKind.CHECK else Verdict.NOT_A_CHECK
                flagged = verdict in REPORT_VERDICTS
                scam = item["label"] == "scam"
                by_language[item["lang"]].add(scam, flagged)
                overall.add(scam, flagged)
                if scam != flagged:
                    kind = "MISSED SCAM" if scam else "FALSE ALARM"
                    misses.append(
                        f"  {kind} [{item['lang']}] {item['id']}: {item.get('note', '')} -> {verdict}"
                    )
                record = {"id": item["id"], "lang": item["lang"], "label": item["label"], "verdict": verdict,
                          "flags": list(answer.flags), "english": answer.english}  # fmt: skip
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nScam detection ({mode}) on {SCENARIOS.relative_to(ROOT)}")
    for language, confusion in sorted(by_language.items()):
        print(f"  {language}: {confusion.line()}")
    print(f"  all: {overall.line()}")
    if not rules_only:
        print(f"  latency p50 {percentile(latencies, 0.5):.1f}s  p95 {percentile(latencies, 0.95):.1f}s")
    print("\n".join(misses) or "  no misses")
    print(f"per-item results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.rules_only))
