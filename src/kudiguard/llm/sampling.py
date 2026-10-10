"""LLM decorator that pins sampling settings, so evaluations can compare them on the same engine.

    llm = FixedSampling(OpenAICompatLLM(url), temperature=0.1, seed=None)  # the pre-2026-10-10 behaviour

Only first attempts (requests at ``DEFAULT_TEMPERATURE``) change temperature; a deliberately warmer retry keeps its
own (this relies on ``engine.RETRY_TEMPERATURE != DEFAULT_TEMPERATURE``, which a test pins). The seed applies to
every request.
"""

from __future__ import annotations

import argparse
import dataclasses

from kudiguard.llm.base import DEFAULT_SEED, DEFAULT_TEMPERATURE, LLM, ChatRequest, Completion


class FixedSampling:
    def __init__(self, inner: LLM, *, temperature: float, seed: int | None) -> None:
        if not temperature >= 0:  # also rejects NaN
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        self._inner = inner
        self._temperature = temperature
        self._seed = seed

    async def complete(self, request: ChatRequest) -> Completion:
        first_attempt = request.temperature == DEFAULT_TEMPERATURE
        temperature = self._temperature if first_attempt else request.temperature
        return await self._inner.complete(
            dataclasses.replace(request, temperature=temperature, seed=self._seed)
        )

    async def ready(self) -> bool:
        return await self._inner.ready()


# --- shared command-line options for evaluation scripts ---------------------------------------------------


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sampling (default: production settings)")
    group.add_argument(
        "--temperature", type=float, help="first-attempt temperature, e.g. 0.1 (the old setting)"
    )
    group.add_argument("--no-seed", action="store_true", help="send no seed (the old setting)")
    group.add_argument(
        "--no-cache-prompt", action="store_true", help="disable llama.cpp's prompt cache (bit-exact)"
    )


def with_sampling(llm: LLM, args: argparse.Namespace) -> LLM:
    """``llm`` as the parsed options ask: unchanged for production settings, else wrapped in ``FixedSampling``."""
    if args.temperature is None and not args.no_seed:
        return llm
    temperature = DEFAULT_TEMPERATURE if args.temperature is None else args.temperature
    return FixedSampling(llm, temperature=temperature, seed=None if args.no_seed else DEFAULT_SEED)


def describe(args: argparse.Namespace) -> str:
    """One line for reports, e.g. ``temperature=0.1 seed=none cache_prompt=on``."""
    temperature = DEFAULT_TEMPERATURE if args.temperature is None else args.temperature
    seed = "none" if args.no_seed else DEFAULT_SEED
    return f"temperature={temperature:g} seed={seed} cache_prompt={'off' if args.no_cache_prompt else 'on'}"
