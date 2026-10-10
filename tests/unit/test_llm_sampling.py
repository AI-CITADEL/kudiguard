import argparse
import dataclasses
import math
from typing import Any

import pytest

from kudiguard.engine import RETRY_TEMPERATURE
from kudiguard.llm.base import DEFAULT_SEED, DEFAULT_TEMPERATURE, ChatRequest, Completion, Purpose
from kudiguard.llm.sampling import FixedSampling, add_sampling_arguments, describe, with_sampling


class Recorder:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return Completion(text="ok", finish_reason="stop")

    async def ready(self) -> bool:
        return True


def request(**changes: Any) -> ChatRequest:
    base = ChatRequest(messages=({"role": "user", "content": "hi"},), max_tokens=5, purpose=Purpose.COMPOSE)
    return dataclasses.replace(base, **changes)


# --- defaults: greedy and seeded ------------------------------------------------------------------------


def test_defaults_are_greedy_and_seeded() -> None:
    req = request()
    assert (req.temperature, req.seed) == (0.0, DEFAULT_SEED) == (DEFAULT_TEMPERATURE, DEFAULT_SEED)
    body = req.payload()
    assert body["temperature"] == 0.0 and body["seed"] == DEFAULT_SEED


def test_seed_none_is_left_to_the_backend() -> None:
    assert "seed" not in request(seed=None).payload()


@pytest.mark.parametrize("seed", [0, 7, 2**31 - 1])
def test_explicit_seed_is_sent(seed: int) -> None:
    assert request(seed=seed).payload()["seed"] == seed


@pytest.mark.parametrize("bad", [-0.01, -1.0, math.nan])
def test_negative_temperature_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        request(temperature=bad)


# --- FixedSampling (evaluation decorator) ---------------------------------------------------------------


async def test_first_attempts_get_the_pinned_temperature_and_seed() -> None:
    inner = Recorder()
    llm = FixedSampling(inner, temperature=0.1, seed=None)
    completion = await llm.complete(request())
    assert completion.text == "ok"
    [sent] = inner.requests
    assert (sent.temperature, sent.seed) == (0.1, None)
    assert "seed" not in sent.payload()


async def test_warmer_retries_keep_their_temperature_but_take_the_seed() -> None:
    inner = Recorder()
    await FixedSampling(inner, temperature=0.1, seed=7).complete(request(temperature=0.5))
    [sent] = inner.requests
    assert (sent.temperature, sent.seed) == (0.5, 7)


async def test_everything_else_is_untouched() -> None:
    inner = Recorder()
    original = request(max_tokens=99, timeout_s=3.0, stop=("\n4.",), json_schema={"type": "object"})
    await FixedSampling(inner, temperature=0.0, seed=DEFAULT_SEED).complete(original)
    assert inner.requests == [original]  # pinning the production values is a no-op


async def test_ready_is_delegated() -> None:
    assert await FixedSampling(Recorder(), temperature=0.0, seed=1).ready() is True


@pytest.mark.parametrize("bad", [-0.1, math.nan])
def test_negative_pinned_temperature_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        FixedSampling(Recorder(), temperature=bad, seed=None)


def test_retry_temperature_differs_from_the_default() -> None:
    assert RETRY_TEMPERATURE != DEFAULT_TEMPERATURE  # FixedSampling tells first attempts from retries by it


# --- command-line options -------------------------------------------------------------------------------


def parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_sampling_arguments(parser)
    return parser.parse_args(argv)


def test_production_options_leave_the_llm_unwrapped() -> None:
    inner = Recorder()
    args = parse()
    assert with_sampling(inner, args) is inner
    assert describe(args) == f"temperature=0 seed={DEFAULT_SEED} cache_prompt=on"


@pytest.mark.parametrize(
    ("argv", "temperature", "seed", "label"),
    [
        (("--temperature", "0.1", "--no-seed"), 0.1, None, "temperature=0.1 seed=none cache_prompt=on"),
        (("--temperature", "0.3"), 0.3, DEFAULT_SEED, f"temperature=0.3 seed={DEFAULT_SEED} cache_prompt=on"),
        (("--no-seed", "--no-cache-prompt"), 0.0, None, "temperature=0 seed=none cache_prompt=off"),
    ],
)
async def test_options_pin_sampling(
    argv: tuple[str, ...], temperature: float, seed: int | None, label: str
) -> None:
    inner = Recorder()
    args = parse(*argv)
    await with_sampling(inner, args).complete(request())
    [sent] = inner.requests
    assert (sent.temperature, sent.seed) == (temperature, seed)
    assert describe(args) == label
