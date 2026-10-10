"""Engine + grounding check: ungrounded replies become fixed text; a check that cannot run fails open."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from fakes import BagOfWordsEmbedder, FakeLLM, analysis
from kudiguard.domain import LanguageCode, Outcome, Verdict
from kudiguard.engine import MIN_GROUNDING_S, Engine
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import Purpose
from kudiguard.safety.grounding import GroundingChecker, GroundingError, GroundingReport
from kudiguard.safety.output_guard import OutputIssue

PACKS = Path(__file__).resolve().parents[2] / "data" / "packs"
OTP_SCAM = "Someone called and asked me to send the OTP sent to my phone. Is this a scam?"
GROUNDED = "This looks like a scam.\n1. Do not share the code.\n2. Change your PIN and password."
INVENTED = "1. Do not share the code.\n2. The EFCC will refund your money within 48 hours."


@pytest.fixture
async def knowledge() -> AsyncIterator[KnowledgeBase]:
    async with httpx.AsyncClient() as client:
        yield knowledge_base_from_packs(PACKS, client=client, include=["kudiguard-core"])


class Recording:
    """GroundingChecker stand-in: records calls; raises or reports as told."""

    def __init__(self, *, error: Exception | None = None, languages: Sequence[str] = ("en",)) -> None:
        self.error = error
        self.languages = languages
        self.calls: list[tuple[str, str, list[str], float | None]] = []

    def supports(self, language: LanguageCode) -> bool:
        return language in self.languages

    async def check(
        self, reply: str, language: LanguageCode, evidence: Sequence[str], *, deadline: float | None = None
    ) -> GroundingReport:
        self.calls.append((reply, language, list(evidence), deadline))
        if self.error:
            raise self.error
        return GroundingReport(reply, ())


def engine(registry: LanguageRegistry, knowledge: KnowledgeBase, llm: FakeLLM, grounding: object) -> Engine:
    return Engine(registry=registry, llm=llm, knowledge=knowledge, grounding=grounding)  # type: ignore[arg-type]


def real_checker(registry: LanguageRegistry) -> GroundingChecker:
    return GroundingChecker(None, BagOfWordsEmbedder(), registry)


async def test_grounded_reply_is_sent(registry: LanguageRegistry, knowledge: KnowledgeBase) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check", ["asks_credentials"]), compose=GROUNDED)
    answer = await engine(registry, knowledge, llm, real_checker(registry)).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.ANSWERED and answer.body == GROUNDED and answer.issues == ()
    assert answer.grounding_ms >= 0


async def test_grounding_time_is_logged_and_reported_apart_from_model_time(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    class Slow(Recording):
        async def check(self, *args: object, **kwargs: object) -> GroundingReport:
            await asyncio.sleep(0.03)
            return await super().check(*args, **kwargs)  # type: ignore[arg-type]

    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=GROUNDED)
    with caplog.at_level(logging.INFO, logger="kudiguard.engine"):
        answer = await engine(registry, knowledge, llm, Slow()).check(OTP_SCAM, "en")
    assert answer.grounding_ms >= 25 and answer.llm_ms == 200
    assert f"grounding_ms={answer.grounding_ms} language=en grounded=True" in caplog.text
    hausa = await engine(
        registry, knowledge, FakeLLM(analyse=analysis("x"), translate="x", compose="Kada."), Slow()
    ).check("Sun ce in ba su OTP", "ha")
    assert hausa.grounding_ms == 0  # not checked live


async def test_ungrounded_reply_becomes_fixed_text_without_a_retry(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check", ["asks_credentials"]), compose=INVENTED)
    with caplog.at_level(logging.INFO, logger="kudiguard.engine"):
        answer = await engine(registry, knowledge, llm, real_checker(registry)).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.FALLBACK and answer.verdict is Verdict.LIKELY_SCAM
    assert answer.body == Messages(registry).text("en", Msg.FALLBACK_SCAM)
    assert answer.issues == (OutputIssue.UNGROUNDED,)
    assert len(llm.calls(Purpose.COMPOSE)) == 1  # no retry
    assert answer.contacts and answer.llm_ms == 200  # report channels and model time still there
    assert "not grounded (1 of 2 claims)" in caplog.text and "EFCC" not in caplog.text


async def test_guard_retry_issues_are_kept_when_the_retry_is_ungrounded(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=["This is not a scam, go ahead.", INVENTED])
    answer = await engine(registry, knowledge, llm, real_checker(registry)).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.FALLBACK
    assert answer.issues == (OutputIssue.SAFETY_CLAIM, OutputIssue.UNGROUNDED)


async def test_checker_gets_the_reply_and_its_evidence(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    grounding = Recording()
    llm = FakeLLM(analyse=analysis("x", "scam_check", ["asks_credentials"]), compose=GROUNDED)
    deadline = time.monotonic() + 60
    answer = await engine(registry, knowledge, llm, grounding).check(OTP_SCAM, "en", deadline=deadline)
    ((reply, language, evidence, passed_deadline),) = grounding.calls
    assert (reply, language, passed_deadline) == (GROUNDED, "en", deadline)
    assert OTP_SCAM in evidence  # the user's message (English input: their own words)
    assert any("never ask for your OTP" in e for e in evidence)  # the facts compose was shown
    assert "This looks like a scam." in evidence and answer.outcome is Outcome.ANSWERED


@pytest.mark.parametrize("error", [GroundingError("out of time"), RuntimeError("embedding model crashed")])
async def test_a_failing_check_fails_open(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=GROUNDED)
    with caplog.at_level(logging.WARNING, logger="kudiguard.engine"):
        answer = await engine(registry, knowledge, llm, Recording(error=error)).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.ANSWERED and answer.body == GROUNDED and answer.issues == ()
    assert "grounding check failed, sending the guarded reply" in caplog.text


async def test_no_time_left_skips_the_check(
    registry: LanguageRegistry,
    knowledge: KnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from kudiguard import engine as engine_module

    grounding = Recording()
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=GROUNDED)
    turn = engine(registry, knowledge, llm, grounding)
    deadline = time.monotonic() + 10
    real_complete = turn._complete

    async def complete_then_run_out_of_time(*args: object) -> object:
        completion = await real_complete(*args)  # type: ignore[arg-type]
        if args[0].purpose is Purpose.COMPOSE:  # type: ignore[attr-defined]
            # the engine's clock only (asyncio keeps the real one)
            later = SimpleNamespace(monotonic=lambda: deadline - MIN_GROUNDING_S / 2)
            monkeypatch.setattr(engine_module, "time", later)
        return completion

    monkeypatch.setattr(turn, "_complete", complete_then_run_out_of_time)
    with caplog.at_level(logging.WARNING, logger="kudiguard.engine"):
        answer = await turn.check(OTP_SCAM, "en", deadline=deadline)
    assert answer.outcome is Outcome.ANSWERED and grounding.calls == []
    assert "grounding check skipped: too little time left" in caplog.text


async def test_unsupported_language_is_not_checked(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    grounding = Recording(languages=("en",))
    llm = FakeLLM(
        analyse=analysis("x", "scam_check"), translate="They asked for my OTP", compose="Kada ku ba su."
    )
    answer = await engine(registry, knowledge, llm, grounding).check("Sun ce in ba su OTP", "ha")
    assert answer.outcome is Outcome.ANSWERED and grounding.calls == []


async def test_fixed_text_and_follow_ups_are_never_checked(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    grounding = Recording()
    llm = FakeLLM(
        analyse=analysis("x", "scam_check"), compose="It is safe to send the code.", warn="Be careful."
    )
    turn = engine(registry, knowledge, llm, grounding)
    blocked = await turn.check(OTP_SCAM, "en")  # guard fallback: fixed text
    assert blocked.outcome is Outcome.FALLBACK and blocked.issues == (OutputIssue.SAFETY_CLAIM,) * 2
    assert (await turn.check("Hello!", "en")).outcome is Outcome.FIXED_REPLY
    await turn.next_steps(OTP_SCAM, (), "en")
    await turn.warning(OTP_SCAM, (), "en")
    assert grounding.calls == []


async def test_without_a_checker_nothing_changes(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=INVENTED)
    answer = await engine(registry, knowledge, llm, None).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.ANSWERED and answer.body == INVENTED
