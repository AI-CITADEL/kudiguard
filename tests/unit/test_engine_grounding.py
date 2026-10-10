"""Engine + grounding check: shadow records, enforce drops unsupported steps or sends fixed text; a check that
cannot run fails open. Also the output guard's unsafe-instruction retry."""

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
from kudiguard.safety.grounding import Claim, GroundingChecker, GroundingError, GroundingMode, GroundingReport
from kudiguard.safety.output_guard import OutputIssue

PACKS = Path(__file__).resolve().parents[2] / "data" / "packs"
OTP_SCAM = "Someone called and asked me to send the OTP sent to my phone. Is this a scam?"
GROUNDED = "This looks like a scam.\n1. Do not share the code.\n2. Change your PIN and password."
INVENTED_STEP = "This looks like a scam.\n1. Do not share the code.\n2. The EFCC will refund your money within 48 hours.\n3. Change your PIN and password."
INVENTED_PROSE = "The EFCC will refund your money within 48 hours.\n1. Do not share the code."
ENFORCE = GroundingMode.ENFORCE


@pytest.fixture
async def knowledge() -> AsyncIterator[KnowledgeBase]:
    async with httpx.AsyncClient() as client:
        yield knowledge_base_from_packs(PACKS, client=client, include=["kudiguard-core"])


class Recording:
    """GroundingChecker stand-in: records calls; raises or reports as told."""

    def __init__(self, *, error: Exception | None = None, languages: Sequence[str] = ("en",)) -> None:
        self.error = error
        self.languages = languages
        self.calls: list[tuple[str, str, list[str], str, float | None]] = []

    def supports(self, language: LanguageCode) -> bool:
        return language in self.languages

    async def check(
        self,
        reply: str,
        language: LanguageCode,
        evidence: Sequence[str],
        *,
        message: str = "",
        deadline: float | None = None,
    ) -> GroundingReport:
        self.calls.append((reply, language, list(evidence), message, deadline))
        if self.error:
            raise self.error
        return GroundingReport(reply, ())


def engine(
    registry: LanguageRegistry,
    knowledge: KnowledgeBase,
    llm: FakeLLM,
    grounding: object,
    mode: GroundingMode = GroundingMode.SHADOW,
) -> Engine:
    return Engine(registry=registry, llm=llm, knowledge=knowledge, grounding=grounding, grounding_mode=mode)  # type: ignore[arg-type]


def real_checker(registry: LanguageRegistry) -> GroundingChecker:
    return GroundingChecker(None, BagOfWordsEmbedder(), registry)


def scam_llm(compose: object) -> FakeLLM:
    return FakeLLM(analyse=analysis("x", "scam_check", ["asks_credentials"]), compose=compose)  # type: ignore[arg-type]


async def test_grounded_reply_is_sent_and_recorded(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    for mode in (GroundingMode.SHADOW, ENFORCE):
        answer = await engine(registry, knowledge, scam_llm(GROUNDED), real_checker(registry), mode).check(
            OTP_SCAM, "en"
        )
        assert answer.outcome is Outcome.ANSWERED and answer.body == GROUNDED and answer.issues == ()
        assert answer.grounded is True and answer.unsupported_claims == 0 and answer.grounding_ms >= 0


async def test_shadow_mode_records_but_never_changes_the_reply(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    llm = scam_llm(INVENTED_PROSE)
    with caplog.at_level(logging.INFO, logger="kudiguard.engine"):
        answer = await engine(registry, knowledge, llm, real_checker(registry)).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.ANSWERED and answer.body == INVENTED_PROSE and answer.issues == ()
    assert answer.grounded is False and answer.unsupported_claims == 1
    assert "mode=shadow grounded=False unsupported=1/2" in caplog.text and "EFCC" not in caplog.text


async def test_enforce_drops_only_the_unsupported_step(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(INVENTED_STEP)
    answer = await engine(registry, knowledge, llm, real_checker(registry), ENFORCE).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.ANSWERED and answer.issues == (OutputIssue.UNGROUNDED,)
    assert answer.body == GROUNDED and answer.grounded is False and answer.unsupported_claims == 1
    assert len(llm.calls(Purpose.COMPOSE)) == 1  # no retry


async def test_enforce_falls_back_when_prose_is_unsupported(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(INVENTED_PROSE)
    answer = await engine(registry, knowledge, llm, real_checker(registry), ENFORCE).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.FALLBACK and answer.verdict is Verdict.LIKELY_SCAM
    assert answer.body == Messages(registry).text("en", Msg.FALLBACK_SCAM)
    assert answer.issues == (OutputIssue.UNGROUNDED,) and len(llm.calls(Purpose.COMPOSE)) == 1
    assert answer.contacts and answer.llm_ms == 200  # report channels and model time still there


async def test_enforce_on_a_back_translated_reply_never_trims(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    class Translated(Recording):
        async def check(self, *args: object, **kwargs: object) -> GroundingReport:
            await super().check(*args, **kwargs)  # type: ignore[arg-type]
            return GroundingReport("1. Pay the fee.", (Claim("Pay the fee.", 0.1, False),))

    llm = FakeLLM(
        analyse=analysis("x", "scam_check"), translate="They asked for my OTP", compose="1. Ku biya."
    )
    answer = await engine(registry, knowledge, llm, Translated(languages=("ha",)), ENFORCE).check(
        "Sun ce in ba su OTP", "ha"
    )
    assert answer.outcome is Outcome.FALLBACK and answer.issues == (OutputIssue.UNGROUNDED,)


async def test_guard_retry_issues_are_kept_when_the_retry_is_ungrounded(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(
        analyse=analysis("x", "scam_check"), compose=["This is not a scam, go ahead.", INVENTED_PROSE]
    )
    answer = await engine(registry, knowledge, llm, real_checker(registry), ENFORCE).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.FALLBACK
    assert answer.issues == (OutputIssue.SAFETY_CLAIM, OutputIssue.UNGROUNDED)


async def test_unsafe_instruction_retries_then_falls_back(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(["Share the code with the caller.", GROUNDED])
    answer = await engine(registry, knowledge, llm, None).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.ANSWERED and answer.body == GROUNDED
    assert answer.issues == (OutputIssue.UNSAFE_INSTRUCTION,)
    llm = scam_llm("Pay the processing fee, then click the link.")
    answer = await engine(registry, knowledge, llm, None).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.FALLBACK and answer.issues == (OutputIssue.UNSAFE_INSTRUCTION,) * 2


async def test_checker_gets_the_reply_evidence_and_message(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    grounding = Recording()
    deadline = time.monotonic() + 60
    turn = engine(registry, knowledge, scam_llm(GROUNDED), grounding)
    answer = await turn.check(OTP_SCAM, "en", deadline=deadline)
    ((reply, language, evidence, message, passed_deadline),) = grounding.calls
    assert (reply, language, message, passed_deadline) == (GROUNDED, "en", OTP_SCAM, deadline)
    assert OTP_SCAM not in evidence  # the user's message is not evidence for meaning
    assert any("never ask for your OTP" in e for e in evidence)  # the facts compose was shown
    assert "This looks like a scam." in evidence and answer.outcome is Outcome.ANSWERED


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
    assert f"grounding_ms={answer.grounding_ms} language=en mode=shadow grounded=True" in caplog.text
    llm = FakeLLM(analyse=analysis("x"), translate="x", compose="Kada.")
    hausa = await engine(registry, knowledge, llm, Slow()).check("Sun ce in ba su OTP", "ha")
    assert hausa.grounding_ms == 0 and hausa.grounded is None  # not checked live


@pytest.mark.parametrize("error", [GroundingError("out of time"), RuntimeError("embedding model crashed")])
async def test_a_failing_check_fails_open(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=GROUNDED)
    with caplog.at_level(logging.WARNING, logger="kudiguard.engine"):
        answer = await engine(registry, knowledge, llm, Recording(error=error), ENFORCE).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.ANSWERED and answer.body == GROUNDED and answer.issues == ()
    assert answer.grounded is None and "grounding check failed, sending the guarded reply" in caplog.text


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
    assert blocked.outcome is Outcome.FALLBACK and OutputIssue.SAFETY_CLAIM in blocked.issues
    assert (await turn.check("Hello!", "en")).outcome is Outcome.FIXED_REPLY
    await turn.next_steps(OTP_SCAM, (), "en")
    await turn.warning(OTP_SCAM, (), "en")
    assert grounding.calls == []


async def test_off_mode_or_no_checker_changes_nothing(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    grounding = Recording()
    for checker, mode in ((None, ENFORCE), (grounding, GroundingMode.OFF)):
        answer = await engine(registry, knowledge, scam_llm(INVENTED_PROSE), checker, mode).check(
            OTP_SCAM, "en"
        )
        assert (
            answer.outcome is Outcome.ANSWERED and answer.body == INVENTED_PROSE and answer.grounded is None
        )
    assert grounding.calls == []
