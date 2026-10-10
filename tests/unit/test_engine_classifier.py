"""Engine + learned guard layer: enforce retries/replaces a flagged reply, shadow only logs after sending."""

import logging
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from test_guard_classifier import hand_model

from fakes import BagOfWordsEmbedder, FakeLLM, analysis
from kudiguard.domain import Outcome
from kudiguard.engine import Engine
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import Purpose
from kudiguard.safety.guard_classifier import ClassifierError, ClassifierMode, UnsafeAdviceClassifier
from kudiguard.safety.output_guard import OutputGuard, OutputIssue

PACKS = Path(__file__).resolve().parents[2] / "data" / "packs"
OTP_SCAM = "Someone called and asked me to send the OTP sent to my phone. Is this a scam?"
GOOD = "This looks like a scam.\n1. Do not share the code.\n2. Call your bank."
# the regex guard lets this through (no verb-first instruction); the hand model flags "send the code" + imperative
SLIPS = "This looks like a scam.\nDrop the code for them so we can check it."


@pytest.fixture
async def knowledge() -> AsyncIterator[KnowledgeBase]:
    async with httpx.AsyncClient() as client:
        yield knowledge_base_from_packs(PACKS, client=client, include=["kudiguard-core"])


def classifier(registry: LanguageRegistry) -> UnsafeAdviceClassifier:
    return UnsafeAdviceClassifier(hand_model(), BagOfWordsEmbedder(), OutputGuard(registry))


def engine(
    registry: LanguageRegistry,
    knowledge: KnowledgeBase,
    llm: FakeLLM,
    model: object,
    mode: ClassifierMode,
    pidgin: ClassifierMode | None = None,
) -> Engine:
    return Engine(registry=registry, llm=llm, knowledge=knowledge, classifier=model, classifier_mode=mode,  # type: ignore[arg-type]
                  classifier_pidgin_mode=pidgin)  # fmt: skip


def scam(compose: object) -> FakeLLM:
    return FakeLLM(analyse=analysis("x", "scam_check", ["asks_credentials"]), compose=compose)  # type: ignore[arg-type]


def test_the_example_slips_past_the_regex_but_not_the_model(registry: LanguageRegistry) -> None:
    assert OutputIssue.UNSAFE_INSTRUCTION not in OutputGuard(registry).check(SLIPS, "en")
    assert classifier(registry).score(SLIPS, "en").flagged


async def test_enforce_retries_then_falls_back(registry: LanguageRegistry, knowledge: KnowledgeBase) -> None:
    llm = scam([SLIPS, GOOD])
    answer = await engine(registry, knowledge, llm, classifier(registry), ClassifierMode.ENFORCE).check(
        OTP_SCAM, "en"
    )
    assert answer.outcome is Outcome.ANSWERED and answer.body == GOOD
    assert OutputIssue.CLASSIFIED_UNSAFE in answer.issues and len(llm.calls(Purpose.COMPOSE)) == 2
    llm = scam(SLIPS)
    answer = await engine(registry, knowledge, llm, classifier(registry), ClassifierMode.ENFORCE).check(
        OTP_SCAM, "en"
    )
    assert answer.outcome is Outcome.FALLBACK and answer.body == Messages(registry).text(
        "en", Msg.FALLBACK_SCAM
    )


async def test_enforce_fails_open(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken(UnsafeAdviceClassifier):
        async def check(self, text: str, language: str, *, wait: bool = True) -> object:  # type: ignore[override]
            raise ClassifierError("busy", "busy")

    broken = Broken(hand_model(), BagOfWordsEmbedder(), OutputGuard(registry))
    with caplog.at_level(logging.WARNING, logger="kudiguard.engine"):
        answer = await engine(registry, knowledge, scam(SLIPS), broken, ClassifierMode.ENFORCE).check(
            OTP_SCAM, "en"
        )
    assert answer.outcome is Outcome.ANSWERED and answer.body == SLIPS
    assert "guard classifier not run, regex verdict only: busy" in caplog.text


async def test_shadow_logs_after_sending(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    turn = engine(registry, knowledge, scam(SLIPS), classifier(registry), ClassifierMode.SHADOW)
    with caplog.at_level(logging.INFO, logger="kudiguard.engine"):
        answer = await turn.check(OTP_SCAM, "en", turn_id="k1")
        assert answer.outcome is Outcome.ANSWERED and answer.body == SLIPS
        await turn.aclose()
    assert (
        "guard_classifier turn=k1 mode=shadow flagged=True" in caplog.text
        and "drop the code" not in caplog.text
    )


async def test_shadow_failure_is_logged(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken(UnsafeAdviceClassifier):
        async def check(self, text: str, language: str, *, wait: bool = True) -> object:  # type: ignore[override]
            raise ClassifierError("timeout", "out of time")

    broken = Broken(hand_model(), BagOfWordsEmbedder(), OutputGuard(registry))
    turn = engine(registry, knowledge, scam(SLIPS), broken, ClassifierMode.SHADOW)
    with caplog.at_level(logging.WARNING, logger="kudiguard.engine"):
        await turn.check(OTP_SCAM, "en")
        await turn.aclose()
    assert "guard classifier not run (shadow): out of time" in caplog.text


async def test_off_or_other_language_skips_the_model(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    answer = await engine(registry, knowledge, scam(SLIPS), classifier(registry), ClassifierMode.OFF).check(
        OTP_SCAM, "en"
    )
    assert answer.body == SLIPS and OutputIssue.CLASSIFIED_UNSAFE not in answer.issues
    llm = FakeLLM(
        analyse=analysis("x", "scam_check"), translate="They asked for my OTP", compose="Ku kira bankinku."
    )
    turn = engine(registry, knowledge, llm, classifier(registry), ClassifierMode.ENFORCE)
    assert (await turn.check("Sun ce in ba su OTP", "ha")).outcome is Outcome.ANSWERED
    shadow = engine(registry, knowledge, FakeLLM(analyse=analysis("x", "scam_check"), translate="x", compose="Ku kira bankinku."),
                    classifier(registry), ClassifierMode.SHADOW)  # fmt: skip
    await shadow.check("Sun ce in ba su OTP", "ha")
    assert not shadow._shadow


PIDGIN_SLIPS = "Na scam be this.\nDrop the code for them make dem check am."


async def test_pidgin_replies_can_run_in_their_own_mode(
    registry: LanguageRegistry, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    # English enforced, Pidgin in shadow: the Pidgin slip is sent (and logged), the English one is replaced
    turn = engine(registry, knowledge, scam(PIDGIN_SLIPS), classifier(registry), ClassifierMode.ENFORCE,
                  ClassifierMode.SHADOW)  # fmt: skip
    with caplog.at_level(logging.INFO, logger="kudiguard.engine"):
        answer = await turn.check(OTP_SCAM, "en", turn_id="p1")
        await turn.aclose()
    assert answer.body == PIDGIN_SLIPS and "guard_classifier turn=p1 mode=shadow flagged=True" in caplog.text
    english = engine(registry, knowledge, scam(SLIPS), classifier(registry), ClassifierMode.ENFORCE,
                     ClassifierMode.SHADOW)  # fmt: skip
    assert (await english.check(OTP_SCAM, "en")).outcome is Outcome.FALLBACK
    # Pidgin only: English replies are not scored at all
    only = engine(
        registry, knowledge, scam(SLIPS), classifier(registry), ClassifierMode.OFF, ClassifierMode.ENFORCE
    )
    assert only.classifier is not None
    assert (await only.check(OTP_SCAM, "en")).body == SLIPS and not only._shadow
    off = engine(
        registry, knowledge, scam(SLIPS), classifier(registry), ClassifierMode.OFF, ClassifierMode.OFF
    )
    assert off.classifier is None
