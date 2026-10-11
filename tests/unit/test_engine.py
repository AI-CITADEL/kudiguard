from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from fakes import FakeLLM, analysis
from kudiguard.domain import Flag, Outcome, Verdict
from kudiguard.engine import RETRY_TEMPERATURE, AnswerKind, Engine
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import DEFAULT_SEED, LLMError, Purpose
from kudiguard.prompts import Intent
from kudiguard.safety.output_guard import OutputIssue

PACKS = Path(__file__).resolve().parents[2] / "data" / "packs"
OTP_SCAM = "Someone called and asked me to send the OTP sent to my phone. Is this a scam?"
HA_OTP = "Wani ya kira ni ya ce in ba shi lambar da aka turo min"
HA_OTP_EN = "Someone called me and said I should give him the OTP code that was sent to me"
GOOD_REPLY = "1. Do not share the code.\n2. Call your bank on the number on your card."


@pytest.fixture
async def knowledge() -> AsyncIterator[KnowledgeBase]:
    async with httpx.AsyncClient() as client:
        yield knowledge_base_from_packs(PACKS, client=client, include=["kudiguard-core"])


def engine_with(registry: LanguageRegistry, knowledge: KnowledgeBase, llm: FakeLLM) -> Engine:
    return Engine(registry=registry, llm=llm, knowledge=knowledge)


async def test_content_free_and_greetings_never_call_the_model(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM()
    engine = engine_with(registry, knowledge, llm)
    assert (await engine.check("👍 ??", "en")).kind is AnswerKind.CONTENT_FREE
    for text, lang in (
        ("Hello!", "en"),
        ("Sannu, na gode", "ha"),
        ("Ẹ kú àárọ̀", "yo"),
        ("hi KudiGuard", "en"),
    ):
        answer = await engine.check(text, lang)
        assert answer.kind is AnswerKind.GREETING and answer.outcome is Outcome.FIXED_REPLY, text
    assert llm.requests == []


async def test_english_scam_is_answered_grounded_with_verified_contacts(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(
        analyse=analysis("ignored for English", "scam_check", ["asks_credentials"]), compose=GOOD_REPLY
    )
    answer = await engine_with(registry, knowledge, llm).check(OTP_SCAM, "en")
    assert answer.kind is AnswerKind.CHECK and answer.outcome is Outcome.ANSWERED
    assert answer.verdict is Verdict.LIKELY_SCAM and Flag.ASKS_CREDENTIALS in answer.flags
    assert answer.english == OTP_SCAM  # English input: the user's own words, never the model's paraphrase
    assert answer.body.startswith("1. Do not share")
    assert answer.contacts and all(c.verified for c in answer.contacts)
    assert answer.sources and answer.actions
    assert answer.llm_ms == 200


async def test_guard_hit_retries_warmer_then_succeeds(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=["This is not a scam, go ahead.", GOOD_REPLY])
    answer = await engine_with(registry, knowledge, llm).check(OTP_SCAM, "en")
    composes = llm.calls(Purpose.COMPOSE)
    assert [c.temperature for c in composes] == [0.0, RETRY_TEMPERATURE]
    assert {c.seed for c in composes} == {DEFAULT_SEED}  # the warmer retry is reproducible too
    assert answer.outcome is Outcome.ANSWERED and answer.body.startswith("1.")
    assert OutputIssue.SAFETY_CLAIM in answer.issues


async def test_repeated_guard_hits_fall_back_to_fixed_text(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose="It is safe to send the code.")
    answer = await engine_with(registry, knowledge, llm).check(OTP_SCAM, "en")
    assert answer.outcome is Outcome.FALLBACK and answer.verdict is Verdict.LIKELY_SCAM
    assert answer.body == Messages(registry).text("en", Msg.FALLBACK_SCAM)
    assert len(llm.calls(Purpose.COMPOSE)) == 2
    assert answer.contacts  # report channels still shown


async def test_model_down_degrades_to_rules_without_further_calls(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(analyse=LLMError("connection refused"), compose=GOOD_REPLY)
    answer = await engine_with(registry, knowledge, llm).check(OTP_SCAM, "en")
    assert answer.verdict is Verdict.LIKELY_SCAM and answer.outcome is Outcome.FALLBACK
    assert llm.calls(Purpose.COMPOSE) == []  # do not wait for a second timeout


async def test_hausa_uses_translation_fallback_when_analysis_echoes(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(
        analyse=analysis(HA_OTP, "scam_check"),
        translate=HA_OTP_EN,
        compose="Do not share the code.",
        reply_translate="Kada ka bayar da lambar.",
    )
    answer = await engine_with(registry, knowledge, llm).check(HA_OTP, "ha")
    assert len(llm.calls(Purpose.TRANSLATE)) == 1
    assert answer.english == HA_OTP_EN
    assert answer.verdict is Verdict.LIKELY_SCAM and answer.outcome is Outcome.ANSWERED
    assert "Reply in English" in llm.calls(Purpose.COMPOSE)[0].messages[-1]["content"]  # english_first
    assert answer.body == "Kada ka bayar da lambar." and answer.reply_english == "Do not share the code."


async def test_hausa_without_any_translation_answers_with_fixed_text(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(analyse="not json", translate=LLMError("timeout"), compose=GOOD_REPLY)
    answer = await engine_with(registry, knowledge, llm).check("Yaya zan buɗe asusun ajiya?", "ha")
    assert answer.english == "" and answer.outcome is Outcome.FALLBACK
    assert answer.body == Messages(registry).text("ha", Msg.FALLBACK_ANSWER)
    assert llm.calls(Purpose.COMPOSE) == []


async def test_model_labelled_greeting_gets_fixed_reply(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(analyse=analysis("Good evening to you and your family", "greeting"))
    answer = await engine_with(registry, knowledge, llm).check("Good evening to you and your family", "en")
    assert answer.kind is AnswerKind.GREETING and llm.calls(Purpose.COMPOSE) == []


async def test_no_flags_fallback_has_no_body(registry: LanguageRegistry, knowledge: KnowledgeBase) -> None:
    llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=LLMError("timeout"))
    answer = await engine_with(registry, knowledge, llm).check(
        "Is this genuine? Your parcel is waiting", "en"
    )
    assert answer.verdict is Verdict.NO_FLAGS and answer.outcome is Outcome.FALLBACK and answer.body == ""
    assert answer.contacts == ()  # report channels only for scam verdicts


async def test_question_is_not_a_check(registry: LanguageRegistry, knowledge: KnowledgeBase) -> None:
    llm = FakeLLM(analyse=analysis("x", "question"), compose="Visit a branch with your ID.")
    answer = await engine_with(registry, knowledge, llm).check("How do I open a savings account?", "en")
    assert answer.verdict is Verdict.NOT_A_CHECK and answer.intent is Intent.QUESTION
    assert answer.contacts == ()


async def test_next_steps_english_needs_no_model(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM()
    steps = await engine_with(registry, knowledge, llm).next_steps(OTP_SCAM, (Flag.ASKS_CREDENTIALS,), "en")
    assert steps.outcome is Outcome.ANSWERED and steps.text.startswith("1. ") and llm.requests == []


async def test_next_steps_translated_or_fixed(registry: LanguageRegistry, knowledge: KnowledgeBase) -> None:
    llm = FakeLLM(steps="1. Kada ka bayar da lambar OTP.\n2. Ka kira bankinka.")
    engine = engine_with(registry, knowledge, llm)
    steps = await engine.next_steps(HA_OTP_EN, (Flag.ASKS_CREDENTIALS,), "ha")
    assert steps.outcome is Outcome.ANSWERED and "OTP" in steps.text
    llm.script[Purpose.STEPS] = [LLMError("down")]
    steps = await engine.next_steps(HA_OTP_EN, (Flag.ASKS_CREDENTIALS,), "ha")
    assert steps.outcome is Outcome.FALLBACK and steps.text == Messages(registry).text(
        "ha", Msg.FALLBACK_STEPS
    )


async def test_warning_and_report(registry: LanguageRegistry, knowledge: KnowledgeBase) -> None:
    llm = FakeLLM(warn="⚠️ Kada ku bayar da OTP ga kowa.")
    engine = engine_with(registry, knowledge, llm)
    warned = await engine.warning(HA_OTP_EN, (Flag.ASKS_CREDENTIALS,), "ha")
    assert warned.outcome is Outcome.ANSWERED and "OTP" in warned.text
    fallback = await engine.warning(
        "", (), "ha"
    )  # source turn gone (deleted/purged): generic warning, no model
    assert fallback.outcome is Outcome.FALLBACK and len(llm.calls(Purpose.WARN)) == 1
    contacts = await engine.report_contacts("", (Flag.ASKS_CREDENTIALS,), "en")
    assert contacts and all(c.verified for c in contacts)


async def test_transcribe_requires_speech_configuration(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    with pytest.raises(RuntimeError, match="speech is not configured"):
        await engine_with(registry, knowledge, FakeLLM()).transcribe(b"x", "en")


async def test_next_steps_without_any_guidance_is_fixed_text_and_no_model(registry: LanguageRegistry) -> None:
    from kudiguard.knowledge.models import PackKind, PackManifest, Query, SourceResult

    class EmptyCore:
        manifest = PackManifest(id="core", name="Core", owner="x", kind=PackKind.CORE)

        async def search(self, query: Query) -> SourceResult:
            return SourceResult()

        async def refresh(self) -> None:
            return None

    llm = FakeLLM(steps="should never be used")
    engine = engine_with(registry, KnowledgeBase([EmptyCore()]), llm)
    for lang in ("en", "ha"):
        steps = await engine.next_steps(OTP_SCAM, (Flag.ASKS_CREDENTIALS,), lang)
        assert steps.outcome is Outcome.FALLBACK
        assert steps.text == Messages(registry).text(lang, Msg.FALLBACK_STEPS)
        assert steps.llm_ms == 0
    assert llm.requests == []


async def test_guarded_warning_falls_back_but_reports_model_time(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(warn="It is safe to send the code.")  # blocked by the output guard twice
    warned = await engine_with(registry, knowledge, llm).warning(HA_OTP_EN, (Flag.ASKS_CREDENTIALS,), "ha")
    assert warned.outcome is Outcome.FALLBACK
    assert warned.text == Messages(registry).text("ha", Msg.WARN_FAMILY_FALLBACK)
    assert warned.llm_ms == 200  # both attempts' latency is still accounted for
    warns = llm.calls(Purpose.WARN)
    assert [c.temperature for c in warns] == [0.0, RETRY_TEMPERATURE]
    assert {c.seed for c in warns} == {DEFAULT_SEED}  # follow-up retries reproduce too
