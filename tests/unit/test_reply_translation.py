"""English-first replies: compose in English, check the English, translate, check the translation."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from fakes import FakeLLM, analysis
from kudiguard import engine as engine_module
from kudiguard.config import Settings
from kudiguard.domain import Flag, Outcome, Verdict
from kudiguard.engine import MIN_GENERATE_S, MIN_TRANSLATE_S, RETRY_TEMPERATURE, Engine
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import DEFAULT_SEED, LLMError, Purpose
from kudiguard.reply_translation import (
    MAX_TRANSLATE_TOKENS,
    MIN_TRANSLATE_TOKENS,
    ComposePolicy,
    ComposeStrategy,
    reply_translate_request,
    step_count,
    strip_reply_tags,
    translate_max_tokens,
    translation_problems,
)
from kudiguard.safety.output_guard import OutputIssue

PACKS = Path(__file__).resolve().parents[2] / "data" / "packs"
YO_OTP = "Ẹnìkan pè mí, ó ní kí n fi koodu OTP tí wọ́n fi ránṣẹ́ sí mi ránṣẹ́ sí òun."
YO_OTP_EN = "Someone called me and asked me to send him the OTP code they sent me."
ENGLISH = (
    "This is a scam. Your bank will never ask for your OTP.\n1. Do not share the code.\n2. Call your bank."
)
YORUBA = "Ẹ̀tàn ni èyí. Báǹkì yín kò ní béèrè OTP yín.\n1. Ẹ má fi kóòdù náà fún ẹnikẹ́ni.\n2. Ẹ pe báǹkì yín."


@pytest.fixture
async def knowledge() -> AsyncIterator[KnowledgeBase]:
    async with httpx.AsyncClient() as client:
        yield knowledge_base_from_packs(PACKS, client=client, include=["kudiguard-core"])


def scam_llm(**script: object) -> FakeLLM:
    base: dict[str, object] = {
        "analyse": analysis(YO_OTP_EN, "scam_check", ["asks_credentials"]),
        "compose": ENGLISH,
        "reply_translate": YORUBA,
    }
    return FakeLLM(**(base | script))  # type: ignore[arg-type]


def make(registry: LanguageRegistry, knowledge: KnowledgeBase, llm: FakeLLM, spec: str = "") -> Engine:
    policy = ComposePolicy.parse(spec) if spec else None
    return Engine(registry=registry, llm=llm, knowledge=knowledge, compose_policy=policy)


# --- policy ------------------------------------------------------------------------------------------------------


def test_policy_defaults_to_english_first_except_english() -> None:
    policy = ComposePolicy()
    assert policy.for_language("yo") is ComposeStrategy.ENGLISH_FIRST
    assert policy.for_language("en") is ComposeStrategy.DIRECT  # nothing to translate


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("english_first", {"ha": "english_first", "yo": "english_first", "en": "direct"}),
        ("DIRECT", {"ha": "direct", "ig": "direct", "en": "direct"}),
        ("english_first, ig=direct,", {"ha": "english_first", "ig": "direct"}),
        ("ha=direct", {"ha": "direct", "yo": "english_first"}),
        ("direct,yo=english_first,en=english_first", {"yo": "english_first", "ig": "direct", "en": "direct"}),
    ],
)
def test_policy_parse(spec: str, expected: dict[str, str]) -> None:
    policy = ComposePolicy.parse(spec)
    assert {lang: policy.for_language(lang).value for lang in expected} == expected


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("english-first", "expected one of"),
        ("ig=translate", "expected one of"),
        ("ig=direct,ig=english_first", "given twice"),
        ("direct,english_first", "more than one default"),
    ],
)
def test_policy_parse_rejects_typos(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ComposePolicy.parse(spec)


def test_settings_validate_the_strategy_at_startup() -> None:
    settings = Settings(compose_strategy="english_first,yo=direct", _env_file=None)  # type: ignore[call-arg]
    assert settings.compose_policy().for_language("yo") is ComposeStrategy.DIRECT
    assert Settings(_env_file=None).compose_policy() == ComposePolicy.parse("english_first")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="compose strategy"):
        Settings(compose_strategy="english", _env_file=None)  # type: ignore[call-arg]


# --- request and fidelity ----------------------------------------------------------------------------------------


def test_translate_request_shape() -> None:
    request = reply_translate_request(ENGLISH, "Yoruba")
    system, user = request.messages  # zero-shot: no worked example to copy
    assert "into Yoruba" in system["content"] and "numbered step" in system["content"]
    assert user["content"] == f"<reply>\n{ENGLISH}\n</reply>"
    assert request.purpose is Purpose.REPLY_TRANSLATE
    assert request.temperature == 0.0 and request.seed == DEFAULT_SEED


def test_translate_budget_follows_the_english_length() -> None:
    assert translate_max_tokens("Do not pay.") == MIN_TRANSLATE_TOKENS
    assert translate_max_tokens("x" * 5000) == MAX_TRANSLATE_TOKENS
    assert MIN_TRANSLATE_TOKENS < translate_max_tokens(ENGLISH * 3) < MAX_TRANSLATE_TOKENS


def test_step_count_reads_localised_step_words() -> None:
    assert step_count("Intro\n1. a\n2) b\n**3.** c") == 3
    assert step_count("Mataki 1: a\nMataki na 2: b\nN5,000.") == 2


def test_faithful_translation_passes() -> None:
    assert translation_problems(ENGLISH, YORUBA) == []
    # amounts and number words are not "new numbers": N5,000 -> 5000, "three months" -> "oṣù 3"
    assert translation_problems("Pay N5,000 within three months.", "San 5000 láàrin oṣù 3 o.") == []


@pytest.mark.parametrize(
    ("translated", "problem"),
    [
        ("   ", "empty"),
        ("Ẹ̀tàn ni èyí. Báǹkì yín kò ní béèrè OTP yín láé, ẹ má fi kóòdù fún ẹnikẹ́ni.", "steps 2 -> 0"),
        ("Ẹ̀tàn.\n1. Rárá.\n2. Pè é.", "length ratio"),
        (YORUBA * 4, "length ratio"),
        (YORUBA + " Ẹ pe 0803 [number].", "new numbers"),
        (YORUBA + " Ẹ lọ sí [link removed].", "new link or number"),
        (ENGLISH.replace("This is", "Eyi ni"), "untranslated"),
    ],
)
def test_unfaithful_translations_are_caught(translated: str, problem: str) -> None:
    assert any(p.startswith(problem) for p in translation_problems(ENGLISH, translated)), problem


# Live failures (docs/SPIKES.md H): each flipped the advice and passed every other check.
@pytest.mark.parametrize(
    ("lang", "english", "translated", "problem"),
    [
        # "You should keep all your money at home"
        ("yo", "You should not keep all your money.", "Ó yẹ kí o máa tọ́jú gbogbo owó rẹ.", "lost"),
        # "Give any money"
        ("ig", "Be careful.\n1. Do not pay anything.", "Kpachara anya.\n1. Nye ego ọ bụla.", "lost"),
        # "Do not block your SIM"
        ("ha", "You should block your SIM now.", "Kada ku rufe katin SIM yanzu.", "added"),
        # "Do not end the call"
        ("ha", "End the call.\n1. Hang up now.", "Ka katse kiran.\n1. Kada ka gama kiran.", "added"),
    ],
)
def test_flipped_negation_is_caught(lang: str, english: str, translated: str, problem: str) -> None:
    assert any(p.startswith(f"negation {problem}") for p in translation_problems(english, translated, lang))


@pytest.mark.parametrize(
    ("lang", "english", "translated"),
    [
        ("ha", "Check your balance, not an SMS alert.", "Duba ma'aunin ku, ba sakon SMS ba."),  # ba ... ba
        ("ig", "Be careful of strangers who call you.", "Kpachara anya maka ndị na-amaghị ama na-akpọ gị."),
        ("yo", "Stop replying to them.", "Ẹ dúró fífèsì sí wọn."),
        ("ig", "1. Stop replying to them.", "1. Kwụsị ịza ha ozi."),
        ("fr", "Do not pay.", "Payez maintenant."),  # no negator list: not checked
    ],
)
def test_kept_negation_passes(lang: str, english: str, translated: str) -> None:
    assert not [p for p in translation_problems(english, translated, lang) if p.startswith("negation")]


def test_echoed_or_invented_tags_are_removed() -> None:
    text = "<ìtumọ̀>\nẸ má sanwó.\n</ translation>\n</reply>"
    assert strip_reply_tags(text).strip() == "Ẹ má sanwó."
    assert strip_reply_tags("Lo <b>app</b> náà") == "Lo <b>app</b> náà"  # inline text is kept


def test_short_replies_are_not_echo_tested() -> None:
    assert translation_problems("1. Dial *565*0# now.", "1. Dial *565*0# now o.") == []


# --- engine ------------------------------------------------------------------------------------------------------


async def test_english_first_composes_in_english_then_translates(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(reply_translate=YORUBA + "\n</reply>")  # an echoed data tag is dropped
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    (compose,) = llm.calls(Purpose.COMPOSE)
    assert compose.messages[-1]["content"].startswith("Reply in English.\n")
    assert compose.messages[1]["content"].startswith("Reply in English.\n")  # the English example
    (translate,) = llm.calls(Purpose.REPLY_TRANSLATE)
    assert "into Yoruba" in translate.messages[0]["content"]
    assert translate.messages[-1]["content"] == f"<reply>\n{ENGLISH}\n</reply>"
    assert answer.outcome is Outcome.ANSWERED and answer.verdict is Verdict.LIKELY_SCAM
    assert answer.body == YORUBA and answer.reply_english == ENGLISH
    assert answer.llm_ms == 300  # analyse + compose + translate


async def test_english_replies_are_never_translated(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm()
    answer = await make(registry, knowledge, llm).check(YO_OTP_EN, "en")
    assert llm.calls(Purpose.REPLY_TRANSLATE) == [] and answer.body == ENGLISH and answer.reply_english == ""


async def test_direct_override_composes_in_the_users_language(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(compose=YORUBA)
    answer = await make(registry, knowledge, llm, "english_first,yo=direct").check(YO_OTP, "yo")
    (compose,) = llm.calls(Purpose.COMPOSE)
    assert compose.messages[-1]["content"].startswith("Reply in Yoruba.\n")
    assert llm.calls(Purpose.REPLY_TRANSLATE) == [] and answer.body == YORUBA and answer.reply_english == ""


async def test_the_guard_checks_the_english_before_translation(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(compose=["This is not a scam, go ahead and send it.", ENGLISH])
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    assert [c.temperature for c in llm.calls(Purpose.COMPOSE)] == [0.0, RETRY_TEMPERATURE]
    assert OutputIssue.SAFETY_CLAIM in answer.issues  # caught by the English guard
    (translate,) = llm.calls(Purpose.REPLY_TRANSLATE)  # only the clean English reply is translated
    assert ENGLISH in translate.messages[-1]["content"] and answer.body == YORUBA


async def test_unsafe_english_twice_gives_fixed_text_without_translating(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(compose="It is safe to send the code.")
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    assert answer.outcome is Outcome.FALLBACK and answer.body == Messages(registry).text(
        "yo", Msg.FALLBACK_SCAM
    )
    assert llm.calls(Purpose.REPLY_TRANSLATE) == [] and answer.reply_english == ""


async def test_unfaithful_translation_is_retried_warmer(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    dropped_steps = "Ẹ̀tàn ni èyí. Báǹkì yín kò ní béèrè OTP yín láé. Ẹ má fi kóòdù náà fún ẹnikẹ́ni."
    llm = scam_llm(reply_translate=[dropped_steps, YORUBA])
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    translations = llm.calls(Purpose.REPLY_TRANSLATE)
    assert [t.temperature for t in translations] == [0.0, RETRY_TEMPERATURE]
    assert {t.seed for t in translations} == {DEFAULT_SEED}
    assert answer.outcome is Outcome.ANSWERED and answer.body == YORUBA


async def test_native_guard_is_a_second_net_on_the_translation(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    unsafe = YORUBA.replace("Ẹ̀tàn ni èyí.", "Kò sí ewu.")  # "there is no danger": a Yoruba safety claim
    llm = scam_llm(reply_translate=[unsafe, YORUBA])
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    assert OutputIssue.SAFETY_CLAIM in answer.issues and answer.body == YORUBA


@pytest.mark.parametrize("failure", ["Ẹ̀tàn ni.", LLMError("timeout")])
async def test_failed_translation_gives_fixed_text_in_the_users_language(
    registry: LanguageRegistry, knowledge: KnowledgeBase, failure: object
) -> None:
    llm = scam_llm(reply_translate=failure)
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    assert answer.outcome is Outcome.FALLBACK and answer.verdict is Verdict.LIKELY_SCAM
    assert answer.body == Messages(registry).text("yo", Msg.FALLBACK_SCAM)
    assert answer.contacts  # report channels still shown
    assert len(llm.calls(Purpose.REPLY_TRANSLATE)) == (2 if isinstance(failure, str) else 1)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def test_too_little_time_for_both_steps_skips_the_model(
    registry: LanguageRegistry, knowledge: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(engine_module.time, "monotonic", clock)
    llm = scam_llm()
    deadline = clock.now + MIN_GENERATE_S + MIN_TRANSLATE_S - 0.5  # enough for one step, not two
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo", deadline=deadline)
    assert llm.calls(Purpose.COMPOSE) == [] and llm.calls(Purpose.REPLY_TRANSLATE) == []
    assert answer.outcome is Outcome.FALLBACK and answer.body == Messages(registry).text(
        "yo", Msg.FALLBACK_SCAM
    )


async def test_deadline_reached_during_compose_skips_the_translation(
    registry: LanguageRegistry, knowledge: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(engine_module.time, "monotonic", clock)

    def slow_compose(_: object) -> str:
        clock.now += MIN_GENERATE_S + 1.0  # the English reply took long
        return ENGLISH

    llm = scam_llm(compose=slow_compose)
    deadline = clock.now + MIN_GENERATE_S + MIN_TRANSLATE_S + 0.5
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo", deadline=deadline)
    assert len(llm.calls(Purpose.COMPOSE)) == 1 and llm.calls(Purpose.REPLY_TRANSLATE) == []
    assert answer.outcome is Outcome.FALLBACK and answer.body == Messages(registry).text(
        "yo", Msg.FALLBACK_SCAM
    )


async def test_translation_budget_is_capped_by_the_deadline(
    registry: LanguageRegistry, knowledge: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(engine_module.time, "monotonic", clock)
    llm = scam_llm()
    deadline = clock.now + MIN_GENERATE_S + MIN_TRANSLATE_S + 1.0
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo", deadline=deadline)
    (translate,) = llm.calls(Purpose.REPLY_TRANSLATE)
    assert translate.timeout_s == pytest.approx(deadline - clock.now)
    assert answer.outcome is Outcome.ANSWERED and Flag.ASKS_CREDENTIALS in answer.flags
