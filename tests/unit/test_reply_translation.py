"""English-first replies: compose in English, check the English, translate, check the translation."""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from test_guard_classifier import hand_model

from fakes import BagOfWordsEmbedder, FakeLLM, analysis
from kudiguard import engine as engine_module
from kudiguard.config import Settings
from kudiguard.domain import Flag, Outcome, Verdict
from kudiguard.engine import MIN_GENERATE_S, MIN_TRANSLATE_S, RETRY_TEMPERATURE, Engine
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.service import KnowledgeBase, knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import DEFAULT_SEED, ChatRequest, LLMError, Purpose
from kudiguard.reply_translation import (
    DEFAULT_COMPOSE_STRATEGY,
    MAX_TRANSLATE_TOKENS,
    MIN_TRANSLATE_TOKENS,
    RETRYABLE_PROBLEMS,
    ComposePolicy,
    ComposeStrategy,
    problem_kind,
    reply_translate_request,
    step_count,
    strip_reply_tags,
    translate_max_tokens,
    translation_problems,
)
from kudiguard.safety.grounding import Claim, GroundingChecker, GroundingMode, GroundingReport
from kudiguard.safety.guard_classifier import ClassifierMode, UnsafeAdviceClassifier
from kudiguard.safety.output_guard import OutputGuard, OutputIssue

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


def test_shipped_default_is_english_first_for_yoruba_and_igbo_direct_for_hausa() -> None:
    policy = ComposePolicy.parse(DEFAULT_COMPOSE_STRATEGY)
    assert {lang: policy.for_language(lang).value for lang in ("en", "ha", "yo", "ig")} == {
        "en": "direct",  # nothing to translate
        "ha": "direct",  # English-first cut Hausa coverage 0.54 -> 0.34 (docs/SPIKES.md H)
        "yo": "english_first",
        "ig": "english_first",
    }
    assert (
        ComposePolicy().for_language("ha") is ComposeStrategy.ENGLISH_FIRST
    )  # the bare policy: no overrides


def test_engine_uses_the_shipped_default(registry: LanguageRegistry) -> None:
    engine = Engine(registry=registry, llm=FakeLLM(), knowledge=None)  # type: ignore[arg-type]
    assert engine.compose_policy == ComposePolicy.parse(DEFAULT_COMPOSE_STRATEGY)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("english_first", {"ha": "english_first", "yo": "english_first", "en": "direct"}),
        ("DIRECT", {"ha": "direct", "ig": "direct", "en": "direct"}),
        ("english_first, ig=direct,", {"ha": "english_first", "ig": "direct"}),
        ("ha=direct", {"ha": "direct", "yo": "english_first"}),
        ("direct,yo=english_first,en=english_first", {"yo": "english_first", "ig": "direct", "en": "direct"}),
        ("English_First, HA=Direct", {"ha": "direct", "yo": "english_first"}),  # case-insensitive
    ],
)
def test_policy_parse(spec: str, expected: dict[str, str]) -> None:
    policy = ComposePolicy.parse(spec, ["en", "ha", "yo", "ig"])
    assert {lang: policy.for_language(lang).value for lang in expected} == expected


def test_unknown_language_override_fails_at_startup(registry: LanguageRegistry) -> None:
    codes = [lang.code for lang in registry.enabled]
    with pytest.raises(ValueError, match="unknown language 'yr'"):
        ComposePolicy.parse("english_first,yr=direct", codes)
    with pytest.raises(ValueError, match="unknown language"):
        Settings(compose_strategy="english_first,yr=direct", _env_file=None).compose_policy(codes)  # type: ignore[call-arg]
    assert (
        ComposePolicy.parse("english_first,yr=direct").for_language("yr") is ComposeStrategy.DIRECT
    )  # unchecked


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
    default = Settings(_env_file=None).compose_policy()  # type: ignore[call-arg]
    assert default == ComposePolicy.parse("english_first,ha=direct")
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
        ("yo", "Stop replying to them.", "Ẹ má fèsì sí wọn mọ́."),  # a soft negation may become a prohibitive
        ("ig", "1. Stop replying to them.", "1. Kwụsị ịza ha ozi."),
        ("fr", "Do not pay.", "Payez maintenant."),  # no negator list: not checked
        # "instead" is no negation: no false fallback
        (
            "yo",
            "Do not click the link. Use your bank app instead.",
            "Ẹ má tẹ ìjápọ̀ náà. Ẹ lo app báǹkì yín dípò rẹ̀.",
        ),
        # "not even" is emphasis: one negation to keep
        (
            "yo",
            "Don't give the PIN to anyone, not even bank staff.",
            "Má fún ẹnikẹ́ni ní PIN, kódà òṣìṣẹ́ báǹkì.",
        ),
        # "don't forget" asks for the action on both sides
        ("ig", "Don't forget to call your bank.", "Echefula ịkpọ ụlọ akụ gị."),
        (
            "yo",
            "Pay the fee only at the bank.",
            "Ẹ máa san owó náà ní báǹkì nìkan.",
        ),  # máa: future, not "do not"
    ],
)
def test_kept_negation_passes(lang: str, english: str, translated: str) -> None:
    assert not [p for p in translation_problems(english, translated, lang) if p.startswith("negation")]


# H review cases: counting, not presence
@pytest.mark.parametrize(
    ("lang", "english", "translated", "problem"),
    [
        # one of two negations dropped ("Do not pay. Reply to them.")
        ("yo", "Do not pay. Do not reply.", "Ẹ má ṣe sanwó. Ẹ fèsì sí wọn.", "lost"),
        # a negation moved to the other sentence ("Do not call your bank. Pay.")
        ("yo", "1. Call your bank. Do not pay.", "1. Ẹ má pe báǹkì yín. Ẹ sanwó.", "added"),
        ("yo", "1. Call your bank. Do not pay.", "1. Ẹ má pe báǹkì yín. Ẹ sanwó.", "lost"),
        # a double negative loses one ("Ignore a debit you made")
        ("yo", "Never ignore a debit you did not make.", "Ẹ má fojú fo owó tí ẹ ná.", "lost"),
        # kọ (write) is not kò (not): "Pay; write the date down"
        ("yo", "Do not pay, write the date down.", "Ẹ sanwó, kọ ọjọ́ náà sílẹ̀.", "lost"),
        # sentence-initial Má + verb is an added prohibitive ("Do not pay the fee now")
        ("yo", "Pay the fee at the bank.", "Má san owó náà ní báǹkì.", "added"),
        # the warm-retry translation from review: "Don't forget to steal money..."
        (
            "ig",
            "Never send money to someone you only met online.",
            "Echefula izu ohi ego n'aka onye ị maara naanị online.",
            "lost",
        ),
    ],
)
def test_review_negation_cases(lang: str, english: str, translated: str, problem: str) -> None:
    assert any(p.startswith(f"negation {problem}") for p in translation_problems(english, translated, lang))


@pytest.mark.parametrize(
    ("english", "igbo"),
    [
        (
            "Call your bank today, do not wait until tomorrow.",
            "Kpọọ ụlọ akụ gị taa, ekwela ka ị chere ruo echi.",
        ),
        ("Call your bank today.", "Kpọọ ụlọ akụ gị taa. Ekwela ka ị chere ruo echi."),  # added by the model
        ("Call your bank today.", "Kpọọ ụlọ akụ gị taa. Echela ruo echi."),
    ],
)
def test_igbo_dont_wait_is_no_negation(english: str, igbo: str) -> None:
    """Round-2 live false fallback: "Ekwela ka ị chere ruo echi" (do not wait until tomorrow) asks for action."""
    assert not [p for p in translation_problems(english, igbo, "ig") if p.startswith("negation")]


def test_whole_step_left_in_english_is_untranslated() -> None:
    translated = YORUBA.replace("Ẹ pe báǹkì yín.", "Call your bank on the number on your card.")
    english = ENGLISH.replace("Call your bank.", "Call your bank on the number on your card.")
    assert "untranslated step 2" in translation_problems(english, translated, "yo")


def test_echoed_or_invented_tags_are_removed() -> None:
    text = "<ìtumọ̀>\nẸ má sanwó.\n</ translation>\n</reply>"
    assert strip_reply_tags(text).strip() == "Ẹ má sanwó."
    assert strip_reply_tags("<ìtumọ̀>Ẹ má sanwó.</ìtumọ̀>") == "Ẹ má sanwó."  # inline tags too
    assert strip_reply_tags("Ẹ ní <5k o") == "Ẹ ní <5k o"  # not a tag


def test_problem_kinds() -> None:
    problems = {
        "empty": "empty",
        "negation lost in step 2": "negation_lost",
        "negation added in intro": "negation_added",
        "length ratio 4": "length",
        "new numbers ['5']": "new_numbers",
        "new link or number": "new_link",
        "steps 2 -> 3": "steps",
        "untranslated step 1": "untranslated",
    }
    assert {problem: problem_kind(problem) for problem in problems} == problems
    assert {"empty", "length", "untranslated"} == RETRYABLE_PROBLEMS


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


async def test_formatting_failure_is_retried_warmer(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(reply_translate=[ENGLISH.replace("This is", "Eyi ni"), YORUBA])  # an untranslated echo
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    translations = llm.calls(Purpose.REPLY_TRANSLATE)
    assert [t.temperature for t in translations] == [0.0, RETRY_TEMPERATURE]
    assert {t.seed for t in translations} == {DEFAULT_SEED}
    assert answer.outcome is Outcome.ANSWERED and answer.body == YORUBA
    assert answer.translation_rejections == ("untranslated",)


# A content failure goes straight to fixed text: the warm retry is where "don't forget to steal money" came from.
@pytest.mark.parametrize(
    ("translation", "reasons"),
    [
        ("Ẹ̀tàn ni èyí. Báǹkì yín kò ní béèrè OTP yín láé. Ẹ má fi kóòdù náà fún ẹnikẹ́ni.", ("steps",)),
        (YORUBA.replace("Ẹ má fi kóòdù náà fún ẹnikẹ́ni.", "Ẹ fi kóòdù náà fún wọn."), ("negation_lost",)),
        (YORUBA.replace("Ẹ̀tàn ni èyí.", "Kò sí ewu."), ("guard",)),  # a Yoruba safety claim: the second net
    ],
)
async def test_content_failure_is_not_retried(
    registry: LanguageRegistry, knowledge: KnowledgeBase, translation: str, reasons: tuple[str, ...]
) -> None:
    llm = scam_llm(reply_translate=[translation, YORUBA])
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    assert len(llm.calls(Purpose.REPLY_TRANSLATE)) == 1
    assert answer.outcome is Outcome.FALLBACK and answer.body == Messages(registry).text(
        "yo", Msg.FALLBACK_SCAM
    )
    assert answer.translation_rejections == reasons and answer.reply_english == ENGLISH


@pytest.mark.parametrize("failure", ["Ẹ̀tàn ni.", LLMError("timeout")])
async def test_failed_translation_gives_fixed_text_in_the_users_language(
    registry: LanguageRegistry, knowledge: KnowledgeBase, failure: object
) -> None:
    llm = scam_llm(reply_translate=failure)
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    assert answer.outcome is Outcome.FALLBACK and answer.verdict is Verdict.LIKELY_SCAM
    assert answer.body == Messages(registry).text("yo", Msg.FALLBACK_SCAM)
    assert answer.contacts  # report channels still shown
    assert len(llm.calls(Purpose.REPLY_TRANSLATE)) == 1


async def test_formatting_failure_twice_gives_fixed_text(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    too_long = YORUBA.replace("Ẹ̀tàn ni èyí.", "Ẹ̀tàn ni èyí. " + "Ẹ ṣọ́ra gidigidi. " * 20)
    llm = scam_llm(reply_translate=too_long)  # too long, twice
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    assert len(llm.calls(Purpose.REPLY_TRANSLATE)) == 2 and answer.outcome is Outcome.FALLBACK
    assert answer.translation_rejections == ("length", "length")


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
    assert answer.translation_rejections == ("no_time",)
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


# --- workstream B's checks see the English draft --------------------------------------------------------------


class RecordingClassifier:
    """UnsafeAdviceClassifier stand-in for shadow mode: records what it scores."""

    budget_s = 1.0

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def supports(self, language: str) -> bool:
        return True

    async def check(self, text: str, language: str, *, wait: bool = True) -> SimpleNamespace:
        self.calls.append((text, language))
        return SimpleNamespace(flagged=False, probability=0.1, ms=1, scored=1, unscored=0)


class RecordingGrounding:
    budget_s = 1.0

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def supports(self, language: str) -> bool:
        return language == "en"

    async def check(self, reply: str, language: str, evidence: Sequence[str], **_: object) -> GroundingReport:
        self.calls.append((reply, language))
        return GroundingReport(reply, ())


async def test_shadow_checks_see_the_english_draft(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    classifier, grounding = RecordingClassifier(), RecordingGrounding()
    engine = Engine(registry=registry, llm=scam_llm(), knowledge=knowledge, classifier=classifier,  # type: ignore[arg-type]
                    grounding=grounding)  # type: ignore[arg-type]  # fmt: skip
    answer = await engine.check(YO_OTP, "yo", turn_id="t1")
    await engine.aclose()
    assert answer.body == YORUBA
    assert classifier.calls == [(ENGLISH, "en")] and grounding.calls == [(ENGLISH, "en")]


async def test_shadow_checks_see_a_direct_reply_in_its_language(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    classifier, grounding = RecordingClassifier(), RecordingGrounding()
    engine = Engine(registry=registry, llm=scam_llm(compose=YORUBA), knowledge=knowledge, classifier=classifier,  # type: ignore[arg-type]
                    grounding=grounding, compose_policy=ComposePolicy.parse("direct"))  # type: ignore[arg-type]  # fmt: skip
    await engine.check(YO_OTP, "yo")
    await engine.aclose()
    assert classifier.calls == [(YORUBA, "yo")] and grounding.calls == []  # grounding: English only


async def test_enforce_classifier_blocks_the_english_before_translation(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    model = UnsafeAdviceClassifier(hand_model(), BagOfWordsEmbedder(), OutputGuard(registry))
    slips = "This looks like a scam.\nDrop the code for them so we can check it."  # passes the regex guard
    llm = scam_llm(compose=slips)
    engine = Engine(registry=registry, llm=llm, knowledge=knowledge, classifier=model,
                    classifier_mode=ClassifierMode.ENFORCE)  # fmt: skip
    answer = await engine.check(YO_OTP, "yo")
    assert OutputIssue.CLASSIFIED_UNSAFE in answer.issues and answer.outcome is Outcome.FALLBACK
    assert llm.calls(Purpose.REPLY_TRANSLATE) == []  # nothing unsafe was translated
    assert answer.body == Messages(registry).text("yo", Msg.FALLBACK_SCAM)


async def test_enforce_grounding_trims_the_english_before_translation(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    invented = (
        "This looks like a scam.\n1. Do not share the code.\n2. The EFCC will refund your money within 48 hours.\n"
        "3. Change your PIN and password."
    )

    def translate(request: ChatRequest) -> str:
        english = request.messages[-1]["content"]
        assert "EFCC" not in english  # the unsupported step went before translation
        return "Èyí jọ ẹ̀tàn.\n1. Ẹ má ṣe fi kóòdù náà fún ẹnikẹ́ni.\n2. Ẹ yí PIN àti ọ̀rọ̀ aṣínà yín padà."

    llm = scam_llm(compose=invented, reply_translate=translate)
    engine = Engine(registry=registry, llm=llm, knowledge=knowledge, compose_policy=None,
                    grounding=GroundingChecker(None, BagOfWordsEmbedder(), registry),
                    grounding_mode=GroundingMode.ENFORCE)  # fmt: skip
    answer = await engine.check(YO_OTP, "yo")
    assert OutputIssue.UNGROUNDED in answer.issues and answer.grounded is False
    assert answer.outcome is Outcome.ANSWERED and "EFCC" not in answer.reply_english
    assert len(llm.calls(Purpose.REPLY_TRANSLATE)) == 1


async def test_enforce_grounding_without_steps_to_keep_gives_fixed_text(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = scam_llm(compose="The EFCC will refund your money within 48 hours.\n1. Do not share the code.")
    engine = Engine(registry=registry, llm=llm, knowledge=knowledge,
                    grounding=GroundingChecker(None, BagOfWordsEmbedder(), registry),
                    grounding_mode=GroundingMode.ENFORCE)  # fmt: skip
    answer = await engine.check(YO_OTP, "yo")
    assert answer.outcome is Outcome.FALLBACK and llm.calls(Purpose.REPLY_TRANSLATE) == []


async def test_enforce_grounding_of_a_direct_non_english_reply_gives_fixed_text(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    class Ungrounded(RecordingGrounding):
        def supports(self, language: str) -> bool:
            return True

        async def check(
            self, reply: str, language: str, evidence: Sequence[str], **_: object
        ) -> GroundingReport:
            return GroundingReport(reply, (Claim(reply, 0.1, False),))

    engine = Engine(registry=registry, llm=scam_llm(compose=YORUBA), knowledge=knowledge, grounding=Ungrounded(),  # type: ignore[arg-type]
                    grounding_mode=GroundingMode.ENFORCE, compose_policy=ComposePolicy.parse("direct"))  # fmt: skip
    answer = await engine.check(YO_OTP, "yo")
    assert answer.outcome is Outcome.FALLBACK and OutputIssue.UNGROUNDED in answer.issues


# --- follow-up buttons ------------------------------------------------------------------------------------------


async def test_next_steps_are_translated_and_checked(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    def translate(request: ChatRequest) -> str:
        english = request.messages[-1]["content"].removeprefix("<reply>\n").removesuffix("\n</reply>")
        return english  # left in English: an untranslated echo, twice

    llm = FakeLLM(reply_translate=translate)
    follow = await make(registry, knowledge, llm).next_steps(YO_OTP_EN, (Flag.ASKS_CREDENTIALS,), "yo")
    assert follow.outcome is Outcome.FALLBACK and follow.text == Messages(registry).text(
        "yo", Msg.FALLBACK_STEPS
    )
    (first, _retry) = llm.calls(Purpose.REPLY_TRANSLATE)
    assert "1. Do not share the code." in first.messages[-1]["content"]  # the knowledge base's English steps
    assert llm.calls(Purpose.STEPS) == []


async def test_next_steps_with_a_lost_negation_fall_back(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    def translate(request: ChatRequest) -> str:
        lines = request.messages[-1]["content"].splitlines()[1:-1]
        return "\n".join(f"{line[:3]}Ẹ ṣe é kíákíá." for line in lines)  # every step "do it quickly"

    llm = FakeLLM(reply_translate=translate)
    follow = await make(registry, knowledge, llm).next_steps(YO_OTP_EN, (Flag.ASKS_CREDENTIALS,), "yo")
    assert follow.outcome is Outcome.FALLBACK and len(llm.calls(Purpose.REPLY_TRANSLATE)) == 1


async def test_next_steps_translated(registry: LanguageRegistry, knowledge: KnowledgeBase) -> None:
    def translate(request: ChatRequest) -> str:
        lines = request.messages[-1]["content"].splitlines()[1:-1]
        filler = " dáadáa" * 12  # keep the length ratio of a real translation
        return "\n".join(
            f"{line[:3]}Ẹ má ṣe é rárá{filler}." if " not " in line else f"{line[:3]}Ẹ ṣe é kíákíá{filler}."
            for line in lines
        )

    llm = FakeLLM(reply_translate=translate)
    follow = await make(registry, knowledge, llm).next_steps(YO_OTP_EN, (Flag.ASKS_CREDENTIALS,), "yo")
    assert follow.outcome is Outcome.ANSWERED and follow.text.startswith("1. Ẹ má ṣe")


async def test_hausa_next_steps_stay_direct(registry: LanguageRegistry, knowledge: KnowledgeBase) -> None:
    llm = FakeLLM(steps="1. Kada ka bayar da lambar.")
    follow = await make(registry, knowledge, llm).next_steps(YO_OTP_EN, (Flag.ASKS_CREDENTIALS,), "ha")
    assert follow.outcome is Outcome.ANSWERED and llm.calls(Purpose.REPLY_TRANSLATE) == []


WARNING_EN = "Scammers call and ask for your OTP. Never share your OTP with anyone."
WARNING_YO = "Àwọn oníjìbìtì ń pè láti béèrè OTP yín. Ẹ má ṣe fún ẹnikẹ́ni ní OTP yín láé."


async def test_warning_is_written_in_english_then_translated(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(warn=WARNING_EN, reply_translate=WARNING_YO)
    follow = await make(registry, knowledge, llm).warning(YO_OTP_EN, (Flag.ASKS_CREDENTIALS,), "yo")
    assert follow.outcome is Outcome.ANSWERED and follow.text == WARNING_YO and follow.llm_ms == 200
    (warn,) = llm.calls(Purpose.WARN)
    assert "in English" in warn.messages[0]["content"]


@pytest.mark.parametrize(
    "script",
    [
        {
            "warn": WARNING_EN,
            "reply_translate": "Àwọn oníjìbìtì ń pè. Ẹ fún wọn ní OTP yín.",
        },  # negation lost
        {"warn": "It is safe to share your OTP with the caller."},  # the English guard blocks it
    ],
)
async def test_unsafe_warning_falls_back(
    registry: LanguageRegistry, knowledge: KnowledgeBase, script: dict[str, str]
) -> None:
    llm = FakeLLM(**script)  # type: ignore[arg-type]
    follow = await make(registry, knowledge, llm).warning(YO_OTP_EN, (Flag.ASKS_CREDENTIALS,), "yo")
    assert follow.outcome is Outcome.FALLBACK
    assert follow.text == Messages(registry).text("yo", Msg.WARN_FAMILY_FALLBACK)
    if "reply_translate" in script:
        assert "negation_lost" in follow.translation_rejections


async def test_next_steps_record_translation_rejections(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    llm = FakeLLM(reply_translate="Ẹ ṣe é.")  # one line for three steps
    follow = await make(registry, knowledge, llm).next_steps(YO_OTP_EN, (Flag.ASKS_CREDENTIALS,), "yo")
    assert follow.outcome is Outcome.FALLBACK and follow.translation_rejections[0] == "steps"


async def test_outcome_claim_in_a_translation_is_log_only_like_in_english(
    registry: LanguageRegistry, knowledge: KnowledgeBase
) -> None:
    """Shadow mode: an outcome claim is recorded, not blocking (mirrors ``Engine._blocks`` for English replies)."""
    english = "This looks like a scam.\n1. Do not share the code.\n2. You will get your money back."
    yoruba = "Èyí jọ ẹ̀tàn.\n1. Ẹ má ṣe fi kóòdù náà fún ẹnikẹ́ni.\n2. Ẹ ó rí owó yín gbà, you will get your money back."
    llm = scam_llm(compose=english, reply_translate=yoruba)
    answer = await make(registry, knowledge, llm).check(YO_OTP, "yo")
    assert answer.outcome is Outcome.ANSWERED and answer.body == yoruba
    assert OutputIssue.OUTCOME_CLAIM in answer.issues and answer.translation_rejections == ()
