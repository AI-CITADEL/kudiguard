"""Advice quality (docs/ARCHITECTURE.md "Advice quality"): entry model answers and the copy guard, genuine-notice
handling, Pidgin replies, the relevance cut-off and its settings."""

import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from fakes import FakeLLM
from kudiguard.api.services import advice_policy, min_relevance
from kudiguard.config import Settings
from kudiguard.domain import Flag, Verdict
from kudiguard.engine import AdvicePolicy, Engine
from kudiguard.knowledge.file_pack import FilePackSource, KnowledgeConfigError, read_pack_files
from kudiguard.knowledge.models import Entry, KnowledgeResult, Query, Relevance, SourcedEntry
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import ChatRequest, Purpose
from kudiguard.prompts import (
    EXAMPLE_LABEL,
    NO_FACTS_TASK,
    PIDGIN_HINT,
    compose_request,
    example_overlap,
    parse_analysis,
)

PACKS = Path(__file__).resolve().parents[2] / "data" / "packs"
LOGIN = "I just logged in on my new phone and my bank sent this: New login to your account on a new device."
EXAMPLE = "This looks like the alert for your own login.\n1. If it was you, there is nothing to do."
NOTICE = Entry(
    id="notice-login-alert",
    title="A login alert after you logged in",
    facts=("Banks send a login alert each time you log in on a new device.",),
    actions=("If it was not you, call your bank on the number on your card.",),
    example=EXAMPLE,
)
FLAG_ENTRY = Entry(
    id="flag-asks-credentials",
    title="Asks for a code",
    facts=("Never share a code.",),
    flags=(Flag.ASKS_CREDENTIALS,),
)
OTHER = Entry(id="saving-basics", title="Saving", facts=("Save first.",))
REPLY = (
    "You started this login yourself, so this message is expected.\n1. If that was not you, call your bank."
)


def analysis(flags: Sequence[str] = (), *, user_initiated: bool = False, intent: str = "scam_check") -> str:
    return json.dumps(
        {"english": "x", "intent": intent, "flags": list(flags), "user_initiated": user_initiated}
    )


class StubKnowledge:
    """A knowledge base that returns a fixed result (relevance on the dense scale)."""

    sources: tuple[object, ...] = ()

    def __init__(self, *items: tuple[Entry, float | None], dense: bool = True) -> None:
        self.result = KnowledgeResult(
            entries=tuple(SourcedEntry(entry, "core", "Core", relevance=rel) for entry, rel in items),
            dense=dense,
        )

    async def search(self, query: Query) -> KnowledgeResult:
        return self.result


def engine(registry: LanguageRegistry, llm: FakeLLM, knowledge: object, **advice: object) -> Engine:
    return Engine(registry=registry, llm=llm, knowledge=knowledge, advice=AdvicePolicy(**advice))  # type: ignore[arg-type]


def last_user(request: ChatRequest) -> str:
    return request.messages[-1]["content"]


# --- genuine notices -------------------------------------------------------------------------------------------

SOFT = [
    "impersonation",
    "urgency",
    "secrecy",
]  # model-only flags a real bank alert can attract (score 3: likely scam)


async def test_notice_started_by_the_user_gets_no_red_flags_and_its_own_check(
    registry: LanguageRegistry,
) -> None:
    llm = FakeLLM(analyse=analysis(SOFT, user_initiated=True), compose=REPLY)
    kb = StubKnowledge((FLAG_ENTRY, None), (NOTICE, 0.8))
    answer = await engine(registry, llm, kb).check(LOGIN, "en")
    assert answer.verdict is Verdict.NO_FLAGS and answer.flags == ()
    assert answer.sources == ("notice-login-alert",)  # the red-flag entry no longer contradicts the notice
    assert "call your bank on the number on your card" in last_user(llm.calls(Purpose.COMPOSE)[0])


async def test_notice_not_started_by_the_user_is_capped_at_caution(registry: LanguageRegistry) -> None:
    llm = FakeLLM(analyse=analysis(SOFT), compose=REPLY)
    answer = await engine(registry, llm, StubKnowledge((NOTICE, 0.8))).check(LOGIN, "en")
    assert answer.verdict is Verdict.CAUTION and Flag.URGENCY in answer.flags


# Look-alike scams that share notice words: the rules, a link/number or the model's request flags keep the verdict.
LOOK_ALIKES = [
    (
        "I got a credit alert of N50,000 and the man says he sent it by mistake, I should send back the excess.",
        SOFT,
    ),
    # first person ("my account") is outside the rules' wrong_transfer phrases: the model's flag keeps the verdict
    (
        "A woman called that she sent money to my account by mistake, she wants me to reverse my money to her.",
        ["wrong_transfer", *SOFT],
    ),
    ("Your BVN will be blocked today. Update your BVN here: bvn-update.ng/verify", SOFT),
    ("Your parcel is held at customs. Pay a customs fee of N15,000 to release your delivery.", SOFT),
    ("Your account has been restricted under the new CBN rule. Pay N5,000 to unlock it.", SOFT),
    ("Mummy it's me, I lost my phone. Send money to this new account 0123456789 urgently.", SOFT),
    ("New login to your account on a new device. If this was not you call 08031234567.", SOFT),  # number only
    (LOGIN, ["asks_credentials", *SOFT]),  # the model saw a request for a code
    (LOGIN, ["money_request"]),
]


@pytest.mark.parametrize(("text", "flags"), LOOK_ALIKES)
async def test_look_alike_scams_are_never_lowered(
    registry: LanguageRegistry, text: str, flags: list[str]
) -> None:
    def answer_with(**advice: object) -> object:
        llm = FakeLLM(analyse=analysis(flags, user_initiated=True), compose=REPLY)
        return engine(registry, llm, StubKnowledge((NOTICE, 0.95)), **advice).check(text, "en")

    with_notices, without = await answer_with(), await answer_with(notice_prefix="")  # type: ignore[misc]
    assert without.verdict in (Verdict.LIKELY_SCAM, Verdict.CAUTION)  # type: ignore[attr-defined]
    assert with_notices.verdict is without.verdict  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("items", "advice"),
    [
        (((NOTICE, 0.7),), {}),  # below the notice level
        (((OTHER, 0.9),), {}),  # not a notice
        (((NOTICE, 0.9),), {"notice_prefix": ""}),  # notice handling off
        (((FLAG_ENTRY, None),), {}),  # no search hit at all
    ],
)
async def test_notice_handling_needs_a_close_notice_hit(
    registry: LanguageRegistry, items: tuple[tuple[Entry, float | None], ...], advice: dict[str, object]
) -> None:
    llm = FakeLLM(analyse=analysis(SOFT, user_initiated=True), compose=REPLY)
    answer = await engine(registry, llm, StubKnowledge(*items), **advice).check(LOGIN, "en")
    assert answer.verdict is Verdict.LIKELY_SCAM


async def test_notice_handling_leaves_questions_alone(registry: LanguageRegistry) -> None:
    llm = FakeLLM(analyse=analysis(intent="question"), compose=REPLY)
    answer = await engine(registry, llm, StubKnowledge((NOTICE, 0.9))).check("What is a login alert?", "en")
    assert answer.verdict is Verdict.NOT_A_CHECK


async def test_example_only_from_a_top_hit_at_the_example_level(registry: LanguageRegistry) -> None:
    for items in (((NOTICE, 0.65),), ((NOTICE.model_copy(update={"flags": (Flag.URGENCY,)}), None),)):
        llm = FakeLLM(analyse=analysis(intent="question"), compose=REPLY)
        await engine(registry, llm, StubKnowledge(*items)).check("What is a login alert?", "en")
        assert "<model_answer>" not in last_user(llm.calls(Purpose.COMPOSE)[0])


# --- entry model answers and the copy guard ---------------------------------------------------------------------


async def test_top_entry_example_is_shown_labelled_instead_of_the_fixed_example(
    registry: LanguageRegistry,
) -> None:
    llm = FakeLLM(analyse=analysis(intent="question"), compose=REPLY)
    await engine(registry, llm, StubKnowledge((NOTICE, 0.9))).check("What is a login alert?", "en")
    (request,) = llm.calls(Purpose.COMPOSE)
    assert len(request.messages) == 2  # system + this message: no fixed worked example
    assert EXAMPLE_LABEL in last_user(request) and "<model_answer>" in last_user(request)


async def test_examples_can_be_switched_off(registry: LanguageRegistry) -> None:
    llm = FakeLLM(analyse=analysis(intent="question"), compose=REPLY)
    await engine(registry, llm, StubKnowledge((NOTICE, 0.9)), examples=False).check(
        "What is a login alert?", "en"
    )
    assert "<model_answer>" not in last_user(llm.calls(Purpose.COMPOSE)[0])


async def test_a_copied_example_is_composed_again_without_it(registry: LanguageRegistry) -> None:
    llm = FakeLLM(analyse=analysis(intent="question"), compose=[EXAMPLE, REPLY])
    answer = await engine(registry, llm, StubKnowledge((NOTICE, 0.9))).check("What is a login alert?", "en")
    first, second = llm.calls(Purpose.COMPOSE)
    assert "<model_answer>" in last_user(first) and "<model_answer>" not in last_user(second)
    assert answer.body == REPLY and answer.llm_ms == 300


async def test_copy_guard_in_english_first_checks_the_english_draft(registry: LanguageRegistry) -> None:
    llm = FakeLLM(analyse=analysis(intent="question"), compose=[EXAMPLE, REPLY], translate="x")
    llm.script[Purpose.REPLY_TRANSLATE] = ["Ọ bụ ozi maka nbanye gị.\n1. Ọ bụrụ na ọ bụghị gị, kpọọ bank gị."]
    await engine(registry, llm, StubKnowledge((NOTICE, 0.9))).check("Kedu ihe bụ ozi nbanye?", "ig")
    assert len(llm.calls(Purpose.COMPOSE)) == 2


def test_example_overlap() -> None:
    assert example_overlap(EXAMPLE, EXAMPLE) == 1.0
    assert example_overlap("Call your bank.", EXAMPLE) == 0.0
    assert example_overlap("ok", EXAMPLE) == 0.0  # too short for a trigram
    assert example_overlap(EXAMPLE, EXAMPLE, grounded=EXAMPLE) == 0.0  # wording from the facts is not a copy


# --- Pidgin replies ------------------------------------------------------------------------------------------


PIDGIN = "Abeg, dem call me say make I send dem the code wey bank send me, wetin I go do?"


@pytest.mark.parametrize(("enabled", "name"), [(True, "Nigerian Pidgin English"), (False, "English")])
async def test_pidgin_input_gets_a_pidgin_reply(registry: LanguageRegistry, enabled: bool, name: str) -> None:
    llm = FakeLLM(analyse=analysis(), compose=REPLY)
    await engine(registry, llm, StubKnowledge((OTHER, 0.9)), pidgin_replies=enabled).check(PIDGIN, "en")
    request = last_user(llm.calls(Purpose.COMPOSE)[0])
    assert request.startswith(f"Reply in {name}.") and (PIDGIN_HINT in request) is enabled


# --- no relevant knowledge, prompts ----------------------------------------------------------------------------


def test_question_without_facts_gets_honest_guidance_and_a_clarifying_question() -> None:
    request = compose_request(
        language_name="English", english="my card was stolen", verdict=Verdict.NOT_A_CHECK, flags=(),
        knowledge=KnowledgeResult(),
    )  # fmt: skip
    text = " ".join(m["content"] for m in request.messages)
    assert NO_FACTS_TASK in last_user(request) and "*346#" not in text


def test_analysis_reads_user_initiated() -> None:
    parsed = parse_analysis(analysis(user_initiated=True))
    assert parsed is not None and parsed.user_initiated
    parsed = parse_analysis(json.dumps({"english": "x", "intent": "other", "flags": []}))
    assert parsed is not None and not parsed.user_initiated


# --- relevance cut-off ---------------------------------------------------------------------------------------


async def test_cut_off_drops_weak_hits_and_keeps_red_flag_entries() -> None:
    async with httpx.AsyncClient() as client:
        kb = knowledge_base_from_packs(
            PACKS, client=client, include=["kudiguard-core"], min_relevance=Relevance(dense=0.62, bm25=0.60)
        )
        weak = await kb.search(Query(text="my card was stolen at the football match"))
        strong = await kb.search(Query(text="How do I check my NIN?"))
        flagged = await kb.search(Query(text="unrelated words", flags=(Flag.ASKS_CREDENTIALS,)))
    assert all(item.relevance is not None and item.relevance >= 0.60 for item in weak.entries)
    assert "nin-basics" not in [item.entry.id for item in weak.entries]
    assert strong.entries[0].entry.id == "nin-basics" and not strong.dense
    assert flagged.entries and flagged.entries[0].entry.flags


def test_relevance_levels() -> None:
    level = Relevance(dense=0.6, bm25=0.5)
    assert level.met(0.6, dense=True) and not level.met(0.55, dense=True) and level.met(0.55, dense=False)
    assert not level.met(None, dense=True)


def test_settings_build_the_policy_and_cut_off() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert min_relevance(settings) == Relevance(dense=0.62, bm25=0.60)
    assert advice_policy(settings) == AdvicePolicy()
    off = Settings(_env_file=None, knowledge_min_relevance_dense=0, knowledge_min_relevance_bm25=0)  # type: ignore[call-arg]
    assert min_relevance(off) is None


# --- examples.yaml -----------------------------------------------------------------------------------------------


def _pack(tmp_path: Path, examples: str | None, inline: str = "") -> Path:
    (tmp_path / "pack.yaml").write_text("id: p\nname: P\nowner: KudiGuard\nkind: core\n", encoding="utf-8")
    (tmp_path / "topics.yaml").write_text(
        f"- id: notice-otp\n  title: Your own OTP\n  facts: [You asked for it.]\n{inline}"
        "- id: other\n  title: Other\n  facts: [Other.]\n",
        encoding="utf-8",
    )
    if examples is not None:
        (tmp_path / "examples.yaml").write_text(examples, encoding="utf-8")
    return tmp_path


def test_examples_file_attaches_model_answers(tmp_path: Path) -> None:
    entries, _ = read_pack_files(_pack(tmp_path, "notice-otp: Use it only on the page you opened.\n"))
    assert [e.example for e in entries] == ["Use it only on the page you opened.", None]
    assert FilePackSource.load(tmp_path).entries[0].example


def test_inline_example(tmp_path: Path) -> None:
    entries, _ = read_pack_files(_pack(tmp_path, None, "  example: Inline answer.\n"))
    assert entries[0].example == "Inline answer."
    assert read_pack_files(_pack(tmp_path, ""))[0][1].example is None  # an empty examples.yaml


@pytest.mark.parametrize(
    ("examples", "inline", "error"),
    [
        ("missing: text\n", "", "unknown entry ids"),
        ("notice-otp: text\n", "  example: Inline.\n", "already have an inline example"),
        ("- a list\n", "", "examples.yaml"),
        (f"notice-otp: {'x' * 1001}\n", "", "examples.yaml"),
    ],
)
def test_examples_file_errors(tmp_path: Path, examples: str, inline: str, error: str) -> None:
    with pytest.raises(KnowledgeConfigError, match=error):
        read_pack_files(_pack(tmp_path, examples, inline))


async def test_a_copied_fixed_worked_example_is_composed_again_without_any_example(
    registry: LanguageRegistry,
) -> None:
    from kudiguard.prompts import worked_example_reply

    copied = worked_example_reply(
        Verdict.CAUTION, (), KnowledgeResult(entries=StubKnowledge((OTHER, 0.9)).result.entries), "English"
    )
    llm = FakeLLM(analyse=analysis(["upfront_fee"]), compose=[copied, REPLY])
    answer = await engine(registry, llm, StubKnowledge((OTHER, 0.9))).check(
        "They said I must pay first.", "en"
    )
    first, second = llm.calls(Purpose.COMPOSE)
    assert len(first.messages) == 4 and len(second.messages) == 2  # the retry has no worked example
    assert answer.body == REPLY


# --- rules: a credential REQUEST, not a bank's own warning; first-person wrong transfers -------------------------


@pytest.mark.parametrize(
    ("text", "flag", "expected"),
    [
        ("Your OTP is 123456. Do not share it with anyone.", Flag.ASKS_CREDENTIALS, False),
        (
            "Your OTP is 482913. Don't share it with anyone, including bank staff.",
            Flag.ASKS_CREDENTIALS,
            False,
        ),
        ("Never give your PIN or OTP to anyone.", Flag.ASKS_CREDENTIALS, False),
        ("Na your code be this, no go send am give anybody.", Flag.ASKS_CREDENTIALS, False),
        ("send me the OTP", Flag.ASKS_CREDENTIALS, True),
        ("read me the code that came to your phone", Flag.ASKS_CREDENTIALS, True),
        ("share your PIN to unlock your account", Flag.ASKS_CREDENTIALS, True),
        ("my account will be blocked if I no send the OTP now", Flag.ASKS_CREDENTIALS, True),
        (
            "She sent money to my account by mistake and wants me to reverse my money to her.",
            Flag.WRONG_TRANSFER,
            True,
        ),
        (
            "He said I got an alert of 50k by mistake and I should send back the excess.",
            Flag.WRONG_TRANSFER,
            True,
        ),
        ("My transfer failed and the bank reversed my money.", Flag.WRONG_TRANSFER, False),
    ],
)
def test_rule_requests(registry: LanguageRegistry, text: str, flag: Flag, expected: bool) -> None:
    from kudiguard.safety.scam_rules import assess

    assert (flag in assess([text], registry.lexicon_for("en")).flags) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("what is my bank's customer care number?", False),
        ("How do I reach GTBank customer care?", False),
        ("This is GTBank customer care, your account will be blocked", True),
        ("A man called me, he said he is from customer care and needs my OTP", True),
        ("Customer care called me and asked for my PIN", True),
        ("one man call me say him be customer care", True),
    ],
)
def test_impersonation_needs_a_claim(registry: LanguageRegistry, text: str, expected: bool) -> None:
    from kudiguard.safety.scam_rules import assess

    assert (Flag.IMPERSONATION in assess([text], registry.lexicon_for("en")).flags) is expected


# --- situational packs in general search ---------------------------------------------------------------------------


def _two_packs(tmp_path: Path) -> list[FilePackSource]:
    core, situational = tmp_path / "core", tmp_path / "sit"
    for folder, extra, topics in (
        (
            core,
            "kind: core\npriority: 10\n",
            "- id: saving\n  title: Saving money every month\n  facts: [Save a little money every month.]\n",
        ),
        (
            situational,
            "kind: core\npriority: 10\nuncued_weight: 0.5\n",
            "- id: sit-fee\n  title: Monthly account fee on your money\n  facts: [Banks take a small fee from your money every month.]\n"
            "  keywords: [maintenance fee]\n",
        ),
    ):
        folder.mkdir()
        (folder / "pack.yaml").write_text(
            f"id: {folder.name}\nname: {folder.name}\nowner: KudiGuard\n{extra}", encoding="utf-8"
        )
        (folder / "topics.yaml").write_text(topics, encoding="utf-8")
    return [FilePackSource.load(core), FilePackSource.load(situational)]


@pytest.mark.parametrize(
    ("text", "flags", "top"),
    [
        ("money every month fee", (), "saving"),  # no cue: the situational hit is down-weighted
        ("why is there a maintenance fee on my money every month", (), "sit-fee"),  # its keyword is a cue
        ("money every month fee", (Flag.UPFRONT_FEE,), "sit-fee"),  # a red flag: no down-weighting
    ],
)
async def test_situational_hits_need_a_cue_in_general_search(
    tmp_path: Path, text: str, flags: tuple[Flag, ...], top: str
) -> None:
    from kudiguard.knowledge.service import KnowledgeBase

    kb = KnowledgeBase(_two_packs(tmp_path))
    result = await kb.search(Query(text=text, flags=flags))
    assert next(item.entry.id for item in result.entries if item.relevance is not None) == top


# --- Pidgin: worked example and the wrong-language retry -----------------------------------------------------------


async def test_pidgin_scam_gets_the_pidgin_worked_example(registry: LanguageRegistry) -> None:
    llm = FakeLLM(analyse=analysis(["upfront_fee"]), compose=REPLY)
    await engine(registry, llm, StubKnowledge((OTHER, 0.9))).check(PIDGIN, "en")
    assert "Real offer no dey ask you to pay first" in llm.calls(Purpose.COMPOSE)[0].messages[2]["content"]


async def test_pidgin_input_with_a_hausa_reply_is_retried_in_english(registry: LanguageRegistry) -> None:
    hausa = "Wannan zamba ne. Kada ka ba su lambar."
    llm = FakeLLM(analyse=analysis(), compose=[hausa, REPLY])
    answer = await engine(registry, llm, StubKnowledge((OTHER, 0.9))).check(PIDGIN, "en")
    first, second = llm.calls(Purpose.COMPOSE)
    assert last_user(first).startswith("Reply in Nigerian Pidgin") and last_user(second).startswith(
        "Reply in English."
    )
    assert answer.body == REPLY and answer.llm_ms == 300


def test_reads_as_hausa_yoruba_igbo() -> None:
    from kudiguard.prompts import reads_as_hausa_yoruba_igbo

    assert reads_as_hausa_yoruba_igbo("Wannan naa jẹ aṣiwere. Banki rẹ ko beere OTP")
    assert not reads_as_hausa_yoruba_igbo("Na scam be this. Your bank no go ever ask you for your OTP.")


async def test_a_copied_pidgin_worked_example_is_composed_again(registry: LanguageRegistry) -> None:
    from kudiguard.prompts import PIDGIN_NAME, worked_example_reply

    knowledge = StubKnowledge((OTHER, 0.9))
    copied = worked_example_reply(Verdict.LIKELY_SCAM, (), knowledge.result, PIDGIN_NAME)
    assert "no dey ask" in copied
    llm = FakeLLM(analyse=analysis(["upfront_fee"]), compose=[copied, REPLY])
    answer = await engine(registry, llm, knowledge).check(PIDGIN, "en")
    assert len(llm.calls(Purpose.COMPOSE)) == 2 and answer.body == REPLY


async def test_at_most_one_extra_compose_for_pidgin(registry: LanguageRegistry) -> None:
    """The English retry for Pidgin input has no example at all (nothing to copy), and no copy retry follows."""
    llm = FakeLLM(analyse=analysis(), compose=["Wannan zamba ne. Kada ka ba su lambar.", REPLY, "unused"])
    answer = await engine(registry, llm, StubKnowledge((OTHER, 0.9))).check(PIDGIN, "en")
    _first, second = llm.calls(Purpose.COMPOSE)
    assert len(second.messages) == 2 and "<model_answer>" not in last_user(second)
    assert answer.body == REPLY
