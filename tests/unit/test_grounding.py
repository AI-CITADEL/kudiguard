"""Grounding check: claim splitting, evidence, scoring and the checker (fake embedder; real model: -m live)."""

import asyncio
import logging
import time

import pytest

from fakes import BagOfWordsEmbedder, FakeLLM
from kudiguard.domain import Flag, LanguageCode, Verdict
from kudiguard.knowledge.embeddings import CachedEmbedder
from kudiguard.knowledge.models import Entry, KnowledgeResult, PackKind, SourcedEntry
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import LLMError, Purpose
from kudiguard.prompts import FLAG_TEXT
from kudiguard.safety.grounding import (
    DEFAULT_THRESHOLD,
    GENERIC_FRAMING,
    VERDICT_LINE,
    Claim,
    GroundingChecker,
    GroundingError,
    GroundingReport,
    LLMTranslator,
    evidence_for,
    evidence_units,
    is_trivial,
    names_in,
    numbers_in,
    score_claims,
    split_claims,
)

FACTS = (
    "Your bank will never ask for your OTP, PIN or password. Anyone who asks is trying to steal your money.",
    "An OTP is like the key to your account.",
)
ACTIONS = ("Do not share the code. End the call or chat.", "Change your PIN and password.")


def knowledge(*entries: Entry, kind: PackKind = PackKind.CORE) -> KnowledgeResult:
    return KnowledgeResult(entries=tuple(SourcedEntry(e, "pack", "Demo Bank", kind=kind) for e in entries))


OTP = knowledge(Entry(id="otp", title="OTP", facts=FACTS, actions=ACTIONS))


# --- claim splitting ------------------------------------------------------------------------------------------


def test_split_claims_handles_steps_bullets_emoji_and_markup() -> None:
    reply = (
        "🔴 *This looks like a scam.* Your bank will never ask for your OTP.\n"
        "\n"
        "What to do:\n"
        "1. Do not share the code.\n"
        "2) End the call or chat.\n"
        "Step 3: Change your PIN and password.\n"
        "- Tell someone you trust before you act.\n"
        "• ➡️ _Check your balance in the app._\n"
        "**4.** Report it to your bank today!"
    )
    assert split_claims(reply) == [
        "This looks like a scam.",
        "Your bank will never ask for your OTP.",
        "Do not share the code.",
        "End the call or chat.",
        "Change your PIN and password.",
        "Tell someone you trust before you act.",
        "Check your balance in the app.",
        "Report it to your bank today!",
    ]


def test_split_claims_skips_filler_and_lead_ins() -> None:
    assert split_claims("Stay calm. Be careful! Okay.\nHere is what you can do:\n") == []
    assert split_claims("") == [] and split_claims("  \n ") == []
    assert (
        is_trivial("Don't worry.")
        and is_trivial("Don\u2019t worry.")
        and is_trivial("Next steps:")
        and not is_trivial("Do not pay.")
    )


def test_split_claims_keeps_numbers_inside_sentences() -> None:
    assert split_claims("Dial *346# from your phone. Wait 2.5 hours, then call.") == [
        "Dial *346# from your phone.",
        "Wait 2.5 hours, then call.",
    ]
    assert split_claims("12. Keep your receipt safe.") == ["Keep your receipt safe."]


def test_numbers_in_reads_digits_and_english_number_words() -> None:
    assert numbers_in("Pay N5,000 within 24 hours or three days") == {"5000", "24", "3"}
    assert numbers_in("Dial *565*0# now") == {"565", "0"}
    assert numbers_in("No one will call, not one person") == set()  # "one" is not a quantity here
    assert numbers_in("a hundred thousand naira") == {"100", "1000"}


def test_evidence_units_add_single_sentences_without_duplicates() -> None:
    units = evidence_units(["Do not pay. Check the offer.", "Do not pay.", "  ", "1. Call your bank."])
    assert units == ["Do not pay. Check the offer.", "Do not pay.", "Check the offer.", "1. Call your bank.", "Call your bank."]  # fmt: skip


# --- evidence ---------------------------------------------------------------------------------------------------


def test_evidence_for_is_what_compose_was_shown_plus_verdict_flags_and_message() -> None:
    evidence = evidence_for(
        OTP, verdict=Verdict.LIKELY_SCAM, flags=(Flag.ASKS_CREDENTIALS,), message="  They asked for my OTP.  "
    )
    assert evidence == [
        *FACTS,
        *ACTIONS,
        VERDICT_LINE[Verdict.LIKELY_SCAM],
        f"The message {FLAG_TEXT[Flag.ASKS_CREDENTIALS]}.",
        "They asked for my OTP.",
    ]


def test_evidence_for_without_verdict_line_or_message() -> None:
    assert evidence_for(OTP, verdict=Verdict.NOT_A_CHECK, message=" ") == [*FACTS, *ACTIONS]
    assert evidence_for(OTP) == [*FACTS, *ACTIONS]
    assert evidence_for(KnowledgeResult()) == []


def test_evidence_for_keeps_partner_citations() -> None:
    partner = knowledge(Entry(id="p", title="P", facts=("Call 0700 for help.",)), kind=PackKind.PARTNER)
    assert evidence_for(partner) == ["Call 0700 for help. (source: Demo Bank)"]


# --- scoring ----------------------------------------------------------------------------------------------------


def test_score_claims_supports_restated_sentences_and_flags_inventions() -> None:
    claims = ["Do not share the code.", "The police will refund you quickly."]
    scored = score_claims(claims, evidence_for(OTP), BagOfWordsEmbedder())
    assert scored[0] == Claim("Do not share the code.", 1.0, True)
    assert scored[1].supported is False and scored[1].support < DEFAULT_THRESHOLD
    assert scored[1].unmatched == ()


def test_numbers_must_appear_in_the_evidence() -> None:
    evidence = ["Most failed transfers are reversed within 24 hours."]
    scored = score_claims(
        ["Most failed transfers are reversed within 48 hours.", "Most failed transfers are reversed within 24 hours."],
        evidence,
        BagOfWordsEmbedder(),
    )  # fmt: skip
    assert scored[0].supported is False and scored[0].unmatched == ("48",)
    assert scored[0].support >= DEFAULT_THRESHOLD  # same words: only the number gives it away
    assert scored[1].supported is True


def test_score_claims_edge_cases() -> None:
    embedder = BagOfWordsEmbedder()
    assert score_claims([], ["anything"], embedder) == ()
    assert embedder.batches == []  # nothing embedded
    (claim,) = score_claims(["Do not share the code."], [" "], embedder)
    assert claim == Claim("Do not share the code.", 0.0, False)


def test_threshold_is_configurable() -> None:
    claims = ["Do not share the code with anyone."]
    loose, strict = (score_claims(claims, ACTIONS, BagOfWordsEmbedder(), threshold=t)[0] for t in (0.5, 0.99))
    assert loose.supported and not strict.supported and loose.support == strict.support


def test_report_properties() -> None:
    good, bad = Claim("a b c", 0.9, True), Claim("d e f", 0.2, False)
    assert GroundingReport("x", (good, bad)).unsupported == ("d e f",)
    assert not GroundingReport("x", (good, bad)).grounded
    assert (
        GroundingReport("x", (good,)).grounded and GroundingReport("", ()).grounded
    )  # nothing to contradict


# --- checker ----------------------------------------------------------------------------------------------------


def checker(registry: LanguageRegistry, **kwargs: object) -> GroundingChecker:
    return GroundingChecker(None, BagOfWordsEmbedder(), registry, **kwargs)  # type: ignore[arg-type]


async def test_english_reply_is_checked_without_a_model(registry: LanguageRegistry) -> None:
    grounding = checker(registry)
    reply = "This looks like a scam.\n1. Do not share the code.\n2. Change your PIN and password."
    report = await grounding.check(reply, "en", evidence_for(OTP, verdict=Verdict.LIKELY_SCAM))
    assert report.english == reply and report.grounded and len(report.claims) == 3
    assert report.ms >= 0 and grounding.supports("en") and not grounding.supports("ha")


async def test_invented_claim_is_reported(registry: LanguageRegistry) -> None:
    reply = "Do not share the code.\nYour bank will refund all your money within 24 hours."
    report = await checker(registry).check(reply, "en", evidence_for(OTP))
    assert not report.grounded
    assert report.unsupported == ("Your bank will refund all your money within 24 hours.",)


async def test_generic_framing_and_fixed_advice_are_always_evidence(registry: LanguageRegistry) -> None:
    grounding = checker(registry)
    assert set(GENERIC_FRAMING) <= set(grounding.framing)
    assert any("number printed on your card" in text for text in grounding.framing)  # en fallback_scam
    reply = f"{GENERIC_FRAMING[0]} Call your bank on the number printed on your card."
    assert (await grounding.check(reply, "en", [])).grounded


def test_invalid_threshold(registry: LanguageRegistry) -> None:
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match="threshold"):
            checker(registry, threshold=bad)


def test_a_cached_embedder_is_not_wrapped_twice(registry: LanguageRegistry) -> None:
    cached = CachedEmbedder(BagOfWordsEmbedder())
    assert GroundingChecker(None, cached, registry)._embedder is cached
    assert isinstance(checker(registry)._embedder, CachedEmbedder)


async def test_evidence_vectors_are_reused_across_turns(registry: LanguageRegistry) -> None:
    embedder = BagOfWordsEmbedder()
    grounding = GroundingChecker(None, embedder, registry)
    evidence = evidence_for(OTP)
    await grounding.check("Do not share the code.", "en", evidence)
    embedder.batches.clear()
    await grounding.check("Change your PIN and password now.", "en", evidence)
    assert embedder.batches == [["Change your PIN and password now."]]  # evidence came from the cache


async def test_other_languages_need_a_translator(registry: LanguageRegistry) -> None:
    with pytest.raises(GroundingError, match="only English"):
        await checker(registry).check("Kada ku ba su lambar.", "ha", evidence_for(OTP))


async def test_back_translation_with_the_model(registry: LanguageRegistry) -> None:
    llm = FakeLLM(translate="KudiGuard: Do not share the code.\n<|eot_id|>")
    grounding = GroundingChecker(llm, BagOfWordsEmbedder(), registry)
    assert grounding.supports("ha") and grounding.supports("yo")
    report = await grounding.check("Kada ku ba su lambar.", "ha", evidence_for(OTP))
    assert report.english == "Do not share the code." and report.grounded
    (request,) = llm.calls(Purpose.TRANSLATE)
    assert (
        "Hausa" in request.messages[0]["content"]
        and "Kada ku ba su lambar." in request.messages[1]["content"]
    )


async def test_injected_translator_wins_over_the_model(registry: LanguageRegistry) -> None:
    class Dictionary:
        async def to_english(
            self, text: str, language: LanguageCode, *, deadline: float | None = None
        ) -> str:
            return "Do not share the code."

    llm = FakeLLM()
    grounding = GroundingChecker(llm, BagOfWordsEmbedder(), registry, translator=Dictionary())
    assert (await grounding.check("Kar ka bayar.", "ha", evidence_for(OTP))).grounded
    assert llm.requests == []


@pytest.mark.parametrize(
    ("reply", "match"), [(LLMError("timeout"), "back-translation failed"), ("<|eot_id|>  ", "empty")]
)
async def test_back_translation_failures(registry: LanguageRegistry, reply: object, match: str) -> None:
    translator = LLMTranslator(FakeLLM(translate=reply), registry)  # type: ignore[arg-type]
    with pytest.raises(GroundingError, match=match):
        await translator.to_english("Kada ku ba su lambar.", "ha")


async def test_back_translation_respects_the_deadline(registry: LanguageRegistry) -> None:
    llm = FakeLLM(translate="Do not share the code.")
    translator = LLMTranslator(llm, registry)
    with pytest.raises(GroundingError, match="skipped"):
        await translator.to_english("Kada ku ba su lambar.", "ha", deadline=time.monotonic() + 0.5)
    assert llm.requests == []  # never started
    await translator.to_english("Kada ku ba su lambar.", "ha", deadline=time.monotonic() + 30)
    assert llm.requests[0].timeout_s is not None and llm.requests[0].timeout_s <= 30


async def test_scoring_out_of_time_is_an_error(registry: LanguageRegistry) -> None:
    class Slow(BagOfWordsEmbedder):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            time.sleep(0.2)
            return super().embed_documents(texts)

    grounding = GroundingChecker(None, Slow(), registry)
    with pytest.raises(GroundingError, match="out of time"):
        await grounding.check("Do not share the code.", "en", ACTIONS, deadline=time.monotonic() + 0.05)
    with pytest.raises(GroundingError, match="out of time"):
        await grounding.check("Do not share the code.", "en", ACTIONS, deadline=time.monotonic() - 1)


async def test_check_logs_counts_and_timing_not_text(
    registry: LanguageRegistry, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="kudiguard.safety.grounding"):
        await checker(registry).check("Send the money to my cousin today.", "en", evidence_for(OTP))
    (message,) = [r.getMessage() for r in caplog.records]
    assert "1 claims, 1 unsupported" in message and " ms" in message and "cousin" not in message


async def test_concurrent_checks_share_the_cache(registry: LanguageRegistry) -> None:
    grounding = checker(registry)
    replies = [f"Do not share the code. Change your PIN and password {n} times." for n in range(8)]
    reports = await asyncio.gather(*(grounding.check(r, "en", evidence_for(OTP)) for r in replies))
    assert all(not r.grounded for r in reports)  # the number of times is invented
    assert all(r.claims[0].supported for r in reports)


async def test_time_budget_caps_the_check(registry: LanguageRegistry) -> None:
    class Slow(BagOfWordsEmbedder):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            time.sleep(0.2)
            return super().embed_documents(texts)

    grounding = GroundingChecker(None, Slow(), registry, budget_s=0.05)
    start = time.perf_counter()
    with pytest.raises(GroundingError, match="out of time"):
        await grounding.check("Do not share the code.", "en", ACTIONS, deadline=time.monotonic() + 60)
    assert time.perf_counter() - start < 0.15  # returned at the budget, not when the embedding finished
    with pytest.raises(ValueError, match="budget_s"):
        checker(registry, budget_s=0)


async def test_warm_embeds_evidence_ahead_of_the_turn(registry: LanguageRegistry) -> None:
    embedder = BagOfWordsEmbedder()
    grounding = GroundingChecker(None, embedder, registry)
    count = grounding.warm([*FACTS, *ACTIONS])
    assert count == len(embedder.batches[0]) and "Do not share the code." in embedder.batches[0]
    embedder.batches.clear()
    await grounding.check("Do not share the code.", "en", [*FACTS, *ACTIONS])
    assert embedder.batches == []  # reply sentence was warmed too (it is a fact); nothing left to embed
    await grounding.check("Change your PIN today.", "en", [*FACTS, *ACTIONS])
    assert embedder.batches == [["Change your PIN today."]]  # one batch: only the reply


def test_names_in_finds_agencies_not_everyday_words() -> None:
    assert names_in("Report the call to the Nigerian Financial Fraud Bureau.") == {
        "nigerian",
        "financial",
        "fraud",
        "bureau",
    }
    assert names_in("The EFCC and the CBN will help.") == {"efcc", "cbn"}
    assert names_in("Do not share your OTP, PIN or BVN on WhatsApp in Nigeria.") == set()
    assert names_in("Call First Bank, Access Bank or GTBank.") == {
        "first",
        "bank",
        "access",
    }  # runs end at commas
    assert names_in("Police Force officers never ask.") == {
        "force"
    }  # the first word may be capitalised anyway
    assert names_in("") == set()


def test_invented_agency_is_unsupported_even_when_the_meaning_is_close() -> None:
    evidence = [
        "Report to your bank first, as fast as possible. You can also report financial crimes to the EFCC."
    ]
    agency, real = score_claims(
        ["Report to the NFIU as fast as possible.", "Report financial crimes to the EFCC."],
        evidence,
        BagOfWordsEmbedder(),
    )
    assert agency.unmatched == ("nfiu",) and not agency.supported
    assert real.supported and real.unmatched == ()
