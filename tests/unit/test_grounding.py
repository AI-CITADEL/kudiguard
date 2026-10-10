"""Grounding check: claim splitting, specifics, polarity, attribution, scoring and the checker.

A bag-of-words embedder stands in for the model (same words -> cosine 1); the real model runs in
tests/integration/test_grounding_quality.py (-m live).
"""

import asyncio
import logging
import threading
import time

import numpy as np
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
    attributed,
    evidence_for,
    evidence_units,
    is_trivial,
    message_sentences,
    names_in,
    negated,
    numbers_in,
    score_claims,
    split_claims,
    without_unsupported_steps,
)

FACTS = (
    "Your bank will never ask for your OTP, PIN or password. Anyone who asks is trying to steal your money.",
    "An OTP is like the key to your account.",
)
ACTIONS = ("Do not share the code. End the call or chat.", "Change your PIN and password.")


def knowledge(*entries: Entry, kind: PackKind = PackKind.CORE) -> KnowledgeResult:
    return KnowledgeResult(entries=tuple(SourcedEntry(e, "pack", "Demo Bank", kind=kind) for e in entries))


OTP = knowledge(Entry(id="otp", title="OTP", facts=FACTS, actions=ACTIONS))


def score(
    claims: list[str], evidence: list[str] | tuple[str, ...], message: str = "", **kw: float
) -> tuple[Claim, ...]:
    return score_claims(claims, evidence, BagOfWordsEmbedder(), message=message, **kw)


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
        "**4.** Report it to your bank today!\n"
        "1️⃣ Keep your receipt safe.\n"
        "5 - Hang up the phone now.\n"
        "Step 6 - Block the caller's number.\n"
        "7 \u2013 Delete the message."
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
        "Keep your receipt safe.",
        "Hang up the phone now.",
        "Block the caller's number.",
        "Delete the message.",
    ]


def test_split_claims_keeps_abbreviations_inside_a_sentence() -> None:
    assert split_claims("Never share it, e.g. by SMS or call. Ask Mr. Okafor at the branch.") == [
        "Never share it, e.g. by SMS or call.",
        "Ask Mr. Okafor at the branch.",
    ]


def test_split_claims_skips_filler_and_lead_ins() -> None:
    assert split_claims("Stay calm. Be careful! Okay.\nHere is what you can do:\n") == []
    assert split_claims("") == [] and split_claims("  \n ") == []
    assert is_trivial("Don't worry.") and is_trivial("Don\u2019t worry.") and is_trivial("Next steps:")
    assert not is_trivial("Do not pay.")


def test_sentences_with_numbers_or_names_are_never_trivial() -> None:
    assert split_claims("Dial *565*0# now.") == ["Dial *565*0# now."]
    assert split_claims("Call the EFCC.") == ["Call the EFCC."]
    assert split_claims("Wait 3 days.") == ["Wait 3 days."]
    assert split_claims("12. Keep your receipt safe.") == ["Keep your receipt safe."]


# --- specifics ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("Pay N5,000 within 24 hours", {"5000", "24"}),
        ("Send 50k or N50,000 or fifty thousand naira", {"50000"}),
        ("Wait seven days, or three weeks", {"7", "3"}),
        ("You have 24hrs to act", {"24"}),
        ("They want 2 million naira or N2m", {"2000000"}),
        ("A hundred times", {"100"}),
        ("twenty five days", {"25"}),
        ("one week", {"1"}),
        ("Dial *565*0# now", {"565", "0"}),
        ("Wait 2.5 hours", {"2.5"}),
        ("Do it in 5 minutes", {"5"}),
        ("No one will call, not one person", set()),
        ("Your PIN has four digits", set()),
        ("Wait a day or two before you pay", set()),
        ("Version 1.2.3", set()),
    ],
)
def test_numbers_in_normalises_amounts_and_needs_units_for_words(text: str, numbers: set[str]) -> None:
    assert numbers_in(text) == numbers


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("Report the call to the Nigerian Financial Fraud Bureau.", {"nigerian financial fraud bureau"}),
        ("The EFCC and the CBN will help.", {"efcc", "cbn"}),
        ("Do not share your OTP, PIN or BVN on WhatsApp in Nigeria.", set()),
        ("Call First Bank, Access Bank or GTBank.", {"first bank", "access bank"}),
        ("Nigerian Police Force will help.", {"police", "police force"}),
        ("Report it to the Police.", {"police"}),
        ("Report it to the police.", set()),  # lower case: an ordinary word
        ("Say NO and tell them STOP.", set()),
        ("Use the Play Store.", {"play store"}),
        ("", set()),
    ],
)
def test_names_in_finds_agencies_as_phrases(text: str, names: set[str]) -> None:
    assert names_in(text) == names


def test_polarity_and_attribution_markers() -> None:
    assert negated("Do not share the code.") and negated("Never pay a fee.") and negated("Don\u2019t click.")
    assert negated("Avoid strangers.") and not negated("Share the code.") and not negated("Keep it secret.")
    assert attributed("They said your account will be blocked.") and attributed(
        "The caller asked for your OTP."
    )
    assert attributed("Scammers want you to act fast.") and not attributed("Your account will be blocked.")


def test_message_sentences_are_bounded() -> None:
    long_message = " ".join(f"Sentence number {n} " + "x" * 400 + "." for n in range(30))
    sentences = message_sentences(long_message)
    assert len(sentences) == 10 and all(len(s) <= 300 for s in sentences)
    assert message_sentences("") == []


def test_evidence_units_are_single_sentences_without_duplicates() -> None:
    units = evidence_units(["Do not pay. Check the offer.", "Do not pay.", "  ", "1. Call your bank."])
    assert units == ["Do not pay.", "Check the offer.", "Call your bank."]


# --- evidence ---------------------------------------------------------------------------------------------------


def test_evidence_for_is_what_compose_was_shown_plus_verdict_and_flags() -> None:
    evidence = evidence_for(OTP, verdict=Verdict.LIKELY_SCAM, flags=(Flag.ASKS_CREDENTIALS,))
    assert evidence == [
        *FACTS,
        *ACTIONS,
        VERDICT_LINE[Verdict.LIKELY_SCAM],
        f"The message {FLAG_TEXT[Flag.ASKS_CREDENTIALS]}.",
    ]
    assert evidence_for(OTP, verdict=Verdict.NOT_A_CHECK) == [*FACTS, *ACTIONS] == evidence_for(OTP)
    assert evidence_for(KnowledgeResult()) == []


def test_evidence_for_keeps_partner_citations() -> None:
    partner = knowledge(Entry(id="p", title="P", facts=("Call 0700 for help.",)), kind=PackKind.PARTNER)
    assert evidence_for(partner) == ["Call 0700 for help. (source: Demo Bank)"]


# --- scoring ----------------------------------------------------------------------------------------------------


def test_score_claims_supports_restated_sentences_and_flags_inventions() -> None:
    scored = score(["Do not share the code.", "The police will refund you quickly."], evidence_for(OTP))
    assert scored[0] == Claim("Do not share the code.", 1.0, True)
    assert (
        scored[1].supported is False and scored[1].support < DEFAULT_THRESHOLD and scored[1].unmatched == ()
    )


def test_polarity_inversion_is_not_supported_by_the_negated_fact() -> None:
    (inverted, kept) = score(["Share the code.", "Do not share the code."], ["Do not share the code."])
    assert not inverted.supported and inverted.support < DEFAULT_THRESHOLD
    assert kept.supported and kept.support == 1.0
    (positive,) = score(["Change your PIN and password."], ["Do not change your PIN and password."])
    assert not positive.supported


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("Your account will be blocked today unless you send them the OTP.",
         "They said my account will be blocked today unless I send them the OTP."),
        ("You won N500,000 and you should pay the N5,000 processing fee to claim it.",
         "They said I won N500,000 and I should pay N5,000 processing fee to claim it."),
        ("Download AnyDesk and share the code that comes out.",
         "Opay customer care told me to download AnyDesk and share the code that comes out."),
    ],
)  # fmt: skip
def test_the_users_message_never_supports_an_unattributed_claim(reply: str, message: str) -> None:
    (claim,) = score([reply], evidence_for(OTP), message=message)
    assert not claim.supported and claim.support < DEFAULT_THRESHOLD


def test_an_attributed_claim_may_restate_the_message() -> None:
    message = "They said my account will be blocked today unless I send them the OTP."
    (claim,) = score(
        ["They said your account will be blocked today unless you send them the OTP."], [], message
    )
    assert claim.supported and claim.support >= DEFAULT_THRESHOLD


def test_message_specifics_are_allowed_but_not_its_meaning() -> None:
    message = "First Bank called about my N50,000 transfer."
    evidence = ["Call your bank on the number on your card about the transfer."]
    (named,) = score(
        ["Call First Bank on the number on your card about the 50k transfer."], evidence, message
    )
    assert named.unmatched == ()
    (invented,) = score(
        ["Call Access Bank on the number on your card about the transfer."], evidence, message
    )
    assert invented.unmatched == ("access bank",) and not invented.supported


def test_numbers_and_names_must_appear_in_the_evidence() -> None:
    evidence = ["Most failed transfers are reversed within 24 hours. Report fraud to the EFCC."]
    wrong, right, agency = score(
        ["Most failed transfers are reversed within 48 hours.", "Most failed transfers are reversed within 24 hours.",
         "Report fraud to the NFIU."],
        evidence,
    )  # fmt: skip
    assert wrong.unmatched == ("48",) and not wrong.supported and wrong.support >= DEFAULT_THRESHOLD
    assert right.supported
    assert agency.unmatched == ("nfiu",) and not agency.supported


def test_score_claims_edge_cases() -> None:
    embedder = BagOfWordsEmbedder()
    assert score_claims([], ["anything"], embedder) == ()
    assert embedder.batches == []  # nothing embedded
    (claim,) = score_claims(["Do not share the code."], [" "], embedder)
    assert claim == Claim("Do not share the code.", 0.0, False)


def test_threshold_is_configurable() -> None:
    claims = ["Do not share the code with anyone."]
    loose, strict = (score(claims, ACTIONS, threshold=t)[0] for t in (0.5, 0.99))
    assert loose.supported and not strict.supported and loose.support == strict.support


def test_report_properties() -> None:
    good, bad = Claim("a b c", 0.9, True), Claim("d e f", 0.2, False)
    assert GroundingReport("x", (good, bad)).unsupported == ("d e f",)
    assert not GroundingReport("x", (good, bad)).grounded
    assert (
        GroundingReport("x", (good,)).grounded and GroundingReport("", ()).grounded
    )  # nothing to contradict


def test_many_claims_are_embedded_in_length_sorted_chunks() -> None:
    embedder = BagOfWordsEmbedder()
    claims = [f"Claim {'word ' * (40 - n)}number {n}." for n in range(20)]
    scored = score_claims(claims, ["Claim number."], CachedEmbedder(embedder))
    assert [c.text for c in scored] == claims  # results stay in reply order
    assert [len(b) for b in embedder.batches] == [16, 4, 1]  # two chunks of claims, then the evidence
    assert embedder.batches[0][0] == claims[-1]  # shortest first


# --- dropping unsupported steps -----------------------------------------------------------------------------------


def report_for(reply: str, unsupported: set[str]) -> GroundingReport:
    return GroundingReport(reply, tuple(Claim(c, 0.5, c not in unsupported) for c in split_claims(reply)))


def test_unsupported_step_is_dropped_and_steps_renumbered() -> None:
    reply = "This is a scam.\n1. Do not share the code.\n2. Pay the NFIU 5k.\n**3.** Change your PIN and password."
    trimmed = without_unsupported_steps(reply, report_for(reply, {"Pay the NFIU 5k."}))
    assert trimmed == "This is a scam.\n1. Do not share the code.\n**2.** Change your PIN and password."
    assert without_unsupported_steps(reply, report_for(reply, set())) == reply


def test_unsupported_prose_or_nothing_left_means_fixed_text() -> None:
    reply = "Your bank will refund you.\n1. Do not share the code."
    assert without_unsupported_steps(reply, report_for(reply, {"Your bank will refund you."})) is None
    only_step = "1. Pay the NFIU 5k."
    assert without_unsupported_steps(only_step, report_for(only_step, {"Pay the NFIU 5k."})) is None


# --- checker ----------------------------------------------------------------------------------------------------


def checker(registry: LanguageRegistry, **kwargs: object) -> GroundingChecker:
    return GroundingChecker(None, BagOfWordsEmbedder(), registry, **kwargs)  # type: ignore[arg-type]


async def test_english_reply_is_checked_without_a_model(registry: LanguageRegistry) -> None:
    grounding = checker(registry)
    reply = "This looks like a scam.\n1. Do not share the code.\n2. Change your PIN and password."
    report = await grounding.check(reply, "en", evidence_for(OTP, verdict=Verdict.LIKELY_SCAM))
    assert report.english == reply and report.grounded and len(report.claims) == 3
    assert report.ms >= 0 and grounding.supports("en") and not grounding.supports("ha")
    await grounding.aclose()


async def test_invented_claim_is_reported(registry: LanguageRegistry) -> None:
    reply = "Do not share the code.\nYour bank will refund all your money within 24 hours."
    report = await checker(registry).check(reply, "en", evidence_for(OTP))
    assert report.unsupported == ("Your bank will refund all your money within 24 hours.",)


async def test_generic_framing_and_fixed_advice_are_always_evidence(registry: LanguageRegistry) -> None:
    grounding = checker(registry)
    assert set(GENERIC_FRAMING) <= set(grounding.framing)
    assert not any("worry" in f and "don" in f.lower() for f in GENERIC_FRAMING)  # overlaps the guard
    assert any("number printed on your card" in text for text in grounding.framing)  # en fallback_scam
    reply = f"{GENERIC_FRAMING[0]} Tell your family and friends about this trick. Call your bank on the number printed on your card."
    assert (await grounding.check(reply, "en", [])).grounded


def test_invalid_threshold_and_budget(registry: LanguageRegistry) -> None:
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match="threshold"):
            checker(registry, threshold=bad)
    with pytest.raises(ValueError, match="budget_s"):
        checker(registry, budget_s=0)


def test_a_cached_embedder_is_not_wrapped_twice(registry: LanguageRegistry) -> None:
    cached = CachedEmbedder(BagOfWordsEmbedder())
    assert GroundingChecker(None, cached, registry)._embedder is cached
    assert isinstance(checker(registry)._embedder, CachedEmbedder)


async def test_only_shared_evidence_is_cached_never_user_text(registry: LanguageRegistry) -> None:
    embedder = BagOfWordsEmbedder()
    grounding = GroundingChecker(None, embedder, registry)
    count = grounding.warm([*FACTS, *ACTIONS])
    assert count == len(embedder.batches[0]) and "Do not share the code." in embedder.batches[0]
    assert (
        VERDICT_LINE[Verdict.CAUTION].split(". ")[0] + "." in embedder.batches[0]
    )  # verdict lines warmed too
    embedder.batches.clear()
    message = "They said my account will be blocked."
    for _ in range(2):
        await grounding.check("They said your account is blocked.", "en", [*FACTS, *ACTIONS], message=message)
    # each turn: one batch with the reply and the message, embedded again (not cached), no evidence re-embedded
    assert embedder.batches == [["They said your account is blocked.", message]] * 2
    cache = grounding._embedder
    assert isinstance(cache, CachedEmbedder) and message not in cache._vectors
    assert "They said your account is blocked." not in cache._vectors


async def test_the_message_is_embedded_only_for_attributed_claims(registry: LanguageRegistry) -> None:
    embedder = BagOfWordsEmbedder()
    grounding = GroundingChecker(None, embedder, registry)
    grounding.warm([*FACTS, *ACTIONS])
    embedder.batches.clear()
    await grounding.check("Do not share the code.", "en", [*FACTS, *ACTIONS], message="x " * 2000)
    assert embedder.batches == [["Do not share the code."]]


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


class Gated(BagOfWordsEmbedder):
    """Blocks inside the model call until released (a slow or stuck embedding)."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        self.release.wait(5)
        return super().embed_documents(texts)


async def test_time_budget_and_deadline_cap_the_check(registry: LanguageRegistry) -> None:
    embedder = Gated()
    grounding = GroundingChecker(None, embedder, registry, budget_s=0.05)
    start = time.perf_counter()
    with pytest.raises(GroundingError, match="out of time"):
        await grounding.check("Do not share the code.", "en", ACTIONS, deadline=time.monotonic() + 60)
    assert time.perf_counter() - start < 0.5  # returned at the budget, not when the embedding finished
    embedder.release.set()
    await asyncio.sleep(0.1)
    unbudgeted = GroundingChecker(None, BagOfWordsEmbedder(), registry)
    with pytest.raises(GroundingError, match="out of time"):
        await unbudgeted.check("Do not share the code.", "en", ACTIONS, deadline=time.monotonic() - 1)


async def test_a_busy_worker_skips_the_check_instead_of_queueing(registry: LanguageRegistry) -> None:
    embedder = Gated()
    grounding = GroundingChecker(None, embedder, registry, budget_s=0.05)
    with pytest.raises(GroundingError, match="out of time"):
        await grounding.check("Do not share the code.", "en", ACTIONS)  # still running in the worker
    with pytest.raises(GroundingError, match="busy"):
        await grounding.check("Change your PIN and password.", "en", ACTIONS)
    embedder.release.set()
    for _ in range(50):  # the worker finishes in the background; then checks run again
        await asyncio.sleep(0.02)
        if grounding._running is not None and grounding._running.done():
            break
    assert (await grounding.check("Change your PIN and password.", "en", ACTIONS)).grounded
    await grounding.aclose()


async def test_check_logs_counts_and_timing_not_text(
    registry: LanguageRegistry, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="kudiguard.safety.grounding"):
        await checker(registry).check("Send the money to my cousin today.", "en", evidence_for(OTP))
    (message,) = [r.getMessage() for r in caplog.records]
    assert "1 claims, 1 unsupported" in message and " ms" in message and "cousin" not in message


def test_cached_embedder_reports_its_dimension_when_empty() -> None:
    cached = CachedEmbedder(BagOfWordsEmbedder())
    assert cached.embed_documents([]).shape == (0, 0)
    cached.embed_documents(["a b"])
    assert cached.embed_documents([]).shape == (0, BagOfWordsEmbedder.DIM)
    assert np.allclose(np.linalg.norm(cached.embed_query("a b")), 1.0)
