import pytest

from kudiguard.domain import Flag, Verdict
from kudiguard.languages import LanguageRegistry
from kudiguard.prompts import Analysis, Intent, select_flags
from kudiguard.safety.decision import decide, is_content_free, is_lookup, resolve_analysis
from kudiguard.safety.scam_rules import Lexicon


@pytest.fixture(scope="module")
def ha(registry: LanguageRegistry) -> Lexicon:
    return registry.lexicon_for("ha")


@pytest.fixture(scope="module")
def en(registry: LanguageRegistry) -> Lexicon:
    return registry.lexicon_for("en")


def test_english_source_uses_the_users_own_text() -> None:
    invented = Analysis("They want you to keep it secret and pay.", Intent.SCAM_CHECK, (Flag.SECRECY,))
    resolved = resolve_analysis(invented, "abeg pay 5k registration fee", "en")
    assert resolved.english == "abeg pay 5k registration fee"


def test_fallback_translation_discards_failed_flags() -> None:
    failed = Analysis("<empty>", Intent.OTHER, (Flag.UPFRONT_FEE,), untranslated=True)
    resolved = resolve_analysis(failed, "Menene BVN?", "ha", fallback_translation="What is BVN?")
    assert resolved.english == "What is BVN?" and resolved.flags == () and not resolved.untranslated


def test_untranslated_analysis_contributes_no_flags() -> None:
    assert select_flags((), Analysis("", Intent.OTHER, (Flag.UPFRONT_FEE,), untranslated=True)) == ()


def test_plain_question_after_failed_analysis_is_not_a_scam(ha: Lexicon) -> None:
    """Regression (M3 verification): a BVN question became 'likely scam' from a failed analysis' flags."""
    failed = Analysis("<empty>", Intent.OTHER, (Flag.ASKS_CREDENTIALS, Flag.UPFRONT_FEE), untranslated=True)
    original = "Menene BVN kuma me yasa nake bukatarsa?"
    resolved = resolve_analysis(
        failed, original, "ha", fallback_translation="What is BVN and why do I need it?"
    )
    assert decide(original, resolved, ha).verdict is Verdict.NOT_A_CHECK


def test_model_only_flags_on_other_are_capped_at_caution(en: Lexicon) -> None:
    analysis = Analysis("Some message", Intent.OTHER, (Flag.ASKS_CREDENTIALS,))
    assert decide("Some message", analysis, en).verdict is Verdict.CAUTION


def test_model_flags_with_scam_check_intent_can_be_likely_scam(en: Lexicon) -> None:
    analysis = Analysis("Some message", Intent.SCAM_CHECK, (Flag.ASKS_CREDENTIALS,))
    assert decide("Some message", analysis, en).verdict is Verdict.LIKELY_SCAM


def test_rule_flags_decide_even_when_intent_is_wrong(en: Lexicon) -> None:
    text = "Customer care asked me to send the OTP they sent to my phone"
    analysis = Analysis(text, Intent.QUESTION, ())
    assert decide(text, analysis, en).verdict is Verdict.LIKELY_SCAM


def test_rules_on_native_original_count(ha: Lexicon) -> None:
    original = "Sun ce na ci kyautar naira dubu dari biyar, amma sai na biya kuɗin rajista tukuna."
    analysis = Analysis("They said I won something.", Intent.OTHER, ())
    assert decide(original, analysis, ha).verdict is Verdict.LIKELY_SCAM


def test_general_question_stays_general(en: Lexicon) -> None:
    analysis = Analysis("How do I save money?", Intent.QUESTION, ())
    assert decide("How do I save money?", analysis, en).verdict is Verdict.NOT_A_CHECK


@pytest.mark.parametrize("text", ["👍🙏", "?", "...", "  ", "k", "🙏🏾🙏🏾🙏🏾"])
def test_content_free_messages(text: str) -> None:
    assert is_content_free(text)


@pytest.mark.parametrize("text", ["hi", "ok", "N5k?", "OTP"])
def test_short_real_messages_are_not_content_free(text: str) -> None:
    assert not is_content_free(text)


@pytest.mark.parametrize(
    ("text", "lookup"),
    [
        ("08031234567", True),
        ("+234 803 123 4567 ?", True),
        ("is this number real 08031234567", False),  # 4 words left: a check via the lexicon instead
        ("bit.ly/claim-cbn-grant", True),
        ("fraud@scam.example", True),
        ("How do I save money every month?", False),
        ("Call your bank on 0700 000 0000 if your transfer failed and you need help", False),
        ("", False),
    ],
)
def test_lookup_detection(text: str, lookup: bool) -> None:
    assert is_lookup(text) is lookup


def test_bare_number_is_checked_but_never_vouched_for(en: Lexicon) -> None:
    analysis = Analysis("08031234567", Intent.OTHER, ())
    assert decide("08031234567", analysis, en).verdict is Verdict.NO_FLAGS


def test_bare_suspicious_link_is_flagged(en: Lexicon) -> None:
    analysis = Analysis("bit.ly/claim-cbn-grant", Intent.OTHER, ())
    assert decide("bit.ly/claim-cbn-grant", analysis, en).verdict in {Verdict.CAUTION, Verdict.LIKELY_SCAM}


def test_is_this_number_real_is_a_check(en: Lexicon) -> None:
    text = "is this number real 08031234567"
    assert decide(text, Analysis(text, Intent.QUESTION, ()), en).verdict is Verdict.NO_FLAGS


def test_explicit_check_phrase_counts_even_if_model_says_question(en: Lexicon) -> None:
    """Regression: the final scoring pass ignored the text, so "is this a scam?" lost its check intent."""
    text = "Is this a scam? My friend wants to sell me his phone."
    assert decide(text, Analysis(text, Intent.QUESTION, ()), en).verdict is Verdict.NO_FLAGS


@pytest.mark.parametrize(
    "text",
    ["Do banks ever ask for OTP?", "When do banks ask for your PIN?", "Why is the bank asking for my BVN?"],
)
def test_general_questions_mentioning_red_flags_are_capped(en: Lexicon, text: str) -> None:
    assert decide(text, Analysis(text, Intent.QUESTION, ()), en).verdict in {
        Verdict.NOT_A_CHECK,
        Verdict.CAUTION,
    }


@pytest.mark.parametrize(
    "text",
    [
        "Someone called me and asked for the OTP on my phone, what should I do?",
        "dem call me say make I send the code wey dem send me",
    ],
)
def test_reported_encounters_stay_likely_scam_even_as_questions(en: Lexicon, text: str) -> None:
    assert decide(text, Analysis(text, Intent.QUESTION, ()), en).verdict is Verdict.LIKELY_SCAM


@pytest.mark.parametrize("text", ["Good   morning!", "thanks a lot", "Ẹ káàbọ̀"])  # noqa: RUF001 - no-break space on purpose
def test_whitespace_and_unicode_variants_are_not_lookups(text: str) -> None:
    assert not is_lookup(text)


def test_lookup_check_is_fast_on_pathological_input() -> None:
    import time

    started = time.perf_counter()
    is_lookup("a." * 10_000)
    is_lookup("a. " * 10_000)
    assert time.perf_counter() - started < 0.2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://www.cbn.gov.ng/", Verdict.NO_FLAGS),  # a bare link is a check; official -> no red flags
        ("Check cbn.gov.ng for the guidelines", Verdict.NOT_A_CHECK),
        ("I won 5 million naira and they say I must pay a processing fee", Verdict.LIKELY_SCAM),
    ],
)
def test_official_links_and_prize_reports(en: Lexicon, text: str, expected: Verdict) -> None:
    assert decide(text, Analysis(text, Intent.OTHER, ()), en).verdict is expected
