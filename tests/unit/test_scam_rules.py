import time

import pytest

from kudiguard.domain import Flag, Verdict
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.scam_rules import WEIGHTS, Lexicon, assess
from kudiguard.text import normalize


@pytest.fixture(scope="module")
def en(registry: LanguageRegistry) -> Lexicon:
    return registry.lexicon_for("en")


# --- normalisation ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Give me your  O.T.P\nNOW", "give me your otp now"),
        ("b-v-n", "bvn"),
        ("send the o t p", "send the otp"),
        ("Ẹ jọ̀ọ́ kíákíá", "e joo kiakia"),
        ("Ɗan damfara ne", "dan damfara ne"),
        ("Don’t tell anyone", "don't tell anyone"),  # noqa: RUF001
        ("visit bit.ly/x", "visit bit.ly/x"),
        ("N2.5 million", "n2.5 million"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


# --- lexicon mechanics --------------------------------------------------------------------------------


def test_terms_match_whole_words_only() -> None:
    lex = Lexicon.from_terms({"asks_credentials": ["pin"]}, [])
    assert assess(["what is my pin"], lex).flags == (Flag.ASKS_CREDENTIALS,)
    assert assess(["spinach and pinto beans"], lex).flags == ()


@pytest.mark.parametrize(
    ("term", "message"),
    [
        ("(unclosed", "invalid regex"),
        ("a|", "empty string"),
        ("(?:x)?", "empty string"),
        ("kíákíá", "kiakia"),
    ],
)
def test_bad_terms_rejected(term: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Lexicon.from_terms({"urgency": [term]}, [])


def test_unknown_flag_rejected() -> None:
    with pytest.raises(ValueError, match="unknown flag"):
        Lexicon.from_terms({"nope": ["x"]}, [])


def test_merge_combines_terms() -> None:
    a = Lexicon.from_terms({"urgency": ["now"]}, ["scam"])
    b = Lexicon.from_terms({"urgency": ["sharp sharp"], "secrecy": ["no tell anybody"]}, ["wayo"])
    merged = a.merge(b)
    assert assess(["do am sharp sharp, no tell anybody"], merged).flags == (Flag.URGENCY, Flag.SECRECY)
    assert assess(["na wayo?"], merged).verdict is Verdict.NO_FLAGS


def test_extra_flags_accept_strings_and_dedupe(en: Lexicon) -> None:
    result = assess(["hello"], en, extra_flags=["asks_credentials", Flag.ASKS_CREDENTIALS])
    assert result.flags == (Flag.ASKS_CREDENTIALS,)
    assert result.verdict is Verdict.LIKELY_SCAM


def test_unknown_extra_flag_raises(en: Lexicon) -> None:
    with pytest.raises(ValueError):
        assess(["hello"], en, extra_flags=["made_up"])


def test_none_and_empty_texts_are_ignored(en: Lexicon) -> None:
    assert assess([None, "", "  "], en).verdict is Verdict.NOT_A_CHECK


def test_every_flag_has_weight_and_english_terms(en: Lexicon) -> None:
    assert set(WEIGHTS) == set(Flag)
    assert set(en.flags) == set(Flag)
    assert all(en.flags[f] for f in Flag)


@pytest.mark.parametrize(
    "text",
    [
        "a " * 2000,
        "send " + "x " * 2000 + "!",
        "pay " + "1," * 2000,
        "your " + "account " * 800,
        "www." + "a" * 4000,
        "invest " + "1 " * 2000,
    ],
)
def test_adversarial_input_is_fast(registry: LanguageRegistry, text: str) -> None:
    start = time.perf_counter()
    assess([text, text], registry.lexicon_for("ha"))
    assert time.perf_counter() - start < 0.5


# --- verdict policy -----------------------------------------------------------------------------------


def test_single_strong_signal_is_caution(en: Lexicon) -> None:
    result = assess(["They sent me this link bit.ly/abc123 to update my details"], en)
    assert result.verdict is Verdict.CAUTION
    assert result.flags == (Flag.SUSPICIOUS_LINK,)


def test_weak_signal_on_request_is_caution_never_no_flags(en: Lexicon) -> None:
    result = assess(["Is this real? Police officer said I should come to the station"], en)
    assert result.verdict is Verdict.CAUTION
    assert result.flags == (Flag.IMPERSONATION,)


def test_weak_signal_without_request_is_not_a_check(en: Lexicon) -> None:
    assert assess(["How do I contact my bank customer care?"], en).verdict is Verdict.NOT_A_CHECK


def test_caller_can_declare_a_check(en: Lexicon) -> None:
    text = "EFCC: you are under investigation"
    assert assess([text], en).verdict is Verdict.NOT_A_CHECK
    assert assess([text], en, is_check=True).verdict is Verdict.CAUTION
    assert assess(["hello there"], en, is_check=True).verdict is Verdict.NO_FLAGS


def test_explicit_check_without_signals_is_no_flags(en: Lexicon) -> None:
    result = assess(["Is this a scam? My friend offered to sell me his phone"], en)
    assert result.verdict is Verdict.NO_FLAGS
    assert result.flags == ()


def test_two_weak_signals_are_caution(en: Lexicon) -> None:
    assert assess(["This is customer care, call us back urgently"], en).verdict is Verdict.CAUTION


def test_flags_unique_and_ordered_by_weight(en: Lexicon) -> None:
    result = assess(["URGENT!!! urgent: share your OTP now, you won a prize"], en)
    assert len(result.flags) == len(set(result.flags))
    assert result.flags[0] is Flag.ASKS_CREDENTIALS
    assert result.flags[-1] is Flag.URGENCY
    assert result.score == 3 + 2 + 1


# --- detection quality on the shipped English pivot lexicon -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Your BVN has been blocked. Send the OTP we sent to you to reactivate it immediately.",
        "Congratulations! You won N500,000 in the MTN promo. Pay N5,000 processing fee to claim your prize.",
        "Invest N50,000 and get guaranteed 30% return weekly, double your money in one month",
        "I sent N20,000 to your account by mistake, please reverse it to this number",
        "Customer care here, kindly tell me your ATM card PIN so we can fix your account",
        "Download AnyDesk so our bank officer can fix your app",
        "They told me to download an app so they can help me fix my account",
        "Someone called and asked me for the OTP on my phone",
        "Click this link https://bit.ly/x urgently to update your details",
        # Pidgin and review-found misses
        "Abeg send me the code wey dem send you",
        "Abeg forward the 6 digit code wey enter your phone",
        "Your WhatsApp will be banned. Send the 6-digit code we sent",
        "We noticed suspicious login. Reply YES and the code sent to your phone",
        "Hi this is GTBank support on Twitter, DM us your account number and the OTP",
        "Glo promo: you have been selected. Send your account details",
        "Your SIM card will be deactivated in 24hrs. Reply with your PIN",
        "Join our cooperative, put 20k and get 40k in 7 days",
        "Forex investment: invest $100 and earn $1000 weekly",
        "Send 0.1 BTC and get 0.2 BTC back guaranteed",
        "Someone on Facebook said if I invest 50,000 naira I will get 150,000 naira in one week",
        "Abeg invest 20k, you go collect 60k after 2 weeks",
        "Your NIN is not linked, visit nimc-verify.com.ng or your SIM will be barred",
        "CBN is giving N250,000 grant, apply now via this link",
        "My dear, I am stuck at the airport, send me $500 for customs clearance",
    ],
)
def test_scams_are_likely_scam(en: Lexicon, text: str) -> None:
    assert assess([text], en).verdict is Verdict.LIKELY_SCAM


@pytest.mark.parametrize(
    ("text", "flag"),
    [
        ("Get instant loan, pay N3,000 to activate your loan", Flag.UPFRONT_FEE),
        ("Federal government Tradermoni loan available, pay N2000 registration", Flag.UPFRONT_FEE),
        ("Shell Nigeria recruitment. Pay N10,000 for your medical test", Flag.UPFRONT_FEE),
        ("Baby, pay the delivery fee to the courier", Flag.UPFRONT_FEE),
        ("FG empowerment program grant at grants-ng.xyz", Flag.SUSPICIOUS_LINK),
        ("Credit Alert! Amt: NGN150,000. Check your alert and send the goods", Flag.FAKE_PAYMENT),
    ],
)
def test_suspicious_messages_are_at_least_caution(en: Lexicon, text: str, flag: Flag) -> None:
    result = assess([text], en)
    assert result.verdict in {Verdict.CAUTION, Verdict.LIKELY_SCAM}
    assert flag in result.flags


@pytest.mark.parametrize(
    "text",
    [
        "How do I reset my ATM PIN?",
        "How do I get a BVN?",
        "What is BVN and why do I need it?",
        "Should I give my password to anyone?",
        "How can I enter my token code on the app",
        "How do I block my card if I lose it?",
        "My account was blocked because of BVN mismatch, what do I do?",
        "How much is the transfer fee for N10,000?",
        "I sent money to the wrong account by mistake, how do I get it back?",
        "How do I start saving money every month?",
        "How do I contact my bank customer care?",
        "What is a good interest rate for a loan?",
        "How do I check my account balance?",
        "Can I put money in savings to earn interest monthly?",
        "Visit cbn.gov.ng to read the guidelines",
        "What is the CBN interest rate this year?",
    ],
)
def test_legitimate_questions_are_not_flagged(en: Lexicon, text: str) -> None:
    assert assess([text], en).verdict is Verdict.NOT_A_CHECK


# --- multilingual -------------------------------------------------------------------------------------


def test_original_and_translation_are_combined(registry: LanguageRegistry) -> None:
    ha = registry.lexicon_for("ha")
    result = assess(["An toshe asusun ku, ku yi gaggawa", "Your account has been blocked, act now"], ha)
    assert Flag.ACCOUNT_THREAT in result.flags
    assert Flag.URGENCY in result.flags
    assert result.verdict is Verdict.LIKELY_SCAM


def test_native_check_intent(registry: LanguageRegistry) -> None:
    assert assess(["Wannan zamba ne?"], registry.lexicon_for("ha")).verdict is Verdict.NO_FLAGS


def test_secrecy_amplifies_a_money_request(en: Lexicon) -> None:
    alone = assess(["Keep this secret, don't tell your wife"], en)
    assert alone.flags == (Flag.SECRECY,)
    assert alone.verdict is Verdict.NOT_A_CHECK
    combined = assess(["Send me money today, keep this secret, don't tell your wife"], en)
    assert combined.verdict is Verdict.LIKELY_SCAM


def test_hausa_prize_and_fee_scam_caught_by_native_rules(registry: LanguageRegistry) -> None:
    text = "Sun ce na ci kyautar naira dubu dari biyar, amma sai na biya kuɗin rajista tukuna."
    result = assess([text], registry.lexicon_for("ha"))
    assert {Flag.PRIZE_LOTTERY, Flag.UPFRONT_FEE} <= set(result.flags)
    assert result.verdict is Verdict.LIKELY_SCAM


def test_asking_about_secrecy_is_not_a_red_flag(en: Lexicon) -> None:
    assert Flag.SECRECY not in assess(["Why should I keep it secret?"], en).flags
    assert Flag.SECRECY in assess(["He told me to keep it secret from my wife"], en).flags


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("Keep this secret, send me 5k", True),
        ("He wants me to keep it secret", True),
        ("I want to keep it secret from my husband", False),
    ],
)
def test_secrecy_rule_needs_someone_asking(en: Lexicon, text: str, flagged: bool) -> None:
    assert (Flag.SECRECY in assess([text], en).flags) is flagged
