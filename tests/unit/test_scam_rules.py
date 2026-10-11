import time
from pathlib import Path

import pytest

from kudiguard import prompts
from kudiguard.domain import Flag, Verdict
from kudiguard.evaluation.answers import entry_texts, pack_entries
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.scam_rules import WEIGHTS, Lexicon, assess, compile_term, has_check_intent
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
        ("ά", "α"),  # noqa: RUF001 - Greek on purpose: its accent-free form sorts after it (mutation testing)
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
    assert has_check_intent([None, "", "  "], en) is False
    assert has_check_intent([None, "Is this a scam?"], en) is True


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


# The weight policy, flag by flag: 3 = decisive alone, 2 = strong (caution alone), 1 = amplifier only.
POLICY = {
    Flag.ASKS_CREDENTIALS: 3,
    Flag.REMOTE_ACCESS: 3,
    Flag.GUARANTEED_RETURNS: 3,
    Flag.WRONG_TRANSFER: 3,
    Flag.ACCOUNT_THREAT: 2,
    Flag.UPFRONT_FEE: 2,
    Flag.PRIZE_LOTTERY: 2,
    Flag.FAKE_PAYMENT: 2,
    Flag.MONEY_REQUEST: 2,
    Flag.SUSPICIOUS_LINK: 2,
    Flag.IMPERSONATION: 1,
    Flag.URGENCY: 1,
    Flag.SECRECY: 1,
}
ALONE = {3: Verdict.LIKELY_SCAM, 2: Verdict.CAUTION, 1: Verdict.NOT_A_CHECK}


@pytest.mark.parametrize("flag", list(Flag))
def test_each_flag_alone_follows_the_weight_policy(en: Lexicon, flag: Flag) -> None:
    result = assess(["hello"], en, extra_flags=[flag])
    assert (result.flags, result.score, result.verdict) == ((flag,), POLICY[flag], ALONE[POLICY[flag]])
    checked = assess(["hello"], en, extra_flags=[flag], is_check=True).verdict
    assert checked is (Verdict.LIKELY_SCAM if POLICY[flag] == 3 else Verdict.CAUTION)


def test_strong_plus_amplifier_reaches_the_scam_threshold(en: Lexicon) -> None:
    result = assess(["hello"], en, extra_flags=[Flag.UPFRONT_FEE, Flag.URGENCY])
    assert (result.score, result.verdict) == (3, Verdict.LIKELY_SCAM)


def test_flags_ordered_by_weight_before_declaration_order(en: Lexicon) -> None:
    # UPFRONT_FEE is declared before GUARANTEED_RETURNS but weighs less.
    result = assess(["hello"], en, extra_flags=[Flag.UPFRONT_FEE, Flag.SECRECY, Flag.GUARANTEED_RETURNS])
    assert result.flags == (Flag.GUARANTEED_RETURNS, Flag.UPFRONT_FEE, Flag.SECRECY)


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


@pytest.mark.parametrize(
    ("lang", "text", "flagged"),
    [
        # data review false positives (8 Oct 2026)
        ("en", "what is the prize of rice today", False),
        ("ha", "an toshe hanyar zuwa kasuwa", False),
        ("ha", "dole sai na biya kudin makaranta", False),
        ("yo", "mo fe ra iresi", False),
        # native starter terms (pending native review)
        ("en", "Congratulations you won a cash prize, claim your prize now", True),
        ("ha", "za a toshe asusunka yau", True),
        ("ha", "ya ce sai ka biya kudin rajista kafin a baka bashin", True),
        ("ha", "ya ce in turo masa lambar da aka aiko min", True),
        ("ha", "kada ka fada wa kowa", True),
        ("ha", "zai ninka kudinka ninki biyu", True),
        ("yo", "won ni ki n fi koodu ti won fi ranse si mi ranse", True),
        ("yo", "e ma se so fun enikeni", True),
        ("yo", "won ni akaunti mi maa di", True),
        ("ig", "o gwara m ka m zitere ya koodu ahu", True),
        ("ig", "agwala onye obula", True),
        ("ig", "a ga emechi akauntu gi", True),
    ],
)
def test_native_terms_and_review_false_positives(
    registry: LanguageRegistry, lang: str, text: str, flagged: bool
) -> None:
    assert bool(assess([text], registry.lexicon_for(lang)).flags) is flagged


# --- Yoruba/Igbo parity terms (workstream H; DRAFT, pending native review) ------------------------------------

PACKS = Path(__file__).resolve().parents[2] / "data" / "packs"


def native(registry: LanguageRegistry, code: str) -> Lexicon:
    """The language's own terms, without the English pivot (so a test proves the native term fires)."""
    spec = registry.get(code).lexicon
    return Lexicon.from_terms(spec.flags, spec.check_intent)


# One realistic message per added term (accented as users type; English gloss in the comment).
@pytest.mark.parametrize(
    ("lang", "flag", "text"),
    [
        # pay it right now right now
        ("yo", Flag.URGENCY, "Wọ́n ní kí n san owó náà ní báyìí báyìí."),
        # they will block my account
        ("yo", Flag.ACCOUNT_THREAT, "Wọ́n ní àwọn yóò dí àkáǹtì mi tí mi ò bá fèsì."),
        # he says it is a sure profit
        ("yo", Flag.GUARANTEED_RETURNS, "Ó ní èrè tó dájú ni, ẹ wá dókòwò."),
        # if I put down 50k, I will get 150k within a week
        ("yo", Flag.GUARANTEED_RETURNS, "Tí mo bá fi 50k sílẹ̀, màá rí 150k gbà láàrin ọ̀sẹ̀ kan."),
        # I must first pay N5,000 before they send the money
        ("yo", Flag.UPFRONT_FEE, "Mo gbọ́dọ̀ kọ́kọ́ san ẹgbẹ̀rún márùn-ún náírà kí wọ́n tó fi owó náà ránṣẹ́."),
        # pay the registration fee today
        ("yo", Flag.UPFRONT_FEE, "Wọ́n ní kí n san owó ìforúkọsílẹ̀ lónìí."),
        # I won a N100,000 prize
        ("yo", Flag.PRIZE_LOTTERY, "Wọ́n ní mo ti jẹ ẹ̀bùn ẹgbẹ̀rún lọ́nà ọgọ́rùn-ún náírà."),
        ("ig", Flag.URGENCY, "Ha sị m zitere ya ozugbo ozugbo."),  # send it immediately immediately
        # he wants me to tell him the OTP
        ("ig", Flag.ASKS_CREDENTIALS, "Ọ chọrọ ka m gwa ya OTP ha zitere m."),
        # if I put in 50k, I will get 150k within one week
        ("ig", Flag.GUARANTEED_RETURNS, "Ọ bụrụ na m tinye 50k, m ga-enweta 150k n'ime otu izu."),
        # I must pay a registration fee
        ("ig", Flag.UPFRONT_FEE, "Ha sị na m ga-akwụ ụgwọ ndebanye aha."),
        ("ig", Flag.PRIZE_LOTTERY, "Ha sị na m meriri ego puku naira nari."),  # I won N100,000
    ],
)
def test_native_parity_terms_fire(registry: LanguageRegistry, lang: str, flag: Flag, text: str) -> None:
    assert flag in assess([text], native(registry, lang)).flags
    assert flag not in assess([text], registry.lexicon_for("en")).flags  # the pivot alone would miss it


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        # this money is a gift from my father
        ("yo", "Ẹ̀bùn ni owó yìí, ó jẹ́ ẹ̀bùn láti ọ̀dọ̀ bàbá mi."),
        # I get my salary next week
        ("yo", "Mo fi owó pamọ́ sí báǹkì, màá rí owó oṣù mi gbà ní ọ̀sẹ̀ tó ń bọ̀."),
        # what must I do before opening an account
        ("yo", "Kí ni mo gbọ́dọ̀ ṣe kí n tó ṣí àkáǹtì tuntun?"),
        # my mother gave me money for my birthday
        ("ig", "Nne m nyere m onyinye ego maka ụbọchị ọmụmụ m."),
        # how do I put money in savings monthly
        ("ig", "Kedu ka m ga-esi tinye ego n'akaụntụ nchekwa kwa ọnwa?"),
        # review false positives (H review, 10 Oct 2026)
        # I put money in the bank, I will get interest in a month
        ("yo", "Mo fi owó sílẹ̀ ní báǹkì, màá gba èlé ní oṣù kan."),
        # a sure profit does not exist in investing (a denied mention)
        ("yo", "Èrè tó dájú kò sí nínú ìdókòwò kankan."),
        ("yo", "Kò sí èrè tó dájú nínú ìdókòwò."),
        # I switched my SIM off / it has become my account
        ("yo", "Mo ti pa SIM mi tì."),
        ("yo", "Ó ti di àkáǹtì mi báyìí."),
        # please tell me what OTP means / I did not tell him my PIN
        ("ig", "Biko gwa m ihe OTP pụtara."),
        ("ig", "Agwaghị m ya PIN m."),
    ],
)
def test_native_parity_terms_spare_ordinary_messages(
    registry: LanguageRegistry, lang: str, text: str
) -> None:
    assert assess([text], native(registry, lang)).flags == ()


# Pre-H term, known: warn_family_fallback paraphrases the scam ("never pay money before you receive money").
KNOWN_SAFE_TEXT_HITS = {("yo", "upfront_fee", "warn_family_fallback")}


@pytest.mark.parametrize("lang", ["ha", "yo", "ig"])
def test_native_terms_never_fire_on_safe_advice(registry: LanguageRegistry, lang: str) -> None:
    """Pack facts/actions, the language's fixed strings and the worked-example replies are safe advice."""
    config = registry.get(lang)
    corpus = [("pack", t) for t in entry_texts(pack_entries(PACKS).values())]
    corpus += list(config.strings.items())
    corpus += [("worked example", e.reply) for e in prompts.WORKED_EXAMPLES]
    hits = {
        (lang, flag, key)
        for flag, terms in config.lexicon.flags.items()
        for term in terms
        for key, text in corpus
        if compile_term(term).search(normalize(text))
    }
    assert hits == {h for h in KNOWN_SAFE_TEXT_HITS if h[0] == lang}
