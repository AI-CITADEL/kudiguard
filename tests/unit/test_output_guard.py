import pytest

from kudiguard.languages import LanguageRegistry
from kudiguard.safety.output_guard import OutputGuard, OutputIssue


@pytest.fixture(scope="module")
def guard(registry: LanguageRegistry) -> OutputGuard:
    return OutputGuard(registry)


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "Don't worry, this is safe."),
        ("en", "It looks completely genuine to me."),
        ("en", "This is not a scam."),
        ("en", "It is safe to pay the fee."),
        ("en", "There is nothing to worry about."),
        ("ha", "Wannan ba zamba bane. Ku zauna lafiya!"),
        ("ha", "Babu hadari a nan."),
        ("yo", "Èyí kì í ṣe ẹ̀tàn."),
        ("yo", "Kò sí ewu kankan."),
        ("ig", "Nke a ọ bụghị aghụghọ."),
        ("ha", "This is legit, ku biya."),  # English mixed into a Hausa reply is still caught
    ],
)
def test_safety_claims_are_caught(guard: OutputGuard, language: str, text: str) -> None:
    assert OutputIssue.SAFETY_CLAIM in guard.check(text, language)


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "Keep your PIN safe and never share it."),
        ("en", "This looks like a scam. Do not pay."),
        ("en", "Check with your bank before you act."),
        ("ha", "Wannan damfara ce. Kada ku biya. Ku zauna lafiya!"),
        ("yo", "Èyí jọ ẹ̀tàn. Ẹ má sanwó."),
    ],
)
def test_ordinary_replies_pass(guard: OutputGuard, language: str, text: str) -> None:
    assert guard.check(text, language) == ()


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "I'm sorry, but I can't help you with this request."),
        ("en", "I cannot assist with that."),
        ("en", "As an AI, I do not have opinions."),
        ("yo", "Ẹ jọ̀wọ́, ṣugbọn mi ò lè ràn ẹ́ lọ́wọ́ pẹ̀lú èyí."),
        ("ha", "Yi haƙuri, ba zan iya taimaka ba."),
        ("ig", "Ndo, enweghị m ike inyere gị aka."),
    ],
)
def test_refusals_are_caught(guard: OutputGuard, language: str, text: str) -> None:
    assert OutputIssue.REFUSAL in guard.check(text, language)


def test_empty_reply(guard: OutputGuard) -> None:
    assert guard.check("   ", "en") == (OutputIssue.EMPTY,)


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("ha", "Babu matsala, ga yadda za ku yi."),
        ("yo", "Kò sí wahala, ẹ ṣe báyìí."),
        ("ig", "Enweghị nsogbu, mee nke a."),
        ("en", "Your money is safe only when you keep your PIN secret."),
        ("en", "It is safe to say this is a scam."),
        ("en", "You can pay your bills with USSD."),
    ],
)
def test_everyday_phrases_are_not_safety_claims(guard: OutputGuard, language: str, text: str) -> None:
    assert OutputIssue.SAFETY_CLAIM not in guard.check(text, language)


@pytest.mark.parametrize(
    "text",
    [
        "You can go ahead and pay.",
        "It is okay to send the money.",
        "E dey real.",
        "This offer is real.",
        "It looks real to me.",
        "This is not fraudulent.",
        "There is no scam here.",
        "You can trust them.",
    ],
)
def test_paraphrased_safety_claims_are_caught(guard: OutputGuard, text: str) -> None:
    assert OutputIssue.SAFETY_CLAIM in guard.check(text, "en")


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "There is nothing suspicious about it."),
        ("en", "Feel free to pay them."),
        ("en", "It is perfectly fine for you to send it."),
        ("en", "You don't need to worry about this."),
        ("en", "Na correct business."),
        ("ig", "Nke a abụghị aghụghọ."),
    ],
)
def test_more_paraphrased_claims_are_caught(guard: OutputGuard, language: str, text: str) -> None:
    assert OutputIssue.SAFETY_CLAIM in guard.check(text, language)


@pytest.mark.parametrize(
    "text",
    [
        "That is okay: call your bank first.",
        "It is fine to ask your bank.",
        "A risk-free investment does not exist.",
    ],
)
def test_advice_wording_is_not_a_claim(guard: OutputGuard, text: str) -> None:
    assert OutputIssue.SAFETY_CLAIM not in guard.check(text, "en")
