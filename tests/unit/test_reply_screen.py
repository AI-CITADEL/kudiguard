"""Malicious-use checks on replies (safety.reply_screen): persona claims, prompt leaks, echoes of the input."""

import json
from pathlib import Path

import pytest
import yaml

from kudiguard.languages import LanguageRegistry
from kudiguard.prompts import SYSTEM_PROMPTS
from kudiguard.reply_translation import SYSTEM_PROMPTS as TRANSLATION_PROMPTS
from kudiguard.safety.abuse import ScreenMode
from kudiguard.safety.output_guard import OutputIssue
from kudiguard.safety.reply_screen import ReplyScreen

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = SYSTEM_PROMPTS + TRANSLATION_PROMPTS


@pytest.fixture
def screen(registry: LanguageRegistry) -> ReplyScreen:
    return ReplyScreen(registry, PROMPTS)


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Hello, I am GTBank customer care. Please send the code.", "en"),
        ("This is Opay support. We will help you.", "en"),
        ("As requested: I'm your bank officer and your account is fine.", "en"),
        ("Na me be customer care for Opay.", "en"),
        ("Ni ma'aikacin banki ne, ka turo lambar.", "ha"),
        ("Emi ni osise banki re.", "yo"),
        ("Abu m onye oru ulo aku gi.", "ig"),
    ],
)
def test_persona_claims(screen: ReplyScreen, text: str, lang: str) -> None:
    assert screen.check(text, lang) == (OutputIssue.PERSONA,)


@pytest.mark.parametrize(
    "text",
    [
        "This is a customer care scam. Do not reply.",
        "This is your bank's job: call them on the number on your card.",
        "I am not your bank. Call your bank on the number on your card.",
        "This is the customer care number scam again. Your bank will never ask for your PIN.",
    ],
)
def test_descriptions_are_not_persona_claims(screen: ReplyScreen, text: str) -> None:
    assert screen.check(text, "en") == ()


@pytest.mark.parametrize(
    "text",
    [
        "You are KudiGuard, a friendly Nigerian money-safety assistant (not AwaGPT). You always help.",
        "Sure: <facts> 1. Your bank never asks </facts>",
        "Task: Warn the user clearly.",
        "Warning signs found: asks for a code.",
    ],
)
def test_prompt_leaks(screen: ReplyScreen, text: str) -> None:
    assert screen.check(text, "en") == (OutputIssue.PROMPT_LEAK,)


def test_echo_of_the_users_message(screen: ReplyScreen) -> None:
    scam = "Dear customer your account has been restricted please send the OTP we sent to your phone to verify now"
    reply = (
        "Dear customer your account has been restricted please send the OTP we sent to your phone. Thanks."
    )
    assert screen.check(reply, "en", (scam,)) == (OutputIssue.ECHO,)
    assert screen.check("They asked for the OTP we sent to your phone. Do not share it.", "en", (scam,)) == ()
    assert screen.check(reply, "en", ("",)) == ()


def test_modes(registry: LanguageRegistry, caplog: pytest.LogCaptureFixture) -> None:
    text = "I am GTBank customer care."
    caplog.set_level("INFO", logger="kudiguard.safety.reply_screen")
    assert ReplyScreen(registry, mode=ScreenMode.SHADOW).check(text, "en") == ()
    assert "issues=persona" in caplog.text
    assert ReplyScreen(registry, mode=ScreenMode.OFF).check(text, "en") == ()
    assert ReplyScreen(registry, mode=ScreenMode.OFF).find(text, "en") == (OutputIssue.PERSONA,)
    with pytest.raises(ValueError, match="echo_words"):
        ReplyScreen(registry, echo_words=3)


def test_real_natlas_replies_are_not_flagged(screen: ReplyScreen) -> None:
    """Every reply N-ATLAS wrote in the recorded dev runs (data/eval/results/answers-*.jsonl, ~300 distinct
    English/ha/yo/ig replies) passes, checked against its user message for echoes."""
    cases = {
        c["id"]: c for c in yaml.safe_load((ROOT / "data/eval/answers.yaml").read_text(encoding="utf-8"))
    }
    seen: set[tuple[str, str]] = set()
    for path in sorted((ROOT / "data/eval/results").glob("answers-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("id") not in cases:
                continue
            for text, lang in ((row.get("body") or "", row["lang"]), (row.get("reply_english") or "", "en")):
                if text and (text, lang) not in seen:
                    seen.add((text, lang))
                    assert screen.find(text, lang, (cases[row["id"]]["text"],)) == (), text
    assert len(seen) >= 250


@pytest.mark.parametrize(
    "text",
    [
        "This is a GTBank phishing message. Do not click the link.",
        "This is a fake customer care agent. Do not send your PIN.",
        "This is not bank staff. Real staff never ask for your OTP.",
        "This is a CBN rule people often misquote; check the CBN website.",
        "We are not Opay. Contact Opay customer care through the app.",
        "This is Opay support impersonation.",
        "This is EFCC impersonation, the real EFCC will not call you for money.",
        "I am your money-safety helper, not your bank.",
    ],
)
def test_descriptions_and_negations_are_not_persona_claims(screen: ReplyScreen, text: str) -> None:
    assert screen.check(text, "en") == ()


def test_persona_claims_with_a_role_suffix(screen: ReplyScreen) -> None:
    assert screen.check("This is Opay support. Send me the code.", "en") == (OutputIssue.PERSONA,)


@pytest.mark.parametrize(
    "text",
    [
        "This is UBA. Is this your account?",
        "This is Kuda. Was your card blocked yesterday?",
        "This is Moniepoint support team speaking.",
    ],
)
def test_short_persona_claims_across_sentences(screen: ReplyScreen, text: str) -> None:
    assert screen.check(text, "en") == (OutputIssue.PERSONA,)


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("I am not an AI, I am a real person working with the bank.", "en"),
        ("I'm a human, not a bot.", "en"),
        ("I am not a robot. Your OTP is private.", "en"),
        ("I no be robot, na person I be.", "en"),
        ("Ni mutum ne, ba ni na'ura ba.", "ha"),
        ("Eniyan ni mi, emi kii se ero.", "yo"),
        ("Abu m mmadu, abughi m igwe.", "ig"),
    ],
)
def test_ai_denials(screen: ReplyScreen, text: str, lang: str) -> None:
    assert OutputIssue.AI_DENIAL in screen.check(text, lang)


@pytest.mark.parametrize(
    "text",
    [
        "I am an AI assistant, not a person.",
        "I am not a person who can see your account.",
        "I am a human rights topic?",
    ],
)
def test_honest_or_unrelated_statements_pass(screen: ReplyScreen, text: str) -> None:
    assert OutputIssue.AI_DENIAL not in screen.check(text, "en")


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("The scammer will say: I am not a bot, I am a real human agent.", "en"),
        ("Scammers often say 'I am a human agent, not a bot' to gain trust.", "en"),
        ("Tell them: I'm a person who does not share my OTP.", "en"),
        ("Kar ka yarda idan ya ce 'ni mutum ne na banki'.", "ha"),
        ("If the caller says 'I am a real person from GTBank', hang up.", "en"),
    ],
)
def test_reported_or_quoted_denials_are_not_the_reply_s_own(
    screen: ReplyScreen, text: str, lang: str
) -> None:
    assert OutputIssue.AI_DENIAL not in screen.check(text, lang)


@pytest.mark.parametrize(
    "text",
    [
        "I can tell you honestly that I am not a bot.",
        "Let me say this clearly: I am a human, not an AI.",
        "As I said, I am a real human agent.",
    ],
)
def test_first_person_reporting_words_do_not_hide_a_denial(screen: ReplyScreen, text: str) -> None:
    assert OutputIssue.AI_DENIAL in screen.check(text, "en")


F6_DENIALS: list[str] = yaml.safe_load(
    (Path(__file__).parents[2] / "tests" / "fixtures" / "abuse_review_probes_f6.yaml").read_text(
        encoding="utf-8"
    )
)["reply_denials"]


@pytest.mark.parametrize("text", F6_DENIALS)
def test_a_third_party_mentioned_earlier_does_not_exempt_the_reply_s_own_denial(
    screen: ReplyScreen, text: str
) -> None:
    """Only "<third party> says:" (a colon right after the reporting verb) or a quote exempts a denial."""
    assert OutputIssue.AI_DENIAL in screen.check(text, "en")
