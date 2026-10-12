import pytest

from kudiguard.domain import Verdict
from kudiguard.i18n import VERDICT_HEADLINE, Messages, Msg
from kudiguard.knowledge.models import Contact, ContactKind
from kudiguard.languages import LanguageRegistry
from kudiguard.render import (
    MAX_TRANSCRIPT_CHARS,
    WHATSAPP_MAX_CHARS,
    render_check_reply,
    render_report,
    render_steps,
    render_warning,
)

BANK = Contact(
    label="Your own bank",
    kind=ContactKind.NOTE,
    value="Call the number on the back of your card",
    pinned=True,
    verified=True,
    label_i18n={"ha": "Bankinku"},
    value_i18n={"ha": "Ku kira lambar da ke bayan katinku"},
)
EFCC = Contact(label="EFCC", kind=ContactKind.WEB, value="https://www.efcc.gov.ng", verified=True)
UNVERIFIED = Contact(label="NCC", kind=ContactKind.PHONE, value="622", verified=False)


@pytest.fixture(scope="module")
def messages(registry: LanguageRegistry) -> Messages:
    return Messages(registry)


def test_scam_reply_layout(messages: Messages) -> None:
    text = render_check_reply(
        messages,
        "en",
        verdict=Verdict.LIKELY_SCAM,
        body="This is a scam.\n1. Do not share the code.",
        transcript="they asked for my OTP",
        contacts=(BANK, EFCC),
    )
    assert text == (
        "🔴 *This looks like a SCAM*\n"
        "🗣️ I heard: “they asked for my OTP”\n"
        "\n"
        "This is a scam.\n1. Do not share the code.\n"
        "\n"
        "*Where to report:*\n"
        "• Your own bank: Call the number on the back of your card\n"
        "• EFCC: https://www.efcc.gov.ng\n"
        "\n"
        "_Guidance only, not financial advice._"
    )


def test_non_english_voice_note_with_no_flags_is_not_reassured(messages: Messages) -> None:
    text = render_check_reply(
        messages, "ha", verdict=Verdict.NO_FLAGS, body="Shawara.", transcript="abokina ya kira", voice=True
    )
    assert "🟢" not in text and messages.text("ha", Msg.VERDICT_NO_FLAGS) not in text
    heard = messages.text("ha", Msg.I_HEARD, transcript="abokina ya kira")
    assert text.startswith(f"{heard}\n{messages.text('ha', Msg.VOICE_UNSURE)}\n\nShawara.")


@pytest.mark.parametrize(
    ("language", "verdict", "voice"),
    [
        ("en", Verdict.NO_FLAGS, True),  # English voice: unchanged
        ("ha", Verdict.NO_FLAGS, False),  # typed, or the option off
        ("ha", Verdict.CAUTION, True),
        ("ha", Verdict.LIKELY_SCAM, True),
    ],
)
def test_other_replies_keep_their_headline(
    messages: Messages, language: str, verdict: Verdict, voice: bool
) -> None:
    text = render_check_reply(messages, language, verdict=verdict, body="x", transcript="words", voice=voice)
    headline = messages.text(language, VERDICT_HEADLINE[verdict])
    assert text.startswith(headline) and messages.text(language, Msg.VOICE_UNSURE) not in text


def test_general_question_has_no_headline_or_contacts(messages: Messages) -> None:
    text = render_check_reply(
        messages, "en", verdict=Verdict.NOT_A_CHECK, body="Save a fixed amount.", contacts=(BANK, EFCC)
    )
    assert text == "Save a fixed amount.\n\n_Guidance only, not financial advice._"


def test_no_flags_reply_has_headline_but_no_report_section(messages: Messages) -> None:
    text = render_check_reply(
        messages, "en", verdict=Verdict.NO_FLAGS, body="Looks ordinary.", contacts=(BANK,)
    )
    assert text.startswith("🟢 *No red flags found.*")
    assert "Where to report" not in text


def test_unverified_contacts_are_never_shown(messages: Messages) -> None:
    text = render_check_reply(
        messages, "en", verdict=Verdict.CAUTION, body="Be careful.", contacts=(UNVERIFIED,)
    )
    assert "622" not in text
    assert "Where to report" not in text  # no verified contacts -> no section


def test_contacts_localised_but_numbers_never_change(messages: Messages, registry: LanguageRegistry) -> None:
    text = render_check_reply(
        messages, "ha", verdict=Verdict.LIKELY_SCAM, body="Damfara ce.", contacts=(BANK, EFCC)
    )
    assert "• Bankinku: Ku kira lambar da ke bayan katinku" in text
    assert "• EFCC: https://www.efcc.gov.ng" in text
    assert registry.get("ha").strings[Msg.WHERE_TO_REPORT] in text


def test_long_transcript_is_shortened(messages: Messages) -> None:
    text = render_check_reply(
        messages, "en", verdict=Verdict.CAUTION, body="Careful.", transcript="word " * 200
    )
    heard = next(line for line in text.splitlines() if line.startswith("🗣️"))
    assert len(heard) < MAX_TRANSCRIPT_CHARS + 40
    assert "…" in heard


def test_reply_never_exceeds_whatsapp_limit(messages: Messages) -> None:
    text = render_check_reply(
        messages, "en", verdict=Verdict.LIKELY_SCAM, body="Sentence. " * 1000, contacts=(BANK, EFCC)
    )
    assert len(text) <= WHATSAPP_MAX_CHARS
    assert text.endswith("_Guidance only, not financial advice._")  # disclaimer and contacts survive
    assert "EFCC" in text


def test_empty_body_falls_back_to_fixed_text(messages: Messages) -> None:
    text = render_check_reply(messages, "en", verdict=Verdict.LIKELY_SCAM, body="   ", contacts=(BANK,))
    assert text.startswith("🔴")
    assert "\n\n\n" not in text


def test_render_warning(messages: Messages) -> None:
    text = render_warning(messages, "en", "⚠️ Never share your OTP with anyone who calls you.")
    assert text == (
        "Forward this to your family and friends 👇\n"
        "\n"
        "⚠️ Never share your OTP with anyone who calls you.\n"
        "\n"
        "_Shared from KudiGuard. Check suspicious messages for free on WhatsApp._"
    )


def test_note_translation_rejected_for_numbers() -> None:
    with pytest.raises(ValueError, match="never translated"):
        Contact(label="X", kind=ContactKind.PHONE, value="622", value_i18n={"ha": "623"})


def test_transcript_markup_cannot_restyle_reply(messages: Messages) -> None:
    text = render_check_reply(
        messages, "en", verdict=Verdict.CAUTION, body="Careful.", transcript="*bold* _it_ ~x~ `code`"
    )
    assert "“bold it x code”" in text


def test_plain_text_strips_whatsapp_markup() -> None:
    from kudiguard.render import plain_text

    assert (
        plain_text("🔴 *This looks like a SCAM*\n_Guidance only._")
        == "🔴 This looks like a SCAM\nGuidance only."
    )


# --- exact limits, found by mutation testing (docs/SPIKES.md, G) ----------------------------------------------


def heard(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("🗣️"))


def test_transcript_is_cut_to_exactly_280_characters(messages: Messages) -> None:
    def shown(transcript: str) -> str:
        reply = render_check_reply(messages, "en", verdict=Verdict.CAUTION, body="x", transcript=transcript)
        return heard(reply).split("“", 1)[1].rsplit("”", 1)[0]

    assert shown("a" * 280) == "a" * 280
    assert shown("a" * 281) == "a" * 279 + "…"
    assert shown("Xray *b* " * 2) == "Xray b Xray b"  # only WhatsApp markup is removed


LANGUAGES = ("ha", "yo", "ig", "en")  # their fixed texts differ in length (odd and even budgets)


def scam_reply(messages: Messages, body: str, language: str = "en") -> str:
    return render_check_reply(
        messages,
        language,
        verdict=Verdict.LIKELY_SCAM,
        body=body,
        transcript="they asked",
        contacts=(BANK, EFCC),
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_long_body_fills_the_message_exactly(messages: Messages, language: str) -> None:
    budget = 4096 - (
        len(scam_reply(messages, "x", language)) - 1
    )  # what the fixed sections leave for the body
    fits = scam_reply(messages, "b" * budget, language)
    assert len(fits) == 4096 and "b" * budget in fits
    cut = scam_reply(messages, "b" * (budget + 1), language)
    assert len(cut) == 4096 and "b" * (budget - 1) + "…" in cut and "b" * budget not in cut


@pytest.mark.parametrize(
    ("body", "kept"),
    [
        ("Line one\n" + "c" * 5000, "Line one…"),  # cut at the last line break
        ("First sentence. " + "d" * 5000, "First sentence.…"),  # or after the last full stop
        ("A\n" + "e" * 5000, "A…"),
        ("Odd. " + "g" * 5000, "Odd.…"),  # a full stop at an odd position
        ("Para.\nNext. " + "f" * 5000, "Para.\nNext.…"),  # whichever comes last
    ],
    ids=["line", "sentence", "line-at-1", "odd-stop", "last-wins"],
)
def test_a_long_body_is_cut_at_a_boundary(messages: Messages, body: str, kept: str) -> None:
    text = scam_reply(messages, body)
    assert f"\n\n{kept}\n\n*Where to report:*" in text


def test_a_long_warning_fills_the_message_exactly(messages: Messages) -> None:
    text = render_warning(messages, "en", "w" * 5000)
    assert len(text) == 4096 and text.endswith(
        "…\n\n_Shared from KudiGuard. Check suspicious messages for free on WhatsApp._"
    )


def test_steps_and_report_layout(messages: Messages) -> None:
    steps = render_steps(messages, "en", "1. Call your bank.\n2. Block the card.")
    assert (
        steps
        == "*What to do:*\n1. Call your bank.\n2. Block the card.\n\n_Guidance only, not financial advice._"
    )
    for language in LANGUAGES:  # header and steps share a line, so a full message is 4096 - 1
        long = render_steps(messages, language, "s" * 5000)
        assert len(long) == 4095 and long.endswith(f"…\n\n{messages.text(language, Msg.DISCLAIMER)}")
    report = render_report(messages, "en", (BANK, UNVERIFIED, EFCC))
    assert report == (
        "*Where to report:*\n• Your own bank: Call the number on the back of your card\n• EFCC: https://www.efcc.gov.ng"
    )
