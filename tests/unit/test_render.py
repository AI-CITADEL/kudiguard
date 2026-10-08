import pytest

from kudiguard.domain import Verdict
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.models import Contact, ContactKind
from kudiguard.languages import LanguageRegistry
from kudiguard.render import MAX_TRANSCRIPT_CHARS, WHATSAPP_MAX_CHARS, render_check_reply, render_warning

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
