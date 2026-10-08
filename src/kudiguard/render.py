"""Assemble WhatsApp-ready replies from fixed text, the model's body and verified contacts. Pure functions.

Layout (sections joined by a blank line, empty sections skipped):

    🔴 *headline*                      fixed text for scam checks only
    🗣️ I heard: “transcript”           voice notes only
    body                               model text, already cleaned (prompts.clean_reply)
    *Where to report:*                 likely-scam / caution only, verified contacts only
    • label: value                     numbers and links inserted verbatim, never by the model
    _disclaimer_
"""

from __future__ import annotations

from collections.abc import Sequence

from kudiguard.domain import LanguageCode, Verdict
from kudiguard.i18n import VERDICT_HEADLINE, Messages, Msg
from kudiguard.knowledge.models import Contact

WHATSAPP_MAX_CHARS = 4096
MAX_TRANSCRIPT_CHARS = 280
REPORT_VERDICTS = frozenset({Verdict.LIKELY_SCAM, Verdict.CAUTION})


_FORMATTING = str.maketrans("", "", "*_~`")  # WhatsApp markup; user text must not restyle our layout


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.translate(_FORMATTING).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _fit(body: str, budget: int) -> str:
    """Trim the model's body (never fixed text) to ``budget`` chars at a line, then sentence, boundary."""
    if len(body) <= budget:
        return body
    head = body[: max(budget - 1, 0)]
    cut = max(head.rfind("\n"), head.rfind(". ") + 1)
    return (head[:cut] if cut > 0 else head).rstrip() + "…"


def contact_line(contact: Contact, language: LanguageCode) -> str:
    return f"• {contact.label_for(language)}: {contact.value_for(language)}"


def render_check_reply(
    messages: Messages,
    language: LanguageCode,
    *,
    verdict: Verdict,
    body: str,
    transcript: str | None = None,
    contacts: Sequence[Contact] = (),
) -> str:
    header = []
    if headline := VERDICT_HEADLINE.get(verdict):
        header.append(messages.text(language, headline))
    if transcript and transcript.strip():
        header.append(
            messages.text(language, Msg.I_HEARD, transcript=_shorten(transcript, MAX_TRANSCRIPT_CHARS))
        )

    shown = [c for c in contacts if c.verified] if verdict in REPORT_VERDICTS else []
    report = (
        "\n".join([messages.text(language, Msg.WHERE_TO_REPORT), *(contact_line(c, language) for c in shown)])
        if shown
        else ""
    )
    disclaimer = messages.text(language, Msg.DISCLAIMER)

    fixed = [part for part in ("\n".join(header), report, disclaimer) if part]
    budget = WHATSAPP_MAX_CHARS - sum(map(len, fixed)) - 2 * len(fixed)  # 2 = blank-line separator
    sections = ["\n".join(header), _fit(body.strip(), budget), report, disclaimer]
    return "\n\n".join(part for part in sections if part)


def render_warning(messages: Messages, language: LanguageCode, warning: str) -> str:
    """The forwardable "warn my family" message."""
    intro = messages.text(language, Msg.WARN_FAMILY_INTRO)
    footer = f"_{messages.text(language, Msg.WARN_FAMILY_FOOTER)}_"
    budget = WHATSAPP_MAX_CHARS - len(intro) - len(footer) - 4
    return "\n\n".join((intro, _fit(warning.strip(), budget), footer))
