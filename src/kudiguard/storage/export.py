"""Privacy helpers shared by every Store: protection before writing, and privacy-safe export rows."""

from __future__ import annotations

import dataclasses
from typing import Any

from kudiguard.domain import Interaction
from kudiguard.privacy import UserHasher
from kudiguard.storage.base import aware
from kudiguard.text import redact

# Cells starting with these are evaluated by spreadsheets (the full-width forms are folded to ASCII by some).
FORMULA_PREFIXES = ("=", "+", "-", "@", "\uff1d", "\uff0b", "\uff0d", "\uff20")
_LEADING_BLANKS = " \t\r\n\u3000"  # skipped by spreadsheets before a formula is recognised


def _redacted(value: str | None) -> str | None:
    return redact(value) if value else value


def protect(interaction: Interaction, hasher: UserHasher) -> Interaction:
    """What actually gets stored: hashed message id, redacted free text, UTC timestamp."""
    return dataclasses.replace(
        interaction,
        message_id=hasher.message_key(interaction.channel, interaction.message_id),
        transcript=_redacted(interaction.transcript),
        english=_redacted(interaction.english),
        reply=_redacted(interaction.reply),
        error=_redacted(interaction.error),
        created_at=aware(interaction.created_at),
    )


def spreadsheet_cell(value: str) -> str:
    """Neutralise CSV formula injection (OWASP): user text, e.g. a forwarded scam, may start with ``=HYPERLINK(``.

    A leading apostrophe makes Excel / Google Sheets show the text instead of evaluating it. Leading blanks are
    skipped when looking for a formula prefix (``" =1+1"``, ``"\\n@SUM(...)"``); a leading tab or carriage return is
    escaped on its own (OWASP)."""
    risky = value.lstrip(_LEADING_BLANKS).startswith(FORMULA_PREFIXES) or value.startswith(("\t", "\r"))
    return "'" + value if risky else value


def export_row(interaction: Interaction, helpful: bool | None) -> dict[str, Any]:
    return {
        "id": interaction.id,
        "created_at": interaction.created_at.isoformat(),
        "user": interaction.user_hash,
        "channel": interaction.channel,
        "language": interaction.language,
        "kind": interaction.kind.value,
        "outcome": interaction.outcome.value,
        "intent": interaction.intent,
        "verdict": interaction.verdict.value if interaction.verdict else None,
        "flags": [f.value for f in interaction.flags],
        "transcript": interaction.transcript,
        "english": interaction.english,
        "reply": interaction.reply,
        "reply_to": interaction.reply_to,
        "decode_ms": interaction.decode_ms,
        "asr_ms": interaction.asr_ms,
        "llm_ms": interaction.llm_ms,
        "total_ms": interaction.total_ms,
        "error": interaction.error,
        "helpful": helpful,
    }
