"""Privacy helpers shared by every Store: protection before writing, and privacy-safe export rows."""

from __future__ import annotations

import dataclasses
from typing import Any

from kudiguard.domain import Interaction
from kudiguard.privacy import UserHasher
from kudiguard.storage.base import aware
from kudiguard.text import redact


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
