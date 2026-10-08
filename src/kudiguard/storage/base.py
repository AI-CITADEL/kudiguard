"""Storage port: user state, the interaction log (validation evidence), feedback, de-duplication, retention.

Privacy is enforced by every Store, not left to callers: users are ``privacy.UserHasher`` ids, provider message ids
are replaced by ``UserHasher.message_key`` before they are stored (a WhatsApp wamid embeds the sender's number), and
free-text fields are passed through ``text.redact``. All datetimes must be timezone-aware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from kudiguard.domain import Interaction, LanguageCode, Outcome

RETENTION_DAYS = 90  # stated in the consent text
ACTIVE_USER_WINDOW_DAYS = 30  # N-ATLAS licence: 1,000 active end-users per rolling 30 days


def utcnow() -> datetime:
    return datetime.now(UTC)


def aware(value: datetime) -> datetime:
    """Normalise to UTC; naive datetimes are a bug (local time is ambiguous across servers)."""
    if value.tzinfo is None:
        raise ValueError("naive datetime: pass a timezone-aware value (UTC)")
    return value.astimezone(UTC)


@dataclass(slots=True)
class UserState:
    user_hash: str
    language: LanguageCode | None = None
    consented_at: datetime | None = None  # first contact after which the consent text was shown
    consent_version: str | None = None  # re-prompt when the consent text changes
    stopped: bool = False  # sent STOP: no messages until they write again
    last_interaction_id: int | None = None  # for button follow-ups ("What should I do?", "Warn my family")
    created_at: datetime = field(default_factory=utcnow)
    last_seen_at: datetime = field(default_factory=utcnow)


class StoreClosedError(RuntimeError):
    pass


class Store(Protocol):
    async def get_user(self, user_hash: str) -> UserState | None: ...

    async def save_user(self, state: UserState) -> None: ...

    async def claim_message(self, channel: str, message_id: str) -> bool:
        """Record a provider message (stored only as a keyed hash); False if already claimed (retry/duplicate)."""
        ...

    async def add_interaction(self, interaction: Interaction) -> int:
        """Persist a turn (message id hashed, text redacted) and return its id."""
        ...

    async def get_interaction(self, interaction_id: int) -> Interaction | None: ...

    async def set_outcome(self, interaction_id: int, outcome: Outcome, error: str | None = None) -> None:
        """Correct a logged turn's outcome (e.g. the answer could not be delivered). ``error`` is redacted."""
        ...

    async def add_feedback(self, user_hash: str, interaction_id: int, helpful: bool) -> None:
        """Record 👍/👎 for one of the user's own interactions (latest vote wins)."""
        ...

    async def count_interactions_since(self, user_hash: str, since: datetime) -> int: ...

    async def count_active_users_since(self, since: datetime) -> int:
        """Distinct users with an interaction since ``since`` (licence cap monitoring)."""
        ...

    async def delete_user(self, user_hash: str) -> int:
        """Erase everything stored for a user, atomically (the DELETE command). Returns rows removed."""
        ...

    async def purge_before(self, cutoff: datetime) -> int:
        """Retention: atomically remove data older than ``cutoff``. Returns rows removed."""
        ...

    async def export_interactions(self) -> list[dict[str, Any]]:
        """Rows for validation reporting (pseudonymous ids, redacted text, verdicts, latencies, feedback)."""
        ...

    async def close(self) -> None: ...
