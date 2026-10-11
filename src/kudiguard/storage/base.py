"""Storage port: user state, the interaction log (validation evidence), feedback, de-duplication, retention.

Privacy is enforced by every Store, not left to callers: users are ``privacy.UserHasher`` ids, provider message ids
are replaced by ``UserHasher.message_key`` before they are stored (a WhatsApp wamid embeds the sender's number), and
free-text fields are passed through ``text.redact``. All datetimes must be timezone-aware.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Protocol

from kudiguard.domain import Interaction, LanguageCode, Outcome
from kudiguard.patterns import ScamPattern

RETENTION_DAYS = 90  # stated in the consent text
ACTIVE_USER_WINDOW_DAYS = 30  # N-ATLAS licence: 1,000 active end-users per rolling 30 days
DELETION_MARKER_DAYS = 30  # DELETE markers (only for the sign-up import); stated on /privacy


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
    # The CONSENT_VERSION of the welcome last *shown* (None = never welcomed, e.g. a row created by the sign-up form
    # import): main consent is recorded only after the current welcome was shown.
    welcome_version: str | None = None
    # When that welcome was sent (accepted by WhatsApp); consent comes only from messages sent after it.
    welcome_at: datetime | None = None
    # Voice-note donation (``kudiguard.donation``): separate, opt-in consent. ``donation_version`` is the donation
    # text the user accepted together with the 18+ and own-voice confirmations (the chat's "Yes, I am 18+" button or
    # the sign-up form's ticked boxes); None = not donating. It survives a change of the main consent text; only
    # STOP, STOP DONATING, "No" and DELETE end it. Kept while the user has donated samples, as proof of consent (s.26).
    donation_version: str | None = None
    donation_at: datetime | None = None
    donation_source: str | None = None  # how consent was given: "chat" or "form" (tester sign-up form)
    donation_offered_at: datetime | None = None  # when the offer was last shown
    donation_offered_version: str | None = (
        None  # which offer text: offered once per text, to old donors again
    )
    donation_withdrawn_at: datetime | None = None  # last "No", STOP DONATING or STOP while donating


@dataclass(frozen=True, slots=True)
class VoiceSample:
    """One donated voice note's record (the audio itself is an encrypted file named by ``id``).

    Minimised: a random id, the donor's pseudonymous hash (for withdrawal, erasure and access requests), the week
    (Monday) instead of a timestamp, and the speech-recognition hypothesis, redacted by every Store."""

    id: str
    user_hash: str
    week: date
    language: LanguageCode
    channel: str
    consent_version: str  # the donation text the donor accepted (per-sample proof of consent)
    hypothesis: str
    duration_s: float | None = None


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

    async def count_active_users_since(self, since: datetime, channels: Collection[str] | None = None) -> int:
        """Distinct users with an interaction since ``since`` (licence cap), optionally only on ``channels``."""
        ...

    async def delete_user(self, user_hash: str) -> int:
        """Erase everything stored for a user, atomically (the DELETE command), including scam examples taken from
        their stored messages and their donated voice-sample rows (files: ``donation.VoiceVault.erase``). Leaves only
        a coded DELETE marker (``deleted_at``). Returns rows removed (the marker is not counted)."""
        ...

    async def record_form_consent(self, state: UserState, expected: UserState | None) -> bool:
        """Compare-and-set for the sign-up import: store ``state``'s donation consent (version, time, source and the
        offer marks) only if the user row is still exactly as read (``expected``; None = no row yet), never stopped,
        withdrawn or offered meanwhile. Touches no other field. Returns whether it was recorded."""
        ...

    async def deleted_at(self, user_hash: str) -> datetime | None:
        """When this user last sent DELETE, from the coded marker ``delete_user`` leaves (``DELETION_MARKER_DAYS``).
        Read only by the sign-up form import."""
        ...

    async def purge_deletion_markers(self, cutoff: datetime) -> int:
        """Remove DELETE markers older than ``cutoff``. Returns markers removed."""
        ...

    async def purge_before(self, cutoff: datetime) -> int:
        """Retention: atomically remove data older than ``cutoff``. Returns rows removed. A user row is kept while
        the user has donated voice samples (it holds their donation consent record)."""
        ...

    async def add_pattern(self, pattern: ScamPattern) -> None:
        """Keep a de-identified scam example (no link to any user; see ``kudiguard.patterns``)."""
        ...

    async def purge_patterns_before(self, cutoff: date) -> int:
        """Delete scam examples whose week starts before ``cutoff`` (their own, longer retention): such a week may
        hold an example dated before ``cutoff``, so none outlives it (a week's later examples go a few days early).
        Rows removed."""
        ...

    async def export_patterns(self) -> list[dict[str, Any]]:
        """All scam examples, ordered by week (never by insertion order)."""
        ...

    async def export_interactions(self) -> list[dict[str, Any]]:
        """Rows for validation reporting (pseudonymous ids, redacted text, verdicts, latencies, feedback)."""
        ...

    async def interaction_ids(self) -> frozenset[int]:
        """Ids of every stored interaction (cheap: no text is read), e.g. to spot rows erased since a review export."""
        ...

    async def add_voice_sample(self, sample: VoiceSample) -> None:
        """Record a donated voice note (the hypothesis is redacted). The audio file is the caller's
        (``donation.VoiceVault``). Raises ``ValueError`` for an id that already exists."""
        ...

    async def voice_samples(self, user_hash: str | None = None) -> list[VoiceSample]:
        """Donated samples, all or one donor's, ordered by week then id (never by insertion order)."""
        ...

    async def delete_voice_samples(self, user_hash: str) -> list[str]:
        """Withdrawal/erasure: remove a donor's sample rows; returns their ids so the files can be deleted."""
        ...

    async def delete_voice_sample(self, sample_id: str) -> bool:
        """Remove one sample row (a reviewer found a child's or another person's voice). False if unknown."""
        ...

    async def voice_vault_id(self) -> str:
        """A random id created once per database; the voice folder carries it too, so files are only ever swept
        against the database they belong to."""
        ...

    async def purge_voice_samples(self, cutoff: date) -> list[str]:
        """Retention: remove samples from weeks starting on or before ``cutoff`` (so none outlives its period);
        returns their ids."""
        ...

    async def close(self) -> None: ...
