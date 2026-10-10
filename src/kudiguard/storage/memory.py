"""In-memory Store for tests and local demos; same contract as SQLite (tests/unit/test_store_contract.py)."""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Collection
from datetime import date, datetime
from typing import Any

from kudiguard.domain import Interaction, Outcome, Verdict
from kudiguard.patterns import ScamPattern, deidentify, week_of
from kudiguard.privacy import UserHasher
from kudiguard.storage.base import StoreClosedError, UserState, VoiceSample, aware, utcnow
from kudiguard.storage.export import export_row, protect
from kudiguard.text import redact


class MemoryStore:
    def __init__(self, hasher: UserHasher) -> None:
        self._hasher = hasher
        self._users: dict[str, UserState] = {}
        self._claimed: dict[str, datetime] = {}
        self._interactions: dict[int, Interaction] = {}
        self._feedback: dict[int, bool] = {}
        self._next_id = 1
        self._closed = False
        self._patterns: dict[str, ScamPattern] = {}
        self._voice: dict[str, VoiceSample] = {}
        self._vault_id = uuid.uuid4().hex
        self._deleted: dict[str, datetime] = {}  # DELETE marker -> when

    async def get_user(self, user_hash: str) -> UserState | None:
        self._ensure_open()
        state = self._users.get(user_hash)
        return dataclasses.replace(state) if state else None

    async def save_user(self, state: UserState) -> None:
        self._ensure_open()
        self._users[state.user_hash] = dataclasses.replace(
            state,
            consented_at=aware(state.consented_at) if state.consented_at else None,
            donation_at=aware(state.donation_at) if state.donation_at else None,
            donation_withdrawn_at=aware(state.donation_withdrawn_at) if state.donation_withdrawn_at else None,
            welcome_at=aware(state.welcome_at) if state.welcome_at else None,
            donation_offered_at=aware(state.donation_offered_at) if state.donation_offered_at else None,
            created_at=aware(state.created_at),
            last_seen_at=aware(state.last_seen_at),
        )

    async def claim_message(self, channel: str, message_id: str) -> bool:
        self._ensure_open()
        key = self._hasher.message_key(channel, message_id)
        if key in self._claimed:
            return False
        self._claimed[key] = utcnow()
        return True

    async def add_interaction(self, interaction: Interaction) -> int:
        self._ensure_open()
        interaction_id = self._next_id
        self._next_id += 1
        self._interactions[interaction_id] = dataclasses.replace(
            protect(interaction, self._hasher), id=interaction_id
        )
        return interaction_id

    async def get_interaction(self, interaction_id: int) -> Interaction | None:
        self._ensure_open()
        found = self._interactions.get(interaction_id)
        return dataclasses.replace(found) if found else None

    async def set_outcome(self, interaction_id: int, outcome: Outcome, error: str | None = None) -> None:
        self._ensure_open()
        found = self._interactions.get(interaction_id)
        if found is not None:
            self._interactions[interaction_id] = dataclasses.replace(
                found, outcome=outcome, error=redact(error) if error else found.error
            )

    async def add_feedback(self, user_hash: str, interaction_id: int, helpful: bool) -> None:
        self._ensure_open()
        found = self._interactions.get(interaction_id)
        if found is not None and found.user_hash == user_hash:
            self._feedback[interaction_id] = helpful

    async def count_interactions_since(self, user_hash: str, since: datetime) -> int:
        self._ensure_open()
        cutoff = aware(since)
        return sum(
            1 for i in self._interactions.values() if i.user_hash == user_hash and i.created_at >= cutoff
        )

    async def count_active_users_since(self, since: datetime, channels: Collection[str] | None = None) -> int:
        self._ensure_open()
        cutoff = aware(since)
        return len(
            {
                i.user_hash
                for i in self._interactions.values()
                if i.created_at >= cutoff and (channels is None or i.channel in channels)
            }
        )

    async def delete_user(self, user_hash: str) -> int:
        self._ensure_open()
        self._deleted[self._hasher.deletion_marker(user_hash)] = utcnow()
        removed = int(self._users.pop(user_hash, None) is not None)
        for item in self._interactions.values():  # examples taken from this user's stored messages
            if (
                item.user_hash == user_hash
                and item.transcript
                and item.verdict
                in (
                    Verdict.LIKELY_SCAM,
                    Verdict.CAUTION,
                )
            ):
                key = (week_of(item.created_at.date()), deidentify(item.transcript))
                for pattern_id in [k for k, p in self._patterns.items() if (p.week, p.text) == key]:
                    del self._patterns[pattern_id]
                    removed += 1
        for interaction_id in [k for k, v in self._interactions.items() if v.user_hash == user_hash]:
            del self._interactions[interaction_id]
            removed += 1 + int(self._feedback.pop(interaction_id, None) is not None)
        return removed + len(await self.delete_voice_samples(user_hash))

    async def record_form_consent(self, state: UserState, expected: UserState | None) -> bool:
        self._ensure_open()
        current = self._users.get(state.user_hash)
        if expected is None:
            if current is not None:
                return False
            await self.save_user(state)
            return True
        if (
            current is None or current.last_seen_at != aware(expected.last_seen_at) or current.stopped
            or current.donation_version is not None or current.donation_withdrawn_at is not None
            or current.donation_offered_version is not None
        ):  # fmt: skip
            return False
        await self.save_user(
            dataclasses.replace(
                current, donation_version=state.donation_version, donation_at=state.donation_at,
                donation_source=state.donation_source, donation_offered_at=state.donation_offered_at,
                donation_offered_version=state.donation_offered_version,
            )
        )  # fmt: skip
        return True

    async def deleted_at(self, user_hash: str) -> datetime | None:
        self._ensure_open()
        return self._deleted.get(self._hasher.deletion_marker(user_hash))

    async def purge_deletion_markers(self, cutoff: datetime) -> int:
        self._ensure_open()
        old = [k for k, at in self._deleted.items() if at < aware(cutoff)]
        for key in old:
            del self._deleted[key]
        return len(old)

    async def purge_before(self, cutoff: datetime) -> int:
        self._ensure_open()
        cutoff = aware(cutoff)
        removed = 0
        for interaction_id in [k for k, v in self._interactions.items() if v.created_at < cutoff]:
            del self._interactions[interaction_id]
            removed += 1 + int(self._feedback.pop(interaction_id, None) is not None)
        donors = {s.user_hash for s in self._voice.values()}
        for user_hash in [k for k, v in self._users.items() if v.last_seen_at < cutoff and k not in donors]:
            del self._users[user_hash]
            removed += 1
        for key in [k for k, at in self._claimed.items() if at < cutoff]:
            del self._claimed[key]
            removed += 1
        return removed

    async def add_pattern(self, pattern: ScamPattern) -> None:
        self._ensure_open()
        self._patterns[pattern.id] = pattern

    async def purge_patterns_before(self, cutoff: date) -> int:
        self._ensure_open()
        old = [k for k, p in self._patterns.items() if p.week < week_of(cutoff)]
        for key in old:
            del self._patterns[key]
        return len(old)

    async def export_patterns(self) -> list[dict[str, Any]]:
        self._ensure_open()
        return [p.as_row() for p in sorted(self._patterns.values(), key=lambda p: (p.week, p.id))]

    async def export_interactions(self) -> list[dict[str, Any]]:
        self._ensure_open()
        return [export_row(i, self._feedback.get(k)) for k, i in sorted(self._interactions.items())]

    async def add_voice_sample(self, sample: VoiceSample) -> None:
        self._ensure_open()
        if sample.id in self._voice:
            raise ValueError("duplicate voice sample id")
        self._voice[sample.id] = dataclasses.replace(
            sample, week=week_of(sample.week), hypothesis=redact(sample.hypothesis)
        )

    async def voice_samples(self, user_hash: str | None = None) -> list[VoiceSample]:
        self._ensure_open()
        found = [s for s in self._voice.values() if user_hash is None or s.user_hash == user_hash]
        return sorted(found, key=lambda s: (s.week, s.id))

    async def delete_voice_samples(self, user_hash: str) -> list[str]:
        self._ensure_open()
        return self._drop_voice([k for k, s in self._voice.items() if s.user_hash == user_hash])

    async def delete_voice_sample(self, sample_id: str) -> bool:
        self._ensure_open()
        return bool(self._drop_voice([sample_id] if sample_id in self._voice else []))

    async def voice_vault_id(self) -> str:
        self._ensure_open()
        return self._vault_id

    async def purge_voice_samples(self, cutoff: date) -> list[str]:
        self._ensure_open()
        return self._drop_voice([k for k, s in self._voice.items() if s.week <= cutoff])

    def _drop_voice(self, ids: list[str]) -> list[str]:
        for key in ids:
            del self._voice[key]
        return sorted(ids)

    async def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise StoreClosedError("store is closed")
