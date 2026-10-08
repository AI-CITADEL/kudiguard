"""In-memory Store for tests and local demos; same contract as SQLite (tests/unit/test_store_contract.py)."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

from kudiguard.domain import Interaction
from kudiguard.privacy import UserHasher
from kudiguard.storage.base import StoreClosedError, UserState, aware, utcnow
from kudiguard.storage.export import export_row, protect


class MemoryStore:
    def __init__(self, hasher: UserHasher) -> None:
        self._hasher = hasher
        self._users: dict[str, UserState] = {}
        self._claimed: dict[str, datetime] = {}
        self._interactions: dict[int, Interaction] = {}
        self._feedback: dict[int, bool] = {}
        self._next_id = 1
        self._closed = False

    async def get_user(self, user_hash: str) -> UserState | None:
        self._ensure_open()
        state = self._users.get(user_hash)
        return dataclasses.replace(state) if state else None

    async def save_user(self, state: UserState) -> None:
        self._ensure_open()
        self._users[state.user_hash] = dataclasses.replace(
            state,
            consented_at=aware(state.consented_at) if state.consented_at else None,
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

    async def count_active_users_since(self, since: datetime) -> int:
        self._ensure_open()
        cutoff = aware(since)
        return len({i.user_hash for i in self._interactions.values() if i.created_at >= cutoff})

    async def delete_user(self, user_hash: str) -> int:
        self._ensure_open()
        removed = int(self._users.pop(user_hash, None) is not None)
        for interaction_id in [k for k, v in self._interactions.items() if v.user_hash == user_hash]:
            del self._interactions[interaction_id]
            removed += 1 + int(self._feedback.pop(interaction_id, None) is not None)
        return removed

    async def purge_before(self, cutoff: datetime) -> int:
        self._ensure_open()
        cutoff = aware(cutoff)
        removed = 0
        for interaction_id in [k for k, v in self._interactions.items() if v.created_at < cutoff]:
            del self._interactions[interaction_id]
            removed += 1 + int(self._feedback.pop(interaction_id, None) is not None)
        for user_hash in [k for k, v in self._users.items() if v.last_seen_at < cutoff]:
            del self._users[user_hash]
            removed += 1
        for key in [k for k, at in self._claimed.items() if at < cutoff]:
            del self._claimed[key]
            removed += 1
        return removed

    async def export_interactions(self) -> list[dict[str, Any]]:
        self._ensure_open()
        return [export_row(i, self._feedback.get(k)) for k, i in sorted(self._interactions.items())]

    async def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise StoreClosedError("store is closed")
