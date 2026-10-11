"""The plug-in contract every knowledge source implements (file packs, partner APIs, future vector stores)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from kudiguard.knowledge.models import Contact, Entry, PackManifest, Query, SourceResult


class SourceUnavailable(RuntimeError):
    """A source could not answer (network, timeout, bad payload, open circuit). Callers skip it."""


class KnowledgeSource(Protocol):
    manifest: PackManifest

    async def search(self, query: Query) -> SourceResult:
        """Return red-flag entries, ranked hits and contacts. May raise ``SourceUnavailable``."""
        ...

    async def refresh(self) -> None:
        """Re-read or re-sync content (no-op where not applicable). May raise ``SourceUnavailable``."""
        ...


class ContentScreen(Protocol):
    """Filters pack content before it is indexed (``safety.pack_screen.PackScreen``: planted instructions, unsafe
    advice). Returns what may be used; logs and drops the rest."""

    def entries(self, manifest: PackManifest, entries: Sequence[Entry]) -> list[Entry]: ...

    def contacts(self, manifest: PackManifest, contacts: Sequence[Contact]) -> list[Contact]: ...
