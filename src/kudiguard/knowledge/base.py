"""The plug-in contract every knowledge source implements (file packs, partner APIs, future vector stores)."""

from __future__ import annotations

from typing import Protocol

from kudiguard.knowledge.models import PackManifest, Query, SourceResult


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
