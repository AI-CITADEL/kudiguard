"""Knowledge types. Pydantic models are the validation boundary for pack files and partner APIs."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from kudiguard.domain import Flag, LanguageCode

Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,79}$")]
Short = Annotated[str, StringConstraints(min_length=1, max_length=200, strip_whitespace=True)]
Long = Annotated[str, StringConstraints(min_length=1, max_length=1000, strip_whitespace=True)]
OWN_PACK_OWNER = "KudiGuard"  # owner of the packs we write ourselves


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContactKind(StrEnum):
    PHONE = "phone"
    WHATSAPP = "whatsapp"
    USSD = "ussd"
    WEB = "web"
    EMAIL = "email"
    APP = "app"
    NOTE = "note"  # an instruction rather than an address, e.g. "the number on the back of your card"


_PHONE_VALUE = re.compile(r"^(?=(?:\D*\d){3})[0-9A-Za-z +()./-]{3,40}$")
_CONTACT_VALUE = {
    ContactKind.PHONE: _PHONE_VALUE,  # digits plus vanity letters / extensions: 0700-CALL-CBN, "... ext 2"
    ContactKind.WHATSAPP: _PHONE_VALUE,
    ContactKind.USS***REMOVED*** re.compile(r"^[*#\d]{2,30}$"),
    ContactKind.WEB: re.compile(r"^https://[^\s/$.?#][^\s]*$"),
    ContactKind.EMAIL: re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$"),
}


class Contact(_Strict):
    label: Short
    kind: ContactKind
    value: Short
    flags: tuple[Flag, ...] = ()  # empty = general contact; otherwise relevant to these red flags
    pinned: bool = False  # always shown first (e.g. "your own bank")
    verified: bool = False  # a human checked this against an official source
    note: Short | None = None
    label_i18n: dict[LanguageCode, Short] = Field(default_factory=dict)  # label per language
    value_i18n: dict[LanguageCode, Short] = Field(default_factory=dict)  # only for `note` (instructions)

    @model_validator(mode="after")
    def _value_matches_kind(self) -> Contact:
        pattern = _CONTACT_VALUE.get(self.kind)
        if pattern and not pattern.match(self.value):
            raise ValueError(f"{self.kind.value} contact has an invalid value: {self.value!r}")
        if self.value_i18n and self.kind is not ContactKind.NOTE:
            raise ValueError(
                "value_i18n is only allowed for note contacts: numbers and links are never translated"
            )
        return self

    def label_for(self, language: LanguageCode) -> str:
        return self.label_i18n.get(language, self.label)

    def value_for(self, language: LanguageCode) -> str:
        return self.value_i18n.get(language, self.value)


class Entry(_Strict):
    id: Slug
    title: Short
    facts: tuple[Long, ...] = Field(min_length=1, max_length=12)
    actions: tuple[Long, ...] = Field(default=(), max_length=8)
    flags: tuple[Flag, ...] = ()
    keywords: tuple[Short, ...] = ()
    # A model answer for a typical message on this topic: shown to COMPOSE (top entry only) as something to adapt,
    # never to copy (``prompts.compose_request``; copy guard in ``engine``). Not indexed for search.
    example: Long | None = None


class PackKind(StrEnum):
    CORE = "core"  # our safety content; always ranked first for red-flag entries
    PARTNER = "partner"


class FilesSourceSpec(_Strict):
    type: Literal["files"] = "files"


class HttpAuth(_Strict):
    header: Short = "Authorization"
    scheme: Short | None = "Bearer"
    env: Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$")]  # token comes from this env var


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_link_local(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_link_local
    except ValueError:  # a hostname, not an IP literal
        return False


class HttpSourceSpec(_Strict):
    type: Literal["http"]
    base_url: Annotated[str, StringConstraints(pattern=r"^https?://\S+$")]
    allow_insecure: bool = False  # permit plain http to non-loopback hosts (development only)
    mode: Literal["sync", "live"] = "sync"
    timeout_s: float = Field(default=1.5, gt=0, le=10)
    refresh_s: int = Field(default=3600, ge=60)
    cache_ttl_s: int = Field(default=300, ge=0)
    max_bytes: int = Field(default=1_000_000, gt=0, le=10_000_000)
    auth: HttpAuth | None = None

    @model_validator(mode="after")
    def _safe_url(self) -> HttpSourceSpec:
        parts = urlsplit(self.base_url)
        host = parts.hostname or ""
        if parts.scheme != "https" and host not in _LOOPBACK_HOSTS and not self.allow_insecure:
            raise ValueError(
                f"base_url must use https (got {self.base_url!r}); tokens must not travel in plaintext"
            )
        if _is_link_local(host):
            raise ValueError(f"base_url host {host!r} is link-local (cloud metadata range)")
        return self


class PackManifest(_Strict):
    id: Slug
    name: Short
    owner: Short
    kind: PackKind = PackKind.PARTNER
    priority: int = 0  # higher = earlier for search hits and contacts
    content_language: LanguageCode = "en"
    enabled: bool = True
    sample: bool = False  # fictional demo content; never present as real
    cite: bool = True  # partner facts carry "(source: <name>)" in prompts; False for KudiGuard's own packs
    description: Annotated[str, StringConstraints(max_length=500)] = ""
    source: Annotated[FilesSourceSpec | HttpSourceSpec, Field(discriminator="type")] = FilesSourceSpec()

    @model_validator(mode="after")
    def _partners_are_cited(self) -> PackManifest:
        if not self.cite and self.owner != OWN_PACK_OWNER:
            raise ValueError(f"cite: false is reserved for {OWN_PACK_OWNER}'s own packs")
        return self


class KnowledgePayload(_Strict):
    """Wire format for partner APIs (``/export`` and ``/search`` responses)."""

    entries: tuple[Entry, ...] = Field(default=(), max_length=2000)
    contacts: tuple[Contact, ...] = Field(default=(), max_length=200)


@dataclass(frozen=True, slots=True)
class Query:
    text: str  # English pivot text
    flags: tuple[Flag, ...] = ()
    language: LanguageCode = "en"
    limit: int = 3
    packs: frozenset[str] | None = None  # partner packs in scope (None = all loaded); core is always in scope
    vector: NDArray[np.float32] | None = field(
        default=None, compare=False, repr=False
    )  # precomputed embedding


@dataclass(frozen=True, slots=True)
class Hit:
    entry: Entry
    relevance: float | None = (
        None  # cosine similarity when embeddings are on, else squashed BM25; None = unscored
    )


@dataclass(frozen=True, slots=True)
class SourceResult:
    flag_entries: tuple[Entry, ...] = ()  # entries tagged with the query's red flags
    hits: tuple[Hit, ...] = ()  # text-search matches, best first
    contacts: tuple[Contact, ...] = ()  # flag-relevant first, then general


@dataclass(frozen=True, slots=True)
class SourcedEntry:
    entry: Entry
    source_id: str
    source_name: str
    sample: bool = False
    relevance: float | None = None
    kind: PackKind = PackKind.CORE
    cite: bool = True


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    entries: tuple[SourcedEntry, ...] = ()
    contacts: tuple[Contact, ...] = ()
    failed_sources: tuple[str, ...] = ()
    dense: bool = False  # hit relevance is cosine similarity (embeddings on), else squashed BM25


@dataclass(frozen=True, slots=True)
class Relevance:
    """A relevance level on both scales a hit can carry (``Hit.relevance``)."""

    dense: float  # cosine similarity (bge-small-en-v1.5)
    bm25: float  # squashed BM25, score / (score + 3)

    def met(self, relevance: float | None, *, dense: bool) -> bool:
        """None (an unscored hit, e.g. from a partner API) never meets a level."""
        return relevance is not None and relevance >= (self.dense if dense else self.bm25)
