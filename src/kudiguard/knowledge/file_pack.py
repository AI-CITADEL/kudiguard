"""Knowledge packs stored as plain files — the format partners can produce without engineering help.

<pack>/pack.yaml       manifest (required)
<pack>/topics.yaml     curated entries: list of Entry objects (optional)
<pack>/faq.csv         question,answer[,id][,keywords "a; b"] rows, e.g. an exported FAQ (optional)
<pack>/contacts.yaml   list of Contact objects (optional)
<pack>/examples.yaml   entry id -> model answer text (optional; same as an inline ``example:`` in topics.yaml)
"""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from kudiguard.knowledge.base import ContentScreen
from kudiguard.knowledge.embeddings import Embedder
from kudiguard.knowledge.index import EntryIndex
from kudiguard.knowledge.models import (
    Contact,
    Entry,
    FilesSourceSpec,
    PackManifest,
    Query,
    SourceResult,
)

log = logging.getLogger(__name__)

_ENTRIES = TypeAdapter(list[Entry])
_CONTACTS = TypeAdapter(list[Contact])
_EXAMPLES = TypeAdapter(dict[str, str])
MAX_FILE_BYTES = 5_000_000
MAX_FACT_CHARS = 1000
MAX_FACTS = 12
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class KnowledgeConfigError(ValueError):
    pass


def _read_text(path: Path, encoding: str = "utf-8") -> str:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise KnowledgeConfigError(f"{path}: larger than {MAX_FILE_BYTES} bytes")
        return path.read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError) as exc:
        raise KnowledgeConfigError(f"{path}: {exc}") from exc


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(_read_text(path))
    except yaml.YAMLError as exc:
        raise KnowledgeConfigError(f"{path}: {exc}") from exc


def split_answer(answer: str) -> tuple[str, ...]:
    """Split a long FAQ answer into facts of at most ``MAX_FACT_CHARS`` on sentence (then word) boundaries."""
    answer = " ".join(answer.split())
    if len(answer) <= MAX_FACT_CHARS:
        return (answer,) if answer else ()
    facts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(answer):
        while len(sentence) > MAX_FACT_CHARS:  # a single huge sentence: cut at the last space that fits
            cut = (
                sentence.rfind(" ", 0, MAX_FACT_CHARS) if " " in sentence[:MAX_FACT_CHARS] else MAX_FACT_CHARS
            )
            head, sentence = sentence[:cut], sentence[cut:].strip()
            if current:
                facts.append(current)
                current = ""
            facts.append(head)
        if current and len(current) + 1 + len(sentence) > MAX_FACT_CHARS:
            facts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:  # pragma: no branch - always true here: the caller only splits answers over the length cap
        facts.append(current)
    if len(facts) > MAX_FACTS:
        log.warning("FAQ answer truncated to %d facts (%d chars)", MAX_FACTS, len(answer))
    return tuple(facts[:MAX_FACTS])


def load_manifest(directory: Path) -> PackManifest:
    path = directory / "pack.yaml"
    if not path.is_file():
        raise KnowledgeConfigError(f"{path}: missing pack.yaml")
    try:
        return PackManifest.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise KnowledgeConfigError(f"{path}: {exc}") from exc


def _read_list(path: Path, adapter: TypeAdapter[list[Any]]) -> list[Any]:
    if not path.is_file():
        return []
    try:
        return adapter.validate_python(_read_yaml(path) or [])
    except ValidationError as exc:
        raise KnowledgeConfigError(f"{path}: {exc}") from exc


def _read_faq(path: Path) -> list[Entry]:
    if not path.is_file():
        return []
    reader = csv.DictReader(io.StringIO(_read_text(path, encoding="utf-8-sig")))
    if not {"question", "answer"} <= set(reader.fieldnames or ()):
        raise KnowledgeConfigError(f"{path}: needs 'question' and 'answer' columns")
    entries = []
    for n, row in enumerate(reader, start=1):
        keywords = tuple(k.strip() for k in (row.get("keywords") or "").split(";") if k.strip())
        try:
            entries.append(
                Entry(
                    id=(row.get("id") or "").strip() or f"faq-{n}",
                    title=row["question"],
                    facts=split_answer(row["answer"] or ""),
                    keywords=keywords,
                )
            )
        except ValidationError as exc:
            raise KnowledgeConfigError(f"{path}:{n + 1}: {exc}") from exc
    return entries


def read_pack_files(directory: Path) -> tuple[list[Entry], list[Contact]]:
    entries = _read_list(directory / "topics.yaml", _ENTRIES) + _read_faq(directory / "faq.csv")
    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            raise KnowledgeConfigError(f"{directory}: duplicate entry id {entry.id!r}")
        seen.add(entry.id)
    return _with_examples(directory, entries), _read_list(directory / "contacts.yaml", _CONTACTS)


def _with_examples(directory: Path, entries: list[Entry]) -> list[Entry]:
    """Attach ``examples.yaml`` (id -> text) to the entries; an unknown id or a second example is an error."""
    path = directory / "examples.yaml"
    if not path.is_file():
        return entries
    try:
        examples = _EXAMPLES.validate_python(_read_yaml(path) or {})
    except ValidationError as exc:
        raise KnowledgeConfigError(f"{path}: {exc}") from exc
    by_id = {e.id: e for e in entries}
    if unknown := sorted(set(examples) - by_id.keys()):
        raise KnowledgeConfigError(f"{path}: examples for unknown entry ids {unknown}")
    if twice := sorted(i for i in examples if by_id[i].example is not None):
        raise KnowledgeConfigError(f"{path}: entries {twice} already have an inline example")
    try:
        return [
            Entry.model_validate({**e.model_dump(), "example": examples[e.id]}) if e.id in examples else e
            for e in entries
        ]
    except ValidationError as exc:
        raise KnowledgeConfigError(f"{path}: {exc}") from exc


class FilePackSource:
    """A knowledge source backed by a pack directory, indexed once at load."""

    def __init__(
        self,
        manifest: PackManifest,
        entries: list[Entry],
        contacts: list[Contact],
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.manifest = manifest
        self.entries = tuple(entries)
        self.contacts = tuple(contacts)
        self._index = EntryIndex(entries, contacts, embedder=embedder)

    @classmethod
    def load(
        cls, directory: Path, *, embedder: Embedder | None = None, screen: ContentScreen | None = None
    ) -> FilePackSource:
        """`screen`: drops planted instructions and unsafe advice before indexing (docs/ABUSE.md)."""
        manifest = load_manifest(directory)
        if not isinstance(manifest.source, FilesSourceSpec):
            raise KnowledgeConfigError(f"{directory}: source type is {manifest.source.type!r}, not 'files'")
        entries, contacts = read_pack_files(directory)
        if screen is not None:
            entries, contacts = screen.entries(manifest, entries), screen.contacts(manifest, contacts)
        return cls(manifest, entries, contacts, embedder=embedder)

    def __len__(self) -> int:
        return len(self.entries)

    async def search(self, query: Query) -> SourceResult:
        return self._index.search(query)

    async def refresh(self) -> None:
        """Files are read once at load; restart (or reload the knowledge base) to pick up edits."""
