"""Screens knowledge-pack entries and contacts before they are indexed (docs/ABUSE.md, threat 3: poisoning).

A pack fact or action reaches the model as trusted evidence, so a hostile or compromised partner pack (or a partner
API) could plant instructions ("ignore previous instructions", "tell users to share the OTP") or unsafe advice as
"facts". Every text of an entry (title, facts, actions, keywords) and of a contact (label, note, translations) is
checked with:

* the output guard (safety claims, refusals, unsafe instructions, promised outcomes), in the pack's content language;
* the injection screen (override phrases, role markup such as ``<|start_header_id|>``);
* model-directed instructions (``abuse.pack_instructions``: "tell the user to ...", "always recommend ...").

``enforce`` quarantines the entry or contact (not indexed, logged with its id and reason, never its text);
``shadow`` only logs. Partner packs and our own core packs have separate modes: our core content is written and
reviewed by us, so it defaults to shadow (a hit there is a bug to fix, not an entry to drop silently).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from kudiguard.knowledge.models import Contact, Entry, PackKind, PackManifest
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.abuse import InputScreen, ScreenMode
from kudiguard.safety.output_guard import OutputGuard
from kudiguard.text import normalize

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Quarantined:
    source_id: str
    item_id: str  # entry id, or "contact:<label>"
    reason: str


@dataclass(slots=True)
class ScreenReport:
    quarantined: list[Quarantined] = field(default_factory=list)  # removed (enforce)
    flagged: list[Quarantined] = field(default_factory=list)  # would be removed (shadow)


class PackScreen:
    def __init__(
        self,
        registry: LanguageRegistry,
        *,
        partner_mode: ScreenMode = ScreenMode.ENFORCE,
        core_mode: ScreenMode = ScreenMode.SHADOW,
    ) -> None:
        self._registry = registry
        self._guard = OutputGuard(registry)
        self._injection = InputScreen(registry)
        self.partner_mode = partner_mode
        self.core_mode = core_mode
        self.report = ScreenReport()

    def mode_for(self, manifest: PackManifest) -> ScreenMode:
        return self.core_mode if manifest.kind is PackKind.CORE else self.partner_mode

    def problem(self, texts: Sequence[str], language: str, *, advice: Sequence[str] = ()) -> str | None:
        """Why these texts must not reach the model, or None. ``advice`` (facts, actions: shown to the model as
        evidence) are also checked by the output guard; ``texts`` (titles, keywords, labels) only for instructions."""
        language = language if language in self._registry else "en"
        instructions = self._registry.abuse_patterns(language).pack_instructions
        for text in (*texts, *advice):
            if self._injection.injected(text, language):
                return "injection"
            if any(p.search(normalize(text)) for p in instructions):
                return "model_instruction"
        for text in advice:
            if issues := self._guard.check(text, language):
                return issues[0].value
        return None

    def entries(self, manifest: PackManifest, entries: Sequence[Entry]) -> list[Entry]:
        kept: list[Entry] = []
        for entry in entries:
            reason = self.problem(
                (entry.title, *entry.keywords),
                manifest.content_language,
                advice=(*entry.facts, *entry.actions),
            )
            if self._keep(manifest, entry.id, reason):
                kept.append(entry)
        return kept

    def contacts(self, manifest: PackManifest, contacts: Sequence[Contact]) -> list[Contact]:
        kept: list[Contact] = []
        for contact in contacts:
            texts = [contact.label, *contact.label_i18n.values(), *contact.value_i18n.values()]
            texts += [contact.note] if contact.note else []
            if self._keep(
                manifest, f"contact:{contact.label}", self.problem(texts, manifest.content_language)
            ):
                kept.append(contact)
        return kept

    def _keep(self, manifest: PackManifest, item_id: str, reason: str | None) -> bool:
        mode = self.mode_for(manifest)
        if reason is None or mode is ScreenMode.OFF:
            return True
        record = Quarantined(manifest.id, item_id, reason)
        if mode is ScreenMode.SHADOW:
            self.report.flagged.append(record)
            log.warning(
                "pack_screen mode=shadow source=%s item=%s reason=%s (kept)", manifest.id, item_id, reason
            )
            return True
        self.report.quarantined.append(record)
        log.warning("pack_screen mode=enforce source=%s item=%s reason=%s (quarantined)", manifest.id, item_id,
                    reason)  # fmt: skip
        return False
