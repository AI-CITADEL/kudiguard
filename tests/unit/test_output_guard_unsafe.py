"""Output guard false-positive sweeps: every fixed string, pack fact and action, prompt example, eval fixture and the
per-language safe-advice fixture must pass. The labelled probes (block / pass, 4 languages) live in
tests/fixtures/guard_probes.yaml and are asserted by tests/unit/test_guard_probes.py.
"""

from pathlib import Path
from typing import Any

import yaml

from kudiguard.knowledge.file_pack import FilePackSource
from kudiguard.languages import LanguageRegistry
from kudiguard.prompts import _EXAMPLE_CREDENTIALS, _EXAMPLE_QUESTION, _EXAMPLE_SCAM
from kudiguard.safety.output_guard import OutputGuard, OutputIssue

ROOT = Path(__file__).resolve().parents[2]
UNSAFE = OutputIssue.UNSAFE_INSTRUCTION
OUTCOME = OutputIssue.OUTCOME_CLAIM


def _strings(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return []


def _hits(guard: OutputGuard, pairs: list[tuple[str, str]]) -> list[tuple[str, str, tuple[OutputIssue, ...]]]:
    out = []
    for language, text in pairs:
        issues = tuple(i for i in guard.check(text, language) if i in (UNSAFE, OUTCOME))
        if issues:
            out.append((language, text, issues))
    return out


def test_fixed_text_never_trips_the_guard(registry: LanguageRegistry) -> None:
    pairs = [(lang.code, text) for lang in registry.enabled for text in lang.strings.values()]
    assert _hits(OutputGuard(registry), pairs) == []


def test_pack_facts_actions_and_prompt_examples_pass(registry: LanguageRegistry) -> None:
    texts = [e.reply for e in (_EXAMPLE_CREDENTIALS, _EXAMPLE_SCAM, _EXAMPLE_QUESTION)]
    for pack in (ROOT / "data" / "packs").iterdir():
        manifest = pack / "pack.yaml"
        if manifest.is_file() and "type: http" not in manifest.read_text(encoding="utf-8"):
            for entry in FilePackSource.load(pack).entries:
                texts.extend((*entry.facts, *entry.actions))
    guard = OutputGuard(registry)
    hits = [h for h in _hits(guard, [("en", t) for t in texts]) if UNSAFE in h[2]]
    assert hits == []


def test_safe_advice_in_every_language_passes(registry: LanguageRegistry) -> None:
    fixture = yaml.safe_load((ROOT / "tests" / "fixtures" / "safe_advice.yaml").read_text(encoding="utf-8"))
    assert set(fixture) == {"en", "ha", "yo", "ig"}
    pairs = [(lang, text) for lang, texts in fixture.items() for text in texts]
    assert _hits(OutputGuard(registry), pairs) == []


def test_legitimate_eval_fixtures_pass(registry: LanguageRegistry) -> None:
    scenarios = yaml.safe_load((ROOT / "data" / "eval" / "scenarios.yaml").read_text(encoding="utf-8"))
    claims = yaml.safe_load((ROOT / "data" / "eval" / "grounding_claims.yaml").read_text(encoding="utf-8"))
    pairs = [(s["lang"], s["text"]) for s in scenarios if s["label"] == "legit"]
    pairs += [(c["lang"], c["claim"]) for c in claims["claims"] if c["supported"]]
    pairs += [("en", r["reply"]) for r in claims["replies"] if r["acceptable"]]
    for name in ("prompts.yaml", "retrieval.yaml", "retrieval_blind.yaml", "retrieval_money.yaml"):
        data = yaml.safe_load((ROOT / "data" / "eval" / name).read_text(encoding="utf-8"))
        pairs += [("en", s) for s in _strings(data) if len(s.split()) >= 3]
    hits = [h for h in _hits(OutputGuard(registry), pairs) if UNSAFE in h[2]]
    assert hits == []
