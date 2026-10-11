"""Every probe of the redaction reviews (tests/fixtures/redaction_corpus.yaml) behaves as recorded."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from kudiguard.text import redact

CORPUS: list[dict[str, Any]] = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "fixtures" / "redaction_corpus.yaml").read_text(encoding="utf-8")
)


def test_corpus_is_large_and_well_formed() -> None:
    assert len(CORPUS) >= 300
    assert all(entry["expect"] in {"masked", "kept"} for entry in CORPUS)
    assert len({entry["text"] for entry in CORPUS}) == len(CORPUS)


@pytest.mark.parametrize("entry", CORPUS, ids=[ascii(e["text"])[:60] for e in CORPUS])
def test_corpus_case(entry: dict[str, Any]) -> None:
    text, masked = entry["text"], redact(entry["text"])
    if entry["expect"] == "kept":
        assert masked == text
        return
    assert "[number]" in masked or "[email]" in masked
    rest = masked
    for kept in entry.get("keep", []):
        assert kept in rest
        rest = rest.replace(kept, " ", 1)
    assert not any(ch.isdigit() for ch in rest.replace("[number]", "").replace("[email]", "")), masked
    assert redact(masked) == masked
