"""The unsafe-advice patterns against the permanent probe set and a corpus of safe advice.

Any change to ``kudiguard.evaluation.answers.unsafe_advice`` must keep every probe in
``data/eval/unsafe_advice_probes.yaml`` passing (relabel only on purpose, with a note), and must not fire on any text
that is safe advice by construction (pack content, ``must`` points, calibration pairs, English fixed strings).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kudiguard.evaluation.answers import MONEY_BACK, UNSAFE_INSTRUCTION, load_cases, unsafe_advice

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "eval"
PROBES: list[dict[str, Any]] = yaml.safe_load(
    (EVAL / "unsafe_advice_probes.yaml").read_text(encoding="utf-8")
)
CATEGORY = {"instruction": UNSAFE_INSTRUCTION, "money_back": MONEY_BACK}


def test_probe_file_is_well_formed() -> None:
    texts = [p["text"] for p in PROBES]
    assert len(texts) == len(set(texts)) and len(texts) >= 190
    assert {p["expect"] for p in PROBES} == {"unsafe", "safe"}
    assert {p["category"] for p in PROBES} == set(CATEGORY)
    assert all(p.get("note") for p in PROBES)


def _param(probe: dict[str, Any]) -> Any:
    marks = [pytest.mark.xfail(reason=probe["xfail"], strict=True)] if probe.get("xfail") else []
    return pytest.param(probe, id=probe["text"][:60], marks=marks)


@pytest.mark.parametrize("probe", [_param(p) for p in PROBES])
def test_probe(probe: dict[str, Any]) -> None:
    found = unsafe_advice(probe["text"])
    if probe["expect"] == "unsafe":
        assert CATEGORY[probe["category"]] in found, probe["note"]
    else:
        assert found == set(), probe["note"]


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def safe_corpus() -> list[str]:
    packs = ROOT / "data" / "packs"
    texts = [
        s for path in sorted(packs.rglob("*.yaml")) for s in _strings(yaml.safe_load(path.read_text("utf-8")))
    ]
    for path in sorted(packs.rglob("*.csv")):
        rows = csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines())
        texts += [row[k] for row in rows for k in ("question", "answer") if row.get(k)]
    texts += [p.point for c in load_cases(EVAL / "answers.yaml") for p in c.must]
    texts += [
        t
        for pair in yaml.safe_load((EVAL / "answer_calibration.yaml").read_text("utf-8"))
        for t in (pair["point"], pair["sentence"])
    ]
    english = yaml.safe_load((ROOT / "data" / "languages" / "en.yaml").read_text(encoding="utf-8"))
    texts += list(_strings(english.get("strings", {})))
    return texts


def test_no_hits_on_safe_advice() -> None:
    corpus = safe_corpus()
    assert len(corpus) >= 1000  # 1056 on 2026-10-10
    hits = [(text, sorted(found)) for text in corpus if (found := unsafe_advice(text))]
    assert hits == []
