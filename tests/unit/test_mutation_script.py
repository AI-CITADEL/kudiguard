"""The string-mutant generator of scripts/mutation.py (cosmic-ray has no string operator)."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mutation.py"


@pytest.fixture(scope="module")
def mutation() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mutation_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


SOURCE = '''"""Module docstring."""
import logging
import re

log = logging.getLogger(__name__)
LABEL = b"kudiguard/v1"
PATTERN = re.compile(r"(?<!\\w)\\d{6,8}\\b")


def f(x: str) -> str:
    """Function docstring."""
    if not x:
        raise ValueError("prose for people")
    log.warning("also prose %s", x)
    return f"id:{x}"
'''


def test_wraps_code_strings_but_not_prose(mutation: ModuleType) -> None:
    wraps = {(m.line, m.mutated) for m in mutation.string_mutants(SOURCE) if m.kind == "wrap"}
    assert wraps == {(6, "b'XXkudiguard/v1XX'"), (7, repr("XX(?<!\\w)\\d{6,8}\\bXX")), (15, "'XXid:XX'")}


def test_regex_mutants_change_one_thing_each(mutation: ModuleType) -> None:
    regexes = {m.mutated for m in mutation.string_mutants(SOURCE) if m.kind == "regex"}
    assert regexes == {
        repr(p)
        for p in (
            "(?<!\\w)\\d{7,8}\\b",
            "(?<!\\w)\\d{5,8}\\b",
            "(?<!\\w)\\d{6,9}\\b",
            "(?<!\\w)\\d{6,7}\\b",
            "\\d{6,8}\\b",  # lookbehind removed
            "(?<!\\w)\\d{6,8}",  # word boundary removed
        )
    }


def test_mutated_source_differs_only_in_that_literal(mutation: ModuleType) -> None:
    for m in mutation.string_mutants(SOURCE):
        mutated = ast.parse(m.source)
        assert ast.dump(mutated) != ast.dump(ast.parse(SOURCE))
        assert ast.literal_eval(m.mutated) in {
            n.value for n in ast.walk(mutated) if isinstance(n, ast.Constant)
        }


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (r"^0[789]\d{8}$", {r"0[789]\d{8}$", r"^0[789]\d{8}", r"^0[789]\d{9}$", r"^0[789]\d{7}$"}),
        (r"[^\W\d_]{2,}", {r"[^\W\d_]{3,}", r"[^\W\d_]{1,}"}),  # ^ in a class is not an anchor
        (r"\{1\}\$", set()),  # escaped braces and dollar are literals
        (r"a{0}", {r"a{1}"}),  # no negative bound
        (r"(?=(?:\D*\d){10})x", {r"x", r"(?=(?:\D*\d){11})x", r"(?=(?:\D*\d){9})x"}),
    ],
)
def test_regex_mutations(mutation: ModuleType, pattern: str, expected: set[str]) -> None:
    assert set(mutation.regex_mutations(pattern)) == expected
    for mutated in expected:
        re.compile(mutated)


def test_annotation_spans_cover_annotations_only(mutation: ModuleType) -> None:
    source = "def f(a: int | None, b=1 | 2) -> str | None:\n    x: list[int] = []\n"
    spans = mutation.annotation_spans(source)
    assert mutation.inside((1, 9), (1, 19), spans)  # int | None
    assert mutation.inside((1, 23), (1, 28), spans) is False  # 1 | 2 is a default value
    assert mutation.inside((1, 33), (1, 43), spans)  # return annotation
    assert mutation.inside((2, 7), (2, 16), spans)


def test_annotation_columns_are_characters(mutation: ModuleType) -> None:
    source = 'X = "₦"; y: int | None = None\n'
    ((start, end),) = mutation.annotation_spans(source)
    assert source.splitlines()[0][start[1] : end[1]] == "int | None"


def test_tally_kill_rate(mutation: ModuleType) -> None:
    assert mutation.Tally().kill_rate is None
    assert mutation.Tally(killed=3, survived=1, error=5).kill_rate == 0.75
    assert mutation.Tally(killed=1, timeout=1, survived=2).kill_rate == 0.5
