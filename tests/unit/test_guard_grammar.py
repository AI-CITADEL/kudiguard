"""The output guard against the generated grammar (tests/guard_grammar.py): every cell must be 100% correct, except
cells listed in KNOWN_LIMITS (strict: a limit that starts passing must be removed from the list)."""

from collections import defaultdict

import pytest

from guard_grammar import Case, instruction_cases, safety_cases
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.output_guard import OutputGuard, OutputIssue

MIN_CASES = 2000
# cell -> reason. Each is a known limit, tracked here instead of hidden.
KNOWN_LIMITS: dict[str, str] = {}


def failures_by_cell(guard: OutputGuard, cases: list[Case]) -> dict[str, list[str]]:
    failed: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        hit = OutputIssue(case.issue) in guard.check(case.text, "en")
        if hit is not (case.expect == "block"):
            failed[case.cell].append(f"{case.expect}: {case.text}")
    return dict(failed)


@pytest.fixture(scope="module")
def guard(registry: LanguageRegistry) -> OutputGuard:
    return OutputGuard(registry)


def test_the_grammar_is_large_deterministic_and_labelled() -> None:
    cases = instruction_cases()
    assert len(cases) >= MIN_CASES and cases == instruction_cases()
    assert {c.expect for c in cases} == {"block", "pass"} and len({c.cell for c in cases}) > 200
    assert {c.expect for c in safety_cases()} == {"block", "pass"}


@pytest.mark.parametrize("suite", ["instruction", "safety"])
def test_every_cell_is_correct_outside_known_limits(guard: OutputGuard, suite: str) -> None:
    cases = instruction_cases() if suite == "instruction" else safety_cases()
    failed = failures_by_cell(guard, cases)
    unexpected = {cell: texts for cell, texts in failed.items() if cell not in KNOWN_LIMITS}
    report = "\n".join(
        f"{cell} ({len(t)}):\n  " + "\n  ".join(t[:5]) for cell, t in sorted(unexpected.items())
    )
    assert not unexpected, f"{len(unexpected)} cells with failures:\n{report}"
    cells = {c.cell for c in cases}
    fixed = [cell for cell in KNOWN_LIMITS if cell in cells and cell not in failed]
    assert not fixed, f"known limits that now pass (remove them): {fixed}"
