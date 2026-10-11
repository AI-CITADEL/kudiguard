"""Compare this checkout's ``redact`` with an earlier one: list every digit the old version masked and the new keeps.

    python scripts/redaction_compare.py                 # against b4f96f8 (before workstream G), corpus + fuzz
    python scripts/redaction_compare.py --old abc1234 --fuzz 200000

Inputs: every case in ``tests/fixtures/redaction_corpus.yaml`` plus random identifiers joined to amounts, dates,
times and words (``tests/redaction_fuzz.py``), codes of 6-9 digits followed by small numbers, date runs, mixed
texts and long inputs. A corpus digit kept now but masked before is a regression unless the
case is marked ``changed:`` with the reason (years, dates and amounts the old patterns over-masked). For fuzz cases
only the identifier's digits count: none may be kept. Exit status 1 on any regression.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from kudiguard.text import redact  # noqa: E402
from redaction_fuzz import case, kept_digits, random_text  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "redaction_corpus.yaml"


def old_redact(commit: str) -> Callable[[str], str]:
    source = subprocess.run(
        ["git", "show", f"{commit}:src/kudiguard/text.py"], cwd=ROOT, capture_output=True, check=True, text=True,
        encoding="utf-8",
    ).stdout  # fmt: skip
    module = types.ModuleType("old_text")
    exec(compile(source, f"{commit}:text.py", "exec"), module.__dict__)
    return module.redact  # type: ignore[no-any-return]


def regressions(text: str, old: Callable[[str], str], span: tuple[int, int] | None = None) -> list[int]:
    """Digits (in ``span`` if given) kept by the new redact but masked by the old one."""
    new_kept, old_kept = kept_digits(text, redact(text)), kept_digits(text, old(text))
    lost = sorted(new_kept - old_kept)
    return [i for i in lost if span is None or span[0] <= i < span[1]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--old", default="b4f96f8", help="commit to compare with (default: before workstream G)"
    )
    parser.add_argument("--fuzz", type=int, default=50_000, help="random cases (default 50000)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    old = old_redact(args.old)
    corpus = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))
    failures = changed = 0
    for entry in corpus:
        lost = regressions(entry["text"], old)
        if lost and entry.get("changed"):
            changed += 1
        elif lost:
            failures += 1
            print(f"REGRESSION {entry['text']!r}: old {old(entry['text'])!r}, new {redact(entry['text'])!r}")
    rng = random.Random(args.seed)
    for _ in range(args.fuzz):
        text, span, _context = case(rng)
        if regressions(text, old, span) or any(
            text[i].isdigit() for i in kept_digits(text, redact(text)) if span[0] <= i < span[1]
        ):
            failures += 1
            print(f"FUZZ LEAK {text!r}: old {old(text)!r}, new {redact(text)!r}")
    texts = args.fuzz // 10

    def leaks(text: str, spans: list[tuple[int, int]]) -> list[int]:
        kept = kept_digits(text, redact(text))
        return [i for i in sorted(kept) if any(a <= i < b for a, b in spans)]

    for _ in range(texts):  # mixed text: no digit of an identifier or code may remain
        text, spans = random_text(rng, rng.randint(20, 400))
        if lost := leaks(text, spans):
            failures += 1
            print(f"TEXT LEAK {text!r} at {lost[:5]}: new {redact(text)!r}")
    slowest = 0.0
    for length in (5_000, 20_000, 64_000):  # long inputs: still linear, still no leak
        text, spans = random_text(rng, length)
        started = time.perf_counter()
        redact(text)
        slowest = max(slowest, time.perf_counter() - started)
        if lost := leaks(text, spans):
            failures += 1
            print(f"LONG LEAK ({length} chars) at {lost[:5]}")
    print(
        f"{len(corpus)} corpus cases ({changed} intended changes), {args.fuzz} joined fuzz cases, {texts} mixed texts, "
        f"long inputs up to 64,000 chars (slowest {slowest:.2f} s): {failures} regressions or leaks"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
