"""Mutation testing of the safety-critical modules (workstream G; method and results in docs/SPIKES.md).

    python scripts/mutation.py                      # every target, cosmic-ray + string mutants
    python scripts/mutation.py scam_rules decision  # some targets
    python scripts/mutation.py --workers 8 --no-strings

Coverage only shows that code ran; a mutant (one small change to the code) that the tests still pass shows a result
that no test checks. Two sources of mutants, run against the same focused test files per module (``TARGETS``):

* cosmic-ray (operators: comparisons, arithmetic, boolean and unary operators, numbers, keywords such as
  ``True``/``None``/``is``/``in``, exceptions, ``break``/``continue``, zero-iteration loops, removed decorators).
  Run with its ``http`` distributor: one cosmic-ray http worker per clone of the repository (``--worker``: the same
  worker, but a timeout kills the whole test process tree), so mutants run in parallel without touching this
  checkout. Mutants inside type annotations are skipped (never evaluated under ``from __future__ import annotations``).
* String mutants (cosmic-ray has no string operator, yet regexes, hash/encryption labels and command words are
  strings): every string/bytes literal except docstrings, exception and log messages is wrapped in ``XX``; regex
  literals also get each quantifier bound moved by one and each lookaround, anchor and ``\\b`` removed.

Results go to ``var/mutation/`` (git-ignored): ``report.json`` and ``<target>-survivors.txt`` with each survivor's
diff. Set ``KUDIGUARD_FFMPEG`` and ``KUDIGUARD_EMBEDDINGS_CACHE`` as for the normal test suite.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "var" / "mutation"
COPIED = ("src", "tests", "data", "docs", "scripts", "deploy", "pyproject.toml", "README.md", "Dockerfile")
BASE_PORT = 9870  # default first worker port (--base-port)

# target -> (module, focused tests that exercise it; the full suite is too slow to run per mutant)
TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "scam_rules": (
        "src/kudiguard/safety/scam_rules.py",
        ("test_scam_rules.py", "test_decision.py", "test_engine.py"),
    ),
    "decision": ("src/kudiguard/safety/decision.py", ("test_decision.py", "test_engine.py")),
    "text": (
        "src/kudiguard/text.py",
        (
            "test_text.py",
            "test_scam_rules.py",
            "test_patterns.py",
            "test_commands.py",
            "test_redaction_paths.py",
            "test_redaction_corpus.py",
            "test_m9_support.py",
        ),
    ),
    "privacy": ("src/kudiguard/privacy.py", ("test_privacy.py", "test_review.py", "test_sqlite_store.py")),
    "donation": (
        "src/kudiguard/donation.py",
        (
            "test_donation.py",
            "test_api_donation.py",
            "test_pipeline_donation.py",
            "test_voice_cli.py",
            "test_patterns.py",
        ),
    ),
    "commands": ("src/kudiguard/commands.py", ("test_commands.py", "test_pipeline.py")),
    "hosting": (
        "src/kudiguard/hosting.py",
        ("test_m9_support.py", "test_api_donation.py", "test_pipeline_donation.py"),
    ),
    "render": ("src/kudiguard/render.py", ("test_render.py", "test_pipeline.py")),
    "review": ("src/kudiguard/review.py", ("test_review.py",)),
    "patterns": (
        "src/kudiguard/patterns.py",
        ("test_patterns.py", "test_m9_support.py", "test_donation.py", "test_redaction_paths.py"),
    ),
    "export": (
        "src/kudiguard/storage/export.py",
        ("test_redaction_paths.py", "test_store_contract.py", "test_review.py", "test_sqlite_store.py"),
    ),
    "sqlite": (
        "src/kudiguard/storage/sqlite.py",
        ("test_sqlite_store.py", "test_store_contract.py", "test_redaction_paths.py", "test_patterns.py"),
    ),
    "legacy": ("src/kudiguard/storage/legacy.py", ("test_redaction_paths.py", "test_sqlite_store.py")),
}


# --- string mutants (pure; tested in tests/unit/test_mutation_script.py) ---------------------------------------


@dataclass(frozen=True, slots=True)
class StringMutant:
    line: int
    kind: str  # "wrap" (literal wrapped in XX) or "regex"
    original: str
    mutated: str
    source: str  # the whole mutated module


_LOGGERS = frozenset({"log", "logger", "logging"})
_REGEX_SYNTAX = re.compile(r"\\[a-zA-Z]|\(\?|\[|\{\d|[+*^$|]")


def _excluded_nodes(tree: ast.Module) -> set[int]:
    """ids of nodes whose strings are prose for people (docstrings, exception and log messages)."""
    excluded: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                excluded.add(id(first.value))
        prose = isinstance(node, ast.Raise) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in _LOGGERS
        )
        if prose:
            excluded.update(id(child) for child in ast.walk(node))
    return excluded


def _candidates(tree: ast.Module) -> list[ast.Constant]:
    excluded = _excluded_nodes(tree)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str | bytes)
        and not isinstance(node.value, bool)
        and id(node) not in excluded
    ]


def _in_fstring(tree: ast.Module) -> set[int]:
    return {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in node.values}


def _escaped(text: str, index: int) -> bool:
    backslashes = 0
    while index - backslashes - 1 >= 0 and text[index - backslashes - 1] == "\\":
        backslashes += 1
    return backslashes % 2 == 1


def _group_end(text: str, start: int) -> int | None:
    """Index just past the group opened at ``start`` (aware of escapes and character classes)."""
    depth, in_class, i = 0, False, start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def regex_mutations(pattern: str) -> list[str]:
    """Small edits to a regex: quantifier bounds +/-1, lookarounds, anchors and word boundaries removed."""
    found: list[str] = []
    for m in re.finditer(r"\{(\d+)(?:(,)(\d*))?\}", pattern):
        if _escaped(pattern, m.start()):
            continue
        low, comma, high = int(m.group(1)), m.group(2), m.group(3)
        bounds: list[tuple[int, str]] = [(low + 1, high or ""), (low - 1, high or "")]
        if high:
            bounds += [(low, str(int(high) + 1)), (low, str(int(high) - 1))]
        for new_low, new_high in bounds:
            if new_low < 0 or (new_high and int(new_high) < new_low):
                continue
            text = f"{{{new_low}{comma or ''}{new_high if comma else ''}}}"
            found.append(pattern[: m.start()] + text + pattern[m.end() :])
    for m in re.finditer(r"\(\?(?:<[!=]|[!=])", pattern):
        if _escaped(pattern, m.start()):
            continue
        end = _group_end(pattern, m.start())
        if end is not None:
            found.append(pattern[: m.start()] + pattern[end:])
    for m in re.finditer(r"\\b|\^|\$", pattern):
        anchor = m.group() == "\\b" or _anchor(pattern, m.start())
        if anchor and not _escaped(pattern, m.start()):
            found.append(pattern[: m.start()] + pattern[m.end() :])
    return list(dict.fromkeys(f for f in found if f != pattern))


def _anchor(pattern: str, index: int) -> bool:
    """``^``/``$`` as an anchor (not inside a character class, not a negated class)."""
    if pattern[index] == "^" and index > 0 and pattern[index - 1] == "[":
        return False
    depth = 0
    for i, ch in enumerate(pattern[:index]):
        if ch == "[" and not _escaped(pattern, i):
            depth = 1
        elif ch == "]" and not _escaped(pattern, i):
            depth = 0
    return depth == 0


def _compiles(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def string_mutants(source: str) -> list[StringMutant]:
    tree = ast.parse(source)
    fstring_parts = _in_fstring(tree)
    plans: list[tuple[int, str, str | bytes]] = []
    for index, node in enumerate(_candidates(tree)):
        value = node.value
        wrapped = b"XX" + value + b"XX" if isinstance(value, bytes) else "XX" + value + "XX"
        plans.append((index, "wrap", wrapped))
        if isinstance(value, str) and _REGEX_SYNTAX.search(value):
            whole = id(node) not in fstring_parts
            if whole and not _compiles(value):
                continue
            plans += [(index, "regex", m) for m in regex_mutations(value) if not whole or _compiles(m)]
    mutants = []
    for index, kind, new in plans:
        mutated_tree = copy.deepcopy(tree)
        node = _candidates(mutated_tree)[index]
        original = node.value
        node.value = new
        mutants.append(StringMutant(node.lineno, kind, repr(original), repr(new), ast.unparse(mutated_tree)))
    return mutants


Span = tuple[tuple[int, int], tuple[int, int]]  # (line, column) start and end; lines from 1, columns from 0


def annotation_spans(source: str) -> list[Span]:
    lines = source.splitlines()

    def char_column(line: int, byte_offset: int) -> int:  # ast counts UTF-8 bytes, cosmic-ray characters
        return len(lines[line - 1].encode()[:byte_offset].decode(errors="ignore"))

    spans: list[Span] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.arg | ast.AnnAssign) and node.annotation is not None:
            note = node.annotation
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
            note = node.returns
        else:
            continue
        end_line = note.end_lineno or note.lineno
        end = (end_line, char_column(end_line, note.end_col_offset or note.col_offset))
        spans.append(((note.lineno, char_column(note.lineno, note.col_offset)), end))
    return spans


def inside(start: Sequence[int], end: Sequence[int], spans: Sequence[Span]) -> bool:
    return any(low <= tuple(start) and tuple(end) <= high for low, high in spans)


# --- running ----------------------------------------------------------------------------------------------------


@dataclass
class Tally:
    generated: int = 0
    killed: int = 0
    survived: int = 0
    timeout: int = 0
    error: int = 0  # mutant could not be applied or run (not counted in the kill rate)
    annotation: int = 0  # cosmic-ray mutant inside a type annotation: skipped (never evaluated)

    @property
    def kill_rate(self) -> float | None:
        judged = self.killed + self.survived + self.timeout
        return (self.killed + self.timeout) / judged if judged else None


@dataclass
class TargetResult:
    target: str
    module: str
    seconds: float = 0.0
    cosmic: Tally = field(default_factory=Tally)
    strings: Tally = field(default_factory=Tally)
    survivors: list[dict[str, object]] = field(default_factory=list)


def test_command(tests: Sequence[str]) -> str:
    paths = " ".join(shlex.quote(f"tests/unit/{t}") for t in tests)
    python = shlex.quote(
        Path(sys.executable).as_posix()
    )  # shlex-split again by cosmic-ray: spaces in paths survive
    return f"{python} -m pytest -x -q -p no:cacheprovider {paths}"


def clone_env(clone: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(clone / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):  # parallel runs, one core each
        env[name] = "1"
    return env


def make_clones(count: int) -> list[Path]:
    clones = []
    for i in range(count):
        clone = OUT / "clones" / f"w{i}"
        shutil.rmtree(clone, ignore_errors=True)
        clone.mkdir(parents=True)
        for name in COPIE***REMOVED***
            source = ROOT / name
            if source.is_dir():
                shutil.copytree(source, clone / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            elif source.is_file():
                shutil.copy2(source, clone / name)
        clones.append(clone)
    return clones


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill a timed-out test run and its children. cosmic-ray kills the process group with ``os.killpg``, which does
    not exist on Windows, and falls back to killing the one process; but a venv's ``python.exe`` is a launcher that
    starts the real interpreter as a child, so that would leave a looping mutant running and eating a core. Hence
    ``taskkill /T`` on Windows and the process group (the run is started in its own session) elsewhere."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        with contextlib.suppress(OSError):  # the group is already gone
            os.killpg(proc.pid, 9)
    proc.kill()
    proc.communicate()


def run_tests(command: str, clone: Path, timeout: float) -> str:
    """'killed', 'survived' or 'timeout'."""
    proc = subprocess.Popen(
        shlex.split(command),
        cwd=clone,
        env=clone_env(clone),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
    )
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        return "timeout"
    return "survived" if returncode == 0 else "killed"


def run_worker(port: int) -> None:
    """A cosmic-ray http worker whose timeouts kill the whole test process tree (see ``kill_tree``)."""
    import cosmic_ray.testing  # type: ignore[import-untyped]
    from cosmic_ray.distribution.http import run_worker as serve  # type: ignore[import-untyped]

    cosmic_ray.testing._kill_process_group = kill_tree
    serve(port=port)


def cosmic_ray_exe() -> str:
    found = shutil.which("cosmic-ray", path=str(Path(sys.executable).parent))
    if not found:
        raise SystemExit("cosmic-ray is not installed (pip install -e .[dev])")
    return found


def start_workers(clones: Sequence[Path], base_port: int) -> list[subprocess.Popen[bytes]]:
    workers = []
    for i, clone in enumerate(clones):
        log = (OUT / f"worker-{i}.log").open("wb")
        workers.append(
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(base_port + i)],
                cwd=clone,
                env=clone_env(clone),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
    deadline = time.monotonic() + 60
    for i in range(len(clones)):
        while True:
            try:
                socket.create_connection(("127.0.0.1", base_port + i), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise SystemExit(
                        f"cosmic-ray worker {i} did not start (see {OUT}/worker-{i}.log)"
                    ) from None
                time.sleep(0.2)
    return workers


def run_cosmic(
    name: str, module: str, command: str, timeout: float, workers: int, result: TargetResult, base_port: int
) -> None:
    from cosmic_ray.work_db import WorkDB, use_db  # type: ignore[import-untyped]
    from cosmic_ray.work_item import TestOutcome, WorkerOutcome, WorkResult  # type: ignore[import-untyped]

    config = OUT / f"{name}.toml"
    session = OUT / f"{name}.sqlite"
    urls = ", ".join(f'"http://127.0.0.1:{base_port + i}"' for i in range(workers))
    config.write_text(
        "[cosmic-ray]\n"
        f'module-path = "{module}"\n'
        f"timeout = {timeout}\n"
        "excluded-modules = []\n"
        f"test-command = {json.dumps(command)}\n\n"
        '[cosmic-ray.distributor]\nname = "http"\n\n'
        f"[cosmic-ray.distributor.http]\nworker-urls = [{urls}]\n",
        encoding="utf-8",
    )
    exe = cosmic_ray_exe()
    subprocess.run([exe, "init", str(config), str(session), "--force"], cwd=ROOT, check=True)
    # Filter (as cosmic-ray's own cr-filter-* tools do): with `from __future__ import annotations` a type
    # annotation is never evaluated, so mutating one cannot change behaviour.
    spans = annotation_spans((ROOT / module).read_text(encoding="utf-8"))
    with use_db(str(session), WorkDB.Mode.open) as db:
        for item in list(db.pending_work_items):
            if all(inside(m.start_pos, m.end_pos, spans) for m in item.mutations):
                db.set_result(item.job_id, WorkResult(worker_outcome=WorkerOutcome.SKIPPED))
    subprocess.run([exe, "exec", str(config), str(session)], cwd=ROOT, check=True)
    with use_db(str(session), WorkDB.Mode.open) as db:
        for item, outcome in db.completed_work_items:
            tally = result.cosmic
            if outcome.worker_outcome == WorkerOutcome.SKIPPE***REMOVED***
                tally.annotation += 1
                continue
            tally.generated += 1
            if (
                outcome.worker_outcome != WorkerOutcome.NORMAL
                or outcome.test_outcome == TestOutcome.INCOMPETENT
            ):
                tally.error += 1
            elif outcome.test_outcome == TestOutcome.SURVIVE***REMOVED***
                tally.survived += 1
                mutation = item.mutations[0]
                result.survivors.append(
                    {
                        "source": "cosmic-ray",
                        "operator": mutation.operator_name,
                        "occurrence": mutation.occurrence,
                        "line": mutation.start_pos[0],
                        "diff": outcome.diff,
                    }
                )
            elif outcome.output == "timeout":
                tally.timeout += 1
            else:
                tally.killed += 1


def run_strings(
    module: str, command: str, timeout: float, clones: Sequence[Path], result: TargetResult
) -> None:
    source = (ROOT / module).read_text(encoding="utf-8")
    pool: Queue[Path] = Queue()
    for clone in clones:
        pool.put(clone)

    def attempt(code: str) -> str:
        clone = pool.get()
        target = clone / module
        try:
            target.write_text(code, encoding="utf-8")
            return run_tests(command, clone, timeout)
        finally:
            target.write_text(source, encoding="utf-8")
            pool.put(clone)

    # Control: the module as ast.unparse writes it must pass, or every string mutant would look killed.
    if attempt(ast.unparse(ast.parse(source))) != "survived":
        print(f"  {module}: unparsed module fails its tests; string mutants skipped", flush=True)
        return
    mutants = string_mutants(source)
    with ThreadPoolExecutor(len(clones)) as executor:
        outcomes = list(executor.map(lambda m: attempt(m.source), mutants))
    for mutant, outcome in zip(mutants, outcomes, strict=True):
        result.strings.generated += 1
        setattr(result.strings, outcome, getattr(result.strings, outcome) + 1)
        if outcome == "survived":
            result.survivors.append(
                {"source": "strings", "operator": mutant.kind, "line": mutant.line,
                 "diff": f"- {mutant.original}\n+ {mutant.mutated}"}
            )  # fmt: skip


def baseline(command: str, clone: Path) -> float:
    start = time.monotonic()
    if run_tests(command, clone, 600) != "survived":
        raise SystemExit(f"baseline failed: {command} (in {clone})")
    return time.monotonic() - start


def write_survivors(result: TargetResult) -> None:
    lines = [f"# {result.module}: {len(result.survivors)} surviving mutants", ""]
    for s in sorted(result.survivors, key=lambda s: (int(str(s["line"])), str(s["operator"]))):
        lines += [f"## line {s['line']} [{s['source']}: {s['operator']}]", str(s["diff"]).rstrip(), ""]
    (OUT / f"{result.target}-survivors.txt").write_text("\n".join(lines), encoding="utf-8")


def rate(tally: Tally) -> str:
    return "n/a" if tally.kill_rate is None else f"{tally.kill_rate:.1%}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("targets", nargs="*", help=f"default: all of {', '.join(TARGETS)}")
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--no-cosmic", action="store_true", help="skip cosmic-ray (string mutants only)")
    parser.add_argument("--no-strings", action="store_true", help="skip string mutants")
    parser.add_argument(
        "--base-port", type=int, default=BASE_PORT, help=f"first worker port (default {BASE_PORT})"
    )
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)  # internal: run one http worker
    args = parser.parse_args(argv)
    if args.worker:
        run_worker(args.worker)
        return 0
    names = args.targets or list(TARGETS)
    if unknown := [n for n in names if n not in TARGETS]:
        parser.error(f"unknown targets: {unknown}")
    OUT.mkdir(parents=True, exist_ok=True)
    clones = make_clones(args.workers)
    workers = [] if args.no_cosmic else start_workers(clones, args.base_port)
    results: list[TargetResult] = []
    started = time.monotonic()
    try:
        for name in names:
            module, tests = TARGETS[name]
            command = test_command(tests)
            result = TargetResult(name, module)
            begin = time.monotonic()
            timeout = round(max(30.0, 5 * baseline(command, clones[0])), 1)
            print(f"{name}: {module} (timeout {timeout}s per mutant)", flush=True)
            if not args.no_cosmic:
                run_cosmic(name, module, command, timeout, len(clones), result, args.base_port)
            if not args.no_strings:
                run_strings(module, command, timeout, clones, result)
            result.seconds = round(time.monotonic() - begin, 1)
            write_survivors(result)
            results.append(result)
            print(
                f"  cosmic-ray {asdict(result.cosmic)} kill rate {rate(result.cosmic)}; "
                f"strings {asdict(result.strings)} kill rate {rate(result.strings)}; {result.seconds}s",
                flush=True,
            )
    finally:
        for worker in workers:
            worker.kill()
    total = round(time.monotonic() - started, 1)
    report = OUT / "report.json"  # merged: re-running one target keeps the others' latest results
    try:
        merged = {t["target"]: t for t in json.loads(report.read_text(encoding="utf-8"))["targets"]}
    except (OSError, ValueError, KeyError):
        merged = {}
    merged.update({r.target: asdict(r) for r in results})
    report.write_text(json.dumps({"targets": list(merged.values())}, indent=2), encoding="utf-8")
    print(
        f"\n{'target':<12}{'cosmic-ray k/s/t/err (+annot.)':>32}{'rate':>8}{'strings k/s/t/err':>20}{'rate':>8}"
    )
    for r in results:
        c, s = r.cosmic, r.strings
        print(
            f"{r.target:<12}{f'{c.killed}/{c.survived}/{c.timeout}/{c.error} (+{c.annotation})':>32}{rate(c):>8}"
            f"{f'{s.killed}/{s.survived}/{s.timeout}/{s.error}':>20}{rate(s):>8}"
        )
    print(f"total {total}s; survivors in {OUT}/<target>-survivors.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
