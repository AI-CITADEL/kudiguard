"""scripts/eval_e2e.py prints its summary on a cp1252 console (Windows) without crashing on "≥"."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "eval_e2e.py"


def test_summary_prints_on_a_cp1252_console(tmp_path: Path) -> None:
    results = tmp_path / "e2e-empty.jsonl"
    results.write_text("", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONPATH=str(ROOT / "src"))
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--summary", str(results)],
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert b"UnicodeEncodeError" not in done.stderr
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert "≥".encode() in done.stdout
