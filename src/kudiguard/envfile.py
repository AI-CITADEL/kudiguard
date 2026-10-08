"""Minimal ``KEY=VALUE`` loader for the local, git-ignored ``.env`` (values are never printed or logged)."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Set variables from ``path`` that are not already set in the environment (real env wins)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
