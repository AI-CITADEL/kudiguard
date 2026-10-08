import os
from pathlib import Path

import pytest

from kudiguard.envfile import load_env_file


def test_loads_values_without_overriding_real_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nKG_A=1\nKG_B = 'two'\nKG_C=\"three\"\nnot a pair\n\nKG_D=already\n", encoding="utf-8"
    )
    monkeypatch.setenv("KG_D", "real")
    for key in ("KG_A", "KG_B", "KG_C"):
        monkeypatch.delenv(key, raising=False)
    load_env_file(env)
    assert (os.environ["KG_A"], os.environ["KG_B"], os.environ["KG_C"], os.environ["KG_D"]) == (
        "1",
        "two",
        "three",
        "real",
    )


def test_missing_file_is_fine(tmp_path: Path) -> None:
    load_env_file(tmp_path / "nope.env")
