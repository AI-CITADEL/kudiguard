from pathlib import Path

import pytest

from kudiguard.languages import LanguageRegistry

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(autouse=True)
def _isolated_voice_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a test's purge sweep the checkout's real var/voice (files unknown to a test database)."""
    monkeypatch.setenv("KUDIGUARD_VOICE_DIR", str(tmp_path_factory.getbasetemp() / "voice-isolated"))


@pytest.fixture(scope="session")
def registry() -> LanguageRegistry:
    return LanguageRegistry.load(DATA_DIR / "languages")
