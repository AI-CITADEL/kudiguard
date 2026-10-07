from pathlib import Path

import pytest

from kudiguard.languages import LanguageRegistry

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def registry() -> LanguageRegistry:
    return LanguageRegistry.load(DATA_DIR / "languages")
