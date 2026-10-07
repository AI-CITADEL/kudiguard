from pathlib import Path

import pytest

from kudiguard.domain import Flag
from kudiguard.languages import LanguageConfigError, LanguageRegistry, LanguageStatus


def _write(dir_: Path, name: str, body: str) -> None:
    (dir_ / name).write_text(body, encoding="utf-8")


MINIMAL_EN = """
code: en
name: English
native_name: English
order: 2
status: stable
asr_model: m/en
lexicon:
  flags: {urgency: ['urgent']}
  check_intent: ['scam']
"""


def test_shipped_languages_load_in_display_order(registry: LanguageRegistry) -> None:
    assert [lang.code for lang in registry.enabled] == ["ha", "yo", "ig", "en"]
    assert registry.get("ha").name == "Hausa"
    assert registry.get("yo").status is LanguageStatus.BETA
    assert registry.get("en").asr_model == "NCAIR1/NigerianAccentedEnglish"


def test_unknown_code_raises(registry: LanguageRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get("fr")
    assert "fr" not in registry
    assert "ha" in registry


def test_lexicon_for_merges_pivot(registry: LanguageRegistry) -> None:
    ha = registry.lexicon_for("ha")
    assert Flag.ASKS_CREDENTIALS in ha.flags  # from English pivot
    assert any(p.search("gaggawa") for p in ha.flags[Flag.URGENCY])  # Hausa native term
    assert registry.lexicon_for("ha") is ha  # cached


def test_disabled_languages_are_hidden(tmp_path: Path) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN)
    _write(
        tmp_path,
        "xx.yaml",
        MINIMAL_EN.replace("code: en", "code: xx")
        .replace("stable", "disabled")
        .replace("order: 2", "order: 9"),
    )
    reg = LanguageRegistry.load(tmp_path)
    assert [lang.code for lang in reg.enabled] == ["en"]
    assert "xx" in reg  # still resolvable for historical records


def test_adding_a_language_is_config_only(tmp_path: Path) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN)
    _write(
        tmp_path,
        "pcm.yaml",
        MINIMAL_EN.replace("code: en", "code: pcm")
        .replace("name: English", "name: Nigerian Pidgin")
        .replace("order: 2", "order: 1")
        .replace("'urgent'", "'sharp sharp'"),
    )
    reg = LanguageRegistry.load(tmp_path)
    assert [lang.code for lang in reg.enabled] == ["pcm", "en"]
    assert any(p.search("do am sharp sharp") for p in reg.lexicon_for("pcm").flags[Flag.URGENCY])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (MINIMAL_EN.replace("urgency", "not_a_flag"), "not_a_flag"),
        (MINIMAL_EN.replace("'urgent'", "'(unclosed'"), "invalid regex"),
        (MINIMAL_EN.replace("status: stable", "status: maybe"), "status"),
        (MINIMAL_EN.replace("code: en", "code: EN!"), "code"),
    ],
)
def test_invalid_config_fails_loudly(tmp_path: Path, body: str, message: str) -> None:
    _write(tmp_path, "en.yaml", body)
    with pytest.raises(LanguageConfigError, match=message):
        LanguageRegistry.load(tmp_path)


def test_filename_must_match_code(tmp_path: Path) -> None:
    _write(tmp_path, "ha.yaml", MINIMAL_EN)
    with pytest.raises(LanguageConfigError, match=r"ha\.yaml"):
        LanguageRegistry.load(tmp_path)


def test_pivot_language_required(tmp_path: Path) -> None:
    _write(tmp_path, "yo.yaml", MINIMAL_EN.replace("code: en", "code: yo"))
    with pytest.raises(LanguageConfigError, match="pivot"):
        LanguageRegistry.load(tmp_path)


def test_duplicate_order_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN)
    _write(tmp_path, "yo.yaml", MINIMAL_EN.replace("code: en", "code: yo"))
    with pytest.raises(LanguageConfigError, match="order"):
        LanguageRegistry.load(tmp_path)


def test_accented_term_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN.replace("'urgent'", "'kíákíá'"))
    with pytest.raises(LanguageConfigError, match="without accents"):
        LanguageRegistry.load(tmp_path)


def test_invalid_review_status_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN + "review: {lexicon: maybe}\n")
    with pytest.raises(LanguageConfigError, match="review"):
        LanguageRegistry.load(tmp_path)


def test_strings_must_match_pivot_keys(tmp_path: Path) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN + "strings: {greet: Hello, bye: Bye}\n")
    _write(
        tmp_path,
        "yo.yaml",
        MINIMAL_EN.replace("code: en", "code: yo").replace("order: 2", "order: 1")
        + "strings: {greet: Bawo}\n",
    )
    with pytest.raises(LanguageConfigError, match=r"yo\.yaml: missing strings \['bye'\]"):
        LanguageRegistry.load(tmp_path)


def test_strings_not_in_pivot_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN + "strings: {greet: Hello}\n")
    _write(
        tmp_path,
        "yo.yaml",
        MINIMAL_EN.replace("code: en", "code: yo").replace("order: 2", "order: 1")
        + "strings: {greet: Bawo, typo: x}\n",
    )
    with pytest.raises(LanguageConfigError, match="not in pivot"):
        LanguageRegistry.load(tmp_path)


def test_shipped_languages_have_complete_strings(registry: LanguageRegistry) -> None:
    pivot = registry.get("en").strings.keys()
    assert all(lang.strings.keys() == pivot for lang in registry.enabled)
