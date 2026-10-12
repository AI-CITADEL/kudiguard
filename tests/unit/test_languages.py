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


def test_asr_configuration_is_per_language(registry: LanguageRegistry) -> None:
    assert registry.asr_models()["ha"] == "NCAIR1/Hausa-ASR"
    tokens = registry.asr_whisper_languages()
    assert tokens["NCAIR1/Hausa-ASR"] == "ha"
    # Whisper has no Igbo token; the Igbo model detects "yo" on its own, so it is forced (docs/SPIKES.md H).
    assert tokens["NCAIR1/Igbo-ASR"] == "yo"


@pytest.mark.parametrize("template", ["Send {} now", "Send {0} now", "Hi {user.name}", "Hi {names[0]}"])
def test_positional_or_attribute_placeholders_rejected(tmp_path: Path, template: str) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN + f"strings: {{greet: '{template}'}}\n")
    with pytest.raises(LanguageConfigError, match=r"en\.yaml: 'greet' placeholders must be names"):
        LanguageRegistry.load(tmp_path)


def test_named_placeholders_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN + "strings: {greet: 'Hi {name}, {{literal}} braces ok'}\n")
    assert LanguageRegistry.load(tmp_path).get("en").strings["greet"].startswith("Hi {name}")


# --- spoken-term corrections for voice-note transcripts (asr_terms) -------------------------------------------


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        ("[{pattern: 'bifi[eni', term: BVN}]", "invalid regex"),
        ("[{pattern: 'bifi eni', term: '  '}]", "term"),
        ("[{pattern: 'bifi ẹni', term: BVN}]", "without accents"),
        ("[{pattern: 'x?', term: BVN}]", "empty string"),
        ("[{pattern: 'bifi eni'}]", "term"),
    ],
)
def test_invalid_asr_terms_fail_loudly(tmp_path: Path, terms: str, message: str) -> None:
    _write(tmp_path, "en.yaml", MINIMAL_EN + f"asr_terms: {terms}\n")
    with pytest.raises(LanguageConfigError, match=message):
        LanguageRegistry.load(tmp_path)


def test_asr_terms_are_per_language_without_pivot_merge(tmp_path: Path, registry: LanguageRegistry) -> None:
    assert [t.term for t in registry.asr_terms("yo")] == ["BVN"]
    assert [t.term for t in registry.asr_terms("ha")] == ["BVN", "account number", "account"]
    assert registry.asr_terms("en") == () and registry.asr_terms("ig") == ()
    _write(tmp_path, "en.yaml", MINIMAL_EN)
    assert LanguageRegistry.load(tmp_path).asr_terms("en") == ()


@pytest.mark.parametrize(
    ("heard", "fixed"),
    [
        ("Wọ́n ní kí n fún wọn ní bifi eni mi", "Wọ́n ní kí n fún wọn ní BVN mi"),
        ("Wọ́n ní kí n fún wọn ní bifi ẹni mi", "Wọ́n ní kí n fún wọn ní BVN mi"),
        ("Wọ́n béèrè bifi ẹ́ni", "Wọ́n béèrè BVN"),  # tone mark on the last letter is swallowed with it
        ("BIFI-ENI mi", "BVN mi"),
        ("bifieni, OTP", "BVN, OTP"),
        ("bifi\u200beni mi", "BVN mi"),  # zero-width space
        ("bifi\u2013eni bifi \u2014 ẹni bifi\u2010eni", "BVN BVN BVN"),  # Unicode dashes
        ("bifi  eni àti bifi ẹni", "BVN àti BVN"),
        ("bifi ẹ̣ni", "BVN"),  # decomposed (NFD) under-dot
        ("bifiẹnikan ni", "bifiẹnikan ni"),  # a longer word: not the term
        ("abifi ẹni", "abifi ẹni"),
        ("bifi", "bifi"),
    ],
)
def test_asr_terms_replace_whole_words_ignoring_accents(
    registry: LanguageRegistry, heard: str, fixed: str
) -> None:
    from kudiguard.languages import apply_asr_terms

    text, replaced = apply_asr_terms(heard, registry.asr_terms("yo"))
    assert text == fixed
    assert replaced == heard.lower().count("bifi") - fixed.count("bifi")
    assert apply_asr_terms(heard, ()) == (heard, 0)


@pytest.mark.parametrize(
    ("heard", "fixed"),
    [
        (
            "wani ya kirani ya ce an toshe bibiyen ɗina.",
            "wani ya kirani ya ce an toshe BVN ɗina.",
        ),  # live voice note
        ("Bibiyen dina", "BVN dina"),
        ("bibiyena", "bibiyena"),  # a longer word: not the term
        ("ya ce na tura mai akawun nan ba na.", "ya ce na tura mai account number na."),  # live voice note
        ("an toshe akawun dina", "an toshe account dina"),
        ("akawunta", "akawunta"),
    ],
)
def test_hausa_asr_terms(registry: LanguageRegistry, heard: str, fixed: str) -> None:
    from kudiguard.languages import apply_asr_terms

    assert apply_asr_terms(heard, registry.asr_terms("ha"))[0] == fixed
