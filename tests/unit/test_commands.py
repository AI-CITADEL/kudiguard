from pathlib import Path

import pytest

from kudiguard.commands import Command, CommandMatcher, Match
from kudiguard.languages import LanguageRegistry

BASE_EN = """
code: en
name: English
native_name: {native}
order: 1
status: stable
asr_model: m/en
commands:
{commands}
"""


def registry_with(
    tmp_path: Path, commands: str, native: str = "English", extra: str = ""
) -> LanguageRegistry:
    (tmp_path / "en.yaml").write_text(BASE_EN.format(native=native, commands=commands), encoding="utf-8")
    if extra:
        (tmp_path / "pcm.yaml").write_text(extra, encoding="utf-8")
    return LanguageRegistry.load(tmp_path)


def test_phrase_bound_to_two_commands_is_rejected(tmp_path: Path) -> None:
    reg = registry_with(tmp_path, "  stop: ['Stop']\n  delete: ['STOP!']")  # same canonical form
    with pytest.raises(ValueError, match=r"en\.yaml: 'STOP!' is configured for two commands"):
        CommandMatcher(reg)


def test_same_phrase_for_the_same_command_in_two_languages_is_fine(tmp_path: Path) -> None:
    pcm = BASE_EN.format(native="Naijá", commands="  stop: ['stop']").replace("code: en", "code: pcm")
    pcm = pcm.replace("name: English", "name: Pidgin").replace("order: 1", "order: 2")
    reg = registry_with(tmp_path, "  stop: ['Stop']", extra=pcm)
    assert CommandMatcher(reg).match("STOP") == Match(command=Command.STOP)


def test_language_name_without_words_is_not_a_choice(tmp_path: Path) -> None:
    matcher = CommandMatcher(registry_with(tmp_path, "  stop: ['stop']", native="'—'"))
    assert matcher.match("english") == Match(language="en")
    assert matcher.match("—") == Match()  # an empty canonical name must not become a catch-all key


def test_without_greetings_nothing_is_a_greeting(tmp_path: Path) -> None:
    matcher = CommandMatcher(registry_with(tmp_path, "  greetings: ['!!!', '🙂']"))  # nothing word-like
    assert matcher.match("hello") == Match()
    assert matcher.match("kudiguard") == Match()
    assert matcher.match("!!!") == Match()


@pytest.mark.parametrize("text", ["!!!", "🙂🙂", "  ...  ", "", None, "stop " * 20])
def test_punctuation_empty_or_long_messages_match_nothing(
    registry: LanguageRegistry, text: str | None
) -> None:
    assert CommandMatcher(registry).match(text) == Match()


# --- found by mutation testing (docs/SPIKES.md, G) ----------------------------------------------------------


def test_commands_up_to_sixty_characters(registry: LanguageRegistry) -> None:
    matcher = CommandMatcher(registry)
    assert matcher.match("STOP".ljust(60, "!")) == Match(command=Command.STOP)
    assert matcher.match("STOP".ljust(61, "!")) == Match()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello", Match(greeting_language="en")),
        ("thanks", Match(greeting_language="en")),
        ("hi KudiGuard!", Match(greeting_language="en")),
        ("KudiGuard", Match()),  # the name alone is not a greeting
        ("hellothanks", Match()),  # greetings are whole words
    ],
)
def test_greetings(registry: LanguageRegistry, text: str, expected: Match) -> None:
    assert CommandMatcher(registry).match(text) == expected


def test_a_phrase_for_delete_and_language_is_rejected(tmp_path: Path) -> None:
    reg = registry_with(tmp_path, "  delete: ['menu']\n  language: ['Menu!']")
    with pytest.raises(ValueError, match="two commands"):
        CommandMatcher(reg)


@pytest.mark.parametrize("text", ["akúàṣálẹ̀ o", "Ẹ kú àṣálẹ́ o", "ẹ kú àṣálẹ́"])
def test_yoruba_evening_greetings(text: str) -> None:
    from pathlib import Path

    from kudiguard.commands import CommandMatcher
    from kudiguard.languages import LanguageRegistry

    matcher = CommandMatcher(
        LanguageRegistry.load(Path(__file__).resolve().parents[2] / "data" / "languages")
    )
    assert (
        matcher.match(text).greeting_language == "yo"
    )  # live voice note 2026-10-12 was translated "I'm in trouble"
