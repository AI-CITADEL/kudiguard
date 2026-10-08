from pathlib import Path

import pytest

from kudiguard.i18n import BUTTON_MAX_CHARS, Messages, Msg
from kudiguard.languages import LanguageConfigError, LanguageRegistry

MINIMAL = """
code: {code}
name: {name}
native_name: {name}
order: {order}
status: stable
asr_model: m/{code}
strings:
  greet: "Hello {{name}}"
  btn_ok: "OK"
"""


def write(tmp_path: Path, code: str, order: int, body: str | None = None) -> None:
    text = body if body is not None else MINIMAL.format(code=code, name=code.upper(), order=order)
    (tmp_path / f"{code}.yaml").write_text(text, encoding="utf-8")


# --- shipped catalogue --------------------------------------------------------------------------------


def test_pivot_defines_exactly_the_message_keys(registry: LanguageRegistry) -> None:
    assert set(registry.get("en").strings) == {m.value for m in Msg}


@pytest.mark.parametrize("code", ["ha", "yo", "ig", "en"])
def test_every_language_has_every_message(registry: LanguageRegistry, code: str) -> None:
    assert set(registry.get(code).strings) == {m.value for m in Msg}


@pytest.mark.parametrize("code", ["ha", "yo", "ig", "en"])
def test_button_labels_fit_whatsapp_limit(registry: LanguageRegistry, code: str) -> None:
    for key, text in registry.get(code).strings.items():
        if key.startswith("btn_"):
            assert len(text) <= BUTTON_MAX_CHARS, f"{code}.{key} is {len(text)} chars"


def test_messages_render_with_params(registry: LanguageRegistry) -> None:
    messages = Messages(registry)
    assert "Hausa" in messages.text("en", Msg.LANGUAGE_SET, language="Hausa")
    assert "30" in messages.text("ha", Msg.AUDIO_TOO_LONG, seconds=30)
    heard = messages.text("yo", Msg.I_HEARD, transcript="owo mi")
    assert "owo mi" in heard


def test_missing_param_raises(registry: LanguageRegistry) -> None:
    with pytest.raises(KeyError, match="seconds"):
        Messages(registry).text("en", Msg.AUDIO_TOO_LONG)


def test_unknown_language_raises(registry: LanguageRegistry) -> None:
    with pytest.raises(KeyError):
        Messages(registry).text("fr", Msg.WELCOME)


def test_user_text_in_params_is_not_reinterpreted(registry: LanguageRegistry) -> None:
    # Braces in user-supplied text must come out literally (no second formatting pass).
    heard = Messages(registry).text("en", Msg.I_HEARD, transcript="pay {amount} now {0}")
    assert "pay {amount} now {0}" in heard


def test_shipped_strings_never_promise_safety(registry: LanguageRegistry) -> None:
    for lang in registry.enabled:
        assert "safe" not in lang.strings[Msg.VERDICT_NO_FLAGS].lower()


# --- load-time validation -----------------------------------------------------------------------------


def test_placeholders_must_match_pivot(tmp_path: Path) -> None:
    write(tmp_path, "en", 1)
    write(tmp_path, "yo", 2, MINIMAL.format(code="yo", name="YO", order=2).replace("{name}", "{nme}"))
    with pytest.raises(LanguageConfigError, match=r"yo\.yaml: .*greet.*placeholders"):
        LanguageRegistry.load(tmp_path)


def test_malformed_template_rejected(tmp_path: Path) -> None:
    write(tmp_path, "en", 1, MINIMAL.format(code="en", name="EN", order=1).replace('"OK"', '"O{K"'))
    with pytest.raises(LanguageConfigError, match="btn_ok"):
        LanguageRegistry.load(tmp_path)


def test_long_button_rejected(tmp_path: Path) -> None:
    write(
        tmp_path, "en", 1, MINIMAL.format(code="en", name="EN", order=1).replace('"OK"', '"' + "x" * 25 + '"')
    )
    with pytest.raises(LanguageConfigError, match=r"btn_ok.*20"):
        LanguageRegistry.load(tmp_path)


@pytest.mark.parametrize("code", ["ha", "yo", "ig"])
def test_layout_matches_pivot(registry: LanguageRegistry, code: str) -> None:
    """Same line structure as English (paragraph breaks, list layout) so WhatsApp rendering is consistent."""
    pivot = registry.get("en").strings
    for key, text in registry.get(code).strings.items():
        assert text.count("\n") == pivot[key].count("\n"), key


@pytest.mark.parametrize("code", ["ha", "yo", "ig", "en"])
def test_consent_states_retention_and_controls(registry: LanguageRegistry, code: str) -> None:
    welcome = registry.get(code).strings[Msg.WELCOME]
    assert "90" in welcome and "*DELETE*" in welcome and "*STOP*" in welcome
    assert "Powered by N-ATLAS · Awarri" in welcome


def test_numbered_fixed_steps_keep_their_line_breaks(registry: LanguageRegistry) -> None:
    # A multi-line double-quoted YAML scalar folds line breaks into spaces: steps must stay one per line.
    for lang in registry.enabled:
        assert lang.strings["fallback_steps"].count("\n") == 2, lang.code
