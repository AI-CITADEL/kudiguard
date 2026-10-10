"""Output guard: unsafe instructions ("share the code", "pay the fee", ...) in all four languages, negation-aware.

ha/yo/ig patterns are DRAFT (native review pending); the false-positive sweep below runs every fixed string, every
pack fact and action, the prompt examples and the legitimate eval scenarios through the guard.
"""

from pathlib import Path

import pytest
import yaml

from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.file_pack import FilePackSource
from kudiguard.languages import LanguageRegistry
from kudiguard.prompts import _EXAMPLE_CREDENTIALS, _EXAMPLE_QUESTION, _EXAMPLE_SCAM
from kudiguard.safety.output_guard import OutputGuard, OutputIssue

ROOT = Path(__file__).resolve().parents[2]
UNSAFE = OutputIssue.UNSAFE_INSTRUCTION


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "Share the code. Stay on the call."),
        ("en", "Give them the code they sent you."),
        ("en", "You can send the caller your OTP now."),
        ("en", "Tell him your PIN so he can fix it."),
        ("en", "Pay the N5,000 processing fee to claim your prize."),
        ("en", "Pay the fee so you can claim it."),
        ("en", "Click the link to update your BVN."),
        ("en", "Click [link removed] to update your BVN."),
        ("en", "Install the app they asked for."),
        ("en", "Download AnyDesk and share the code that comes out."),
        ("en", "Download QuickKash.apk from the link."),
        ("en", "Send the money back yourself."),
        ("en", "Refund it to his sister's account."),
        ("en", "1. Do not panic.\n2. Share the OTP with the agent."),  # negation in another sentence
        ("ha", "Ku tura musu lambar OTP da aka turo muku."),
        ("ha", "Ku biya kudin rajista domin ku karbi rancen."),
        ("ha", "Ku danna link din domin gyara BVN."),
        ("ha", "Ku sauke manhajar AnyDesk."),
        ("ha", "Ku mayar musu da kudin da kanku."),
        ("yo", "Ẹ fi kóòdù náà ránṣẹ́ sí wọn."),
        ("yo", "Ẹ fún wọn ní OTP yín."),
        ("yo", "Ẹ san processing fee náà."),
        ("yo", "Ẹ tẹ link yìí kí ẹ lè ṣàtúnṣe."),
        ("yo", "Ẹ download app AnyDesk."),
        ("yo", "Ẹ dá owó náà padà fún un."),
        ("ig", "Nye ha koodu ahụ ozugbo."),
        ("ig", "Zite OTP gị ugbu a."),
        ("ig", "Kwụọ processing fee ahụ taa."),
        ("ig", "Pịa link ahụ."),
        ("ig", "Budata app AnyDesk."),
        ("ig", "Zighachi ha ego ahụ."),
        ("ha", "Send them the OTP now."),  # English mixed into a Hausa reply
    ],
)
def test_unsafe_instructions_are_blocked(registry: LanguageRegistry, language: str, text: str) -> None:
    assert UNSAFE in OutputGuard(registry).check(text, language)


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "Do not share the code. End the call."),
        ("en", "Never share your OTP, PIN or BVN with anyone."),
        ("en", "Don't send them the code, and do not pay any fee."),
        ("en", "Do not click the link or install the app."),
        ("en", "Do not send any money back yourself."),
        ("en", "If they ask you to share the code, end the call."),
        ("en", "Scammers want you to pay the fee first."),
        ("en", "If you already shared it, call your bank now."),
        ("en", "Only download the app from the Play Store."),
        ("en", "Tell someone you trust before you act."),
        ("en", "Tell the person to contact their own bank to reverse it."),
        ("en", "Enter the code only in your own bank app."),
        ("en", "Pay yourself first: save before you spend."),
        ("ha", "Kada ku tura musu lambar OTP."),
        ("ha", "Kar ku biya kudin rajista."),
        ("ha", "Kada ku danna link din, kuma kada ku sauke manhajar."),
        ("ha", "Kada ku mayar da kudin da kanku."),
        ("yo", "Ẹ má ṣe fi kóòdù náà ránṣẹ́ sí wọn."),
        ("yo", "Ẹ má ṣe fún wọn ní OTP yín."),
        ("yo", "Ẹ má ṣe san processing fee náà."),
        ("yo", "Ẹ má tẹ link yìí."),
        ("ig", "Enyela ha koodu ahụ."),
        ("ig", "Ezitela OTP gị."),
        ("ig", "Akwụla processing fee ahụ."),
        ("ig", "Pịala link ahụ."),
        ("ig", "Ụlọ akụ gị agaghị ajụ gị OTP."),
    ],
)
def test_negated_and_reported_forms_pass(registry: LanguageRegistry, language: str, text: str) -> None:
    assert UNSAFE not in OutputGuard(registry).check(text, language)


def test_fixed_text_never_trips_the_guard(registry: LanguageRegistry) -> None:
    guard, messages = OutputGuard(registry), Messages(registry)
    keys = (Msg.FALLBACK_SCAM, Msg.FALLBACK_STEPS, Msg.FALLBACK_ANSWER, Msg.WARN_FAMILY_FALLBACK, Msg.WELCOME)
    for language in registry.enabled:
        for key in keys:
            text = messages.text(language.code, key)
            assert UNSAFE not in guard.check(text, language.code), (language.code, key)
        for key, text in language.strings.items():
            assert UNSAFE not in guard.check(text, language.code), (language.code, key)


def test_pack_facts_actions_and_prompt_examples_pass(registry: LanguageRegistry) -> None:
    guard = OutputGuard(registry)
    texts = [e.reply for e in (_EXAMPLE_CREDENTIALS, _EXAMPLE_SCAM, _EXAMPLE_QUESTION)]
    for pack in (ROOT / "data" / "packs").iterdir():
        if (pack / "pack.yaml").is_file() and "type: http" not in (pack / "pack.yaml").read_text(
            encoding="utf-8"
        ):
            for entry in FilePackSource.load(pack).entries:
                texts.extend((*entry.facts, *entry.actions))
    hits = [t for t in texts if UNSAFE in guard.check(t, "en")]
    assert hits == []


def test_legitimate_scenarios_and_supported_claims_pass(registry: LanguageRegistry) -> None:
    guard = OutputGuard(registry)
    scenarios = yaml.safe_load((ROOT / "data" / "eval" / "scenarios.yaml").read_text(encoding="utf-8"))
    claims = yaml.safe_load((ROOT / "data" / "eval" / "grounding_claims.yaml").read_text(encoding="utf-8"))
    texts = [(s["lang"], s["text"]) for s in scenarios if s["label"] == "legit"]
    texts += [(c["lang"], c["claim"]) for c in claims["claims"] if c["supported"]]
    texts += [("en", r["reply"]) for r in claims["replies"] if r["acceptable"]]
    hits = [t for t in texts if UNSAFE in guard.check(t[1], t[0])]
    assert hits == []
