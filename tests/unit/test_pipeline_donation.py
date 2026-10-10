"""Opt-in voice-note donation in the conversation: the one-time offer, explicit opt-in, withdrawal and erasure."""

import dataclasses
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest
from test_pipeline import OTP_SCAM, PHONE, VOICE_NOTE, Harness, _queue_behind_a_slow_turn, interactions
from test_pipeline import h as _harness_fixture

from fakes import analysis
from kudiguard.domain import Outcome
from kudiguard.donation import DONATION_VERSION, Sealer, VoiceVault
from kudiguard.i18n import Msg
from kudiguard.llm.base import Purpose
from kudiguard.pipeline import CONSENT_VERSION, DONATE_NO, DONATE_YES, Action, Conversation, button_id
from kudiguard.privacy import MIN_SECRET_BYTES
from kudiguard.storage.base import UserState, VoiceSample, utcnow

h = _harness_fixture
QUESTION = "How do I open a savings account?"
YES = button_id(Action.DONATE, DONATE_YES)
NO = button_id(Action.DONATE, DONATE_NO)
COMMANDS = {  # (donate, stop donating), one phrase per language
    "en": ("Donate voice", "STOP DONATING"),
    "ha": ("ba da gudummawar murya", "daina gudummawar murya"),
    "yo": ("fi ohùn mi tọrẹ", "dá ìtọrẹ ohùn dúró"),
    "ig": ("nye onyinye olu", "kwụsị onyinye olu"),
}


@pytest.fixture
async def d(h: Harness, tmp_path: Path) -> AsyncIterator[Harness]:
    """The shared harness, with a voice vault wired in (``h.vault``)."""
    vault = VoiceVault(tmp_path / "voice", h.store, Sealer(b"k" * MIN_SECRET_BYTES))
    h.conversation = Conversation(
        engine=h.engine, store=h.store, messenger=h.messenger, hasher=h.hasher, donations=vault
    )
    h.vault = vault  # type: ignore[attr-defined]
    h.llm.script[Purpose.ANALYSE] = [analysis("x", "question")]
    h.asr.text = QUESTION
    yield h


def vault(h: Harness) -> VoiceVault:
    return h.vault  # type: ignore[attr-defined, no-any-return]


async def voice(h: Harness, *, forwarded: bool = False, user: str = PHONE) -> list[str]:
    h.messenger.clear()
    message = dataclasses.replace(h.msg(audio=VOICE_NOTE.read_bytes(), user=user), forwarded=forwarded)
    await h.conversation.handle(message)
    return h.messenger.texts


async def samples(h: Harness) -> list[VoiceSample]:
    return await h.store.voice_samples()


async def user(h: Harness, phone: str = PHONE) -> UserState:
    found = await h.store.get_user(h.hasher.user_id(phone))
    assert found is not None
    return found


def offer_ids(h: Harness) -> list[str]:
    return [b.id for _, m in h.messenger.sent for b in m.buttons if b.id.startswith("donate:")]


async def opt_in(h: Harness, language: str = "en") -> None:
    await h.onboard(language)
    assert await h.say(button=YES) == [h.t(Msg.DONATION_ON, language)]


# --- the offer -------------------------------------------------------------------------------------------------


async def test_offer_follows_the_first_answered_own_voice_note_only_once(d: Harness) -> None:
    await d.onboard("en")
    texts = await voice(d)
    assert texts[0] == d.t(Msg.LISTENING) and texts[-1] == d.t(Msg.DONATION_OFFER)
    assert offer_ids(d) == [YES, NO] and f"donate:yes-{DONATION_VERSION}" == YES
    titles = [b.title for b in d.messenger.sent[-1][1].buttons]
    assert titles == [d.t(Msg.BTN_DONATE_YES), d.t(Msg.BTN_DONATE_NO)] and "18" in titles[0]
    assert (await user(d)).donation_offered_at is not None
    assert (await user(d)).last_interaction_id == (await interactions(d))[-1][
        "id"
    ]  # the turn's save survives
    assert await samples(d) == []  # never retroactive: the note before the offer is not kept
    assert d.t(Msg.DONATION_OFFER) not in await voice(d)  # at most once


async def test_offer_states_what_the_law_requires(d: Harness) -> None:
    for language in COMMANDS:
        text = d.t(Msg.DONATION_OFFER, language)
        assert "18" in text and "12" in text and "STOP DONATING" in text and "DELETE" in text
    english = d.t(Msg.DONATION_OFFER)
    for promise in ("own voice", "forwarded voice notes are never kept", "never use your voice to identify you",
                    "written confidentiality", "encrypted", "whether you say yes or no"):  # fmt: skip
        assert promise in english


async def test_no_offer_right_after_a_scam_warning(d: Harness) -> None:
    await d.onboard("en")
    d.asr.text = OTP_SCAM
    d.llm.script[Purpose.ANALYSE] = [analysis("x", "scam_check")]
    texts = await voice(d)
    assert d.t(Msg.VERDICT_LIKELY_SCAM) in texts[1] and d.t(Msg.DONATION_OFFER) not in texts
    assert (await user(d)).donation_offered_at is None
    d.asr.text = QUESTION
    d.llm.script[Purpose.ANALYSE] = [analysis("x", "question")]
    assert (await voice(d))[-1] == d.t(Msg.DONATION_OFFER)  # a calmer moment


async def test_no_offer_after_text_or_a_forwarded_voice_note(d: Harness) -> None:
    await d.onboard("en")
    assert d.t(Msg.DONATION_OFFER) not in await d.say(QUESTION)
    assert d.t(Msg.DONATION_OFFER) not in await voice(d, forwarded=True)
    assert (await user(d)).donation_offered_at is None


@pytest.mark.parametrize("heard", ["", "hello"])
async def test_no_offer_when_the_voice_note_was_not_really_answered(d: Harness, heard: str) -> None:
    await d.onboard("en")
    d.asr.text = heard  # nothing understood, or only a greeting
    texts = await voice(d)
    assert d.t(Msg.DONATION_OFFER) not in texts and (await user(d)).donation_offered_at is None


async def test_no_offer_when_the_voice_note_was_unusable(d: Harness) -> None:
    await d.onboard("en")
    d.messenger.clear()
    await d.conversation.handle(d.msg(audio=b"not audio at all"))
    assert d.messenger.texts[-1] == d.t(Msg.NOT_UNDERSTOOD) and offer_ids(d) == []


# --- opting in and what is kept ----------------------------------------------------------------------------------


async def test_yes_records_versioned_consent_and_keeps_later_voice_notes_after_the_reply(
    d: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await d.onboard("en")
    await voice(d)  # offered; this note is not kept
    before = utcnow()
    assert await d.say(button=YES) == [d.t(Msg.DONATION_ON)]
    consent = await user(d)
    assert consent.donation_version == DONATION_VERSION
    assert consent.donation_at is not None and consent.donation_at >= before
    assert await samples(d) == []

    seen_at: list[int] = []
    original = vault(d).donate

    async def spy(donation):  # type: ignore[no-untyped-def]
        seen_at.append(len(d.messenger.sent))
        return await original(donation)

    monkeypatch.setattr(vault(d), "donate", spy)
    texts = await voice(d)
    assert seen_at == [len(texts)]  # written only after every reply message was sent
    [kept] = await samples(d)
    assert kept.user_hash == d.user_hash and kept.language == "en" and kept.channel == "whatsapp"
    assert kept.consent_version == DONATION_VERSION and kept.hypothesis == QUESTION
    assert kept.duration_s is not None and kept.duration_s > 0
    assert await vault(d).read(kept.id) == VOICE_NOTE.read_bytes()  # the original bytes, as received
    assert d.t(Msg.DONATION_OFFER) not in texts


async def test_text_and_forwarded_voice_notes_are_never_donated(d: Harness) -> None:
    await opt_in(d)
    await d.say(QUESTION)
    await voice(d, forwarded=True)
    assert await samples(d) == []


async def test_voice_notes_merged_into_one_check_are_each_kept(d: Harness) -> None:
    await opt_in(d)
    note = VOICE_NOTE.read_bytes()
    forwarded = dataclasses.replace(d.msg(audio=note), forwarded=True)
    await _queue_behind_a_slow_turn(d, d.msg(audio=note), [d.msg(audio=note), forwarded, d.msg("and this")])
    assert len(await samples(d)) == 2  # both own voice notes; the forwarded one and the text are not


async def test_a_yes_under_an_outdated_offer_shows_the_current_offer(d: Harness) -> None:
    await d.onboard("en")
    texts = await d.say(button="donate:yes-2020-01-01")
    assert texts == [d.t(Msg.DONATION_OFFER)] and offer_ids(d) == [YES, NO]
    assert (await user(d)).donation_version is None


async def test_no_thanks_declines_and_is_remembered(d: Harness) -> None:
    await d.onboard("en")
    assert await d.say(button=NO) == [d.t(Msg.DONATION_DECLINED)]
    state = await user(d)
    assert state.donation_version is None and state.donation_offered_at is not None
    assert d.t(Msg.DONATION_OFFER) not in await voice(d)  # not asked again
    assert await samples(d) == []


async def test_consent_buttons_are_never_rate_limited(d: Harness) -> None:
    d.configure(per_user_per_hour=1)
    d.conversation._donations = vault(d)
    await d.onboard("en")
    await d.say(QUESTION)
    assert await d.say(QUESTION) == [d.t(Msg.RATE_LIMITED)]
    assert await d.say(button=YES) == [d.t(Msg.DONATION_ON)]
    assert await d.say("stop donating") == [d.t(Msg.DONATION_OFF)]


# --- withdrawal and erasure ------------------------------------------------------------------------------------


async def test_no_while_donating_withdraws_and_deletes_everything(d: Harness) -> None:
    await opt_in(d)
    await voice(d)
    [kept] = await samples(d)
    assert vault(d).path(kept.id).exists()
    assert await d.say(button=NO) == [d.t(Msg.DONATION_OFF)]
    assert await samples(d) == [] and not vault(d).path(kept.id).exists()
    assert (await user(d)).donation_version is None
    await voice(d)
    assert await samples(d) == []  # nothing after withdrawal


@pytest.mark.parametrize("language", sorted(COMMANDS))
async def test_commands_opt_in_and_out_in_every_language(d: Harness, language: str) -> None:
    donate, stop = COMMANDS[language]
    await d.onboard(language)
    assert await d.say(donate) == [d.t(Msg.DONATION_OFFER, language)]
    assert offer_ids(d) == [YES, NO]
    assert [b.title for b in d.messenger.sent[0][1].buttons] == [
        d.t(Msg.BTN_DONATE_YES, language), d.t(Msg.BTN_DONATE_NO, language),
    ]  # fmt: skip
    offered = (await user(d)).donation_offered_at
    assert await d.say(donate) == [d.t(Msg.DONATION_OFFER, language)]  # shown again on request
    assert (await user(d)).donation_offered_at == offered  # the first offer's time is kept
    assert await d.say(button=YES) == [d.t(Msg.DONATION_ON, language)]
    await voice(d)
    [kept] = await samples(d)
    assert kept.language == language
    assert await d.say(stop) == [d.t(Msg.DONATION_OFF, language)]
    assert await samples(d) == [] and not vault(d).path(kept.id).exists()
    assert (await user(d)).donation_version is None


async def test_stop_donating_works_for_anyone_at_any_time(d: Harness) -> None:
    assert await d.say("stop donating") == [
        d.t(Msg.DONATION_OFF)
    ]  # a stranger: nothing kept, nothing created
    assert await d.store.get_user(d.user_hash) is None
    await d.say("hello")  # welcomed, not yet consented
    assert await d.say("stop donating") == [d.t(Msg.DONATION_OFF)]
    state = await user(d)
    assert state.consented_at is None and state.donation_offered_at is not None  # never offered later


async def test_stop_ends_donation_and_delete_erases_the_audio(d: Harness) -> None:
    await opt_in(d)
    await voice(d)
    [kept] = await samples(d)
    assert await d.say("STOP") == [d.t(Msg.STOPPED)]
    assert (await user(d)).donation_version is None
    assert len(await samples(d)) == 1  # STOP keeps data until it expires or DELETE (as stated on /privacy)
    await d.onboard("en")  # back again: the main consent is renewed, donation is not
    texts = await voice(d)
    assert len(await samples(d)) == 1 and d.t(Msg.DONATION_OFFER) not in texts
    assert await d.say("DELETE") == [d.t(Msg.DELETED)]
    assert await samples(d) == [] and not vault(d).path(kept.id).exists()
    assert await d.store.get_user(d.user_hash) is None


async def test_a_new_main_consent_text_ends_donation_but_not_the_offered_mark(d: Harness) -> None:
    await opt_in(d)
    state = await user(d)
    await d.store.save_user(dataclasses.replace(state, consent_version="2026-10-08"))
    assert (await d.say(QUESTION))[0] == d.t(Msg.WELCOME)  # asked again for the new text
    state = await user(d)
    assert state.donation_version is None and state.donation_at is None
    assert state.donation_offered_at is not None
    await d.say(button="lang:en")
    assert (await user(d)).consent_version == CONSENT_VERSION
    await voice(d)
    assert await samples(d) == []


async def test_choosing_a_language_to_renew_consent_also_ends_donation(d: Harness) -> None:
    await opt_in(d)
    state = await user(d)
    await d.store.save_user(dataclasses.replace(state, consent_version="2026-10-08"))
    await d.say(button="lang:ha")
    renewed = await user(d)
    assert renewed.consent_version == CONSENT_VERSION and renewed.donation_version is None


async def test_choosing_a_language_while_donating_keeps_donation(d: Harness) -> None:
    await opt_in(d)
    await d.say(button="lang:yo")
    assert (await user(d)).donation_version == DONATION_VERSION


# --- failures never touch the reply -------------------------------------------------------------------------------


async def test_a_failing_donation_never_affects_the_reply(
    d: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await opt_in(d)

    async def broken(donation):  # type: ignore[no-untyped-def]
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(vault(d), "donate", broken)
    texts = await voice(d)
    assert d.t(Msg.ERROR_RETRY) not in texts
    rows = await interactions(d)
    assert rows[-1]["outcome"] != Outcome.ERROR.value and len(texts) == 2


async def test_a_failing_store_write_inside_the_vault_is_contained(
    d: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await opt_in(d)

    async def broken(sample):  # type: ignore[no-untyped-def]
        raise OSError("database is locked")

    monkeypatch.setattr(d.store, "add_voice_sample", broken)
    texts = await voice(d)
    assert d.t(Msg.ERROR_RETRY) not in texts and await samples(d) == []
    assert list(vault(d).directory.iterdir()) == []


async def test_an_offer_that_cannot_be_sent_is_not_an_error(d: Harness) -> None:
    await d.onboard("en")
    d.messenger.fail_on = {d.t(Msg.DONATION_OFFER)}
    texts = await voice(d)
    assert d.t(Msg.ERROR_RETRY) not in texts and (await interactions(d))[-1]["outcome"] != "error"


# --- a server without a vault ------------------------------------------------------------------------------------


async def test_without_a_vault_nothing_is_offered_or_kept_but_withdrawal_still_works(h: Harness) -> None:
    await h.onboard("en")
    assert await h.say("donate voice") == [h.t(Msg.GREETING_REPLY)]
    assert await h.say(button=YES) == [h.t(Msg.GREETING_REPLY)]  # cannot opt in where nothing can be kept
    assert (await user(h)).donation_version is None
    await h.store.add_voice_sample(
        VoiceSample("a" * 32, h.user_hash, utcnow().date(), "en", "whatsapp", DONATION_VERSION, "x")
    )
    assert await h.say("stop donating") == [h.t(Msg.DONATION_OFF)]
    assert await h.store.voice_samples() == []
    assert (await user(h)).last_seen_at > utcnow() - timedelta(minutes=1)
