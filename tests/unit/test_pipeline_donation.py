"""Opt-in voice-note donation in the conversation: the offer, explicit opt-in, withdrawal, erasure, background work."""

import asyncio
import dataclasses
import threading
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest
from test_pipeline import OTP_SCAM, PHONE, VOICE_NOTE, Harness, _queue_behind_a_slow_turn, interactions
from test_pipeline import h as _harness_fixture

from fakes import analysis
from kudiguard.domain import InboundMessage, MessageKind, Outcome
from kudiguard.donation import DONATION_VERSION, Sealer, VoiceVault
from kudiguard.hosting import StorageLocation
from kudiguard.i18n import Msg
from kudiguard.llm.base import Purpose
from kudiguard.pipeline import (
    CONSENT_VERSION,
    DONATE_NO,
    DONATE_YES,
    Action,
    Conversation,
    Limits,
    _Turn,
    button_id,
)
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


def wire(h: Harness, *, accept: bool = True, vault: VoiceVault | None = None, **limits: object) -> None:
    h.conversation = Conversation(
        engine=h.engine, store=h.store, messenger=h.messenger, hasher=h.hasher, donations=vault,
        accept_donations=accept, limits=Limits(**limits),  # type: ignore[arg-type]
    )  # fmt: skip


@pytest.fixture
async def d(h: Harness, tmp_path: Path) -> AsyncIterator[Harness]:
    """The shared harness, with a voice vault wired in (``h.vault``) and collection switched on."""
    h.vault = VoiceVault(tmp_path / "voice", h.store, Sealer(b"k" * MIN_SECRET_BYTES))  # type: ignore[attr-defined]
    wire(h, vault=vault(h))
    h.llm.script[Purpose.ANALYSE] = [analysis("x", "question")]
    h.asr.text = QUESTION
    yield h


def vault(h: Harness) -> VoiceVault:
    return h.vault  # type: ignore[attr-defined, no-any-return]


def voice_msg(
    h: Harness, *, recorded: bool = True, forwarded: bool = False, user: str = PHONE
) -> InboundMessage:
    message = h.msg(audio=VOICE_NOTE.read_bytes(), user=user)
    return dataclasses.replace(message, voice_note=recorded, forwarded=forwarded)


async def voice(h: Harness, **kwargs: object) -> list[str]:
    """A voice note, then any background donation work (write or offer)."""
    h.messenger.clear()
    await h.conversation.handle(voice_msg(h, **kwargs))  # type: ignore[arg-type]
    await h.conversation.wait_idle()
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


def gate_writes(v: VoiceVault) -> threading.Event:
    gate = threading.Event()
    original = v._write

    def slow_write(sample_id: str, audio: bytes, expected: str) -> int | None:
        gate.wait(timeout=5)
        return original(sample_id, audio, expected)

    v._write = slow_write  # type: ignore[method-assign]
    return gate


# --- the offer -------------------------------------------------------------------------------------------------


async def test_offer_follows_the_first_answered_own_voice_note_only_once(d: Harness) -> None:
    await d.onboard("en")
    texts = await voice(d)
    assert texts[0] == d.t(Msg.LISTENING) and texts[-1] == d.t(Msg.DONATION_OFFER)
    assert "outside Nigeria during the pilot" in texts[-1]  # where the audio would be stored
    assert offer_ids(d) == [YES, NO] and f"donate:yes-{DONATION_VERSION}" == YES
    titles = [b.title for b in d.messenger.sent[-1][1].buttons]
    assert titles == [d.t(Msg.BTN_DONATE_YES), d.t(Msg.BTN_DONATE_NO)] == ["Yes, I am 18+", "No, thanks"]
    state = await user(d)
    assert state.donation_offered_at is not None and state.donation_offered_version == DONATION_VERSION
    assert state.last_interaction_id == (await interactions(d))[-1]["id"]  # the turn's own save survives
    assert await samples(d) == []  # never retroactive: the note before the offer is not kept
    assert d.t(Msg.DONATION_OFFER) not in await voice(d)  # once per offer text


async def test_offer_states_what_the_law_requires(d: Harness) -> None:
    for language in COMMANDS:
        text = d.t(Msg.DONATION_OFFER, language)
        assert (
            "18" in text and "12" in text and "STOP DONATING" in text and "DELETE" in text and "STOP*" in text
        )
        assert d.t(Msg.WHERE_OUTSIDE, language) in text and "✅" not in d.t(Msg.BTN_DONATE_YES, language)
    english = d.t(Msg.DONATION_OFFER)
    for promise in ("own voice", "We try not to keep forwarded voice notes", "never use your voice to identify you",
                    "written confidentiality", "encrypted", "whether you say yes or no", "may not fully protect"):  # fmt: skip
        assert promise in english
    assert "never kept" not in english  # not claimed until confirmed on the live number


async def test_the_offer_follows_the_configured_storage_location(d: Harness) -> None:
    d.conversation._storage = StorageLocation.NIGERIA
    await d.onboard("ha")
    texts = await voice(d)
    assert d.t(Msg.WHERE_NIGERIA, "ha") in texts[-1] and d.t(Msg.WHERE_OUTSIDE, "ha") not in texts[-1]


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


@pytest.mark.parametrize(("recorded", "forwarded"), [(False, False), (True, True), (False, True)])
async def test_no_offer_after_text_forwarded_notes_or_audio_files(
    d: Harness, recorded: bool, forwarded: bool
) -> None:
    await d.onboard("en")
    assert d.t(Msg.DONATION_OFFER) not in await d.say(QUESTION)
    assert d.t(Msg.DONATION_OFFER) not in await voice(d, recorded=recorded, forwarded=forwarded)
    assert (await user(d)).donation_offered_at is None


@pytest.mark.parametrize("heard", ["", "hello"])
async def test_no_offer_when_the_voice_note_was_not_really_answered(d: Harness, heard: str) -> None:
    await d.onboard("en")
    d.asr.text = heard  # nothing understood, or only a greeting
    assert d.t(Msg.DONATION_OFFER) not in await voice(d)
    assert (await user(d)).donation_offered_at is None


async def test_no_offer_when_the_voice_note_was_unusable(d: Harness) -> None:
    await d.onboard("en")
    d.messenger.clear()
    await d.conversation.handle(dataclasses.replace(d.msg(audio=b"not audio at all"), voice_note=True))
    await d.conversation.wait_idle()
    assert d.messenger.texts[-1] == d.t(Msg.NOT_UNDERSTOOD) and offer_ids(d) == []


async def test_the_offer_is_rechecked_when_its_turn_comes(d: Harness) -> None:
    """The offer runs after messages already queued; a user who stopped meanwhile is not asked."""
    await d.onboard("en")
    stopped = dataclasses.replace(await user(d), stopped=True)
    await d.store.save_user(stopped)
    turn = _Turn(d.msg(QUESTION), d.user_hash, "en", 0.0, deadline=0.0)
    await d.conversation._offer_later(turn)
    await d.store.delete_user(d.user_hash)
    await d.conversation._offer_later(turn)
    assert d.messenger.sent == []


# --- opting in and what is kept ----------------------------------------------------------------------------------


async def test_yes_records_versioned_consent_and_keeps_later_voice_notes_after_the_reply(
    d: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await d.onboard("en")
    await voice(d)  # offered; this note is not kept
    before = utcnow()
    assert await d.say(button=YES) == [d.t(Msg.DONATION_ON)]
    consent = await user(d)
    assert (consent.donation_version, consent.donation_source) == (DONATION_VERSION, "chat")
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


async def test_only_recorded_unforwarded_voice_notes_are_ever_donated(d: Harness) -> None:
    await opt_in(d)
    await d.say(QUESTION)
    await voice(d, forwarded=True)
    await voice(d, recorded=False)  # an audio file, not a note recorded in WhatsApp
    await voice(d, recorded=False, forwarded=True)
    assert await samples(d) == []


async def test_voice_notes_merged_into_one_check_are_each_kept(d: Harness) -> None:
    await opt_in(d)
    rest = [voice_msg(d), voice_msg(d, forwarded=True), voice_msg(d, recorded=False), d.msg("and this")]
    await _queue_behind_a_slow_turn(d, voice_msg(d), rest)
    await d.conversation.wait_idle()
    assert len(await samples(d)) == 2  # both own recordings; not the forwarded one, the file or the text


async def test_a_yes_under_an_outdated_offer_shows_the_current_offer(d: Harness) -> None:
    await d.onboard("en")
    texts = await d.say(button="donate:yes-2020-01-01")
    assert texts == [d.t(Msg.DONATION_OFFER)] and offer_ids(d) == [YES, NO]
    assert (await user(d)).donation_version is None


async def test_no_thanks_declines_and_is_remembered(d: Harness) -> None:
    await d.onboard("en")
    assert await d.say(button=NO) == [d.t(Msg.DONATION_DECLINED)]
    state = await user(d)
    assert state.donation_version is None and state.donation_offered_version == DONATION_VERSION
    assert d.t(Msg.DONATION_OFFER) not in await voice(d)  # not asked again
    assert await samples(d) == []


async def test_consent_buttons_are_never_rate_limited(d: Harness) -> None:
    wire(d, vault=vault(d), per_user_per_hour=1)
    await d.onboard("en")
    await d.say(QUESTION)
    assert await d.say(QUESTION) == [d.t(Msg.RATE_LIMITED)]
    assert await d.say(button=YES) == [d.t(Msg.DONATION_ON)]
    assert await d.say("stop donating") == [d.t(Msg.DONATION_OFF)]


async def form_tester(h: Harness) -> None:
    """A row as `kudiguard voice preconsent` creates it for a tester who never wrote to KudiGuard."""
    at = utcnow() - timedelta(days=1)
    await h.store.save_user(
        UserState(h.user_hash, created_at=at, last_seen_at=at, donation_version=DONATION_VERSION, donation_at=at,
                  donation_source="form", donation_offered_at=at, donation_offered_version=DONATION_VERSION)
    )  # fmt: skip


@pytest.mark.parametrize("first", ["text", "voice", "button", "greeting"])
async def test_a_form_tester_gets_the_welcome_before_anything_is_processed(d: Harness, first: str) -> None:
    """The import never implies main consent: whatever they send first, they see the welcome (retention, reviewers,
    transfer abroad) and the language menu, and nothing is processed, logged or donated."""
    await form_tester(d)
    if first == "voice":
        texts = await voice(d)
    elif first == "button":
        texts = await d.say(button="lang:ha")  # e.g. a forwarded language menu
    else:
        texts = await d.say(QUESTION if first == "text" else "hello")
    assert texts[0].startswith(d.t(Msg.WELCOME).split("\n")[0]) and d.t(Msg.RISK_OUTSIDE) in texts[0]
    assert d.messenger.sent[-1][1].options  # the language menu
    state = await user(d)
    assert state.consented_at is None and state.welcome_version == CONSENT_VERSION
    assert state.donation_version == DONATION_VERSION  # the form's donation consent is kept
    assert await interactions(d) == [] and await samples(d) == [] and d.llm.requests == []
    # Only now does continuing count as consent, and their voice notes are kept from then on.
    await d.say(button="lang:en")
    agreed = await user(d)
    assert agreed.consent_version == CONSENT_VERSION and agreed.consented_at is not None
    texts = await voice(d)
    assert d.t(Msg.DONATION_OFFER) not in texts and len(await samples(d)) == 1


async def test_a_language_button_never_skips_the_welcome(d: Harness) -> None:
    """A stranger tapping a language option (an old or forwarded menu) is welcomed, not signed up."""
    texts = await d.say(button="lang:yo")
    assert texts[0] == d.t(Msg.WELCOME) and (await user(d)).consented_at is None
    await d.store.save_user(
        dataclasses.replace(await user(d), welcome_version="2026-10-08")
    )  # an old welcome
    texts = await d.say(button="lang:yo")
    assert texts[0] == d.t(Msg.WELCOME) and (await user(d)).consented_at is None


@pytest.mark.parametrize(("command", "reply"), [("DELETE", Msg.DELETED), ("STOP", Msg.STOPPED_DONOR)])
async def test_a_form_testers_first_message_can_be_delete_or_stop(
    d: Harness, command: str, reply: Msg
) -> None:
    await form_tester(d)
    assert await d.say(command) == [d.t(reply)]
    state = await d.store.get_user(d.user_hash)
    assert state is None or (state.stopped and state.donation_version is None and state.donation_withdrawn_at)


async def test_testers_who_consented_on_the_sign_up_form_are_not_asked_again(d: Harness) -> None:
    await d.onboard("en")
    now = utcnow()
    await d.store.save_user(
        dataclasses.replace(
            await user(d), donation_version=DONATION_VERSION, donation_at=now, donation_source="form",
            donation_offered_at=now, donation_offered_version=DONATION_VERSION,
        )
    )  # fmt: skip
    texts = await voice(d)
    assert d.t(Msg.DONATION_OFFER) not in texts and len(await samples(d)) == 1


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
    assert await d.say(donate) == [d.t(Msg.DONATION_OFFER, language)]  # shown again on request
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
    assert state.consented_at is None and state.donation_offered_version == DONATION_VERSION  # never offered


async def test_stop_also_deletes_donated_voice_notes(d: Harness) -> None:
    await opt_in(d)
    await voice(d)
    [kept] = await samples(d)
    assert await d.say("STOP") == [d.t(Msg.STOPPED_DONOR)]
    assert await samples(d) == [] and not vault(d).path(kept.id).exists()
    state = await user(d)
    assert state.stopped and state.donation_version is None and state.donation_source is None
    await d.onboard("en")  # back again: the main consent is renewed, donation is not; not asked again
    texts = await voice(d)
    assert await samples(d) == [] and d.t(Msg.DONATION_OFFER) not in texts


async def test_a_stopped_user_who_rejoins_never_gets_donation_consent_back(d: Harness) -> None:
    """Even if consent was (wrongly) recorded on a stopped user, rejoining starts without it: a fresh yes is needed."""
    await opt_in(d)
    await d.say("STOP")
    stopped = await user(d)
    await d.store.save_user(
        dataclasses.replace(
            stopped, donation_version=DONATION_VERSION, donation_at=utcnow(), donation_source="form"
        )
    )
    await d.say("hello")  # back: welcomed again
    rejoined = await user(d)
    assert rejoined.donation_version is None and rejoined.donation_source is None
    assert rejoined.donation_offered_version == DONATION_VERSION  # the offer mark stays: not asked again
    await d.say(button="lang:en")
    await voice(d)
    assert await samples(d) == []


async def test_the_transfer_risk_sentence_follows_the_storage_location(d: Harness) -> None:
    for language in COMMANDS:
        outside = d.t(Msg.WELCOME, language)
        assert d.t(Msg.RISK_OUTSIDE, language) in outside and d.t(Msg.RISK_NIGERIA, language) not in outside
    d.conversation._storage = StorageLocation.NIGERIA
    texts = await d.say("hello")
    assert d.t(Msg.RISK_NIGERIA) in texts[0] and d.t(Msg.RISK_OUTSIDE) not in texts[0]
    assert "Meta and Cloudflare may be processed abroad" in texts[0] and "in Nigeria" in texts[0]
    await d.say(button="lang:en")
    offer = (await voice(d))[-1]
    assert d.t(Msg.RISK_NIGERIA) in offer and d.t(Msg.RISK_OUTSIDE) not in offer


async def test_withdrawals_are_timed_for_the_form_import(d: Harness) -> None:
    await opt_in(d)
    assert (await user(d)).donation_withdrawn_at is None
    await d.say(button=NO)
    assert (await user(d)).donation_withdrawn_at is not None


async def test_every_kind_of_message_counts_as_activity(d: Harness) -> None:
    """last_seen_at moves on unsupported messages, feedback, rate-limited and "not available" replies too, so the
    form import's "active after the form" and the 90-day retention see them."""
    wire(d, vault=vault(d), accept=False, per_user_per_hour=1)
    await d.onboard("en")
    for message in (
        d.msg(kind=MessageKind.UNSUPPORTED), d.msg(button="helpful:1"), d.msg("donate voice"), d.msg(QUESTION),
        d.msg(QUESTION),  # the second one is rate-limited
    ):  # fmt: skip
        state = await user(d)
        await d.store.save_user(
            dataclasses.replace(state, last_seen_at=state.last_seen_at - timedelta(days=1))
        )
        stale = (await user(d)).last_seen_at
        await d.conversation.handle(message)
        assert (await user(d)).last_seen_at > stale, message
    assert d.messenger.texts[-1] == d.t(Msg.RATE_LIMITED)


async def test_stop_for_a_non_donor_is_the_usual_reply(d: Harness) -> None:
    await d.onboard("en")
    assert await d.say("STOP") == [d.t(Msg.STOPPED)]


async def test_delete_erases_the_audio_and_resets_the_offer(d: Harness) -> None:
    await opt_in(d)
    await voice(d)
    [kept] = await samples(d)
    assert await d.say("DELETE") == [d.t(Msg.DELETED)]
    assert await samples(d) == [] and not vault(d).path(kept.id).exists()
    assert await d.store.get_user(d.user_hash) is None
    await d.onboard("en")  # a new start: nothing remembered, so the offer may come once more
    assert (await voice(d))[-1] == d.t(Msg.DONATION_OFFER)


async def test_donation_survives_a_new_main_consent_text(d: Harness) -> None:
    await opt_in(d)
    await d.store.save_user(dataclasses.replace(await user(d), consent_version="2026-10-08"))
    assert (await d.say(QUESTION))[0] == d.t(Msg.WELCOME)  # asked again for the new main text
    state = await user(d)
    assert state.donation_version == DONATION_VERSION and state.donation_source == "chat"
    await d.say(button="lang:en")
    assert (await user(d)).consent_version == CONSENT_VERSION
    await voice(d)
    assert len(await samples(d)) == 1


async def test_choosing_a_language_to_renew_consent_keeps_donation(d: Harness) -> None:
    await opt_in(d)
    await d.store.save_user(dataclasses.replace(await user(d), consent_version="2026-10-08"))
    await d.say(button="lang:ha")
    renewed = await user(d)
    assert renewed.consent_version == CONSENT_VERSION and renewed.donation_version == DONATION_VERSION


async def test_a_new_offer_text_pauses_old_donors_and_asks_them_once(d: Harness) -> None:
    await d.onboard("en")
    old = "2025-01-01"
    await d.store.save_user(
        dataclasses.replace(
            await user(d), donation_version=old, donation_at=utcnow(), donation_offered_version=old
        )
    )
    texts = await voice(d)
    assert texts[-1] == d.t(Msg.DONATION_OFFER) and await samples(d) == []  # paused, asked again
    assert d.t(Msg.DONATION_OFFER) not in await voice(d)  # once
    assert await samples(d) == []  # still paused until they answer
    assert await d.say(button=YES) == [d.t(Msg.DONATION_ON)]
    await voice(d)
    assert len(await samples(d)) == 1


# --- collection switched off (KUDIGUARD_VOICE_DONATION=false, the default) ---------------------------------------


async def test_with_collection_off_nothing_is_offered_or_kept_but_withdrawal_works(d: Harness) -> None:
    wire(d, vault=vault(d), accept=False)
    await d.onboard("en")
    assert await d.say("donate voice") == [d.t(Msg.DONATION_UNAVAILABLE)]
    assert await d.say(button=YES) == [d.t(Msg.DONATION_UNAVAILABLE)]
    assert (await user(d)).donation_version is None
    now = utcnow()
    await d.store.save_user(  # a tester who agreed on the sign-up form
        dataclasses.replace(await user(d), donation_version=DONATION_VERSION, donation_at=now,
                            donation_offered_version=DONATION_VERSION)
    )  # fmt: skip
    texts = await voice(d)
    assert await samples(d) == [] and d.t(Msg.DONATION_OFFER) not in texts
    assert await d.say(button=NO) == [d.t(Msg.DONATION_OFF)]  # declining always works
    assert (await user(d)).donation_version is None


async def test_without_a_vault_withdrawal_still_deletes_rows(h: Harness) -> None:
    await h.onboard("en")
    assert await h.say("donate voice") == [h.t(Msg.DONATION_UNAVAILABLE)]
    await h.store.add_voice_sample(
        VoiceSample("a" * 32, h.user_hash, utcnow().date(), "en", "whatsapp", DONATION_VERSION, "x")
    )
    assert await h.say("stop donating") == [h.t(Msg.DONATION_OFF)]
    assert await h.store.voice_samples() == []
    assert (await user(h)).last_seen_at > utcnow() - timedelta(minutes=1)


# --- background work: off the reply path, ordered with withdrawal ------------------------------------------------


async def test_the_write_does_not_hold_the_turn_slot(d: Harness) -> None:
    wire(d, vault=vault(d), max_concurrent_turns=1)
    await opt_in(d)
    gate = gate_writes(vault(d))
    await d.conversation.handle(voice_msg(d))  # the reply is out; the write waits in the background
    other = "+2348099999999"
    await d.say("hello", user=other)  # another user's turn gets the only slot at once
    assert d.messenger.texts[0] == d.t(Msg.WELCOME)
    gate.set()
    await d.conversation.wait_idle()
    assert len(await samples(d)) == 1


async def test_a_withdrawal_right_after_a_voice_note_always_wins(d: Harness) -> None:
    await opt_in(d)
    gate = gate_writes(vault(d))
    await d.conversation.handle(voice_msg(d))
    withdrawal = asyncio.create_task(d.say("stop donating"))
    await asyncio.sleep(0.05)
    gate.set()
    assert await withdrawal == [d.t(Msg.DONATION_OFF)]
    await d.conversation.wait_idle()
    assert await samples(d) == [] and list(vault(d).directory.glob("*.kgv")) == []


async def test_failing_donation_steps_never_affect_the_reply_or_log_a_turn_failure(
    d: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    await opt_in(d)

    async def broken(donation):  # type: ignore[no-untyped-def]
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(vault(d), "donate", broken)
    texts = await voice(d)
    assert d.t(Msg.ERROR_RETRY) not in texts and len(texts) == 2
    assert (await interactions(d))[-1]["outcome"] != Outcome.ERROR.value
    assert "voice donation step failed" in caplog.text and "turn failed" not in caplog.text


async def test_a_failing_store_write_inside_the_vault_is_contained(
    d: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await opt_in(d)

    async def broken(sample):  # type: ignore[no-untyped-def]
        raise OSError("database is locked")

    monkeypatch.setattr(d.store, "add_voice_sample", broken)
    texts = await voice(d)
    assert d.t(Msg.ERROR_RETRY) not in texts and await samples(d) == []
    assert list(vault(d).directory.glob("*.kgv")) == []


async def test_an_offer_that_cannot_be_sent_is_not_an_error(
    d: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    await d.onboard("en")
    d.messenger.fail_on = {d.t(Msg.DONATION_OFFER)}
    texts = await voice(d)
    assert d.t(Msg.ERROR_RETRY) not in texts and (await interactions(d))[-1]["outcome"] != "error"
    assert "turn failed" not in caplog.text


async def test_shutdown_waits_for_donation_writes_then_cancels_stragglers(d: Harness) -> None:
    await opt_in(d)
    gate = gate_writes(vault(d))
    await d.conversation.handle(voice_msg(d))
    drain = asyncio.create_task(d.conversation.drain(timeout_s=5))
    await asyncio.sleep(0.05)
    assert not drain.done()  # waiting for the write
    gate.set()
    await drain
    assert len(await samples(d)) == 1

    wire(d, vault=vault(d))
    gate = gate_writes(vault(d))
    await d.conversation.handle(voice_msg(d))
    await d.conversation.drain(timeout_s=0.05)  # gives up; the shielded write still finishes in its thread
    gate.set()
    await asyncio.sleep(0.2)
    assert d.conversation._tasks == set()


async def test_a_message_kind_without_voice_never_triggers_background_work(d: Harness) -> None:
    await opt_in(d)
    await d.conversation.handle(d.msg(kind=MessageKind.UNSUPPORTED))
    assert d.conversation._tasks == set()
