import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from fakes import FakeASR, FakeLLM, FakeMessenger, analysis
from kudiguard.domain import Flag, InboundMessage, MessageKind, Outcome, Verdict
from kudiguard.engine import Engine
from kudiguard.i18n import Messages, Msg
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import Purpose
from kudiguard.pipeline import (
    CONSENT_VERSION,
    MAX_COALESCED,
    Action,
    Conversation,
    Limits,
    button_id,
    parse_button,
)
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.speech.audio import find_ffmpeg
from kudiguard.speech.base import ASRError, AudioError, AudioProblem
from kudiguard.storage.memory import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
VOICE_NOTE = ROOT / "tests" / "fixtures" / "voice_note_en_otp.ogg"
PHONE = "+2348031234567"
OTP_SCAM = "Someone called and asked me to send the OTP sent to my phone. Is this a scam?"
REPLY = "1. Do not share the code.\n2. Call your bank on the number on your card."


@dataclass
class Harness:
    conversation: Conversation
    store: MemoryStore
    messenger: FakeMessenger
    llm: FakeLLM
    asr: FakeASR
    hasher: UserHasher
    messages: Messages
    engine: Engine
    counter: int = 0

    def configure(self, **limits: object) -> None:
        """A fresh Conversation with these limits over the same engine, store and messenger."""
        self.conversation = Conversation(
            engine=self.engine, store=self.store, messenger=self.messenger, hasher=self.hasher,
            limits=Limits(**limits),  # type: ignore[arg-type]
        )  # fmt: skip

    def msg(self, text: str | None = None, *, user: str = PHONE, kind: MessageKind | None = None,
            button: str | None = None, audio: bytes | None = None, message_id: str | None = None,
            media_id: str | None = None) -> InboundMessage:  # fmt: skip
        self.counter += 1
        if kind is None:
            kind = (
                MessageKind.BUTTON if button else MessageKind.AUDIO if audio or media_id else MessageKind.TEXT
            )
        return InboundMessage(
            channel="whatsapp", message_id=message_id or f"wamid.{self.counter}", user_id=user, kind=kind,
            text=text, audio=audio, button_id=button, media_id=media_id,
        )  # fmt: skip

    async def say(self, text: str | None = None, **kwargs: object) -> list[str]:
        self.messenger.clear()
        await self.conversation.handle(self.msg(text, **kwargs))  # type: ignore[arg-type]
        return self.messenger.texts

    async def onboard(self, language: str = "en", user: str = PHONE) -> None:
        await self.say("hello", user=user)
        await self.say(button=button_id(Action.LANGUAGE, language), user=user)
        self.messenger.clear()

    def t(self, key: Msg, lang: str = "en", /, **params: object) -> str:
        return self.messages.text(lang, key, **params)

    @property
    def user_hash(self) -> str:
        return self.hasher.user_id(PHONE)


@pytest.fixture
async def h(registry: LanguageRegistry) -> AsyncIterator[Harness]:
    hasher = UserHasher(b"k" * MIN_SECRET_BYTES)
    async with httpx.AsyncClient() as client:
        knowledge = knowledge_base_from_packs(
            ROOT / "data" / "packs", client=client, include=["kudiguard-core"]
        )
        llm = FakeLLM(analyse=analysis("x", "scam_check"), compose=REPLY, translate="x", warn="⚠️ Beware.")
        asr = FakeASR(OTP_SCAM)
        engine = Engine(registry=registry, llm=llm, knowledge=knowledge, asr=asr, ffmpeg=find_ffmpeg())
        store = MemoryStore(hasher)
        messenger = FakeMessenger()
        conversation = Conversation(engine=engine, store=store, messenger=messenger, hasher=hasher)
        yield Harness(conversation, store, messenger, llm, asr, hasher, Messages(registry), engine)


async def interactions(h: Harness) -> list[dict[str, object]]:
    return await h.store.export_interactions()


# --- onboarding and consent ------------------------------------------------------------------------------


async def test_first_contact_gets_welcome_and_language_menu_without_processing(h: Harness) -> None:
    texts = await h.say(OTP_SCAM)
    assert texts[0] == h.t(Msg.WELCOME)
    menu = h.messenger.sent[1][1]
    assert menu.menu_button == h.t(Msg.LANGUAGE_MENU_BUTTON)
    assert [o.id for o in menu.options] == ["lang:ha", "lang:yo", "lang:ig", "lang:en"]
    assert h.llm.requests == [] and await interactions(h) == []  # nothing processed before consent
    user = await h.store.get_user(h.user_hash)
    assert user is not None and user.consented_at is None


async def test_greeting_in_hausa_gets_hausa_welcome(h: Harness) -> None:
    texts = await h.say("Sannu")
    assert texts[0] == h.t(Msg.WELCOME, "ha")


async def test_choosing_a_language_records_consent(h: Harness) -> None:
    await h.say("hello")
    texts = await h.say(button="lang:ha")
    assert texts == [
        "\n\n".join((h.t(Msg.LANGUAGE_SET, "ha", language="Hausa"), h.t(Msg.GREETING_REPLY, "ha")))
    ]
    user = await h.store.get_user(h.user_hash)
    assert user is not None and user.language == "ha" and user.consent_version == CONSENT_VERSION


async def test_language_by_name_works_at_any_time(h: Harness) -> None:
    await h.onboard("en")
    texts = await h.say("Yorùbá")
    assert texts[0].startswith(h.t(Msg.LANGUAGE_SET, "yo", language="Yorùbá"))
    user = await h.store.get_user(h.user_hash)
    assert user is not None and user.language == "yo"


async def test_real_message_after_welcome_is_consent_and_is_answered(h: Harness) -> None:
    await h.say("hello")
    texts = await h.say(OTP_SCAM)
    assert texts[0].startswith(h.t(Msg.VERDICT_LIKELY_SCAM))
    user = await h.store.get_user(h.user_hash)
    assert user is not None and user.consented_at is not None and user.language == "en"


async def test_invalid_language_button_is_ignored(h: Harness) -> None:
    await h.onboard("en")
    texts = await h.say(button="lang:xx")
    assert texts == [h.t(Msg.GREETING_REPLY)]


# --- commands --------------------------------------------------------------------------------------------


async def test_stop_then_any_message_restarts_with_welcome(h: Harness) -> None:
    await h.onboard("ha")
    assert await h.say("STOP") == [h.t(Msg.STOPPED, "ha")]
    user = await h.store.get_user(h.user_hash)
    assert user is not None and user.stopped
    texts = await h.say(OTP_SCAM)
    assert texts[0] == h.t(Msg.WELCOME, "ha") and h.llm.requests == []


async def test_delete_erases_everything(h: Harness) -> None:
    await h.onboard("en")
    await h.say(OTP_SCAM)
    assert await h.say("delete") == [h.t(Msg.DELETED)]
    assert await h.store.get_user(h.user_hash) is None and await interactions(h) == []


async def test_ambiguous_native_words_are_not_commands(h: Harness) -> None:
    await h.onboard("ha")
    h.llm.script[Purpose.ANALYSE] = [analysis("Wait", "other")]
    texts = await h.say("tsaya")  # "wait" in Hausa, not STOP
    assert texts != [h.t(Msg.STOPPED, "ha")]
    user = await h.store.get_user(h.user_hash)
    assert user is not None and not user.stopped


async def test_language_menu_command(h: Harness) -> None:
    await h.onboard("en")
    await h.say("change language")
    assert h.messenger.sent[0][1].options


async def test_duplicate_delivery_is_processed_once(h: Harness) -> None:
    await h.onboard("en")
    message = h.msg(OTP_SCAM, message_id="wamid.same")
    await h.conversation.handle(message)
    await h.conversation.handle(message)
    assert len(h.llm.calls(Purpose.ANALYSE)) == 1


# --- checks and buttons ----------------------------------------------------------------------------------


async def test_scam_reply_has_action_buttons_then_feedback_prompt(h: Harness) -> None:
    await h.onboard("en")
    await h.say(OTP_SCAM)
    (_, reply), (_, prompt) = h.messenger.sent
    rows = await interactions(h)
    interaction_id = rows[-1]["id"]
    assert reply.text.startswith(h.t(Msg.VERDICT_LIKELY_SCAM)) and h.t(Msg.WHERE_TO_REPORT) in reply.text
    assert [b.id for b in reply.buttons] == [
        f"steps:{interaction_id}",
        f"report:{interaction_id}",
        f"warn:{interaction_id}",
    ]
    assert prompt.text == h.t(Msg.FEEDBACK_PROMPT)
    assert [b.id for b in prompt.buttons] == [f"helpful:{interaction_id}", f"unhelpful:{interaction_id}"]
    row = rows[-1]
    assert row["verdict"] == Verdict.LIKELY_SCAM.value and row["outcome"] == Outcome.ANSWERED.value
    assert Flag.ASKS_CREDENTIALS.value in str(row["flags"])
    user = await h.store.get_user(h.user_hash)
    assert user is not None and user.last_interaction_id == interaction_id


async def test_question_reply_carries_feedback_buttons(h: Harness) -> None:
    await h.onboard("en")
    h.llm.script[Purpose.ANALYSE] = [analysis("x", "question")]
    await h.say("How do I open a savings account?")
    [(_, reply)] = h.messenger.sent
    assert [b.id.split(":")[0] for b in reply.buttons] == ["helpful", "unhelpful"]
    assert h.t(Msg.WHERE_TO_REPORT) not in reply.text


async def test_feedback_is_recorded_only_for_own_interactions(h: Harness) -> None:
    await h.onboard("en")
    await h.say(OTP_SCAM)
    interaction_id = (await interactions(h))[-1]["id"]
    assert await h.say(button=f"helpful:{interaction_id}") == [h.t(Msg.FEEDBACK_THANKS)]
    await h.onboard("en", user="+2348099999999")
    await h.say(button=f"unhelpful:{interaction_id}", user="+2348099999999")
    rows = await interactions(h)
    assert next(r for r in rows if r["id"] == interaction_id)["helpful"] is True


async def test_follow_up_buttons(h: Harness) -> None:
    await h.onboard("en")
    await h.say(OTP_SCAM)
    source_id = (await interactions(h))[-1]["id"]
    steps = await h.say(button=f"steps:{source_id}")
    assert steps[0].startswith(h.t(Msg.WHAT_TO_DO)) and "1. " in steps[0]
    report = await h.say(button=f"report:{source_id}")
    assert report[0].startswith(h.t(Msg.WHERE_TO_REPORT)) and "•" in report[0]
    warn = await h.say(button=f"warn:{source_id}")
    assert warn[0].startswith(h.t(Msg.WARN_FAMILY_INTRO)) and "Beware" in warn[0]
    rows = (await interactions(h))[-3:]
    assert [r["intent"] for r in rows] == ["button:steps", "button:report", "button:warn"]
    assert all(r["reply_to"] == source_id for r in rows)


async def test_follow_up_on_someone_elses_interaction_uses_generic_text(h: Harness) -> None:
    await h.onboard("en")
    await h.say(OTP_SCAM)
    source_id = (await interactions(h))[-1]["id"]
    other = "+2348099999999"
    await h.onboard("en", user=other)
    texts = await h.say(button=f"warn:{source_id}", user=other)
    assert h.t(Msg.WARN_FAMILY_FALLBACK) in texts[0]
    assert (await interactions(h))[-1]["reply_to"] is None


async def test_unknown_button_and_unsupported_media(h: Harness) -> None:
    await h.onboard("en")
    assert await h.say(button="bogus") == [h.t(Msg.GREETING_REPLY)]
    assert await h.say(kind=MessageKind.UNSUPPORTED) == [h.t(Msg.UNSUPPORTED)]


async def test_greeting_after_onboarding_is_fixed_and_logged(h: Harness) -> None:
    await h.onboard("en")
    assert await h.say("thank you!") == [h.t(Msg.GREETING_REPLY)]
    assert h.llm.requests == []
    assert (await interactions(h))[-1]["outcome"] == Outcome.FIXED_REPLY.value


# --- voice -----------------------------------------------------------------------------------------------


async def test_voice_note_listening_then_answer_with_echo(h: Harness) -> None:
    await h.onboard("ha")
    h.llm.script[Purpose.ANALYSE] = [analysis(OTP_SCAM, "scam_check")]
    h.llm.script[Purpose.COMPOSE] = ["Kada ka bayar da lambar."]
    texts = await h.say(audio=VOICE_NOTE.read_bytes())
    assert texts[0] == h.t(Msg.LISTENING, "ha")
    assert h.t(Msg.VERDICT_LIKELY_SCAM, "ha") in texts[1] and "“" in texts[1]  # "I heard: …"
    assert h.asr.languages == ["ha"]
    row = (await interactions(h))[-1]
    assert row["kind"] == "audio" and row["asr_ms"] == 50 and row["decode_ms"] is not None


@pytest.mark.parametrize(
    ("problem", "expected"),
    [(AudioProblem.TOO_LONG, Msg.AUDIO_TOO_LONG), (AudioProblem.NO_SPEECH, Msg.NOT_UNDERSTOOD)],
)
async def test_unusable_voice_notes(h: Harness, problem: AudioProblem, expected: Msg) -> None:
    await h.onboard("en")
    h.asr.error = AudioError(problem)
    texts = await h.say(audio=VOICE_NOTE.read_bytes())
    assert texts[1] == h.t(expected, seconds=90)
    assert (await interactions(h))[-1]["outcome"] == Outcome.AUDIO_ERROR.value


async def test_voice_note_with_no_words(h: Harness) -> None:
    await h.onboard("en")
    h.asr.text = " … "
    texts = await h.say(audio=VOICE_NOTE.read_bytes())
    assert texts[1] == h.t(Msg.NOT_UNDERSTOOD, seconds=90)


# --- limits and failures ---------------------------------------------------------------------------------


async def test_rate_limit(h: Harness) -> None:
    h.configure(per_user_per_hour=2)
    await h.onboard("en")
    await h.say(OTP_SCAM)
    await h.say(OTP_SCAM)
    assert await h.say(OTP_SCAM) == [h.t(Msg.RATE_LIMITED)]
    assert await h.say("STOP") == [h.t(Msg.STOPPED)]  # commands are never rate limited


async def test_licence_cap_turns_away_new_users_only(h: Harness) -> None:
    h.configure(active_user_cap=1)
    await h.onboard("en")
    await h.say(OTP_SCAM)
    assert await h.say("hello", user="+2348099999999") == [h.t(Msg.AT_CAPACITY)]
    assert (await h.say(OTP_SCAM))[0].startswith(h.t(Msg.VERDICT_LIKELY_SCAM))  # existing users unaffected


async def test_licence_cap_counts_dormant_and_unconsented_users_as_new(h: Harness) -> None:
    h.configure(active_user_cap=1)
    dormant = "+2348011111111"
    await h.onboard("en", user=dormant)  # consented long ago, no interaction in the window
    await h.say("hello", user="+2348022222222")  # welcomed only
    await h.onboard("en")
    await h.say(OTP_SCAM)  # the one active user
    assert await h.say(OTP_SCAM, user=dormant) == [h.t(Msg.AT_CAPACITY)]
    assert await h.say("thanks", user="+2348022222222") == [h.t(Msg.AT_CAPACITY)]


async def test_consent_text_change_asks_again(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    await h.onboard("ha")
    monkeypatch.setattr("kudiguard.pipeline.CONSENT_VERSION", "2099-01-01")
    texts = await h.say(OTP_SCAM)
    assert texts[0] == h.t(Msg.WELCOME, "ha") and h.llm.requests == []
    user = await h.store.get_user(h.user_hash)
    assert user is not None and user.consented_at is None and user.language == "ha"
    await h.say(button="lang:ha")
    user = await h.store.get_user(h.user_hash)
    assert user is not None and user.consent_version == "2099-01-01"


async def test_voice_note_before_choosing_a_language_gets_the_menu(h: Harness) -> None:
    await h.say("hello")
    await h.say(audio=VOICE_NOTE.read_bytes())
    [(_, menu)] = h.messenger.sent
    assert menu.options and h.asr.languages == []  # not transcribed with a guessed language model


async def test_asr_failure_gets_retry_message_and_is_logged(h: Harness) -> None:
    await h.onboard("en")
    h.asr.error = ASRError("model missing")
    texts = await h.say(audio=VOICE_NOTE.read_bytes())
    assert texts == [h.t(Msg.LISTENING), h.t(Msg.ERROR_RETRY)]
    row = (await interactions(h))[-1]
    assert row["outcome"] == Outcome.ERROR.value and "ASRError" in str(row["error"])


async def test_slow_model_still_delivers_the_rules_verdict_in_time(h: Harness) -> None:
    h.configure(turn_deadline_s=1.0)
    await h.onboard("en")
    h.llm.delay_s = 5.0  # every model call would blow the turn deadline
    texts = await h.say(OTP_SCAM)
    assert texts[0].startswith(h.t(Msg.VERDICT_LIKELY_SCAM)) and h.t(Msg.FALLBACK_SCAM) in texts[0]
    assert h.llm.requests == []  # skipped up front: not enough time left for a model step
    assert (await interactions(h))[-1]["outcome"] == Outcome.FALLBACK.value


async def test_hard_turn_deadline(h: Harness) -> None:
    h.configure(turn_deadline_s=0.2)
    await h.onboard("en")
    h.asr.delay_s = 5.0
    texts = await h.say(audio=VOICE_NOTE.read_bytes())
    assert texts[-1] == h.t(Msg.ERROR_RETRY)
    assert (await interactions(h))[-1]["outcome"] == Outcome.ERROR.value


async def test_failed_send_falls_back_and_receipt_failure_is_harmless(h: Harness) -> None:
    await h.onboard("en")
    h.messenger.fail_acknowledge = True
    h.messenger.fail_sends = 1
    texts = await h.say(OTP_SCAM)
    assert texts == [h.t(Msg.ERROR_RETRY)]


async def test_partial_send_failure_does_not_add_an_error_reply(h: Harness) -> None:
    await h.onboard("en")
    h.messenger.fail_on = {h.t(Msg.FEEDBACK_PROMPT)}  # the answer arrives, the feedback prompt fails
    texts = await h.say(OTP_SCAM)
    assert len(texts) == 1 and texts[0].startswith(h.t(Msg.VERDICT_LIKELY_SCAM))
    rows = await interactions(h)
    assert [r["outcome"] for r in rows] == [Outcome.ANSWERED.value]  # one evidence row, no error row


async def test_hung_channel_cannot_wedge_a_user(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kudiguard.pipeline.FAILURE_DEADLINE_S", 0.05)
    h.configure(turn_deadline_s=0.1)
    await h.onboard("en")
    h.messenger.hang = True
    await asyncio.wait_for(h.say(OTP_SCAM), timeout=2)  # returns although every send hangs
    h.messenger.hang = False
    assert (await h.say("STOP")) == [h.t(Msg.STOPPED)]  # the user's next message is handled


async def test_store_failure_on_accept_still_replies(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(channel: str, message_id: str) -> bool:
        raise RuntimeError("database locked")

    monkeypatch.setattr(h.store, "claim_message", broken)
    assert await h.say(OTP_SCAM) == [h.t(Msg.ERROR_RETRY)]


async def test_overload_gets_busy_reply(h: Harness) -> None:
    h.configure(max_concurrent_turns=1, max_waiting_turns=1)
    users = [f"+23480300000{i:02d}" for i in range(3)]
    for user in users:
        await h.onboard("en", user=user)
    h.llm.delay_s = 0.05
    h.messenger.clear()
    await asyncio.gather(*(h.conversation.handle(h.msg(OTP_SCAM, user=u)) for u in users))
    assert h.messenger.texts.count(h.t(Msg.BUSY)) == 1
    assert len(h.messenger.acknowledged) == 3  # everyone was acknowledged, even the one turned away


async def test_turns_of_one_user_are_serialised(h: Harness) -> None:
    await h.onboard("en")
    h.llm.delay_s = 0.05
    tracker = h.llm.track_concurrency()
    await asyncio.gather(*(h.conversation.handle(h.msg(OTP_SCAM)) for _ in range(3)))
    assert tracker.peak == 1
    assert len(await interactions(h)) == 2  # the 2nd and 3rd queued behind the 1st and were merged


async def test_different_users_run_concurrently_up_to_the_limit(h: Harness) -> None:
    h.configure(max_concurrent_turns=2)
    users = [f"+23480300000{i:02d}" for i in range(4)]
    for user in users:
        await h.onboard("en", user=user)
    h.llm.delay_s = 0.02
    tracker = h.llm.track_concurrency()
    await asyncio.gather(*(h.conversation.handle(h.msg(OTP_SCAM, user=u)) for u in users))
    assert tracker.peak == 2


async def test_submit_and_drain(h: Harness) -> None:
    await h.onboard("en")
    assert h.conversation.submit(h.msg(OTP_SCAM))
    await h.conversation.drain()
    assert h.messenger.texts
    assert not h.conversation.submit(h.msg(OTP_SCAM))  # shutting down: the channel answers "retry later"


async def test_stored_text_is_redacted_and_ids_hashed(h: Harness) -> None:
    await h.onboard("en")
    await h.say("They want my BVN 22212345678 and asked me to call 08031234567", message_id="wamid.HBgNMjM0")
    row = (await interactions(h))[-1]
    assert "22212345678" not in str(row) and "08031234567" not in str(row) and "HBgN" not in str(row)


# --- pure helpers ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("steps:12", (Action.STEPS, "12")),
        ("lang:ha", (Action.LANGUAGE, "ha")),
        ("steps:", None),
        ("nope:1", None),
        ("", None),
        (None, None),
        ("warn:" + "9" * 30, None),
    ],
)
def test_parse_button(raw: str | None, expected: tuple[Action, str] | None) -> None:
    assert parse_button(raw) == expected


def test_limits_validation() -> None:
    with pytest.raises(ValueError):
        Limits(per_user_per_hour=0)


def test_unknown_default_language(registry: LanguageRegistry) -> None:
    engine = Engine(registry=registry, llm=FakeLLM(), knowledge=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default language"):
        Conversation(
            engine=engine, store=MemoryStore(UserHasher(b"k" * 32)), messenger=FakeMessenger(),
            hasher=UserHasher(b"k" * 32), limits=Limits(default_language="fr"),
        )  # fmt: skip


# --- multi-message turns ---------------------------------------------------------------------------------


async def test_messages_sent_while_busy_are_merged_into_one_check(h: Harness) -> None:
    await h.onboard("en")
    release = asyncio.Event()
    first_started = asyncio.Event()
    original = h.llm.complete

    async def gated(request):  # type: ignore[no-untyped-def]
        if not first_started.is_set():
            first_started.set()
            await release.wait()
        return await original(request)

    h.llm.complete = gated  # type: ignore[method-assign]
    first = asyncio.create_task(h.conversation.handle(h.msg("Someone called me from my bank")))
    await first_started.wait()
    rest = [
        asyncio.create_task(h.conversation.handle(h.msg(text)))
        for text in ("he said my account is blocked", "and he wants the OTP they sent me")
    ]
    await asyncio.sleep(0.01)  # both claimed and queued behind the first turn
    release.set()
    await asyncio.gather(first, *rest)
    analyses = [r.messages[-1]["content"] for r in h.llm.calls(Purpose.ANALYSE)]
    assert len(analyses) == 2  # first message alone, then the two follow-ons together
    assert "account is blocked" in analyses[1] and "wants the OTP" in analyses[1]
    assert len(await interactions(h)) == 2
    assert h.conversation._pending == {}


async def test_commands_are_never_merged_and_keep_their_order(h: Harness) -> None:
    await h.onboard("en")
    release = asyncio.Event()
    original = h.llm.complete
    calls = 0

    async def gated(request):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            await release.wait()
        return await original(request)

    h.llm.complete = gated  # type: ignore[method-assign]
    tasks = [asyncio.create_task(h.conversation.handle(h.msg(OTP_SCAM)))]
    await asyncio.sleep(0.01)
    tasks += [asyncio.create_task(h.conversation.handle(h.msg(t))) for t in ("is it real?", "STOP", "hello?")]
    await asyncio.sleep(0.01)
    release.set()
    await asyncio.gather(*tasks)
    texts = h.messenger.texts
    stop_at = texts.index(h.t(Msg.STOPPED))
    assert any(t.startswith(h.t(Msg.VERDICT_LIKELY_SCAM)) for t in texts[:stop_at])
    assert texts[stop_at + 1] == h.t(Msg.WELCOME)  # message after STOP restarts onboarding


async def test_queued_voice_notes_are_transcribed_and_merged(h: Harness) -> None:
    await h.onboard("en")
    release = asyncio.Event()
    original = h.llm.complete
    calls = 0

    async def gated(request):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            await release.wait()
        return await original(request)

    h.llm.complete = gated  # type: ignore[method-assign]
    first = asyncio.create_task(h.conversation.handle(h.msg("Someone called me")))
    await asyncio.sleep(0.01)
    notes = [
        asyncio.create_task(h.conversation.handle(h.msg(audio=VOICE_NOTE.read_bytes()))) for _ in range(2)
    ]
    await asyncio.sleep(0.01)
    release.set()
    await asyncio.gather(first, *notes)
    last = (await interactions(h))[-1]
    assert last["kind"] == "audio" and int(str(last["asr_ms"])) == 100  # two notes, 50 ms each
    assert len(h.asr.languages) == 2


async def test_unusable_merged_voice_note_is_skipped(h: Harness) -> None:
    await h.onboard("en")
    turn_text = await h.conversation._content_of(  # direct: a bad note among several must not fail the turn
        _turn_for(h), h.msg(audio=b"not audio at all")
    )
    assert turn_text == ""


def _turn_for(h: Harness):  # type: ignore[no-untyped-def]
    from kudiguard.pipeline import _Turn

    return _Turn(h.msg("x"), h.user_hash, "en", 0.0, deadline=float("inf"))


# --- review round 2: merge limits, delivery, bounded failure paths -----------------------------------------


async def _queue_behind_a_slow_turn(h: Harness, first: InboundMessage, rest: list[InboundMessage]) -> None:
    """Run ``first``; while its model call is blocked, deliver ``rest`` (they queue for the user's lock)."""
    release = asyncio.Event()
    original = h.llm.complete
    calls = 0

    async def gated(request):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            await release.wait()
        return await original(request)

    h.llm.complete = gated  # type: ignore[method-assign]
    tasks = [asyncio.create_task(h.conversation.handle(first))]
    await asyncio.sleep(0.01)
    tasks += [asyncio.create_task(h.conversation.handle(m)) for m in rest]
    await asyncio.sleep(0.01)
    release.set()
    await asyncio.gather(*tasks)


async def test_merge_is_capped(h: Harness) -> None:
    await h.onboard("en")
    rest = [h.msg(f"part {i}") for i in range(MAX_COALESCED + 2)]
    await _queue_behind_a_slow_turn(h, h.msg(OTP_SCAM), rest)
    merged = [r.messages[-1]["content"] for r in h.llm.calls(Purpose.ANALYSE)][1]
    assert merged.count("part ") == MAX_COALESCED  # one turn takes at most MAX_COALESCED messages
    assert len(await interactions(h)) == 3  # first, a full merged batch, the remainder


async def test_button_between_texts_keeps_its_place(h: Harness) -> None:
    await h.onboard("en")
    await h.say(OTP_SCAM)
    source_id = (await interactions(h))[-1]["id"]
    h.messenger.clear()
    rest = [h.msg("is it real"), h.msg(button=f"report:{source_id}"), h.msg("and the link?")]
    await _queue_behind_a_slow_turn(h, h.msg("someone called me"), rest)
    intents = [r["intent"] for r in (await interactions(h))[1:]]
    assert "button:report" in intents
    report_at = next(i for i, t in enumerate(h.messenger.texts) if t.startswith(h.t(Msg.WHERE_TO_REPORT)))
    assert 0 < report_at < len(h.messenger.texts) - 1  # after the first answer, before the last


async def test_queued_voice_note_absorbs_the_following_text_and_all_are_acknowledged(h: Harness) -> None:
    """The voice note heads its own turn (queued behind the first) and absorbs the text queued after it."""
    await h.onboard("en")
    h.messenger.clear()
    note = VOICE_NOTE.read_bytes()
    rest = [h.msg(audio=note), h.msg("he wants the OTP")]
    await _queue_behind_a_slow_turn(h, h.msg("someone called me"), rest)
    assert len(h.messenger.acknowledged) == 3
    assert h.t(Msg.LISTENING) in h.messenger.texts
    last = (await interactions(h))[-1]
    assert last["kind"] == "audio" and "I heard" not in str(last["transcript"])


async def test_undelivered_answer_is_marked_on_its_row(h: Harness) -> None:
    await h.onboard("en")
    h.llm.script[Purpose.ANALYSE] = [analysis("x", "question")]
    h.messenger.fail_sends = 1  # the answer itself cannot be delivered
    texts = await h.say("How do I open a savings account?")
    assert texts == [h.t(Msg.ERROR_RETRY)]
    rows = await interactions(h)
    assert (
        len(rows) == 1
        and rows[0]["outcome"] == Outcome.ERROR.value
        and "undelivered" in str(rows[0]["error"])
    )


async def test_busy_path_survives_a_hung_store(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kudiguard.pipeline.FAILURE_DEADLINE_S", 0.05)
    h.configure(max_concurrent_turns=1, max_waiting_turns=0)
    users = [f"+23480300000{i:02d}" for i in range(2)]
    for user in users:
        await h.onboard("en", user=user)
    h.llm.delay_s = 0.05
    original = h.store.get_user
    calls = 0

    async def slow_second(user_hash: str):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:  # the BUSY path's language lookup hangs
            await asyncio.Event().wait()
        return await original(user_hash)

    monkeypatch.setattr(h.store, "get_user", slow_second)
    h.messenger.clear()
    await asyncio.wait_for(
        asyncio.gather(*(h.conversation.handle(h.msg(OTP_SCAM, user=u)) for u in users)), timeout=2
    )
    assert h.t(Msg.BUSY) in h.messenger.texts


async def test_transcriptions_are_bounded_across_turns(h: Harness) -> None:
    h.configure(max_concurrent_transcriptions=1)
    users = [f"+23480300000{i:02d}" for i in range(3)]
    for user in users:
        await h.onboard("en", user=user)
    h.asr.delay_s = 0.02
    active = peak = 0
    original = h.asr.transcribe

    async def tracking(audio, language):  # type: ignore[no-untyped-def]
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            return await original(audio, language)
        finally:
            active -= 1

    h.asr.transcribe = tracking  # type: ignore[method-assign]
    note = VOICE_NOTE.read_bytes()
    await asyncio.gather(*(h.conversation.handle(h.msg(audio=note, user=u)) for u in users))
    assert peak == 1


async def test_drain_cancels_stuck_turns_cleanly(h: Harness) -> None:
    await h.onboard("en")
    h.llm.delay_s = 10.0
    assert h.conversation.submit(h.msg(OTP_SCAM))
    await asyncio.sleep(0.01)
    await asyncio.wait_for(h.conversation.drain(timeout_s=0.05), timeout=2)
    assert h.conversation._pending == {}


async def test_fault_in_a_queued_voice_note_is_logged_by_its_real_name(h: Harness) -> None:
    await h.onboard("en")
    h.asr.error = ASRError("model crashed")
    await _queue_behind_a_slow_turn(h, h.msg("someone called me"), [h.msg(audio=VOICE_NOTE.read_bytes())])
    row = (await interactions(h))[-1]
    assert row["outcome"] == Outcome.ERROR.value and str(row["error"]).startswith("ASRError")


# --- coverage gaps: failure-of-failure paths, edge inputs ---------------------------------------------------


async def test_reply_failure_while_store_is_down_is_swallowed(
    h: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(channel: str, message_id: str) -> bool:
        raise RuntimeError("database locked")

    monkeypatch.setattr(h.store, "claim_message", broken)
    h.messenger.fail_sends = 1
    assert await h.say(OTP_SCAM) == []  # handle() never raises, even when both store and channel fail
    assert "could not send" in caplog.text


async def test_unmarkable_undelivered_answer_does_not_crash_or_double_reply(
    h: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    await h.onboard("en")
    h.llm.script[Purpose.ANALYSE] = [analysis("x", "question")]
    h.messenger.fail_sends = 1  # the answer cannot be delivered...

    async def broken(*args: object) -> None:
        raise RuntimeError("database locked")

    monkeypatch.setattr(h.store, "set_outcome", broken)  # ...nor marked as undelivered
    assert await h.say("How do I open a savings account?") == [h.t(Msg.ERROR_RETRY)]
    assert "could not mark the answer undelivered" in caplog.text
    rows = await interactions(h)
    assert len(rows) == 1 and rows[0]["outcome"] != Outcome.ERROR.value  # the row keeps its original outcome


async def test_failure_to_log_the_failure_still_tells_the_user(
    h: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    await h.onboard("en")

    async def broken(*args: object) -> int:
        raise RuntimeError("disk full")

    monkeypatch.setattr(h.store, "add_interaction", broken)
    assert await h.say("hello") == [h.t(Msg.ERROR_RETRY)]  # the fixed reply is logged first, so never sent
    assert "could not log the failure" in caplog.text


async def test_stop_from_an_unknown_user_creates_no_record(h: Harness) -> None:
    assert await h.say("STOP") == [h.t(Msg.STOPPED)]
    assert await h.store.get_user(h.user_hash) is None  # nothing stored about someone we never onboarded


async def test_text_turn_with_merged_voice_note_says_listening(h: Harness) -> None:
    await h.onboard("en")
    h.messenger.clear()
    # The 2nd message (text) starts its own turn and absorbs the 3rd (voice): the turn was not voice until then.
    rest = [h.msg("he said my account is blocked"), h.msg(audio=VOICE_NOTE.read_bytes())]
    await _queue_behind_a_slow_turn(h, h.msg("someone called me"), rest)
    analyses = [r.messages[-1]["content"] for r in h.llm.calls(Purpose.ANALYSE)]
    assert len(analyses) == 2 and "account is blocked" in analyses[1] and "OTP" in analyses[1]
    assert h.messenger.texts.count(h.t(Msg.LISTENING)) == 1
    last = (await interactions(h))[-1]
    assert last["kind"] == "audio" and last["asr_ms"] is not None


async def test_system_fault_in_merged_voice_note_surfaces_unwrapped(h: Harness) -> None:
    await h.onboard("en")
    h.asr.error = ASRError("model crashed")
    rest = [h.msg("he said my account is blocked"), h.msg(audio=VOICE_NOTE.read_bytes())]
    await _queue_behind_a_slow_turn(h, h.msg("someone called me"), rest)
    row = (await interactions(h))[-1]
    assert row["outcome"] == Outcome.ERROR.value
    assert str(row["error"]).startswith("ASRError: model crashed")  # not "ExceptionGroup: ..."
    assert h.messenger.texts[-1] == h.t(Msg.ERROR_RETRY)


async def test_voice_message_without_audio_or_media_reference(h: Harness) -> None:
    await h.onboard("en")
    texts = await h.say(kind=MessageKind.AUDIO)
    assert texts == [h.t(Msg.LISTENING), h.t(Msg.NOT_UNDERSTOOD, seconds=90)]
    row = (await interactions(h))[-1]
    assert row["outcome"] == Outcome.AUDIO_ERROR.value and row["error"] == AudioProblem.EMPTY.value
    assert h.asr.languages == []  # nothing reached speech recognition


async def test_button_with_non_numeric_reference_gets_generic_follow_up(h: Harness) -> None:
    await h.onboard("en")
    texts = await h.say(button="warn:abc")
    assert h.t(Msg.WARN_FAMILY_FALLBACK) in texts[0]
    row = (await interactions(h))[-1]
    assert row["intent"] == "button:warn" and row["reply_to"] is None


async def test_no_licence_cap_skips_the_active_user_count(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import timedelta

    from kudiguard.pipeline import utcnow

    h.configure(active_user_cap=None)
    windows: list[object] = []
    original = h.store.count_interactions_since

    async def spy(user_hash: str, since):  # type: ignore[no-untyped-def]
        windows.append(since)
        return await original(user_hash, since)

    monkeypatch.setattr(h.store, "count_interactions_since", spy)
    await h.onboard("en")
    texts = await h.say(OTP_SCAM)
    assert texts[0].startswith(h.t(Msg.VERDICT_LIKELY_SCAM))
    assert windows  # the hourly rate limit still counts...
    assert all(
        utcnow() - w < timedelta(days=1) for w in windows
    )  # ...but the 30-day licence window never does
