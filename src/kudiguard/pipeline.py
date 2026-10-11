"""Conversation pipeline: one inbound message → consent, commands, rate limits, the engine, replies, evidence log.

Channel-agnostic: channels (WhatsApp, web) turn provider payloads into ``InboundMessage`` and implement ``Messenger``.

Production behaviour:
* At-most-once per provider message id (``Store.claim_message``); webhooks are acknowledged before processing
  (``submit``), and background turns are tracked so shutdown can drain them.
* Per-user ordering: one turn at a time per user (a voice note and a quick follow-up text cannot race on state).
* Bounded concurrency with back-pressure, and a hard deadline per turn: the user always gets *some* reply
  (``error_retry`` in their language), never silence.
* No raw phone numbers or provider ids reach storage (``privacy.UserHasher``); free text is redacted by the store.
* The engine is stateless and the Store is the only shared state. Per-user ordering, message merging and the
  concurrency limits live in this process: to run several instances, route each user to one instance (sticky by
  user hash) or move ordering to a shared queue; user-state saves are last-writer-wins.
* Per-turn store cost: claim, get_user, the licence-cap count (one indexed per-user count; a global count only for
  users new to the 30-day window), the hourly rate-limit count, then the evidence row and one user save.
* Voice-note donation (``kudiguard.donation``, opt-in, off unless ``accept_donations``): offered once per offer text,
  after an answered voice note that was not a scam warning; after the turn has released its slot, a background task
  (tracked, so shutdown waits for it) keeps a donor's voice notes or makes the offer. Never on the reply path.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from kudiguard.admission import Admission, AtCapacity, Reservation
from kudiguard.commands import Command, CommandMatcher, Match
from kudiguard.domain import InboundMessage, Interaction, LanguageCode, MessageKind, Outcome
from kudiguard.donation import DONATION_VERSION, Donation, VoiceVault
from kudiguard.engine import Answer, AnswerKind, Engine
from kudiguard.hosting import StorageLocation
from kudiguard.i18n import Messages, Msg
from kudiguard.languages import LanguageStatus
from kudiguard.patterns import PATTERN_CHANNELS, PATTERN_RETENTION_DAYS, ScamPattern
from kudiguard.privacy import UserHasher
from kudiguard.render import REPORT_VERDICTS, render_check_reply, render_report, render_steps, render_warning
from kudiguard.speech.audio import MAX_SECONDS
from kudiguard.speech.base import AudioError, AudioProblem
from kudiguard.speech.service import SpeechResult
from kudiguard.storage.base import Store, UserState, utcnow
from kudiguard.text import redact

log = logging.getLogger(__name__)

CONSENT_VERSION = "2026-10-11"  # bump when the welcome/consent text changes materially (also in voice.html)
REPLY_RESERVE_S = 3.0  # of the turn deadline, kept for rendering, logging and sending the reply
FAILURE_DEADLINE_S = (
    10.0  # for the error reply/log after a failed turn (a hung channel must not wedge a user)
)
ACK_DEADLINE_S = 3.0
DONATION_DEADLINE_S = 60.0  # the background donation step (write or offer) after a turn
# A message counts as consent only if sent at least this long after the welcome was sent (accepted by WhatsApp):
# nobody reads the notice faster, and it protects against this server's clock running up to this much *behind*
# the provider's (a clock ahead only delays when a reply counts). Keep the clock synced (docs/DEPLOY.md).
WELCOME_READ_S = 5.0
MAX_COALESCED = 5  # quick successive notes/texts merged into one check (people split one story over messages)


# --- outbound messages and the channel port ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Button:
    id: str
    title: str


@dataclass(frozen=True, slots=True)
class MenuOption:
    id: str
    title: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class Outbound:
    """A reply: text, optionally with up to 3 reply buttons or a single-choice menu (WhatsApp list)."""

    text: str
    buttons: tuple[Button, ...] = ()
    menu_button: str | None = None
    options: tuple[MenuOption, ...] = ()


class Messenger(Protocol):
    async def acknowledge(self, message: InboundMessage) -> None:
        """Best-effort "seen"/typing signal for ``message`` (may be a no-op)."""
        ...

    async def send(self, user_id: str, message: Outbound) -> None:
        """Deliver ``message`` to the channel user. Raises on delivery failure."""
        ...

    async def fetch_media(self, message: InboundMessage) -> bytes:
        """Download ``message.media_id``. Raises ``AudioError`` (e.g. TOO_LARGE) for user-facing problems."""
        ...


class Action(StrEnum):
    """Reply-button and menu actions; ids are ``<action>:<ref>`` (ref = interaction id or language code)."""

    HELPFUL = "helpful"
    NOT_HELPFUL = "unhelpful"
    STEPS = "steps"
    REPORT = "report"
    WARN = "warn"
    LANGUAGE = "lang"
    DONATE = "donate"  # ref: DONATE_YES (bound to the offer's text version) or DONATE_NO


# A "Yes" under an older offer text asks again: consent is to the text that was shown.
DONATE_YES = f"yes-{DONATION_VERSION}"
DONATE_NO = "no"


def button_id(action: Action, ref: int | str) -> str:
    return f"{action.value}:{ref}"


def parse_button(raw: str | None) -> tuple[Action, str] | None:
    action, _, ref = (raw or "").partition(":")
    if action not in Action._value2member_map_ or not ref or len(ref) > 20:
        return None
    return Action(action), ref


# --- configuration --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Limits:
    """Operational limits. Note: model steps need a few seconds each (``engine.MIN_*_S``); with a turn deadline under
    ~5 s the engine never calls the model and every check is answered by the rules with fixed text."""

    per_user_per_hour: int = 20
    active_user_cap: int | None = (
        1000  # N-ATLAS licence: active end-users per rolling 30 days (None = no cap)
    )
    max_concurrent_turns: int = 8  # back-pressure: further turns wait for a slot
    max_waiting_turns: int = 200  # beyond this, new turns get a "busy, try again" reply (bounded memory)
    turn_deadline_s: float = 60.0  # hard limit per turn, including model queueing
    default_language: LanguageCode = "en"
    # Shown in the welcome (consent) text: the configured retention periods (Settings), so the text never drifts.
    retention_days: int = 90
    pattern_retention_days: int = PATTERN_RETENTION_DAYS

    def __post_init__(self) -> None:
        if (
            self.per_user_per_hour < 1
            or self.max_concurrent_turns < 1
            or self.max_waiting_turns < 0
            or self.turn_deadline_s <= 0
        ):
            raise ValueError("limits must be positive")


DEFAULT_LIMITS = Limits()


@dataclass(slots=True)
class _Turn:
    """Per-turn facts gathered for the evidence log."""

    message: InboundMessage
    user_hash: str
    language: LanguageCode
    started: float
    deadline: float  # time.monotonic() by which model work must end (turn deadline minus the reply reserve)
    transcript: str | None = None
    decode_ms: int | None = None
    asr_ms: int | None = None
    voice: bool = False  # any voice note in this turn (the reply echoes "I heard: …")
    replied: bool = False  # the user has received this turn's answer (or a fixed reply)
    logged: bool = False  # an evidence row exists for this turn
    logged_at: datetime | None = None  # its timestamp (scam examples are dated from it)
    interaction_id: int | None = None  # the turn's evidence row
    calm_answer: bool = False  # answered, and not a scam warning: a fair moment to offer voice donation
    received_at: datetime | None = None  # when the turn started (donation consent must stand at this time)
    # Own (not forwarded) voice notes of this turn with their transcripts, for donation after the reply.
    voice_notes: list[tuple[bytes, SpeechResult]] = dataclasses.field(default_factory=list)
    # A new user's licence-cap slot, held until this turn's interaction is logged (or the turn ends).
    reservation: Reservation = dataclasses.field(default_factory=Reservation)


class _KeyedLocks:
    """One lock per user, dropped when nobody holds or waits for it (bounded memory)."""

    def __init__(self) -> None:
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    async def run(self, key: str, work: Callable[[], Coroutine[Any, Any, None]]) -> None:
        lock, users = self._locks.get(key, (asyncio.Lock(), 0))
        self._locks[key] = (lock, users + 1)
        try:
            async with lock:
                await work()
        finally:
            lock, users = self._locks[key]
            if users == 1:
                del self._locks[key]
            else:
                self._locks[key] = (lock, users - 1)


# --- the pipeline ---------------------------------------------------------------------------------------


class Conversation:
    def __init__(
        self,
        *,
        engine: Engine,
        store: Store,
        messenger: Messenger,
        hasher: UserHasher,
        limits: Limits = DEFAULT_LIMITS,
        clock: Callable[[], datetime] = utcnow,
        privacy_url: str = "",
        donations: VoiceVault | None = None,
        accept_donations: bool = False,
        storage: StorageLocation = StorageLocation.OUTSIDE_NIGERIA,
        admission: Admission | None = None,
    ) -> None:
        """``admission``: the licence-cap gate shared with the website and the partner API (``Services``); without
        one, a private gate for ``limits.active_user_cap`` (tests, single-channel hosts)."""
        self._engine = engine
        self._registry = engine.registry
        self._messages: Messages = engine.messages
        self._store = store
        self._messenger = messenger
        self._hasher = hasher
        self._limits = limits
        self._clock = clock
        self._privacy_url = privacy_url
        self._donations = donations  # also used to erase when collection is off
        self._collecting = donations is not None and accept_donations  # KUDIGUARD_VOICE_DONATION
        self._storage = storage
        self._admission = admission or Admission(store, limits.active_user_cap, clock=clock)
        self._matcher = CommandMatcher(self._registry)
        self._slots = asyncio.Semaphore(limits.max_concurrent_turns)
        self._waiting = 0  # turns queued for a slot
        self._user_locks = _KeyedLocks()
        self._tasks: set[asyncio.Task[None]] = set()
        self._pending: dict[str, list[InboundMessage]] = {}  # claimed, waiting for the user's lock (in order)
        self._closing = False
        if limits.default_language not in self._registry:
            raise ValueError(f"default language {limits.default_language!r} is not configured")

    # --- entry points -----------------------------------------------------------------------------------

    def submit(self, message: InboundMessage) -> bool:
        """Process in the background (webhooks must answer within seconds). Must be called on the event loop.

        Returns False while shutting down: the channel should then answer with a retryable error (e.g. HTTP 503)
        so the provider redelivers the message to a running instance.
        """
        if self._closing:
            return False
        task = asyncio.create_task(self.handle(message), name=f"turn:{message.channel}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def wait_idle(self) -> None:
        """Wait until every background turn submitted so far has finished (tests, deploy health checks)."""
        while self._tasks:
            await asyncio.gather(*set(self._tasks), return_exceptions=True)

    async def drain(self, timeout_s: float = 30.0) -> None:
        """Graceful shutdown: refuse new turns, wait for running ones, cancel what is left after ``timeout_s``."""
        self._closing = True
        deadline = time.monotonic() + timeout_s
        while self._tasks and (left := deadline - time.monotonic()) > 0:  # turns may start donation tasks
            await asyncio.wait(set(self._tasks), timeout=left)
        pending = set(self._tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def handle(self, message: InboundMessage) -> None:
        """Process one inbound message to completion. Never raises (except cancellation)."""
        try:
            async with asyncio.timeout(FAILURE_DEADLINE_S):
                claimed = await self._store.claim_message(message.channel, message.message_id)
            if not claimed:
                return  # provider retry or duplicate delivery
            user_hash = self._hasher.user_id(message.user_id)
        except Exception:
            log.exception("could not accept message")
            await self._best_effort_reply(message.user_id, self._limits.default_language)
            return
        self._pending.setdefault(user_hash, []).append(message)
        try:
            await self._user_locks.run(user_hash, lambda: self._next(message, user_hash))
        finally:  # also on cancellation: never leave a message (or an empty queue) behind
            queue = self._pending.get(user_hash, [])
            queue[:] = [m for m in queue if m is not message]
            if not queue:
                self._pending.pop(user_hash, None)

    async def _next(self, message: InboundMessage, user_hash: str) -> None:
        queue = self._pending.get(user_hash, [])
        if not any(m is message for m in queue):
            return  # already merged into an earlier turn of this user
        queue[:] = [m for m in queue if m is not message]
        await self._guarded(message, user_hash)

    def _take_follow_ons(self, turn: _Turn) -> list[InboundMessage]:
        """Content messages that queued up behind this turn (sent while we were busy): merged into this check.
        Stops at the first command, language choice or button so those keep their order."""
        queue = self._pending.get(turn.user_hash, [])
        taken: list[InboundMessage] = []
        for queued in queue:
            if len(taken) >= MAX_COALESCED - 1:
                break
            is_text = queued.kind is MessageKind.TEXT
            match = self._matcher.match(queued.text) if is_text else Match()
            content = queued.kind is MessageKind.AUDIO or (
                is_text and match.command is None and not match.language
            )
            if not content:
                break
            taken.append(queued)
        queue[:] = queue[len(taken) :]
        return taken

    async def _guarded(self, message: InboundMessage, user_hash: str) -> None:
        now = time.monotonic()
        turn = _Turn(
            message, user_hash, self._limits.default_language, time.perf_counter(),
            deadline=now + max(self._limits.turn_deadline_s - REPLY_RESERVE_S, self._limits.turn_deadline_s / 2),
        )  # fmt: skip
        await self._acknowledge(message)  # before queueing for a slot: the user sees "read" within a second
        if self._slots.locked() and self._waiting >= self._limits.max_waiting_turns:
            log.warning("overloaded: %d turns waiting", self._waiting)
            await self._best_effort_reply(message.user_id, await self._language_of(user_hash), Msg.BUSY)
            return
        try:
            async with asyncio.timeout(self._limits.turn_deadline_s):
                self._waiting += 1
                try:
                    await self._slots.acquire()
                finally:
                    self._waiting -= 1
                try:
                    await self._turn(turn)
                finally:
                    self._slots.release()
        except Exception as exc:
            log.exception("turn failed (%s)", type(exc).__name__)
            await self._fail(turn, exc)
        finally:
            turn.reservation.release()  # normally already released by _log

    async def _fail(self, turn: _Turn, exc: Exception) -> None:
        """Never silence, never two answers: an error reply only if no answer was delivered; one log row."""
        delivered = turn.replied
        try:
            async with asyncio.timeout(FAILURE_DEADLINE_S):
                if not delivered:
                    await self._send(turn, self._messages.text(turn.language, Msg.ERROR_RETRY))
        except Exception:
            log.exception("could not tell the user about the failure")
        if turn.logged:
            if turn.interaction_id is not None and not delivered:  # logged as answered, never delivered
                try:
                    async with asyncio.timeout(FAILURE_DEADLINE_S):
                        await self._store.set_outcome(
                            turn.interaction_id, Outcome.ERROR, f"undelivered: {type(exc).__name__}"
                        )
                except Exception:
                    log.exception("could not mark the answer undelivered")
            return
        try:
            async with asyncio.timeout(FAILURE_DEADLINE_S):
                await self._log(turn, outcome=Outcome.ERROR, error=f"{type(exc).__name__}: {exc}"[:300])
        except Exception:
            log.exception("could not log the failure")

    async def _best_effort_reply(
        self, user_id: str, language: LanguageCode, key: Msg = Msg.ERROR_RETRY
    ) -> None:
        try:
            async with asyncio.timeout(FAILURE_DEADLINE_S):
                await self._messenger.send(user_id, Outbound(self._messages.text(language, key)))
        except Exception:
            log.exception("could not send %s", key)

    async def _language_of(self, user_hash: str) -> LanguageCode:
        try:
            async with asyncio.timeout(FAILURE_DEADLINE_S):
                user = await self._store.get_user(user_hash)
        except Exception:
            return self._limits.default_language
        return user.language if user is not None and user.language else self._limits.default_language

    # --- one turn ---------------------------------------------------------------------------------------

    async def _turn(self, turn: _Turn) -> None:
        turn.received_at = (
            utcnow()
        )  # real time, compared with withdrawal tombstones (not the injectable clock)
        message = turn.message
        match = self._matcher.match(message.text) if message.kind is MessageKind.TEXT else Match()
        user = await self._store.get_user(turn.user_hash)
        if user is not None and user.language:
            turn.language = user.language

        if match.command is Command.STOP:  # also ends voice donation and deletes what was donated
            removed = await self._erase_donations(turn.user_hash)
            donor = removed > 0 or (user is not None and user.donation_version is not None)
            if user is not None:
                now = self._clock()
                await self._store.save_user(
                    dataclasses.replace(
                        user, stopped=True, last_seen_at=now, donation_version=None, donation_at=None,
                        donation_source=None, donation_withdrawn_at=now if donor else user.donation_withdrawn_at,
                    )
                )  # fmt: skip
            await self._send(
                turn, self._messages.text(turn.language, Msg.STOPPED_DONOR if donor else Msg.STOPPED)
            )
            return
        if match.command is Command.DELETE:
            if self._donations is not None:
                await self._donations.erase(turn.user_hash)  # files first: delete_user drops their rows
            await self._store.delete_user(turn.user_hash)
            await self._send(turn, self._messages.text(turn.language, Msg.DELETED))
            return
        if match.command is Command.STOP_DONATING:  # withdrawal works for everyone, at any time
            await self._withdraw_donation(turn, user)
            return
        try:
            turn.reservation = await self._admission.reserve(turn.user_hash, message.channel)
        except AtCapacity:
            await self._send(turn, self._messages.text(turn.language, Msg.AT_CAPACITY))
            return

        chosen = match.language or self._chosen_language(message)
        if user is None or user.stopped or not _consented(user):
            user = await self._onboard(turn, user, match, chosen)
            if user is None:
                return
        elif chosen:
            await self._set_language(turn, user, chosen)
            return
        await self._dispatch(turn, dataclasses.replace(user, last_seen_at=self._clock()), match)

    async def _onboard(
        self, turn: _Turn, user: UserState | None, match: Match, chosen: LanguageCode | None
    ) -> UserState | None:
        """First contact, after STOP, or after a consent-text change: welcome + language menu.

        Returns the user if this message should be processed now: a text message after the welcome counts as
        consent ("by continuing, you agree"). A voice note before any language was chosen gets the menu again,
        because transcribing it with a guessed language model would mostly produce garbage.

        Consent needs the *current* welcome to have been shown (``welcome_version``): a user who exists without it
        (e.g. a tester row created by the sign-up form import, or a language button from an old menu) gets the
        welcome first, whatever they sent.
        """
        now = self._clock()
        # Shown = this text's welcome was sent (accepted by WhatsApp) at a known time; a row without the time is
        # welcomed again.
        welcomed = (
            user is not None and user.welcome_version == CONSENT_VERSION and user.welcome_at is not None
        )
        if (
            welcomed
            and (chosen or turn.message.kind is MessageKind.TEXT)
            and _before_welcome(turn.message, user)
        ):
            await self._send(
                turn, self._messages.text(turn.language, Msg.READ_WELCOME)
            )  # not consent: not seen yet
            return None
        if chosen and welcomed:
            assert user is not None
            await self._set_language(turn, user, chosen)
            return None
        pending = welcomed and user is not None and not user.stopped and user.consented_at is None
        if pending and turn.message.kind is MessageKind.AUDIO:
            await self._messenger.send(turn.message.user_id, self._language_menu(turn.language))
            turn.replied = True
            return None
        if pending and turn.message.kind is MessageKind.TEXT and not match.greeting_language:
            assert user is not None
            consented = dataclasses.replace(
                user,
                language=turn.language,
                consented_at=now,
                consent_version=CONSENT_VERSION,
                last_seen_at=now,
            )
            await self._store.save_user(consented)
            return consented
        turn.language = match.greeting_language or turn.language
        # The welcome's language is kept provisionally (no consent yet) so a reply in it is understood. Donation
        # consent is separate and survives a new main consent text, but never a STOP: a stopped user who comes back
        # starts without it (a fresh yes is required); the "offered" marks stay.
        # welcome_version stays unset until the welcome was sent (accepted by WhatsApp): a failed send is not "shown".
        state = UserState(turn.user_hash, language=turn.language, created_at=user.created_at if user else now,
                          last_seen_at=now)  # fmt: skip
        if user is not None:
            state = dataclasses.replace(
                state,
                donation_offered_at=user.donation_offered_at,
                donation_offered_version=user.donation_offered_version,
                donation_withdrawn_at=user.donation_withdrawn_at,
            )
            if not user.stopped:
                state = dataclasses.replace(
                    state, donation_version=user.donation_version, donation_at=user.donation_at,
                    donation_source=user.donation_source,
                )  # fmt: skip
        await self._store.save_user(state)
        welcome = self._messages.text(turn.language, Msg.WELCOME, **self._hosting(turn.language))
        if self._privacy_url:
            welcome += "\n" + self._messages.text(turn.language, Msg.PRIVACY_LINK, url=self._privacy_url)
        await self._send(turn, welcome)
        delivered = dataclasses.replace(
            state, welcome_version=CONSENT_VERSION, welcome_at=utcnow()
        )  # now shown
        await self._store.save_user(delivered)
        await self._messenger.send(turn.message.user_id, self._language_menu(turn.language))
        return None

    async def _set_language(self, turn: _Turn, user: UserState, code: LanguageCode) -> None:
        now = self._clock()
        new_consent = user.stopped or not _consented(user)  # choosing a language = continuing = consent
        await self._store.save_user(
            dataclasses.replace(
                user,
                language=code,
                stopped=False,
                last_seen_at=now,
                consented_at=now if new_consent else user.consented_at,
                consent_version=CONSENT_VERSION if new_consent else user.consent_version,
            )
        )
        turn.language = code
        config = self._registry.get(code)
        lines = [self._messages.text(code, Msg.LANGUAGE_SET, language=config.native_name)]
        if config.status is LanguageStatus.BETA:
            lines.append(self._messages.text(code, Msg.BETA_LANGUAGE, language=config.native_name))
        lines.append(self._messages.text(code, Msg.GREETING_REPLY))
        await self._send(turn, "\n\n".join(lines))

    async def _dispatch(self, turn: _Turn, user: UserState, match: Match) -> None:
        kind = turn.message.kind
        if match.command is Command.LANGUAGE_MENU:
            await self._store.save_user(user)
            await self._messenger.send(turn.message.user_id, self._language_menu(turn.language))
            turn.replied = True
            return
        if kind is MessageKind.UNSUPPORTE***REMOVED***
            await self._store.save_user(user)  # last_seen_at: every message counts as activity
            await self._send(turn, self._messages.text(turn.language, Msg.UNSUPPORTED))
            return
        parsed = parse_button(turn.message.button_id) if kind is MessageKind.BUTTON else None
        if parsed and parsed[0] in (Action.HELPFUL, Action.NOT_HELPFUL):
            await self._store.save_user(user)
            await self._feedback(turn, parsed[0], parsed[1])
            return
        if parsed and parsed[0] is Action.DONATE:  # consent choices are never rate-limited
            await self._donation_choice(turn, user, parsed[1])
            return
        if match.command is Command.DONATE:
            if self._collecting:
                await self._offer_donation(turn, user)
            else:
                await self._store.save_user(user)
                await self._send(turn, self._messages.text(turn.language, Msg.DONATION_UNAVAILABLE))
            return
        if await self._rate_limited(turn):
            await self._store.save_user(user)
            await self._send(turn, self._messages.text(turn.language, Msg.RATE_LIMITED))
            return
        if kind is MessageKind.BUTTON:
            await self._store.save_user(user)
            await self._follow_up(turn, parsed)
        elif match.greeting_language:
            await self._store.save_user(user)
            await self._fixed(turn, Msg.GREETING_REPLY, transcript=turn.message.text)
        else:
            await self._check(turn, user)
            self._after_voice(turn, user)

    # --- checks -----------------------------------------------------------------------------------------

    async def _check(self, turn: _Turn, user: UserState) -> None:
        message = turn.message
        if message.kind is MessageKind.AUDIO:
            await self._send(turn, self._messages.text(turn.language, Msg.LISTENING), ack=True)
            turn.voice = True
            try:
                # Network download first, outside the engine's CPU-bound speech slots.
                data = await self._audio_of(message, turn.deadline)
                speech = await self._transcribe(data, turn)
                self._keep_for_donation(turn, message, data, speech)
            except AudioError as exc:
                too_big = exc.problem in (AudioProblem.TOO_LONG, AudioProblem.TOO_LARGE)
                key = Msg.AUDIO_TOO_LONG if too_big else Msg.NOT_UNDERSTOOD
                await self._store.save_user(user)
                await self._log(turn, outcome=Outcome.AUDIO_ERROR, error=exc.problem.value)
                await self._send(turn, self._messages.text(turn.language, key, seconds=int(MAX_SECONDS)))
                return
            turn.transcript = speech.transcript.text.strip()
            turn.decode_ms, turn.asr_ms = speech.decode_ms, speech.asr_ms
        else:
            turn.transcript = (message.text or "").strip()
        follow_ons = self._take_follow_ons(turn)
        if follow_ons:
            for merged in follow_ons:
                await self._acknowledge(merged)
            if not turn.voice and any(m.kind is MessageKind.AUDIO for m in follow_ons):
                await self._send(turn, self._messages.text(turn.language, Msg.LISTENING), ack=True)
            try:
                async with asyncio.TaskGroup() as group:  # a system fault in one cancels the others
                    tasks = [group.create_task(self._content_of(turn, m)) for m in follow_ons]
            except (
                ExceptionGroup
            ) as group_error:  # surface the real fault (e.g. ASRError) in logs and evidence
                raise group_error.exceptions[0] from group_error
            extra = [task.result() for task in tasks]
            turn.transcript = "\n".join(p for p in (turn.transcript, *extra) if p)

        answer = await self._engine.check(
            turn.transcript,
            turn.language,
            deadline=turn.deadline,
            # hashed: a provider message id may embed the sender's number (privacy.UserHasher.message_key)
            turn_id=self._hasher.message_key(turn.message.channel, turn.message.message_id),
        )
        if answer.kind is not AnswerKind.CHECK:
            await self._store.save_user(user)
            voice_unclear = turn.voice and answer.kind is AnswerKind.CONTENT_FREE
            await self._fixed(
                turn, Msg.NOT_UNDERSTOOD if voice_unclear else Msg.GREETING_REPLY, answer=answer
            )
            return

        turn.calm_answer = answer.verdict not in REPORT_VERDICTS
        reply = render_check_reply(
            self._messages,
            turn.language,
            verdict=answer.verdict,
            body=answer.body,
            transcript=turn.transcript if turn.voice else None,
            contacts=answer.contacts,
        )
        interaction_id = await self._log(turn, outcome=answer.outcome, answer=answer, reply=reply)
        assert turn.logged_at is not None
        await record_pattern(self._store, turn.message.channel, answer, turn.transcript or "", turn.logged_at)
        await self._store.save_user(dataclasses.replace(user, last_interaction_id=interaction_id))
        feedback = (
            Button(
                button_id(Action.HELPFUL, interaction_id), self._messages.text(turn.language, Msg.BTN_HELPFUL)
            ),
            Button(
                button_id(Action.NOT_HELPFUL, interaction_id),
                self._messages.text(turn.language, Msg.BTN_NOT_HELPFUL),
            ),
        )
        if answer.verdict in REPORT_VERDICTS:
            actions = tuple(
                Button(button_id(action, interaction_id), self._messages.text(turn.language, title))
                for action, title in (
                    (Action.STEPS, Msg.BTN_WHAT_TO_DO),
                    (Action.REPORT, Msg.BTN_REPORT),
                    (Action.WARN, Msg.BTN_WARN_FAMILY),
                )
            )
            await self._send(turn, reply, actions)
            await self._send(turn, self._messages.text(turn.language, Msg.FEEDBACK_PROMPT), feedback)
        else:
            await self._send(turn, reply, feedback)

    async def _follow_up(self, turn: _Turn, parsed: tuple[Action, str] | None) -> None:
        if parsed is None or parsed[0] is Action.LANGUAGE:  # stale/unknown button, or an invalid language id
            await self._fixed(turn, Msg.GREETING_REPLY)
            return
        action, ref = parsed
        source = await self._own_interaction(turn, ref)
        english = (source.english or "") if source else ""
        flags = source.flags if source else ()
        llm_ms = 0
        if action is Action.STEPS:
            steps = await self._engine.next_steps(english, flags, turn.language, deadline=turn.deadline)
            reply, outcome, llm_ms = (
                render_steps(self._messages, turn.language, steps.text), steps.outcome, steps.llm_ms
            )  # fmt: skip
        elif action is Action.REPORT:
            contacts = await self._engine.report_contacts(english, flags, turn.language)
            reply, outcome = render_report(self._messages, turn.language, contacts), Outcome.FIXED_REPLY
        else:
            warning = await self._engine.warning(english, flags, turn.language, deadline=turn.deadline)
            reply, outcome = render_warning(self._messages, turn.language, warning.text), warning.outcome
            llm_ms = warning.llm_ms
        await self._log(
            turn, outcome=outcome, intent=f"button:{action.value}", reply=reply, llm_ms=llm_ms,
            reply_to=source.id if source else None,
        )  # fmt: skip
        await self._send(turn, reply)

    async def _feedback(self, turn: _Turn, action: Action, ref: str) -> None:
        source = await self._own_interaction(turn, ref)
        if source is not None and source.id is not None:
            await self._store.add_feedback(turn.user_hash, source.id, action is Action.HELPFUL)
        await self._send(turn, self._messages.text(turn.language, Msg.FEEDBACK_THANKS))

    async def _fixed(
        self, turn: _Turn, key: Msg, *, answer: Answer | None = None, transcript: str | None = None
    ) -> None:
        reply = self._messages.text(turn.language, key, seconds=int(MAX_SECONDS))
        if transcript is not None:
            turn.transcript = transcript
        await self._log(turn, outcome=Outcome.FIXED_REPLY, answer=answer, reply=reply)
        await self._send(turn, reply)

    # --- voice-note donation (opt-in; kudiguard.donation) ----------------------------------------------

    def _keep_for_donation(
        self, turn: _Turn, message: InboundMessage, data: bytes, speech: SpeechResult
    ) -> None:
        # Only notes recorded in the app and not marked forwarded: never someone else's recording, if we can tell.
        if self._collecting and message.voice_note and not message.forwarded:
            turn.voice_notes.append((data, speech))

    def _after_voice(self, turn: _Turn, user: UserState) -> None:
        """After the reply was sent: keep a donor's voice notes or make the offer, in a tracked background task that
        does not hold the turn's slot. A withdrawal later in the queue still wins (``VoiceVault`` tombstones)."""
        notes, turn.voice_notes = turn.voice_notes, []
        if not notes or not turn.replied:
            return
        if user.donation_version == DONATION_VERSION:  # consent stood when these notes arrived
            work = self._donate_notes(turn, notes)
        elif turn.calm_answer and _offer_due(user):
            work = self._user_locks.run(turn.user_hash, lambda: self._offer_later(turn))
        else:
            return
        task = asyncio.create_task(self._background(work), name="donation")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def _background(work: Coroutine[Any, Any, None]) -> None:
        try:
            async with asyncio.timeout(DONATION_DEADLINE_S):
                await work
        except Exception:
            log.exception("voice donation step failed (the reply was already sent)")

    async def _donate_notes(self, turn: _Turn, notes: list[tuple[bytes, SpeechResult]]) -> None:
        assert self._donations is not None and turn.received_at is not None
        for data, speech in notes:
            await self._donations.donate(
                Donation(
                    user_hash=turn.user_hash, channel=turn.message.channel, language=turn.language, audio=data,
                    hypothesis=speech.transcript.text, consent_version=DONATION_VERSION,
                    duration_s=speech.transcript.duration_s, consented_at=turn.received_at,
                )
            )  # fmt: skip

    async def _offer_later(self, turn: _Turn) -> None:
        """Runs under the user's lock, after any message already queued: the state is re-read and re-checked."""
        latest = await self._store.get_user(turn.user_hash)
        if latest is not None and not latest.stopped and _consented(latest) and _offer_due(latest):
            await self._offer_donation(turn, latest)

    async def _offer_donation(self, turn: _Turn, user: UserState) -> None:
        """The donation request: its own message and buttons, separate from the main consent. Recorded before it is
        sent, so a quick "Yes" can never be overwritten by this save."""
        buttons = tuple(
            Button(button_id(Action.DONATE, ref), self._messages.text(turn.language, title))
            for ref, title in ((DONATE_YES, Msg.BTN_DONATE_YES), (DONATE_NO, Msg.BTN_DONATE_NO))
        )
        await self._store.save_user(
            dataclasses.replace(
                user, donation_offered_at=self._clock(), donation_offered_version=DONATION_VERSION
            )
        )
        text = self._messages.text(turn.language, Msg.DONATION_OFFER, **self._hosting(turn.language))
        await self._send(turn, text, buttons)

    async def _donation_choice(self, turn: _Turn, user: UserState, ref: str) -> None:
        if ref == DONATE_NO:  # declining or withdrawing always works, even with collection switched off
            await self._withdraw_donation(turn, user, declined=True)
        elif not self._collecting:
            await self._send(turn, self._messages.text(turn.language, Msg.DONATION_UNAVAILABLE))
        elif ref == DONATE_YES:
            now = self._clock()
            await self._store.save_user(
                dataclasses.replace(
                    user, donation_version=DONATION_VERSION, donation_at=now, donation_source="chat",
                    donation_offered_at=user.donation_offered_at or now, donation_offered_version=DONATION_VERSION,
                )
            )  # fmt: skip
            await self._send(turn, self._messages.text(turn.language, Msg.DONATION_ON))
        else:  # "Yes" under an older offer text (or an unknown ref): show the current offer
            await self._offer_donation(turn, user)

    async def _withdraw_donation(
        self, turn: _Turn, user: UserState | None, *, declined: bool = False
    ) -> None:
        """Withdrawal = stop collecting AND delete everything donated (most protective)."""
        removed = await self._erase_donations(turn.user_hash)
        was_donating = user is not None and user.donation_version is not None
        if user is not None:
            now = self._clock()
            await self._store.save_user(
                dataclasses.replace(
                    user, donation_version=None, donation_at=None, donation_source=None, last_seen_at=now,
                    donation_offered_at=user.donation_offered_at or now,
                    donation_offered_version=user.donation_offered_version or DONATION_VERSION,
                    donation_withdrawn_at=now,
                )
            )  # fmt: skip
        key = Msg.DONATION_DECLINED if declined and not (was_donating or removed) else Msg.DONATION_OFF
        await self._send(turn, self._messages.text(turn.language, key))

    async def _erase_donations(self, user_hash: str) -> int:
        if self._donations is not None:
            return await self._donations.erase(user_hash)
        return len(await self._store.delete_voice_samples(user_hash))  # no files on this server; the rows go

    def _hosting(self, language: LanguageCode) -> dict[str, str]:
        """``{where}`` and ``{risk}`` (the configured storage location) and the retention periods of the welcome and
        offer texts."""
        return {
            "where": self._messages.text(language, self._storage.message),
            "risk": self._messages.text(language, self._storage.risk),
            "retention_days": str(self._limits.retention_days),
            "pattern_retention_days": str(self._limits.pattern_retention_days),
        }

    # --- helpers ----------------------------------------------------------------------------------------

    async def _content_of(self, turn: _Turn, message: InboundMessage) -> str:
        """Text of a merged follow-on message; an unusable voice note among several is skipped, not fatal."""
        if message.kind is not MessageKind.AUDIO:
            return (message.text or "").strip()
        turn.voice = True
        try:
            data = await self._audio_of(message, turn.deadline)
            speech = await self._transcribe(data, turn)
        except AudioError as exc:
            log.info("merged voice note skipped: %s", exc.problem)
            return ""
        self._keep_for_donation(turn, message, data, speech)
        turn.asr_ms = (turn.asr_ms or 0) + speech.asr_ms
        turn.decode_ms = (turn.decode_ms or 0) + speech.decode_ms
        return speech.transcript.text.strip()

    async def _transcribe(self, data: bytes, turn: _Turn) -> SpeechResult:
        """Speech recognition within the turn's deadline: the engine's speech slots are shared with web/API voice,
        so a long wait becomes "please resend" rather than a generic failure."""
        left = turn.deadline - time.monotonic()
        if left <= 0:
            raise AudioError(AudioProblem.TIMEOUT, "no time left to transcribe")
        try:
            async with asyncio.timeout(left):
                return await self._engine.transcribe(data, turn.language)
        except TimeoutError as exc:
            raise AudioError(AudioProblem.TIMEOUT, "transcription did not finish in time") from exc

    async def _audio_of(self, message: InboundMessage, deadline: float) -> bytes:
        """Inline bytes (web, API) or a lazy channel download (WhatsApp) — only ever for consented users."""
        if message.audio is not None:
            return message.audio
        if message.media_id is None:
            raise AudioError(AudioProblem.EMPTY, "no audio and no media reference")
        left = deadline - time.monotonic()
        if left <= 0:
            raise AudioError(AudioProblem.TIMEOUT, "no time left to download the voice note")
        try:
            async with asyncio.timeout(left):  # a stuck download must not eat the whole turn
                return await self._messenger.fetch_media(message)
        except TimeoutError as exc:
            raise AudioError(AudioProblem.TIMEOUT, "voice note download timed out") from exc

    def _chosen_language(self, message: InboundMessage) -> LanguageCode | None:
        parsed = parse_button(message.button_id) if message.kind is MessageKind.BUTTON else None
        if parsed and parsed[0] is Action.LANGUAGE and parsed[1] in {c.code for c in self._registry.enabled}:
            return parsed[1]
        return None

    def _language_menu(self, language: LanguageCode) -> Outbound:
        options = tuple(
            MenuOption(
                button_id(Action.LANGUAGE, lang.code),
                lang.native_name,
                lang.name if lang.name != lang.native_name else "",
            )
            for lang in self._registry.enabled
        )
        return Outbound(
            self._messages.text(language, Msg.CHOOSE_LANGUAGE),
            menu_button=self._messages.text(language, Msg.LANGUAGE_MENU_BUTTON),
            options=options,
        )

    async def _own_interaction(self, turn: _Turn, ref: str) -> Interaction | None:
        """The interaction a button refers to, only if it belongs to this user (ids come from the client)."""
        if not ref.isdigit():
            return None
        found = await self._store.get_interaction(int(ref))
        return found if found is not None and found.user_hash == turn.user_hash else None

    async def _rate_limited(self, turn: _Turn) -> bool:
        since = self._clock() - timedelta(hours=1)
        count = await self._store.count_interactions_since(turn.user_hash, since)
        return count >= self._limits.per_user_per_hour

    async def _acknowledge(self, message: InboundMessage) -> None:
        try:
            async with asyncio.timeout(ACK_DEADLINE_S):
                await self._messenger.acknowledge(message)
        except Exception:
            log.warning("acknowledge failed", exc_info=True)

    async def _send(
        self, turn: _Turn, text: str, buttons: tuple[Button, ...] = (), *, ack: bool = False
    ) -> None:
        """``ack``: an interim message ("Listening…") that does not count as the turn's answer."""
        await self._messenger.send(turn.message.user_id, Outbound(text, buttons))
        if not ack:
            turn.replied = True

    async def _log(
        self,
        turn: _Turn,
        *,
        outcome: Outcome,
        answer: Answer | None = None,
        reply: str | None = None,
        error: str | None = None,
        intent: str | None = None,
        llm_ms: int | None = None,
        reply_to: int | None = None,
    ) -> int:
        logged_at = self._clock()
        interaction_id = await self._store.add_interaction(
            Interaction(
                channel=turn.message.channel,
                message_id=turn.message.message_id,
                user_hash=turn.user_hash,
                language=turn.language,
                kind=AUDIO if turn.voice else turn.message.kind,
                outcome=outcome,
                intent=intent or (answer.intent.value if answer and answer.intent else None),
                transcript=turn.transcript,
                english=(answer.english or None) if answer else None,
                verdict=answer.verdict if answer and answer.kind is AnswerKind.CHECK else None,
                flags=answer.flags if answer else (),
                reply=reply,
                decode_ms=turn.decode_ms,
                asr_ms=turn.asr_ms,
                llm_ms=llm_ms if llm_ms is not None else (answer.llm_ms if answer else None),
                total_ms=int((time.perf_counter() - turn.started) * 1000),
                error=error,
                created_at=logged_at,
                reply_to=reply_to,
            )
        )
        turn.logged = True
        turn.logged_at = logged_at
        turn.interaction_id = interaction_id
        turn.reservation.release()  # the store counts this user now
        return interaction_id


AUDIO = MessageKind.AUDIO


async def record_pattern(store: Store, channel: str, answer: Answer, text: str, now: datetime) -> None:
    """Keep a de-identified example of a 🔴/🟠 check (consented channels only). Never fails the turn.

    Built from ``redact(text)``, exactly what the store keeps as the transcript, and dated with the interaction's own
    timestamp, so DELETE can find it again (``storage``: same function, same input, same week).
    """
    if answer.verdict not in REPORT_VERDICTS or channel not in PATTERN_CHANNELS or not text.strip():
        return
    try:
        await store.add_pattern(
            ScamPattern.from_check(
                language=answer.language, channel=channel, verdict=answer.verdict, flags=answer.flags,
                text=redact(text), english=redact(answer.english) if answer.english else None,
                day=now.astimezone(UTC).date(),
            )
        )  # fmt: skip
    except Exception:
        log.exception("could not record a scam pattern")


def _before_welcome(message: InboundMessage, user: UserState | None) -> bool:
    """Sent before the user could have read the welcome (e.g. a second message queued behind the first): such a
    message is not consent. Compared on absolute time (provider timestamp vs. when we sent the welcome) with a
    reading margin (``WELCOME_READ_S``) that also covers our clock running behind the provider's. Comparing with the
    triggering message's own timestamp would accept any later message sent before the welcome arrived, which is
    exactly the race to close."""
    if message.sent_at is None or user is None or user.welcome_at is None:
        return False
    return message.sent_at < user.welcome_at.timestamp() + WELCOME_READ_S


def _offer_due(user: UserState) -> bool:
    """Never offered; or donating under an older offer text (collection pauses until they answer the new one)."""
    if user.donation_version is None:
        return user.donation_offered_version is None
    return user.donation_version != DONATION_VERSION and user.donation_offered_version != DONATION_VERSION


def _consented(user: UserState) -> bool:
    """Consent stands only for the current consent text (a material change asks again)."""
    return user.consented_at is not None and user.consent_version == CONSENT_VERSION
