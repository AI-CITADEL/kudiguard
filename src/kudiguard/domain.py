"""Core value types shared by every module. No I/O here.

Languages are configuration (see ``kudiguard.languages``), so they are plain ISO-639 code strings here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

LanguageCode = str


class MessageKind(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    BUTTON = "button"  # a tap on one of our reply buttons or list options (``button_id`` is our own id)
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class InboundMessage:
    channel: str
    message_id: str
    user_id: str
    kind: MessageKind
    text: str | None = None
    audio: bytes | None = None
    audio_mime: str | None = None
    button_id: str | None = None


class Flag(StrEnum):
    """Scam red flags. Shared by the lexicon rules, the LLM extraction schema and the knowledge base."""

    ASKS_CREDENTIALS = "asks_credentials"  # OTP, PIN, CVV, password, BVN/NIN, card details
    REMOTE_ACCESS = "remote_access"  # install AnyDesk / share screen
    ACCOUNT_THREAT = "account_threat"  # account/BVN/SIM blocked or suspended
    UPFRONT_FEE = "upfront_fee"  # pay a fee to receive money, a loan, a job or a prize
    PRIZE_LOTTERY = "prize_lottery"  # you have won / been selected
    GUARANTEED_RETURNS = "guaranteed_returns"  # double your money, fixed high % returns
    WRONG_TRANSFER = "wrong_transfer"  # "I sent money to you by mistake, reverse it"
    FAKE_PAYMENT = "fake_payment"  # fake credit alert, "check your alert, release the goods"
    MONEY_REQUEST = "money_request"  # romance / "stuck at the airport" / contact asking for money
    SUSPICIOUS_LINK = "suspicious_link"  # shortened links, unknown links, APK downloads
    IMPERSONATION = "impersonation"  # claims to be bank staff, CBN, EFCC, police, customer care
    URGENCY = "urgency"  # act now, within 24 hours
    SECRECY = "secrecy"  # "don't tell anyone"


class Verdict(StrEnum):
    LIKELY_SCAM = "likely_scam"
    CAUTION = "caution"
    NO_FLAGS = "no_flags"  # user asked for a check; no flags at all (never "safe")
    NOT_A_CHECK = "not_a_check"  # general question, no scam signals


@dataclass(frozen=True, slots=True)
class ScamAssessment:
    verdict: Verdict
    flags: tuple[Flag, ...] = ()
    score: int = 0


@dataclass(frozen=True, slots=True)
class Reply:
    text: str


class Outcome(StrEnum):
    """How a turn ended (validation evidence and monitoring)."""

    ANSWERED = "answered"  # model-composed reply
    FIXED_REPLY = "fixed_reply"  # greeting / content-free / command: fixed text, no model
    FALLBACK = "fallback"  # model failed or was blocked by the output guard: fixed fallback text
    AUDIO_ERROR = "audio_error"  # unusable voice note (too long, no speech, undecodable)
    ERROR = "error"  # system fault


@dataclass(slots=True)
class Interaction:
    """One user turn, persisted as validation evidence. Text fields are stored redacted (no numbers/emails)."""

    channel: str
    message_id: str
    user_hash: str
    language: LanguageCode
    kind: MessageKind
    outcome: Outcome = Outcome.ANSWERED
    intent: str | None = None
    transcript: str | None = None
    english: str | None = None
    verdict: Verdict | None = None
    flags: tuple[Flag, ...] = ()
    reply: str | None = None
    decode_ms: int | None = None
    asr_ms: int | None = None
    llm_ms: int | None = None
    total_ms: int | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # The interaction a button press refers to ("What should I do?", "Warn my family").
    reply_to: int | None = None
    id: int | None = None  # assigned by the store
