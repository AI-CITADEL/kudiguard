"""Runtime settings from the environment / ``.env`` (prefix ``KUDIGUARD_``; WhatsApp keys use ``WHATSAPP_``).

Everything has a safe default for local development except the secrets, which are validated at startup so a
misconfigured deployment fails fast instead of half-working.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from kudiguard.hosting import StorageLocation
from kudiguard.reply_translation import DEFAULT_COMPOSE_STRATEGY, ComposePolicy

ROOT = Path(__file__).resolve().parents[2]  # source checkout; set KUDIGUARD_DATA_DIR etc. for other installs
ScreenModeName = Literal["off", "shadow", "enforce"]  # = safety.abuse.ScreenMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KUDIGUARD_", env_file=ROOT / ".env", extra="ignore")

    # --- data and storage ---------------------------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    db_path: Path = ROOT / "var" / "kudiguard.db"
    hash_secret: SecretStr = SecretStr("")  # >= 32 bytes; pseudonymous user ids (privacy.UserHasher)
    retention_days: int = Field(default=90, ge=1)
    pattern_retention_days: int = Field(default=730, ge=1)  # de-identified scam examples (kudiguard.patterns)
    # Voice-note donation (kudiguard.donation; docs/DPIA_VOICE.md). Off until the DPIA is signed off.
    voice_donation: bool = False
    voice_dir: Path = ROOT / "var" / "voice"  # encrypted audio; must belong to db_path (marker file)
    voice_retention_days: int = Field(default=365, ge=1, le=365)  # the offer promises at most 12 months
    voice_key: SecretStr | None = None  # >= 32 bytes; default: derived from hash_secret
    voice_max_per_day: int = Field(default=20, ge=1)  # samples per donor per day
    voice_max_gb: float = Field(default=5.0, gt=0)  # stop collecting beyond this folder size
    # Where the server stores data, as told to users (kudiguard.hosting): outside-nigeria | nigeria.
    storage_location: StorageLocation = StorageLocation.OUTSIDE_NIGERIA
    allow_sample_packs: bool = False  # fictional demo packs (never in production)
    packs: Annotated[tuple[str, ...], NoDecode] = ()  # empty = every enabled pack

    # --- models -------------------------------------------------------------------------------------------
    llm_url: str = "http://127.0.0.1:8081"
    llm_api_key: SecretStr | None = None
    model_cache: Path = ROOT / "models"  # converted CTranslate2 speech models (scripts/prepare_asr.py)
    embeddings: bool = True  # hybrid retrieval (fastembed, CPU); False = BM25 only
    embeddings_cache: Path = ROOT / "models" / "fastembed"
    # Check English replies claim by claim against their facts (needs ``embeddings``; CPU, fails open; see
    # safety/grounding.py). shadow = record only; enforce = drop unsupported steps or send fixed text. Enforce only
    # after >= 100 held-out real N-ATLAS English replies show <= 5% false replacements (docs/SPIKES.md S8).
    grounding_mode: Literal["off", "shadow", "enforce"] = "shadow"
    grounding_threshold: float = Field(
        default=0.75, gt=0, lt=1
    )  # = grounding.DEFAULT_THRESHOLD (bge-small cosine)
    grounding_budget_ms: int = Field(
        default=150, gt=0
    )  # = grounding.DEFAULT_BUDGET_S; past it: send the reply
    # Second guard layer: a learned classifier for unsafe advice in English replies (safety/guard_classifier.py;
    # needs ``embeddings``). shadow = score after sending, log only; enforce = a flagged reply is retried, then fixed
    # text. Pidgin replies (language "en") use ``guard_classifier_pidgin_mode`` when set (e.g. en enforce, Pidgin
    # shadow). Threshold: unset or empty = the artefact's (chosen on dev). Past the budget: the regex verdict alone.
    guard_classifier_mode: Literal["off", "shadow", "enforce"] = "shadow"
    guard_classifier_pidgin_mode: Literal["off", "shadow", "enforce"] | None = None
    guard_classifier_path: Path = ROOT / "data" / "guard" / "en_classifier.npz"
    guard_classifier_threshold: float | None = Field(default=None, gt=0, lt=1)
    guard_classifier_budget_ms: int = Field(default=250, gt=0)  # = guard_classifier.DEFAULT_BUDGET_S
    speech: bool = True
    preload_languages: Annotated[tuple[str, ...], NoDecode] = ("ha", "en")  # warm speech models at startup
    ffmpeg: str | None = None
    # How replies are written (kudiguard.reply_translation): english_first (compose in English, check, translate) or
    # direct (compose in the user's language), with per-language overrides. Default "english_first,ha=direct":
    # Yoruba/Igbo English-first, Hausa direct (docs/SPIKES.md H); English is always direct.
    compose_strategy: str = DEFAULT_COMPOSE_STRATEGY

    # --- conversation limits ------------------------------------------------------------------------------
    per_user_per_hour: int = Field(default=20, ge=1)
    # N-ATLAS licence: active end-users per rolling 30 days, project-wide (every channel counts).
    active_user_cap: int | None = Field(default=1000, ge=1)
    web_session_cap: int = Field(default=200, ge=1)  # share of that cap open to (anonymous) web sessions
    turn_deadline_s: float = Field(default=60.0, gt=0)
    default_language: str = "en"

    # --- public API ---------------------------------------------------------------------------------------
    # Comma-separated "name:key" pairs, e.g. "demo-bank:3f9…,ngo:…". Keys >= 24 chars. Empty = API disabled.
    api_keys: Annotated[tuple[str, ...], NoDecode] = ()
    api_rate_per_minute: int = Field(default=30, ge=1)  # per API key
    web_rate_per_minute: int = Field(default=6, ge=1)  # per client IP on the public web voice page
    # Client IP from CF-Connecting-IP (else rightmost X-Forwarded-For). Only behind Cloudflare: any other proxy
    # passes a client-sent CF-Connecting-IP through, which would make the web rate limit spoofable.
    trust_proxy_headers: bool = False

    # --- public pages ----------------------------------------------------------------------------------------
    operator_name: str = "the KudiGuard team"  # data controller named on /privacy
    public_url: str = ""  # e.g. https://kudiguard.theaicitadel.com (privacy link in the WhatsApp welcome)
    privacy_contact: str = (
        ""  # email for privacy requests on /privacy (empty: "contact the operator" + warning)
    )

    # --- WhatsApp (optional: the channel is mounted only when configured) -----------------------------------
    whatsapp_enabled: bool = True

    # --- malicious use (docs/ABUSE.md). Modes: off | shadow (log only) | enforce ------------------------------
    abuse_crisis_mode: ScreenModeName = "enforce"  # self-harm messages: humane fixed reply with help numbers
    abuse_harassment_mode: ScreenModeName = (
        "enforce"  # insults / hate / sexual messages at the bot: calm reply
    )
    abuse_scam_tool_mode: ScreenModeName = "enforce"  # "write a message that sounds like my bank": refusal
    abuse_injection_mode: ScreenModeName = (
        "enforce"  # override / role-play / prompt extraction: rules-only answer
    )
    reply_screen_mode: ScreenModeName = (
        "enforce"  # replies: persona claims, prompt leaks, long echoes of the input
    )
    reply_echo_words: int = Field(default=12, ge=4)  # a copied run this long counts as an echo
    asr_hallucination_mode: ScreenModeName = (
        "enforce"  # "thanks for watching" on music/noise is not a message
    )
    pack_screen_mode: ScreenModeName = (
        "enforce"  # partner knowledge: quarantine planted instructions/unsafe facts
    )
    pack_screen_core_mode: ScreenModeName = "shadow"  # our own reviewed core packs: log only
    learning_gate_mode: ScreenModeName = "enforce"  # pattern library and voice donations
    pattern_max_per_user_per_day: int = Field(default=5, ge=1)
    pattern_max_similar_per_week: int = Field(default=20, ge=1)
    new_user_mode: ScreenModeName = "enforce"  # rate of users new to the licence window ("busy, try later")
    new_users_per_hour: int | None = Field(default=40, ge=1)
    new_users_per_day: int | None = Field(default=200, ge=1)
    flood_mode: ScreenModeName = (
        "enforce"  # per-user queue cap, one rate-limit notice per interval, voice budget
    )
    max_queued_per_user: int = Field(default=10, ge=1)
    rate_notice_interval_s: float = Field(default=600.0, ge=0)
    voice_notes_per_user_per_hour: int | None = Field(default=30, ge=1)

    @field_validator(
        "new_users_per_hour", "new_users_per_day", "voice_notes_per_user_per_hour", mode="before"
    )
    @classmethod
    def _empty_is_no_cap(cls, value: object) -> object:
        """An empty value (``KUDIGUARD_NEW_USERS_PER_DAY=``) means no cap."""
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _new_user_caps_consistent(self) -> Settings:
        hour, day = self.new_users_per_hour, self.new_users_per_day
        if hour is not None and day is not None and hour > day:
            raise ValueError("KUDIGUARD_NEW_USERS_PER_HOUR cannot exceed KUDIGUARD_NEW_USERS_PER_DAY")
        return self

    @field_validator("guard_classifier_threshold", "guard_classifier_pidgin_mode", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        """``KUDIGUARD_GUARD_CLASSIFIER_THRESHOLD=`` (an empty line in .env) means "not set", not a parse error."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("packs", "preload_languages", "api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("compose_strategy")
    @classmethod
    def _valid_compose_strategy(cls, value: str) -> str:
        ComposePolicy.parse(value)  # fail fast at startup on a typo
        return value

    def compose_policy(self, languages: Iterable[str] | None = None) -> ComposePolicy:
        """``languages``: the configured codes (startup); an override for another code raises ``ValueError``."""
        return ComposePolicy.parse(self.compose_strategy, languages)

    def voice_secret(self) -> bytes:
        """Input key for donated-audio encryption: ``voice_key`` if set, else the hash secret."""
        own = self.voice_key.get_secret_value() if self.voice_key else ""  # empty in .env = not set
        return (own or self.hash_secret.get_secret_value()).encode()

    def api_key_map(self) -> dict[str, str]:
        """``{key: client name}``; raises on malformed or weak entries (fail fast at startup)."""
        keys: dict[str, str] = {}
        for item in self.api_keys:
            name, sep, key = item.partition(":")
            if not sep or not name.strip() or len(key.strip()) < 24:
                raise ValueError(
                    "KUDIGUARD_API_KEYS entries must be 'name:key' with keys of at least 24 chars"
                )
            keys[key.strip()] = name.strip()
        return keys
