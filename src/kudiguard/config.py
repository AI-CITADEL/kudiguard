"""Runtime settings from the environment / ``.env`` (prefix ``KUDIGUARD_``; WhatsApp keys use ``WHATSAPP_``).

Everything has a safe default for local development except the secrets, which are validated at startup so a
misconfigured deployment fails fast instead of half-working.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from kudiguard.hosting import StorageLocation
from kudiguard.reply_translation import DEFAULT_COMPOSE_STRATEGY, ComposePolicy

ROOT = Path(__file__).resolve().parents[2]  # source checkout; set KUDIGUARD_DATA_DIR etc. for other installs


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KUDIGUARD_", env_file=ROOT / ".env", extra="ignore")

    # --- data and storage ---------------------------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    db_path: Path = ROOT / "var" / "kudiguard.db"
    hash_secret: SecretStr = SecretStr("")  # >= 32 bytes; pseudonymous user ids (privacy.UserHasher)
    # Upper bounds = the pilot's published promises ("up to 90 days", "up to 2 years"): the user-facing texts render
    # the configured values, and a longer period would need a new consent text (docs/DEPLOY.md). Lower bound 30: the
    # licence cap counts active users from the evidence log over 30 days, so it must be kept at least that long.
    retention_days: int = Field(default=90, ge=30, le=90)
    # de-identified scam examples (kudiguard.patterns)
    pattern_retention_days: int = Field(default=730, ge=30, le=730)
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
    # Partner packs in ``mode: live`` send the English text of each user message to the partner's server
    # (knowledge/http_source.py). Refused unless allowed here AND disclosed on /privacy (rendered when on).
    allow_live_packs: bool = False
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
    # The partner REST API (/v1/*) is OFF in the pilot: its routes are not mounted at all (404, absent from /docs).
    # When on, every check must name a pseudonymous end user (X-KudiGuard-End-User), who counts toward the N-ATLaS
    # licence cap (``active_user_cap``) together with WhatsApp and web users (docs/DEPLOY.md, "Partner API").
    partner_api: bool = False
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
