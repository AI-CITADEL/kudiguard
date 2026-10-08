"""Runtime settings from the environment / ``.env`` (prefix ``KUDIGUARD_``; WhatsApp keys use ``WHATSAPP_``).

Everything has a safe default for local development except the secrets, which are validated at startup so a
misconfigured deployment fails fast instead of half-working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]  # source checkout; set KUDIGUARD_DATA_DIR etc. for other installs


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KUDIGUARD_", env_file=ROOT / ".env", extra="ignore")

    # --- data and storage ---------------------------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    db_path: Path = ROOT / "var" / "kudiguard.db"
    hash_secret: SecretStr = SecretStr("")  # >= 32 bytes; pseudonymous user ids (privacy.UserHasher)
    retention_days: int = Field(default=90, ge=1)
    allow_sample_packs: bool = False  # fictional demo packs (never in production)
    packs: Annotated[tuple[str, ...], NoDecode] = ()  # empty = every enabled pack

    # --- models -------------------------------------------------------------------------------------------
    llm_url: str = "http://127.0.0.1:8081"
    llm_api_key: SecretStr | None = None
    model_cache: Path = ROOT / "models"  # converted CTranslate2 speech models (scripts/prepare_asr.py)
    embeddings: bool = True  # hybrid retrieval (fastembed, CPU); False = BM25 only
    embeddings_cache: Path = ROOT / "models" / "fastembed"
    speech: bool = True
    preload_languages: Annotated[tuple[str, ...], NoDecode] = ("ha", "en")  # warm speech models at startup
    ffmpeg: str | None = None

    # --- conversation limits ------------------------------------------------------------------------------
    per_user_per_hour: int = Field(default=20, ge=1)
    active_user_cap: int | None = Field(
        default=1000, ge=1
    )  # N-ATLAS licence: active users per rolling 30 days
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

    # --- WhatsApp (optional: the channel is mounted only when configured) -----------------------------------
    whatsapp_enabled: bool = True

    @field_validator("packs", "preload_languages", "api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

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
