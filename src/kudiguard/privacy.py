"""Pseudonymous identifiers (NDPA 2023): phone numbers and anything derived from them never reach storage readable.

``UserHasher`` computes HMAC-SHA256 under a server secret:
* ``user_id``: the same person gets the same id across formats (``0803…``, ``+234 803…``); without the secret an id
  cannot be matched to a number (keep the secret out of database backups: the phone-number space is small).
* ``message_key``: provider message ids are hashed too — a WhatsApp ``wamid`` is an encoded protobuf that contains
  the sender's number, so storing it raw would defeat ``user_id``.
Rotating the secret orphans existing records (they then age out under the retention policy).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re

MIN_SECRET_BYTES = 32
ID_HEX_CHARS = 32  # 128 bits: collision-free at any realistic user count
REVIEW_ID_HEX_CHARS = 10  # 40 bits: short enough to read out, collision-free for review samples
SECRET_ENV = "KUDIGUARD_HASH_SECRET"
_NIGERIA = "234"
_PREFIX = re.compile(r"^(?:whatsapp|tel):", re.IGNORECASE)
_PHONE_SHAPED = re.compile(r"^\+?(?=(?:\D*\d){10})[\d\s().-]{10,20}$")  # at least 10 digits
_NIGERIAN_LOCAL = re.compile(r"^0[789][01]\d{8}$")  # 080x, 081x, 070x, 090x, 091x mobile prefixes


def normalise_phone(raw: str) -> str:
    """Digits-only E.164 (no '+'). Local Nigerian mobile numbers (e.g. 0803…) get the 234 country code."""
    digits = re.sub(r"\D", "", _PREFIX.sub("", raw.strip()))
    if _NIGERIAN_LOCAL.match(digits):
        return _NIGERIA + digits[1:]
    return digits


class UserHasher:
    def __init__(self, secret: bytes) -> None:
        if len(secret) < MIN_SECRET_BYTES:
            raise ValueError(f"hash secret must be at least {MIN_SECRET_BYTES} bytes")
        self._secret = secret

    @classmethod
    def from_env(cls) -> UserHasher:
        value = os.environ.get(SECRET_ENV, "")
        if len(value.encode()) < MIN_SECRET_BYTES:
            raise ValueError(f"set {SECRET_ENV} to a random secret of at least {MIN_SECRET_BYTES} bytes")
        return cls(value.encode())

    def _hmac(self, value: str) -> str:
        return hmac.new(self._secret, value.encode(), hashlib.sha256).hexdigest()[:ID_HEX_CHARS]

    def user_id(self, external_id: str) -> str:
        """Stable pseudonymous id. Phone-shaped ids are normalised first; anything else (web sessions, API client
        ids) is hashed verbatim in its own namespace, so different sessions never merge into one user."""
        stripped = _PREFIX.sub("", external_id.strip())
        if _PHONE_SHAPED.match(stripped):
            return self._hmac(f"phone:{normalise_phone(stripped)}")
        return self._hmac(f"id:{external_id.strip()}")

    def deletion_marker(self, user_hash: str) -> str:
        """Coded marker of a DELETE (own label): lets the sign-up import skip someone who deleted after the form."""
        return self._hmac(f"deleted:{user_hash}")

    def message_key(self, channel: str, message_id: str) -> str:
        """Opaque de-duplication key for a provider message id (which may embed the sender's number)."""
        return self._hmac(f"msg:{channel}:{message_id}")

    def review_id(self, interaction_id: int) -> str:
        """Opaque label for an interaction shown to human reviewers: stable across exports (so several reviewers'
        sheets line up), unlinkable to the row or the user without the secret. The ``R-`` prefix stops spreadsheets
        reading hex such as ``12e45`` as a number."""
        return "R-" + self._hmac(f"review:{interaction_id}")[:REVIEW_ID_HEX_CHARS]
