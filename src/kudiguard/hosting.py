"""Where KudiGuard stores data, as disclosed to users (NDPA ss.41-43: an informed consent to a transfer abroad).

During the pilot the server is the team's PC outside Nigeria; set ``KUDIGUARD_STORAGE_LOCATION=nigeria`` only once
the server really is in Nigeria. WhatsApp (Meta) and Cloudflare stay outside Nigeria in either case, and every text
says so. Moving the server *abroad* again requires a new ``pipeline.CONSENT_VERSION`` (users must agree again).
"""

from __future__ import annotations

from enum import StrEnum

from kudiguard.i18n import Msg


class StorageLocation(StrEnum):
    OUTSIDE_NIGERIA = "outside-nigeria"
    NIGERIA = "nigeria"

    @property
    def message(self) -> Msg:
        """The localised phrase completing "stored on our server …"."""
        return Msg.WHERE_NIGERIA if self is StorageLocation.NIGERIA else Msg.WHERE_OUTSIDE

    @property
    def risk(self) -> Msg:
        """The localised transfer-risk sentence: our server abroad, or only Meta/Cloudflare abroad."""
        return Msg.RISK_NIGERIA if self is StorageLocation.NIGERIA else Msg.RISK_OUTSIDE

    @property
    def english(self) -> str:
        """For /privacy (English page)."""
        return "in Nigeria" if self is StorageLocation.NIGERIA else "outside Nigeria (during the pilot)"
