"""Reference implementation of the partner knowledge API (contract v1, see ``http_source``).

Serves any pack directory over HTTP. Partners can run it as-is in front of their exported files, or copy
the two endpoints into their own stack. Used in tests as the contract's executable specification.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from kudiguard.domain import Flag
from kudiguard.knowledge.file_pack import read_pack_files
from kudiguard.knowledge.index import EntryIndex
from kudiguard.knowledge.models import KnowledgePayload, Query


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(max_length=4000)
    flags: list[Flag] = Field(default_factory=list)
    language: str = Field(default="en", max_length=3)
    limit: int = Field(default=3, ge=1, le=20)


def create_app(pack_dir: Path, token: str | None) -> FastAPI:
    entries, contacts = read_pack_files(pack_dir)
    index = EntryIndex(entries, contacts)
    export = KnowledgePayload(entries=tuple(entries), contacts=tuple(contacts))
    app = FastAPI(title="Partner knowledge API (reference)", version="1")

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = f"Bearer {token}".encode()
        if token is not None and not secrets.compare_digest((authorization or "").encode(), expected):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/export", dependencies=[Depends(authorize)])
    def export_all() -> KnowledgePayload:
        return export

    @app.post("/search", dependencies=[Depends(authorize)])
    def search(request: SearchRequest) -> KnowledgePayload:
        result = index.search(
            Query(
                text=request.query, flags=tuple(request.flags), language=request.language, limit=request.limit
            )
        )
        entries = result.flag_entries + tuple(hit.entry for hit in result.hits)
        return KnowledgePayload(entries=entries, contacts=result.contacts)

    return app
