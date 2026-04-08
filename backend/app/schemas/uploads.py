from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    ok: bool
    upload_id: int
    public_id: str
    filename: str
    content_type: str | None
    size_bytes: int | None
    storage_url: str | None
