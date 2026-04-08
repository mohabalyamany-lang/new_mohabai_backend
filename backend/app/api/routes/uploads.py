from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.enums import ArtifactType
from app.db.models import Artifact, Conversation, Upload, User
from app.db.session import get_db_session
from app.schemas.uploads import UploadResponse
from app.services.storage_service import StorageService

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: int | None = Form(default=None),
    db: Session = Depends(get_db_session),
    user: User = Depends(get_current_user),
) -> UploadResponse:
    storage = StorageService()
    content = await file.read()
    storage_url = storage.save_bytes(file.filename, content, subdir="uploads")

    artifact = Artifact(
        conversation_id=conversation_id if conversation_id else None,
        artifact_type=ArtifactType.FILE,
        title=file.filename,
        storage_url=storage_url,
        metadata_json={
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
        },
    )
    db.add(artifact)
    db.flush()

    upload = Upload(
        user_id=user.id,
        conversation_id=conversation_id,
        artifact_id=artifact.id,
        filename=file.filename,
        content_type=file.content_type,
        storage_url=storage_url,
        size_bytes=len(content),
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    return UploadResponse(
        ok=True,
        upload_id=upload.id,
        public_id=upload.public_id,
        filename=upload.filename,
        content_type=upload.content_type,
        size_bytes=upload.size_bytes,
        storage_url=upload.storage_url,
    )
