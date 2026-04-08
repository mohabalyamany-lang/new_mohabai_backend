from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.enums import ArtifactType
from app.db.models import Artifact


class ArtifactService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_last_artifact(self, conversation_id: int) -> Artifact | None:
        return (
            self.db.query(Artifact)
            .filter(Artifact.conversation_id == conversation_id)
            .order_by(Artifact.created_at.desc(), Artifact.id.desc())
            .first()
        )

    def create_artifact(
        self,
        conversation_id: int,
        turn_id: int,
        artifact_type: ArtifactType,
        title: str | None = None,
        storage_url: str | None = None,
        inline_data: str | None = None,
        prompt: str | None = None,
        effective_prompt: str | None = None,
        metadata_json: dict | None = None,
        source_tool_event_id: int | None = None,
        parent_artifact_id: int | None = None,
    ) -> Artifact:
        artifact = Artifact(
            conversation_id=conversation_id,
            turn_id=turn_id,
            source_tool_event_id=source_tool_event_id,
            parent_artifact_id=parent_artifact_id,
            artifact_type=artifact_type,
            title=title,
            storage_url=storage_url,
            inline_data=inline_data,
            prompt=prompt,
            effective_prompt=effective_prompt,
            metadata_json=metadata_json,
        )
        self.db.add(artifact)
        self.db.flush()
        return artifact
