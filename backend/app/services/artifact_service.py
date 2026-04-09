from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.enums import ArtifactType
from app.db.models import Artifact


class ArtifactService:

    # ------------------------------------------------
    # CREATE
    # ------------------------------------------------

    @staticmethod
    def create_image_artifact(
        db: Session,
        *,
        conversation_id: int,
        turn_id: int | None,
        prompt: str,
        effective_prompt: str,
        storage_url: str | None,
        parent_artifact_id: int | None = None,
    ) -> Artifact:

        artifact = Artifact(
            conversation_id=conversation_id,
            turn_id=turn_id,
            artifact_type=ArtifactType.IMAGE,
            prompt=prompt,
            effective_prompt=effective_prompt,
            storage_url=storage_url,
            parent_artifact_id=parent_artifact_id,
        )

        db.add(artifact)
        db.flush()
        return artifact

    # ------------------------------------------------
    # RETRIEVE LAST IMAGE
    # ------------------------------------------------

    @staticmethod
    def get_last_image(db: Session, conversation_id: int) -> Artifact | None:
        return (
            db.query(Artifact)
            .filter(
                Artifact.conversation_id == conversation_id,
                Artifact.artifact_type == ArtifactType.IMAGE,
            )
            .order_by(Artifact.id.desc())
            .first()
        )
