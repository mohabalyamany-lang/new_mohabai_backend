from __future__ import annotations

from sqlalchemy.orm import Session

from app.services.context.context_types import (
    ContextBundle,
    ContextMessage,
)
from app.services.context.conversation_memory_service import memory_service
from app.services.artifact_service import ArtifactService


class ContextBuilder:

    async def build(
        self,
        db: Session,
        conversation_id: int,
        user_message: str,
    ) -> ContextBundle:

        recent = memory_service.load_recent_messages(
            db,
            conversation_id,
        )

        messages = []

        # SYSTEM CORE BEHAVIOR
        messages.append(
            ContextMessage(
                role="system",
                content=(
                    "You are Mohab AI, an intelligent assistant capable of "
                    "conversation, reasoning, and tool usage. "
                    "Respond naturally and infer user intent."
                ),
            )
        )

        # HISTORY
        for m in recent:
            messages.append(
                ContextMessage(
                    role=m.role,
                    content=m.content,
                )
            )

        # CURRENT USER MESSAGE
        messages.append(
            ContextMessage(
                role="user",
                content=user_message,
            )
        )

        last_image = ArtifactService.get_last_image(
            db,
            conversation_id,
        )

        return ContextBundle(
            messages=messages,
            last_image_prompt=(
                last_image.effective_prompt if last_image else None
            ),
        )


context_builder = ContextBuilder()
