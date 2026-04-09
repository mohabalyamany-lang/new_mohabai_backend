from __future__ import annotations

from sqlalchemy.orm import Session

from app.memory.memory_retriever import memory_retriever
from app.services.artifact_service import ArtifactService
from app.services.context.context_types import ContextBundle, ContextMessage
from app.services.context.conversation_memory_service import memory_service


class ContextBuilder:
    async def build(
        self,
        db: Session,
        conversation_id: int,
        user_message: str,
        user_id: int | None = None,
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

        # LONG-TERM MEMORY INJECTION
        if user_id is not None:
            memories = await memory_retriever.retrieve(
                db,
                user_id=user_id,
                query=user_message,
            )
            if memories:
                messages.append(
                    ContextMessage(
                        role="system",
                        content=(
                            "Known user facts:\n"
                            + "\n".join(f"- {m}" for m in memories)
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

        last_image = ArtifactService.get_last_image(db, conversation_id)

        return ContextBundle(
            messages=messages,
            last_image_prompt=(
                last_image.effective_prompt if last_image else None
            ),
        )


context_builder = ContextBuilder()
