from __future__ import annotations
from sqlalchemy.orm import Session
from app.db.models import Message
from app.services.artifact_service import ArtifactService
from app.services.context.context_builder import context_builder
from app.services.intent_engine import intent_engine
from app.services.llm_service import llm_service
from app.tools.tool_registry import tool_registry


class RuntimeOrchestrator:
    async def run_turn(
        self,
        *,
        db: Session,
        conversation_id: int,
        user_message: str,
    ) -> dict:
        last_image = ArtifactService.get_last_image(db, conversation_id)
        intent = intent_engine.detect(
            user_message,
            has_last_image=last_image is not None,
        )

        # ---------------- IMAGE GENERATE ----------------
        if intent.intent == "image_generate":
            result = await tool_registry.generate_image(user_message)
            artifact = ArtifactService.create_image_artifact(
                db=db,
                conversation_id=conversation_id,
                turn_id=None,
                prompt=user_message,
                effective_prompt=result.revised_prompt or user_message,
                storage_url=result.storage_url,
            )
            db.commit()
            return {
                "type": "image",
                "url": artifact.storage_url,
            }

        # ---------------- IMAGE EDIT ----------------
        if intent.intent == "image_edit" and last_image:
            result = await tool_registry.edit_image(
                instruction=user_message,
                parent_prompt=last_image.effective_prompt,
                parent_reference=last_image.storage_url,
            )
            artifact = ArtifactService.create_image_artifact(
                db=db,
                conversation_id=conversation_id,
                turn_id=None,
                prompt=user_message,
                effective_prompt=result.revised_prompt or user_message,
                storage_url=result.storage_url,
                parent_artifact_id=last_image.id,
            )
            db.commit()
            return {
                "type": "image",
                "url": artifact.storage_url,
            }

        # ---------------- CHAT ----------------
        bundle = await context_builder.build(
            db=db,
            conversation_id=conversation_id,
            user_message=user_message,
        )

        reply = await llm_service.chat(
            [
                {"role": m.role, "content": m.content}
                for m in bundle.messages
            ]
        )

        db.add(Message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
        ))
        db.add(Message(
            conversation_id=conversation_id,
            role="assistant",
            content=reply,
        ))
        db.commit()

        return {
            "type": "chat",
            "text": reply,
        }


runtime_orchestrator = RuntimeOrchestrator()
