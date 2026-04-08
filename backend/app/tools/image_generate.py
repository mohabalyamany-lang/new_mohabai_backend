from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Artifact, Conversation, Turn
from app.planner.contracts import PlannerAction, PlannerIntent
from app.tools.base import BaseTool
from app.tools.image_analyze import ImageAnalyzeTool
from app.tools.image_provider_base import BaseImageProvider
from app.tools.image_provider_pollinations import PollinationsImageProvider


class ImageGenerateTool(BaseTool):
    name = "image"

    def __init__(
        self,
        image_analyze_tool: ImageAnalyzeTool,
        image_provider: BaseImageProvider | None = None,
    ) -> None:
        self.image_analyze_tool = image_analyze_tool
        self.image_provider = image_provider or PollinationsImageProvider()

    def _find_parent_artifact(self, db: Session, artifact_public_id: str | None) -> Artifact | None:
        if not artifact_public_id:
            return None
        return db.query(Artifact).filter(Artifact.public_id == artifact_public_id).first()

    async def execute(
        self,
        planner_action: PlannerAction,
        conversation: Conversation,
        turn: Turn,
        db: Session,
    ) -> dict:
        if planner_action.intent == PlannerIntent.IMAGE_QUESTION:
            return await self.image_analyze_tool.execute(
                planner_action=planner_action,
                conversation=conversation,
                turn=turn,
                db=db,
            )

        instruction = planner_action.tool_input.image_instruction or ""
        artifact_id = planner_action.tool_input.artifact_id
        parent_prompt = None
        parent_db_id = None

        parent_artifact = self._find_parent_artifact(db, artifact_id)
        if parent_artifact is not None:
            parent_prompt = parent_artifact.effective_prompt or parent_artifact.prompt
            parent_db_id = parent_artifact.id

        if planner_action.intent == PlannerIntent.IMAGE_EDIT:
            result = await self.image_provider.edit(
                instruction=instruction,
                parent_artifact_reference=artifact_id,
                parent_prompt=parent_prompt,
            )
        elif planner_action.intent == PlannerIntent.IMAGE_RETRY:
            seed_prompt = parent_prompt or instruction
            result = await self.image_provider.generate(seed_prompt)
        else:
            result = await self.image_provider.generate(instruction)

        if not result.ok:
            return {
                "ok": False,
                "tool": self.name,
                "result_type": "tool_result",
                "content": "Image generation failed.",
                "assistant_content_json": None,
                "artifacts": [],
                "tool_payload": {
                    "instruction": instruction,
                    "artifact_id": artifact_id,
                    "error": result.error,
                },
                "state_patch": {},
                "citations": [],
                "latency_ms": 0,
                "error": result.error or "image_generation_failed",
            }

        effective_prompt = result.revised_prompt or instruction or parent_prompt or ""

        return {
            "ok": True,
            "tool": self.name,
            "result_type": "tool_result",
            "content": f"Here is your image.",
            "assistant_content_json": {
                "type": "image_result",
                "intent": planner_action.intent.value,
                "instruction": instruction,
                "artifact_id": artifact_id,
                "storage_url": result.storage_url,
                "inline_data": result.inline_data,
            },
            "artifacts": [
                {
                    "artifact_type": "image",
                    "title": "Generated image",
                    "storage_url": result.storage_url,
                    "inline_data": result.inline_data,
                    "prompt": instruction or parent_prompt,
                    "effective_prompt": effective_prompt,
                    "parent_artifact_id": parent_db_id,
                    "metadata_json": {
                        "intent": planner_action.intent.value,
                        "provider": result.provider,
                        "model": result.model,
                        "source_artifact_public_id": artifact_id,
                    },
                }
            ],
            "tool_payload": {
                "instruction": instruction,
                "artifact_id": artifact_id,
                "storage_url": result.storage_url,
                "inline_data": result.inline_data,
                "provider": result.provider,
                "model": result.model,
                "effective_prompt": effective_prompt,
            },
            "state_patch": {
                "active_mode": "image_iteration",
                "pending_followup_kind": "image",
                "pending_followup_target": artifact_id,
                "allow_context_carryover": True,
            },
            "citations": [],
            "latency_ms": 0,
        }
