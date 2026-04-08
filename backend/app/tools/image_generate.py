from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Conversation, Turn
from app.planner.contracts import PlannerAction, PlannerIntent
from app.tools.base import BaseTool
from app.tools.image_analyze import ImageAnalyzeTool


class ImageGenerateTool(BaseTool):
    name = "image"

    def __init__(self, image_analyze_tool: ImageAnalyzeTool) -> None:
        self.image_analyze_tool = image_analyze_tool

    async def execute(
        self,
        planner_action: PlannerAction,
        conversation: Conversation,
        turn: Turn,
        db: Session,
    ) -> dict[str, Any]:
        if planner_action.intent == PlannerIntent.IMAGE_QUESTION:
            return await self.image_analyze_tool.execute(
                planner_action=planner_action,
                conversation=conversation,
                turn=turn,
                db=db,
            )

        instruction = planner_action.tool_input.image_instruction or ""
        artifact_id = planner_action.tool_input.artifact_id

        # Placeholder execution layer; provider adapter comes next.
        content = f"[Stub] Image action {planner_action.intent.value}: {instruction}".strip()

        return {
            "ok": True,
            "tool": self.name,
            "result_type": "tool_result",
            "content": content,
            "assistant_content_json": {
                "type": "image_result",
                "intent": planner_action.intent.value,
                "instruction": instruction,
                "artifact_id": artifact_id,
            },
            "artifacts": [
                {
                    "artifact_type": "image",
                    "title": "Generated image",
                    "storage_url": None,
                    "inline_data": None,
                    "prompt": instruction,
                    "effective_prompt": instruction,
                    "parent_artifact_id": None,
                    "metadata_json": {
                        "intent": planner_action.intent.value,
                        "artifact_id": artifact_id,
                    },
                }
            ],
            "tool_payload": {
                "instruction": instruction,
                "artifact_id": artifact_id,
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
