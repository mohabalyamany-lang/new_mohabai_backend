from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Conversation, Turn
from app.planner.contracts import PlannerAction
from app.tools.base import BaseTool


class ImageAnalyzeTool(BaseTool):
    name = "image_analyze"

    async def execute(
        self,
        planner_action: PlannerAction,
        conversation: Conversation,
        turn: Turn,
        db: Session,
    ) -> dict[str, Any]:
        target = planner_action.tool_input.analysis_target or "Describe this image."
        artifact_id = planner_action.tool_input.artifact_id

        return {
            "ok": True,
            "tool": self.name,
            "result_type": "tool_result",
            "content": f"[Stub] Analyze image artifact {artifact_id}: {target}",
            "assistant_content_json": None,
            "artifacts": [],
            "tool_payload": {
                "artifact_id": artifact_id,
                "analysis_target": target,
            },
            "state_patch": {},
            "citations": [],
            "latency_ms": 0,
        }
