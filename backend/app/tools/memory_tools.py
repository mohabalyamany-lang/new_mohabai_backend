from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Conversation, Turn
from app.planner.contracts import PlannerAction
from app.tools.base import BaseTool


class MemoryTool(BaseTool):
    name = "memory"

    async def execute(
        self,
        planner_action: PlannerAction,
        conversation: Conversation,
        turn: Turn,
        db: Session,
    ) -> dict[str, Any]:
        operation = planner_action.tool_input.memory_operation or "none"
        content = planner_action.tool_input.memory_content

        return {
            "ok": True,
            "tool": self.name,
            "result_type": "tool_result",
            "content": f"[Stub] Memory operation: {operation}",
            "assistant_content_json": None,
            "artifacts": [],
            "tool_payload": {
                "memory_operation": operation,
                "memory_content": content,
            },
            "state_patch": {},
            "citations": [],
            "latency_ms": 0,
        }
