from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Conversation, Turn
from app.planner.contracts import PlannerAction
from app.tools.base import BaseTool


class FileAnalyzeTool(BaseTool):
    name = "file"

    async def execute(
        self,
        planner_action: PlannerAction,
        conversation: Conversation,
        turn: Turn,
        db: Session,
    ) -> dict[str, Any]:
        target = planner_action.tool_input.analysis_target or "Analyze uploaded files."
        return {
            "ok": True,
            "tool": self.name,
            "result_type": "tool_result",
            "content": f"[Stub] {target}",
            "assistant_content_json": None,
            "artifacts": [],
            "tool_payload": {"analysis_target": target},
            "state_patch": {
                "active_mode": "file_analysis",
                "pending_followup_kind": "file",
                "allow_context_carryover": True,
            },
            "citations": [],
            "latency_ms": 0,
        }
