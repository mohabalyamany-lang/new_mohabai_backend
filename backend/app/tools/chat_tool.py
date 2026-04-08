from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Conversation, Turn
from app.planner.contracts import PlannerAction
from app.tools.base import BaseTool


class ChatTool(BaseTool):
    name = "chat"

    async def execute(
        self,
        planner_action: PlannerAction,
        conversation: Conversation,
        turn: Turn,
        db: Session,
    ) -> dict[str, Any]:
        reply = planner_action.reply_text or ""
        return {
            "ok": True,
            "tool": self.name,
            "result_type": "tool_result",
            "content": reply,
            "assistant_content_json": None,
            "artifacts": [],
            "tool_payload": {},
            "state_patch": {},
            "citations": [],
            "latency_ms": 0,
        }
