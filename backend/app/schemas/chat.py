from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: int
    message: str = Field(min_length=1, max_length=20000)


class ChatResponse(BaseModel):
    ok: bool
    conversation_id: int
    turn_id: int | None
    assistant_text: str | None
    planner_action: dict
    planner_trace: list[dict]
    tool_result: dict | None = None
    error: str | None = None
