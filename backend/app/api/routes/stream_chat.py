from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.models import Conversation
from app.db.session import get_db_session
from app.schemas.chat import ChatRequest
from app.services.model_service import ModelService
from app.services.orchestrator import ConversationOrchestrator
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/stream-chat", tags=["chat"])


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def stream_chat_endpoint(
    payload: ChatRequest,
    db: Session = Depends(get_db_session),
) -> StreamingResponse:
    conversation = db.query(Conversation).filter(Conversation.id == payload.conversation_id).first()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    orchestrator = ConversationOrchestrator(db=db, tool_registry=ToolRegistry())

    async def event_stream() -> AsyncIterator[str]:
        result = await orchestrator.handle_turn(
            conversation=conversation,
            user_message=payload.message,
        )

        yield sse_event(
            {
                "type": "meta",
                "conversation_id": result.conversation_id,
                "turn_id": result.turn_id,
                "planner_action": result.planner_action,
                "planner_trace": result.planner_trace,
            }
        )

        if not result.ok:
            yield sse_event({"type": "error", "error": result.error or "Unknown error"})
            yield sse_event({"type": "done"})
            return

        text = result.assistant_text or ""
        if text:
            yield sse_event({"type": "content", "content": text})

        if result.tool_result:
            yield sse_event({"type": "tool_result", "tool_result": result.tool_result})

        yield sse_event({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
