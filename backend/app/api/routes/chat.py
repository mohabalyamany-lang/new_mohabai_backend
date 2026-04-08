from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Conversation
from app.db.session import get_db_session
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.orchestrator import ConversationOrchestrator
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    db: Session = Depends(get_db_session),
) -> ChatResponse:
    conversation = db.query(Conversation).filter(Conversation.id == payload.conversation_id).first()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    orchestrator = ConversationOrchestrator(db=db, tool_registry=ToolRegistry())
    result = await orchestrator.handle_turn(
        conversation=conversation,
        user_message=payload.message,
    )

    return ChatResponse(
        ok=result.ok,
        conversation_id=result.conversation_id,
        turn_id=result.turn_id,
        assistant_text=result.assistant_text,
        planner_action=result.planner_action,
        planner_trace=result.planner_trace,
        tool_result=result.tool_result,
        error=result.error,
    )