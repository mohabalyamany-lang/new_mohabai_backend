from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import Conversation, User
from app.db.session import get_db_session
from app.services.orchestrator import ConversationOrchestrator
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(
    message: str,
    conversation_id: int,
    db: Session = Depends(get_db_session),
    user: User = Depends(get_current_user),
):
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
        .first()
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    orchestrator = ConversationOrchestrator(
        db=db,
        tool_registry=ToolRegistry(),
    )

    result = await orchestrator.handle_turn(
        conversation=conversation,
        user_message=message,
        user_id=user.id,
    )

    return {
        "ok": result.ok,
        "conversation_id": result.conversation_id,
        "turn_id": result.turn_id,
        "result": {
            "type": "chat",
            "text": result.assistant_text,
            "tool_result": result.tool_result,
            "planner_action": result.planner_action,
        },
        "error": result.error,
    }
