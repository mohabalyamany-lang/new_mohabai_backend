from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import Conversation, User
from app.db.session import get_db_session
from app.services.runtime_orchestrator import runtime_orchestrator

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(
    message: str,
    conversation_id: int,
    db: Session = Depends(get_db_session),
    user: User = Depends(get_current_user),
):
    convo = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
        .first()
    )

    result = await runtime_orchestrator.run_turn(
        db=db,
        conversation_id=convo.id,
        user_message=message,
    )

    return {
        "ok": True,
        "result": result,
    }
