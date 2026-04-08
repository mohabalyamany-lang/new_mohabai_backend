from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    user_id: int
    title: str | None = None


class CreateConversationResponse(BaseModel):
    ok: bool
    conversation_id: int
    title: str | None


@router.post("", response_model=CreateConversationResponse)
async def create_conversation(
    payload: CreateConversationRequest,
    db: Session = Depends(get_db_session),
) -> CreateConversationResponse:
    service = ConversationService(db)
    conversation = service.create_conversation(
        user_id=payload.user_id,
        title=payload.title,
    )
    return CreateConversationResponse(
        ok=True,
        conversation_id=conversation.id,
        title=conversation.title,
    )
