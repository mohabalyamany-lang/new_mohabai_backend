from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db_session
from app.db.models import Conversation, User
from app.schemas.conversations import ConversationResponse

router = APIRouter()


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    db: Session = Depends(get_db_session),
    user: User = Depends(get_current_user),
):
    convos = db.query(Conversation).filter(
        Conversation.user_id == user.id
    ).order_by(Conversation.updated_at.desc()).all()
    return [
        ConversationResponse(
            ok=True,
            conversation_id=c.id,
            public_id=c.public_id,
            title=c.title,
        )
        for c in convos
    ]


@router.patch("/{conversation_id}")
async def rename_conversation(
    conversation_id: int,
    payload: dict,
    db: Session = Depends(get_db_session),
    user: User = Depends(get_current_user),
):
    convo = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    ).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Not found")
    convo.title = payload.get("title", convo.title)
    db.commit()
    return {"ok": True}


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db_session),
    user: User = Depends(get_current_user),
):
    convo = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    ).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(convo)
    db.commit()
    return {"ok": True}
