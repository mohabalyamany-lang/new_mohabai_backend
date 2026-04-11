from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import User
from app.db.session import get_db_session as get_db
from app.runtime.runtime_controller import runtime_controller

router = APIRouter()


@router.post("/chat")
async def chat(
    payload: dict,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    message = payload.get("message", "")
    conversation_id = payload.get("conversation_id")
    state = payload.get("state", {"conversation_id": conversation_id})

    reply = await runtime_controller.handle(
        db=db,
        user_id=user.id,
        message=message,
        conversation_state=state,
    )
    return {"reply": reply}