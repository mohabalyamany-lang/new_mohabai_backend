from fastapi import APIRouter, Depends
from app.runtime.runtime_controller import runtime_controller
from app.db.session import get_db

router = APIRouter()


@router.post("/chat")
async def chat(
    payload: dict,
    db=Depends(get_db),
):
    user_id = payload["user_id"]
    message = payload["message"]
    state = payload.get("state", {})

    reply = await runtime_controller.handle(
        db=db,
        user_id=user_id,
        message=message,
        conversation_state=state,
    )

    return {"reply": reply}
