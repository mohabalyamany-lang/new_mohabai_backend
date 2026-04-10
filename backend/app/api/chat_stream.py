from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.runtime.streaming_controller import streaming_controller
from app.db.session import get_db

router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(
    payload: dict,
    db=Depends(get_db),
):
    user_id = payload["user_id"]
    message = payload["message"]
    state = payload.get("state", {})

    async def event_generator():
        async for chunk in streaming_controller.stream(
            db=db,
            user_id=user_id,
            message=message,
            conversation_state=state,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
