from pydantic import BaseModel


class ConversationResponse(BaseModel):
    ok: bool
    conversation_id: int
    public_id: str
    title: str | None = None
