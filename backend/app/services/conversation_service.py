from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.enums import ConversationMode
from app.db.models import Conversation, User


class ConversationService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        return self.db.query(Conversation).filter(Conversation.id == conversation_id).first()

    def create_conversation(self, user_id: int, title: str | None = None) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=title,
            active_mode=ConversationMode.NORMAL_CHAT,
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
