from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Message


class ConversationMemoryService:

    MAX_RECENT_MESSAGES = 12

    def load_recent_messages(
        self,
        db: Session,
        conversation_id: int,
    ):
        msgs = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(self.MAX_RECENT_MESSAGES)
            .all()
        )

        return list(reversed(msgs))


memory_service = ConversationMemoryService()
