from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.enums import MemoryType
from app.db.models import Memory


class MemoryService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_memory(
        self,
        user_id: int,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata_json: dict | None = None,
    ) -> Memory:
        memory = Memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            metadata_json=metadata_json,
        )
        self.db.add(memory)
        self.db.flush()
        return memory

    def recent_memories(self, user_id: int, limit: int = 20) -> list[Memory]:
        return (
            self.db.query(Memory)
            .filter(Memory.user_id == user_id)
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
            .limit(limit)
            .all()
        )
