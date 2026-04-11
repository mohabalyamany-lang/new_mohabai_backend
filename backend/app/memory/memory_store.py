import json

from sqlalchemy.orm import Session

from app.db.models import Memory
from app.memory.embedding_service import embedding_service


class MemoryStore:

    async def save_memories(
        self,
        db: Session,
        user_id: int,
        memories: list[str],
    ):
        for mem in memories:
            embedding = await embedding_service.embed(mem)

            db.add(
                Memory(
                    user_id=user_id,
                    content=mem,
                    embedding=embedding,
                    importance=0.7,
                )
            )

        db.commit()


memory_store = MemoryStore()
