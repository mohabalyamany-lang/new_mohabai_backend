import json
import numpy as np

from sqlalchemy.orm import Session
from app.memory.memory_models import Memory
from app.memory.embedding_service import embedding_service


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class MemoryRetriever:

    async def retrieve(
        self,
        db: Session,
        user_id: int,
        query: str,
        k: int = 5,
    ):
        query_emb = json.loads(await embedding_service.embed(query))

        memories = db.query(Memory).filter(
            Memory.user_id == user_id
        ).all()

        scored = []

        for m in memories:
            emb = json.loads(m.embedding)
            scored.append((cosine(query_emb, emb), m.content))

        scored.sort(reverse=True)

        return [m for _, m in scored[:k]]


memory_retriever = MemoryRetriever()
