from __future__ import annotations

import asyncio
import json
import time

import numpy as np
from sqlalchemy.orm import Session

from app.memory.embedding_service import embedding_service
from app.memory.memory_models import Memory

# Simple in-process embedding cache
# key: query string → value: (embedding_list, timestamp)
_EMBED_CACHE: dict[str, tuple[list, float]] = {}
_EMBED_CACHE_TTL = 300  # 5 minutes


def _cosine(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(va @ vb / (norm_a * norm_b))


async def _get_embedding(text: str) -> list:
    """Get embedding with cache — avoids re-embedding the same query."""
    now = time.monotonic()
    cached = _EMBED_CACHE.get(text)
    if cached:
        emb, ts = cached
        if now - ts < _EMBED_CACHE_TTL:
            return emb

    raw = await embedding_service.embed(text)
    emb = json.loads(raw)
    _EMBED_CACHE[text] = (emb, now)
    return emb


class MemoryRetriever:

    async def retrieve(
        self,
        db: Session,
        user_id: int,
        query: str,
        k: int = 5,
    ) -> list[str]:

        # Run embedding and DB fetch in parallel
        query_emb, memories = await asyncio.gather(
            _get_embedding(query),
            asyncio.to_thread(
                lambda: db.query(Memory)
                .filter(Memory.user_id == user_id)
                .order_by(Memory.salience_score.desc())
                .limit(200)  # cap at 200 to avoid loading entire table
                .all()
            ),
        )

        if not memories:
            return []

        scored = []
        for m in memories:
            if not m.embedding:
                continue
            try:
                emb = json.loads(m.embedding)
                score = _cosine(query_emb, emb)
                scored.append((score, m.content))
            except Exception:
                continue

        scored.sort(reverse=True)
        return [content for _, content in scored[:k]]


memory_retriever = MemoryRetriever()
