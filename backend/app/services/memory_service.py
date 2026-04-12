"""
Memory Service — Validation layer over the existing memory/ module.

Does NOT duplicate storage or retrieval.
- Validation: unique (should_store_memory)
- Storage: delegates to app.memory.memory_store (stores with embeddings)
- Retrieval: delegates to app.memory.memory_retriever (embedding-based cosine similarity)

This is the correct integration point. The memory/ module handles
embeddings and vector math. This service handles policy (what to store).
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VALIDATION (LLM suggests → system decides)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Reuse limits from memory_retriever's k default
RETRIEVAL_TOP_K = 5
RETRIEVAL_MAX_CHARS = 800

GENERIC_RESPONSES = frozenset({
    "i see", "okay", "ok", "thanks", "thank you", "got it",
    "sure", "cool", "nice", "great", "wow", "hmm", "interesting",
    "i understand", "makes sense", "right", "yeah", "yep",
})


def should_store_memory(content: str, confidence: float = 0.0) -> tuple[bool, str]:
    """Validate whether a planner-suggested memory should be stored."""
    if not content or not content.strip():
        return False, "empty_content"

    content = content.strip()

    if len(content) < 5:
        return False, "too_short"

    if len(content) > 500:
        return False, "too_long"

    if confidence > 0 and confidence < 0.85:
        return False, f"low_confidence_{confidence:.2f}"

    if content.lower() in GENERIC_RESPONSES:
        return False, "generic_response"

    return True, "valid"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SERVICE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MemoryService:
    def __init__(self, db: Session) -> None:
        self.db = db

    async def add_memory_from_planner(
        self,
        user_id: int,
        memory_content: str | None,
        memory_operation: str | None,
        confidence: float = 0.0,
    ) -> Any:
        """Validate planner memory write, then delegate to memory_store."""
        if not memory_content or memory_operation != "store":
            return None

        valid, reason = should_store_memory(memory_content, confidence)
        if not valid:
            return None

        from app.memory.memory_store import memory_store

        try:
            await memory_store.save_memories(
                db=self.db,
                user_id=user_id,
                memories=[memory_content],
            )
            # memory_store does db.commit() internally
            # Return a lightweight object so the orchestrator can log success
            class _Stored:
                id = None
                content = memory_content
            return _Stored()
        except Exception:
            return None

    async def search_relevant(
        self,
        query: str,
        user_id: int,
        top_k: int = RETRIEVAL_TOP_K,
        max_total_chars: int = RETRIEVAL_MAX_CHARS,
    ) -> list[dict[str, Any]]:
        """Delegate to memory_retriever, then apply budget limits."""
        if not query or not query.strip():
            return []

        from app.memory.memory_retriever import memory_retriever

        try:
            raw_results = await memory_retriever.retrieve(
                db=self.db,
                user_id=user_id,
                query=query,
                k=top_k * 3,  # fetch more than needed, then trim by budget
            )
        except Exception:
            return []

        if not raw_results:
            return []

        # Apply character budget
        results: list[dict[str, Any]] = []
        total_chars = 0

        for i, content in enumerate(raw_results):
            content_len = len(content)

            if total_chars + content_len > max_total_chars:
                remaining = max_total_chars - total_chars
                if remaining > 30:
                    results.append({
                        "content": content[:remaining] + "...",
                        "memory_type": "episodic",
                        "relevance": round(1.0 - i / max(len(raw_results), 1), 2),
                    })
                break

            results.append({
                "content": content,
                "memory_type": "episodic",
                "relevance": round(1.0 - i / max(len(raw_results), 1), 2),
            })
            total_chars += content_len

            if len(results) >= top_k:
                break

        return results
