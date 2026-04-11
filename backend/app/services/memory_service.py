"""
Memory Service v2 — Production retrieval + validation.

Rule: LLM suggests → system decides (validation).
Rule: Retrieve before plan → inject into context (retrieval).
Rule: Never dump raw memory → limit aggressively.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.enums import MemoryType
from app.db.models import Memory


class MemoryValidationError(Exception):
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RETRIEVAL CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEMORY_TOP_K = 5
MEMORY_MAX_TOTAL_CHARS = 800
MEMORY_SEARCH_LIMIT = 50

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "i", "me", "my", "myself", "you", "your", "yours",
    "and", "or", "but", "if", "then", "so", "because",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "it", "its", "this", "that", "these", "those",
    "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "should", "may", "might",
    "not", "no", "nor", "very", "just", "also", "too",
    "what", "which", "who", "whom", "how", "when", "where", "why",
})


class MemoryService:
    MIN_CONFIDENCE = 0.85
    MIN_CONTENT_LENGTH = 5
    MAX_CONTENT_LENGTH = 500

    def __init__(self, db: Session) -> None:
        self.db = db

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDATION (LLM suggests → system decides)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def should_store_memory(content: str, confidence: float = 0.0) -> tuple[bool, str]:
        if not content or not content.strip():
            return False, "empty_content"

        content = content.strip()

        if len(content) < MemoryService.MIN_CONTENT_LENGTH:
            return False, "too_short"

        if len(content) > MemoryService.MAX_CONTENT_LENGTH:
            return False, "too_long"

        if confidence > 0 and confidence < MemoryService.MIN_CONFIDENCE:
            return False, f"low_confidence_{confidence:.2f}"

        generic_patterns = frozenset({
            "i see", "okay", "ok", "thanks", "thank you", "got it",
            "sure", "cool", "nice", "great", "wow", "hmm", "interesting",
            "i understand", "makes sense", "right", "yeah", "yep",
        })

        if content.lower() in generic_patterns:
            return False, "generic_response"

        return True, "valid"

    def add_memory(
        self,
        user_id: int,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata_json: dict | None = None,
        confidence: float = 0.0,
        skip_validation: bool = False,
    ) -> Memory | None:
        if not skip_validation:
            should_store, _ = self.should_store_memory(content, confidence)
            if not should_store:
                return None

        memory = Memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content.strip(),
            metadata_json=metadata_json,
        )
        self.db.add(memory)
        self.db.flush()
        return memory

    def add_memory_from_planner(
        self,
        user_id: int,
        memory_content: str | None,
        memory_operation: str | None,
        confidence: float = 0.0,
    ) -> Memory | None:
        if not memory_content or memory_operation != "store":
            return None

        return self.add_memory(
            user_id=user_id,
            content=memory_content,
            confidence=confidence,
            skip_validation=False,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RETRIEVAL (keyword-scoped, budget-limited)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase, split, remove stop words, deduplicate."""
        words = set(text.lower().split())
        return words - STOP_WORDS

    @staticmethod
    def _score_relevance(query_tokens: set[str], content: str) -> float:
        """Score a memory by keyword overlap with query.

        Uses proportional overlap: matching_words / query_words.
        This weights precision over recall — fewer but more relevant matches win.
        """
        content_tokens = MemoryService._tokenize(content)
        if not query_tokens:
            return 0.0

        overlap = query_tokens & content_tokens
        if not overlap:
            return 0.0

        return len(overlap) / len(query_tokens)

    def search_relevant(
        self,
        query: str,
        user_id: int,
        top_k: int = MEMORY_TOP_K,
        max_total_chars: int = MEMORY_MAX_TOTAL_CHARS,
    ) -> list[dict[str, Any]]:
        """Search memories by keyword relevance to query.

        Returns:
            List of {content, memory_type, relevance} dicts.
            Limited by top_k AND max_total_chars — whichever hits first.

        Upgrade path: replace _score_relevance with embedding cosine similarity.
        The interface stays identical.
        """
        if not query or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Fetch candidate memories (recent first, bounded)
        candidates = (
            self.db.query(Memory)
            .filter(Memory.user_id == user_id)
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
            .limit(MEMORY_SEARCH_LIMIT)
            .all()
        )

        if not candidates:
            return []

        # Score and sort
        scored: list[tuple[Memory, float]] = []
        for memory in candidates:
            score = self._score_relevance(query_tokens, memory.content)
            if score > 0:
                scored.append((memory, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Collect results respecting both limits
        results: list[dict[str, Any]] = []
        total_chars = 0

        for memory, score in scored:
            content_len = len(memory.content)

            if total_chars + content_len > max_total_chars:
                # Try to fit a truncated version if space remains
                remaining = max_total_chars - total_chars
                if remaining > 30:
                    results.append({
                        "content": memory.content[:remaining] + "...",
                        "memory_type": memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type),
                        "relevance": round(score, 2),
                    })
                    total_chars = max_total_chars
                break

            results.append({
                "content": memory.content,
                "memory_type": memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type),
                "relevance": round(score, 2),
            })
            total_chars += content_len

            if len(results) >= top_k:
                break

        return results

    def recent_memories(self, user_id: int, limit: int = 20) -> list[Memory]:
        return (
            self.db.query(Memory)
            .filter(Memory.user_id == user_id)
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
            .limit(limit)
            .all()
        )
