"""
Memory Service — Production-grade with validation.

Rule: LLM suggests → system decides.
We NEVER blindly store what the LLM outputs.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.enums import MemoryType
from app.db.models import Memory


class MemoryValidationError(Exception):
    """Raised when memory content fails validation."""
    pass


class MemoryService:
    """Memory service with production validation guardrails."""

    # ━━━ Validation thresholds ━━━
    MIN_CONFIDENCE = 0.85
    MIN_CONTENT_LENGTH = 5
    MAX_CONTENT_LENGTH = 500

    def __init__(self, db: Session) -> None:
        self.db = db

    @staticmethod
    def should_store_memory(content: str, confidence: float = 0.0) -> tuple[bool, str]:
        """Validate whether memory should be stored.
        
        Returns:
            (should_store, reason)
        
        Rules:
        - Confidence must meet threshold (if provided)
        - Content must be meaningful length
        - Content must not be too long
        - Content must not be generic/fluff
        """
        if not content or not content.strip():
            return False, "empty_content"

        content = content.strip()

        if len(content) < MemoryService.MIN_CONTENT_LENGTH:
            return False, "too_short"

        if len(content) > MemoryService.MAX_CONTENT_LENGTH:
            return False, "too_long"

        if confidence > 0 and confidence < MemoryService.MIN_CONFIDENCE:
            return False, f"low_confidence_{confidence:.2f}"

        # Reject generic fluff that isn't worth remembering
        generic_patterns = [
            "i see",
            "okay",
            "ok",
            "thanks",
            "thank you",
            "got it",
            "sure",
            "cool",
            "nice",
            "great",
            "wow",
            "hmm",
            "interesting",
            "i understand",
            "makes sense",
        ]

        lower = content.lower()
        for pattern in generic_patterns:
            if lower == pattern:
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
        """Add a memory with validation.
        
        Args:
            user_id: The user this memory belongs to
            content: The memory content
            memory_type: Type of memory
            metadata_json: Optional structured metadata
            confidence: Planner confidence (0.0-1.0). If > 0, must meet threshold.
            skip_validation: If True, bypass validation (for system-initiated memories)
        
        Returns:
            Memory object if stored, None if validation failed
        """
        if not skip_validation:
            should_store, reason = self.should_store_memory(content, confidence)
            if not should_store:
                # Silently skip — don't store junk
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
        """Convenience method for planner-initiated memory writes.
        
        The planner suggests memory_write with content and confidence.
        This method validates before storing.
        """
        if not memory_content:
            return None

        if memory_operation != "store":
            return None

        return self.add_memory(
            user_id=user_id,
            content=memory_content,
            confidence=confidence,
            skip_validation=False,
        )

    def recent_memories(self, user_id: int, limit: int = 20) -> list[Memory]:
        return (
            self.db.query(Memory)
            .filter(Memory.user_id == user_id)
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
            .limit(limit)
            .all()
        )
