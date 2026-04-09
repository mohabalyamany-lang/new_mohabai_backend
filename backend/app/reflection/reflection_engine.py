from __future__ import annotations

import json

from app.reflection.reflection_prompts import REFLECTION_PROMPT
from app.services.llm_service import llm_service

# Shape of a valid reflection result
_DEFAULTS = {
    "needs_revision": False,
    "fix_strategy": "ok",
    "reason": "",
    "confidence": 1.0,
}


def _parse(raw: str) -> dict:
    """Parse and validate LLM reflection output. Never raises."""
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        return {**_DEFAULTS, **parsed}
    except Exception:
        return dict(_DEFAULTS)


class ReflectionEngine:
    async def evaluate(
        self,
        user_message: str,
        assistant_reply: str,
    ) -> dict:
        try:
            result = await llm_service.chat([
                {"role": "system", "content": REFLECTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User message:\n{user_message}\n\n"
                        f"Assistant reply:\n{assistant_reply}"
                    ),
                },
            ])
            return _parse(result)
        except Exception:
            # Reflection failure must never block a response
            return dict(_DEFAULTS)


reflection_engine = ReflectionEngine()
