import json

from app.services.llm_service import llm_service
from app.reflection.reflection_prompts import REFLECTION_PROMPT


class ReflectionEngine:

    async def evaluate(
        self,
        user_message: str,
        assistant_reply: str,
    ):

        result = await llm_service.chat([
            {"role": "system", "content": REFLECTION_PROMPT},
            {
                "role": "user",
                "content": f"""
User message:
{user_message}

Assistant reply:
{assistant_reply}
"""
            },
        ])

        try:
            return json.loads(result)
        except Exception:
            return {"needs_revision": False}


reflection_engine = ReflectionEngine()
