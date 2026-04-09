from app.services.llm_service import llm_service


EXTRACT_PROMPT = """
Extract permanent user facts or preferences.

Only store information that is:
- stable
- reusable later
- about the user or goals

Return JSON list of memories.
"""


class MemoryExtractor:

    async def extract(self, conversation_text: str):

        result = await llm_service.chat([
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": conversation_text},
        ])

        try:
            import json
            return json.loads(result)
        except Exception:
            return []


memory_extractor = MemoryExtractor()
