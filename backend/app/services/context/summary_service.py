from __future__ import annotations

from app.services.llm_service import llm_service


class SummaryService:

    async def summarize(self, messages: list[dict]) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the conversation briefly while preserving "
                    "facts, user goals, and ongoing tasks."
                ),
            },
            *messages,
        ]

        return await llm_service.chat(prompt)


summary_service = SummaryService()
