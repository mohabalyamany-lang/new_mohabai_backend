from __future__ import annotations

import httpx

from app.config import get_settings

settings = get_settings()


class LLMService:

    async def chat(self, messages: list[dict]) -> str:
        """
        Replace later with OpenAI / local model.
        Currently placeholder compatible layer.
        """

        # Example: OpenRouter compatible
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                },
                json={
                    "model": settings.default_model,
                    "messages": messages,
                },
            )

        data = r.json()
        return data["choices"][0]["message"]["content"]


llm_service = LLMService()
