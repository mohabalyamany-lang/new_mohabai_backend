from __future__ import annotations

import json
import httpx

from app.config import get_settings

settings = get_settings()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMService:

    async def chat(self, messages: list[dict]) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                OPENROUTER_URL,
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

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                },
                json={
                    "model": settings.default_model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                },
            )

        data = r.json()
        msg = data["choices"][0]["message"]

        if msg.get("tool_calls"):
            call = msg["tool_calls"][0]
            return {
                "tool_call": {
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]),
                }
            }

        return {"content": msg["content"]}


llm_service = LLMService()
