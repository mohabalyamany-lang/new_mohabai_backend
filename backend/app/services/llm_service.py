from __future__ import annotations

import json
import httpx
from app.config import get_settings

settings = get_settings()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Shared client with connection pooling — created once, reused across all requests
_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=5.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }


def _parse_response(data: dict) -> dict:
    """Safe response parser — never raises KeyError."""
    if "error" in data:
        raise ValueError(f"OpenRouter error: {data['error']}")
    choices = data.get("choices")
    if not choices:
        raise ValueError(f"No choices in response: {data}")
    return choices[0]["message"]


class LLMService:

    async def chat(self, messages: list[dict]) -> str:
        for attempt in range(3):
            try:
                r = await _client.post(
                    OPENROUTER_URL,
                    headers=_headers(),
                    json={
                        "model": settings.default_model,
                        "messages": messages,
                    },
                )
                r.raise_for_status()
                msg = _parse_response(r.json())
                return msg.get("content") or ""
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise
            except ValueError:
                raise

        return ""

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        for attempt in range(3):
            try:
                r = await _client.post(
                    OPENROUTER_URL,
                    headers=_headers(),
                    json={
                        "model": settings.default_model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                    },
                )
                r.raise_for_status()
                msg = _parse_response(r.json())

                if msg.get("tool_calls"):
                    call = msg["tool_calls"][0]
                    return {
                        "tool_call": {
                            "name": call["function"]["name"],
                            "arguments": json.loads(call["function"]["arguments"]),
                        }
                    }

                return {"content": msg.get("content") or ""}

            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise
            except ValueError:
                raise

        return {"content": ""}


llm_service = LLMService()
