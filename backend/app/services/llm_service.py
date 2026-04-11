from __future__ import annotations

import json

import httpx

from app.config import get_settings

settings = get_settings()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=5.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)


def _groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }


def _openrouter_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }


def _parse_response(data: dict) -> dict:
    if "error" in data:
        raise ValueError(f"API error: {data['error']}")
    choices = data.get("choices")
    if not choices:
        raise ValueError(f"No choices in response: {data}")
    return choices[0]["message"]


class LLMService:

    async def chat(self, messages: list[dict]) -> str:
        # Try Groq first
        if settings.groq_api_key:
            for attempt in range(2):
                try:
                    r = await _client.post(
                        GROQ_URL,
                        headers=_groq_headers(),
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": messages,
                            "max_tokens": 1024,
                        },
                    )
                    r.raise_for_status()
                    msg = _parse_response(r.json())
                    return msg.get("content") or ""
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt == 0:
                        continue
                    break
                except (httpx.TimeoutException, httpx.NetworkError):
                    break
                except ValueError:
                    break

        # Fallback to OpenRouter
        if settings.openrouter_api_key:
            for attempt in range(2):
                try:
                    r = await _client.post(
                        OPENROUTER_URL,
                        headers=_openrouter_headers(),
                        json={
                            "model": "google/gemma-3-12b-it:free",
                            "messages": messages,
                        },
                    )
                    r.raise_for_status()
                    msg = _parse_response(r.json())
                    return msg.get("content") or ""
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt == 0:
                        continue
                    break
                except (httpx.TimeoutException, httpx.NetworkError):
                    break
                except ValueError:
                    break

        return ""

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        # Try Groq first
        if settings.groq_api_key:
            for attempt in range(2):
                try:
                    r = await _client.post(
                        GROQ_URL,
                        headers=_groq_headers(),
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": messages,
                            "tools": tools,
                            "tool_choice": "auto",
                            "max_tokens": 1024,
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
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt == 0:
                        continue
                    break
                except (httpx.TimeoutException, httpx.NetworkError):
                    break
                except ValueError:
                    break

        # Fallback to OpenRouter
        if settings.openrouter_api_key:
            for attempt in range(2):
                try:
                    r = await _client.post(
                        OPENROUTER_URL,
                        headers=_openrouter_headers(),
                        json={
                            "model": "google/gemma-3-12b-it:free",
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
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt == 0:
                        continue
                    break
                except (httpx.TimeoutException, httpx.NetworkError):
                    break
                except ValueError:
                    break

        return {"content": ""}


llm_service = LLMService()