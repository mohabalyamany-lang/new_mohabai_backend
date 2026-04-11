from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings

settings = get_settings()

# Shared persistent clients
_openai_client = httpx.AsyncClient(
    timeout=httpx.Timeout(90.0, connect=6.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)
_groq_client = httpx.AsyncClient(
    timeout=httpx.Timeout(90.0, connect=6.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)
_openrouter_client = httpx.AsyncClient(
    timeout=httpx.Timeout(90.0, connect=6.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)

@dataclass(slots=True)
class ModelChunk:
    text: str

@dataclass(slots=True)
class ModelResponse:
    content: str
    raw: dict[str, Any]

class BaseChatProvider:
    name: str

    async def complete(
        self, messages: list[dict[str, str]], max_tokens: int = 1200
    ) -> ModelResponse:
        raise NotImplementedError

    async def stream(
        self, messages: list[dict[str, str]], max_tokens: int = 1200
    ) -> AsyncIterator[str]:
        raise NotImplementedError

class ZAIChatProvider(BaseChatProvider):
    name = "zai"

    def __init__(self, model: str = "glm-4.5-flash") -> None:
        self.model = model
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self._client = _openai_client

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.zai_api_key}",
            "Content-Type": "application/json",
        }

    async def complete(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> ModelResponse:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        response = await self._client.post(self.url, headers=self._headers(), json=payload)
        response.raise_for_status()
        data = response.json()
        return ModelResponse(content=data["choices"][0]["message"]["content"], raw=data)

    async def stream(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> AsyncIterator[str]:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": True}
        async with self._client.stream("POST", self.url, headers=self._headers(), json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "): continue
                data_str = line[6:].strip()
                if data_str == "[DONE]": break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta: yield delta
                except json.JSONDecodeError: continue

class OpenAIChatProvider(BaseChatProvider):
    name = "openai"
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.url = "https://api.openai.com/v1/chat/completions"
        self._client = _openai_client

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}

    async def complete(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> ModelResponse:
        payload = {"model": self.model, "messages": messages, "temperature": 0.7, "max_tokens": max_tokens, "stream": False}
        response = await self._client.post(self.url, headers=self._headers(), json=payload)
        response.raise_for_status()
        data = response.json()
        return ModelResponse(content=data["choices"][0]["message"]["content"], raw=data)

    async def stream(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> AsyncIterator[str]:
        payload = {"model": self.model, "messages": messages, "temperature": 0.7, "max_tokens": max_tokens, "stream": True}
        async with self._client.stream("POST", self.url, headers=self._headers(), json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "): continue
                data_str = line[6:].strip()
                if data_str == "[DONE]": break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta: yield delta
                except json.JSONDecodeError: continue

class GroqChatProvider(BaseChatProvider):
    name = "groq"
    def __init__(self, model: str = "llama-3.1-8b-instant") -> None:
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self._client = _groq_client

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"}

    async def complete(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> ModelResponse:
        payload = {"model": self.model, "messages": messages, "temperature": 0.6, "max_tokens": max_tokens, "stream": False}
        response = await self._client.post(self.url, headers=self._headers(), json=payload)
        response.raise_for_status()
        data = response.json()
        return ModelResponse(content=data["choices"][0]["message"]["content"], raw=data)

    async def stream(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> AsyncIterator[str]:
        payload = {"model": self.model, "messages": messages, "temperature": 0.6, "max_tokens": max_tokens, "stream": True}
        async with self._client.stream("POST", self.url, headers=self._headers(), json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "): continue
                data_str = line[6:].strip()
                if data_str == "[DONE]": break
                data = json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta: yield delta

class OpenRouterChatProvider(BaseChatProvider):
    name = "openrouter"
    def __init__(self, model: str = "google/gemma-3-12b-it:free") -> None:
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self._client = _openrouter_client

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mohabai.vercel.app",
            "X-Title": "Mohab AI",
        }

    async def complete(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> ModelResponse:
        payload = {"model": self.model, "messages": messages, "temperature": 0.6, "max_tokens": max_tokens, "stream": False}
        response = await self._client.post(self.url, headers=self._headers(), json=payload)
        response.raise_for_status()
        data = response.json()
        return ModelResponse(content=data["choices"][0]["message"]["content"], raw=data)

    async def stream(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> AsyncIterator[str]:
        payload = {"model": self.model, "messages": messages, "temperature": 0.6, "max_tokens": max_tokens, "stream": True}
        async with self._client.stream("POST", self.url, headers=self._headers(), json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "): continue
                data_str = line[6:].strip()
                if data_str == "[DONE]": break
                data = json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta: yield delta

class ModelService:
    def __init__(self) -> None:
        self.providers: list[BaseChatProvider] = []
        # Priority ordering
        if settings.zai_api_key: self.providers.append(ZAIChatProvider())
        if settings.openai_api_key: self.providers.append(OpenAIChatProvider())
        if settings.groq_api_key: self.providers.append(GroqChatProvider())
        if settings.openrouter_api_key: self.providers.append(OpenRouterChatProvider())
        if not self.providers: raise RuntimeError("No chat model providers are configured")

    async def chat(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> str:
        all_errors: list[str] = []
        for provider in self.providers:
            try:
                response = await provider.complete(messages=messages, max_tokens=max_tokens)
                return response.content
            except Exception as exc:
                all_errors.append(f"{provider.name}: {exc}")
                continue
        raise RuntimeError(f"All model providers failed: {'; '.join(all_errors)}")

    async def stream(self, messages: list[dict[str, str]], max_tokens: int = 1200) -> AsyncIterator[str]:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                async for chunk in provider.stream(messages=messages, max_tokens=max_tokens):
                    yield chunk
                return
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"All model providers failed: {last_error}")

model_service = ModelService()
