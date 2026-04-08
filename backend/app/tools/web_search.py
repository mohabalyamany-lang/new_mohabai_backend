from __future__ import annotations

import time
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Conversation, Turn
from app.planner.contracts import PlannerAction
from app.tools.base import BaseTool

settings = get_settings()


class WebSearchTool(BaseTool):
    name = "web"

    async def execute(
        self,
        planner_action: PlannerAction,
        conversation: Conversation,
        turn: Turn,
        db: Session,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        query = (planner_action.tool_input.query or "").strip()

        if not query:
            return {
                "ok": False,
                "tool": self.name,
                "result_type": "tool_result",
                "content": "I need a search query.",
                "assistant_content_json": None,
                "artifacts": [],
                "tool_payload": {},
                "state_patch": {},
                "citations": [],
                "latency_ms": 0,
                "error": "missing_query",
            }

        if not settings.serper_api_key:
            return {
                "ok": False,
                "tool": self.name,
                "result_type": "tool_result",
                "content": "Web search is not configured.",
                "assistant_content_json": None,
                "artifacts": [],
                "tool_payload": {"query": query},
                "state_patch": {},
                "citations": [],
                "latency_ms": 0,
                "error": "serper_not_configured",
            }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": settings.serper_api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": 5},
                )
                response.raise_for_status()
                data = response.json()

            lines: list[str] = [f'Web results for "{query}"']
            answer_box = data.get("answerBox") or {}
            answer = answer_box.get("answer") or answer_box.get("snippet")
            if answer:
                lines.append(f"\nDirect answer: {answer}")

            organic = data.get("organic") or []
            citations: list[dict[str, str]] = []
            for idx, item in enumerate(organic[:5], start=1):
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                lines.append(f"\n{idx}. {title}\n{snippet}\nSource: {link}")
                if link:
                    citations.append({"title": title, "url": link})

            content = "\n".join(lines).strip()
            latency_ms = int((time.perf_counter() - started) * 1000)

            return {
                "ok": True,
                "tool": self.name,
                "result_type": "tool_result",
                "content": content,
                "assistant_content_json": {
                    "type": "web_result",
                    "query": query,
                    "raw": data,
                },
                "artifacts": [
                    {
                        "artifact_type": "web_result",
                        "title": f"Search: {query}",
                        "storage_url": None,
                        "inline_data": None,
                        "prompt": query,
                        "effective_prompt": query,
                        "metadata_json": {"raw": data},
                    }
                ],
                "tool_payload": {"query": query, "raw": data},
                "state_patch": {
                    "active_mode": "live_info",
                    "pending_followup_kind": "live_info",
                    "pending_followup_target": query,
                    "allow_context_carryover": True,
                },
                "citations": citations,
                "latency_ms": latency_ms,
            }
        except httpx.HTTPError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return {
                "ok": False,
                "tool": self.name,
                "result_type": "tool_result",
                "content": "Web search failed.",
                "assistant_content_json": None,
                "artifacts": [],
                "tool_payload": {"query": query, "error": str(exc)},
                "state_patch": {},
                "citations": [],
                "latency_ms": latency_ms,
                "error": "web_search_failed",
            }
