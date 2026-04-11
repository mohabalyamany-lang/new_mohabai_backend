from __future__ import annotations

from typing import Any, Dict

from app.tools.image_provider_pollinations import PollinationsImageProvider
from app.services.web_search_service import web_search


class BaseTool:
    name: str

    async def execute(self, **kwargs) -> dict:
        raise NotImplementedError


# ---------------- IMAGE TOOL ----------------

class ImageTool(BaseTool):
    name = "image"

    def __init__(self):
        self.provider = PollinationsImageProvider()

    async def execute(self, planner_action, conversation, turn, db):
        tool_input = planner_action.tool_input

        # IMAGE GENERATION
        if planner_action.intent.value == "image_gen":
            result = await self.provider.generate(tool_input.image_instruction)

            return {
                "ok": True,
                "result_type": "image_result",
                "content": "Image generated successfully.",
                "artifacts": [{
                    "artifact_type": "image",
                    "title": "Generated Image",
                    "storage_url": result.get("url"),
                    "prompt": tool_input.image_instruction,
                    "effective_prompt": tool_input.image_instruction,
                }],
                "structured": {
                    "url": result.get("url"),
                    "prompt": tool_input.image_instruction,
                },
                "state_patch": {
                    "pending_followup_kind": "image",
                    "allow_context_carryover": False,
                },
            }

        # IMAGE EDIT
        if planner_action.intent.value == "image_edit":
            result = await self.provider.edit(
                instruction=tool_input.image_instruction,
                parent_prompt=tool_input.metadata.get("parent_prompt"),
                parent_artifact_reference=tool_input.artifact_id,
            )

            return {
                "ok": True,
                "result_type": "image_result",
                "content": "Image edited successfully.",
                "artifacts": [{
                    "artifact_type": "image",
                    "title": "Edited Image",
                    "storage_url": result.get("url"),
                    "prompt": tool_input.image_instruction,
                    "effective_prompt": tool_input.image_instruction,
                }],
                "structured": {
                    "url": result.get("url"),
                    "prompt": tool_input.image_instruction,
                    "edit_parent": tool_input.artifact_id,
                },
                "state_patch": {
                    "pending_followup_kind": "image",
                    "allow_context_carryover": False,
                },
            }

        return {"ok": False, "result_type": "error", "content": None, "error": "Unsupported image action"}


# ---------------- WEB TOOL ----------------

class WebTool(BaseTool):
    name = "web"

    async def execute(self, planner_action, conversation, turn, db):
        # ━── Use previous_output if injected by execution engine ━──
        tool_input = planner_action.tool_input
        query = tool_input.query

        # If no explicit query but we have previous_output, 
        # the user likely said "summarize it" after a search
        previous_output = tool_input.model_dump().get("previous_output")

        if not query and previous_output:
            # This step is consuming previous search results
            return {
                "ok": True,
                "result_type": "web_result",
                "content": previous_output,
                "structured": {"source": "previous_step"},
                "citations": tool_input.model_dump().get("previous_citations", []),
                "state_patch": {
                    "pending_followup_kind": "live_info",
                    "pending_followup_target": query,
                    "allow_context_carryover": True,
                },
            }

        if not query:
            return {
                "ok": False,
                "result_type": "error",
                "content": None,
                "error": "Missing query",
            }

        results = await web_search(query)
        content = results.get("content", "")
        citations = results.get("citations", [])

        return {
            "ok": True,
            "result_type": "web_result",
            "content": content,
            "structured": {
                "query": query,
                "result_count": len(citations),
            },
            "citations": citations,
            "state_patch": {
                "pending_followup_kind": "live_info",
                "pending_followup_target": query,
                "allow_context_carryover": True,
            },
        }


# ---------------- CHAT TOOL ----------------

class ChatTool(BaseTool):
    name = "chat"

    async def execute(self, planner_action, conversation, turn, db):
        tool_input = planner_action.tool_input
        previous_output = tool_input.model_dump().get("previous_output")

        # If chat step has previous_output, pass it through
        # The execution engine's synthesis will handle the actual LLM call
        return {
            "ok": True,
            "result_type": "chat_result",
            "content": previous_output or None,
            "structured": {"source": "previous_step" if previous_output else "direct"},
        }


# ---------------- REGISTRY ----------------

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.register(ImageTool())
        self.register(WebTool())
        self.register(ChatTool())

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    def get(self, tool_name: str) -> BaseTool:
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        return tool


tool_registry = ToolRegistry()
