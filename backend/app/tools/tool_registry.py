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
                "content": "Here is your image.",
                "artifacts": [{
                    "artifact_type": "image",
                    "title": "Generated Image",
                    "storage_url": result.get("url"),
                    "prompt": tool_input.image_instruction,
                    "effective_prompt": tool_input.image_instruction,
                }],
                "state_patch": {
                    "pending_followup_kind": "image",
                    "allow_context_carryover": False,
                }
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
                "content": "Here is the updated image.",
                "artifacts": [{
                    "artifact_type": "image",
                    "title": "Edited Image",
                    "storage_url": result.get("url"),
                    "prompt": tool_input.image_instruction,
                    "effective_prompt": tool_input.image_instruction,
                }],
                "state_patch": {
                    "pending_followup_kind": "image",
                    "allow_context_carryover": False,
                }
            }

        return {"ok": False, "error": "Unsupported image action"}


# ---------------- WEB TOOL ----------------

class WebTool(BaseTool):
    name = "web"

    async def execute(self, planner_action, conversation, turn, db):
        query = planner_action.tool_input.query

        if not query:
            return {"ok": False, "error": "Missing query"}

        results = await web_search(query)

        return {
            "ok": True,
            "content": results.get("content", ""),
            "citations": results.get("citations", []),
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
        return {
            "ok": True,
            "content": None,
        }


# ---------------- REGISTRY ----------------

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

        # Register tools here
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
