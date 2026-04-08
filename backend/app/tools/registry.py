from __future__ import annotations

from app.tools.base import BaseTool
from app.tools.chat_tool import ChatTool
from app.tools.file_analyze import FileAnalyzeTool
from app.tools.image_analyze import ImageAnalyzeTool
from app.tools.image_generate import ImageGenerateTool
from app.tools.memory_tools import MemoryTool
from app.tools.web_search import WebSearchTool


class ToolRegistry:
    def __init__(self) -> None:
        image_analyze = ImageAnalyzeTool()
        self._tools: dict[str, BaseTool] = {
            "chat": ChatTool(),
            "web": WebSearchTool(),
            "file": FileAnalyzeTool(),
            "image": ImageGenerateTool(image_analyze_tool=image_analyze),
            "memory": MemoryTool(),
        }

    def get(self, tool_name: str) -> BaseTool:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Unknown tool: {tool_name}")
        return tool
