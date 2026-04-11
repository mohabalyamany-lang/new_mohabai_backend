from app.tools.registry import ToolRegistry

_registry = ToolRegistry()


class ToolExecutor:

    async def execute(self, tool_name: str, tool_args: dict) -> str:
        try:
            tool = _registry.get(tool_name)
            result = await tool.execute(**tool_args)
            return str(result)
        except Exception as exc:
            return f"Tool execution failed: {exc}"


tool_executor = ToolExecutor()