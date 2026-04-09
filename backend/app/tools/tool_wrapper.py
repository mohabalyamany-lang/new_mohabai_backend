import asyncio


class ToolWrapper:

    async def execute(self, tool_fn, *args, **kwargs):

        for attempt in range(3):
            try:
                return await asyncio.wait_for(
                    tool_fn(*args, **kwargs),
                    timeout=12,
                )
            except Exception:
                if attempt == 2:
                    raise
