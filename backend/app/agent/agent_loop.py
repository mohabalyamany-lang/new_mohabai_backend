from __future__ import annotations

import json

from app.agent.tool_schema import TOOLS
from app.agent.tool_registry import TOOL_REGISTRY
from app.services.llm_service import llm_service


MAX_STEPS = 6


class AgentLoop:

    async def run(self, messages: list[dict]) -> str:

        for _ in range(MAX_STEPS):

            response = await llm_service.chat_with_tools(
                messages=messages,
                tools=TOOLS,
            )

            # Final answer
            if "tool_call" not in response:
                return response["content"]

            tool_call = response["tool_call"]
            tool_name = tool_call["name"]
            args = tool_call["arguments"]

            tool_fn = TOOL_REGISTRY[tool_name]

            result = await tool_fn(**args)

            # append reasoning state
            messages.append({
                "role": "assistant",
                "tool_call": tool_call,
            })

            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
            })

        return "I could not complete the task."
        

agent_loop = AgentLoop()
