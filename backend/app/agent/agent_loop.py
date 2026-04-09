from __future__ import annotations

import json

from app.agent.tool_schema import TOOLS
from app.agent.tool_registry import TOOL_REGISTRY
from app.planner.task_manager import task_manager
from app.planner.step_executor import step_executor
from app.services.llm_service import llm_service

MAX_STEPS = 6


class AgentLoop:

    async def _looks_like_goal(self, text: str) -> bool:
        trigger_words = ["build", "create", "plan", "help me", "make a"]
        return any(w in text.lower() for w in trigger_words)

    async def run(
        self,
        messages: list[dict],
        db=None,
        user_id: int | None = None,
        user_message: str = "",
    ) -> str:

        # Goal detection — start a task if this looks like a multi-step goal
        if db and user_id and user_message:
            if await self._looks_like_goal(user_message):
                await task_manager.start_task(db, user_id, user_message)

        # Inject active task context if one exists
        if db and user_id:
            task = task_manager.get_active_task(db, user_id)
            if task:
                step = step_executor.next_step(task)
                if step:
                    messages.append({
                        "role": "system",
                        "content": (
                            f"Current goal: {task.goal}\n"
                            f"Current step: {step['step']}"
                        ),
                    })

        for _ in range(MAX_STEPS):
            response = await llm_service.chat_with_tools(
                messages=messages,
                tools=TOOLS,
            )

            # Final answer — no tool call needed
            if "tool_call" not in response:
                return response["content"]

            tool_call = response["tool_call"]
            tool_name = tool_call["name"]
            args = tool_call["arguments"]

            tool_fn = TOOL_REGISTRY[tool_name]
            result = await tool_fn(**args)

            # Append reasoning state for next loop iteration
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
