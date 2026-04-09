from app.services.model_service import model_service
from app.planner.semantic_planner import semantic_planner
from app.tools.tool_executor import tool_executor
from app.context.context_builder import context_builder


class ConversationOrchestrator:

    async def handle(
        self,
        db,
        user_id,
        message,
        conversation_state,
    ):
        # Build context
        messages = await context_builder.build(
            db,
            user_id,
            message,
        )

        # Plan
        plan = await semantic_planner.plan(
            message,
            conversation_state,
        )

        # Execute plan
        reply = await self._execute_plan(
            plan,
            messages,
        )

        return reply

    async def _execute_plan(self, plan, messages):

        if plan.requires_tool:
            tool_result = await tool_executor.execute(
                plan.tool_name,
                plan.tool_args,
            )

            messages.append({
                "role": "tool",
                "content": tool_result,
            })

        reply = await model_service.chat(messages)

        return reply


conversation_orchestrator = ConversationOrchestrator()
