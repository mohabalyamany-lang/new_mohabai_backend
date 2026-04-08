from __future__ import annotations

from app.planner.semantic_planner import PlannerResult, SemanticPlanner
from app.planner.state_resolver import ResolvedConversationState


class PlannerService:
    def __init__(self) -> None:
        self.planner = SemanticPlanner()

    async def plan_turn(
        self,
        user_message: str,
        state: ResolvedConversationState,
        recent_messages: list[dict[str, str]] | None = None,
    ) -> PlannerResult:
        return await self.planner.plan(
            user_message=user_message,
            state=state,
            recent_messages=recent_messages,
        )
