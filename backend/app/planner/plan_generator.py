import json
from app.services.llm_service import llm_service
from app.planner.planning_prompts import PLAN_PROMPT


class PlanGenerator:

    async def create_plan(self, goal: str):

        response = await llm_service.chat([
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": goal},
        ])

        return json.loads(response)


plan_generator = PlanGenerator()
