from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Conversation, Turn
from app.planner.contracts import PlannerAction


class BaseTool(ABC):
    name: str

    @abstractmethod
    async def execute(
        self,
        planner_action: PlannerAction,
        conversation: Conversation,
        turn: Turn,
        db: Session,
    ) -> dict[str, Any]:
        raise NotImplementedError
