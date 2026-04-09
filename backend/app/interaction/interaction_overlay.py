from __future__ import annotations
from dataclasses import dataclass


@dataclass
class InteractionOverlay:
    style: str = "normal"
    user_prefers_brief: bool = False
    user_prefers_detail: bool = False
    user_prefers_talkative: bool = False
    last_clarification_turn_id: int | None = None
