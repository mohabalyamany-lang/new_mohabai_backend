from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ResponseStyle = Literal[
    "normal",
    "brief",
    "detailed",
    "technical",
    "casual",
    "talkative",
]


@dataclass
class InteractionPolicy:
    style: ResponseStyle = "normal"
    ask_clarifying_when_ambiguous: bool = True
    preserve_user_tone: bool = True
    avoid_unnecessary_refusals: bool = True
    avoid_capability_narration: bool = True
    keep_context_tight: bool = False
