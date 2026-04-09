from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ContextMessage:
    role: str
    content: str


@dataclass
class ContextBundle:
    messages: List[ContextMessage]
    last_image_prompt: Optional[str] = None
    conversation_summary: Optional[str] = None
