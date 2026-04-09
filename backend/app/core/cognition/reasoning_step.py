from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ReasoningStep:
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
