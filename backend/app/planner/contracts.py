from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlannerIntent(str, Enum):
    CHAT = "chat"
    WEB_SEARCH = "web_search"
    FILE_ANALYSIS = "file_analysis"
    IMAGE_GEN = "image_gen"
    IMAGE_EDIT = "image_edit"
    IMAGE_RETRY = "image_retry"
    IMAGE_QUESTION = "image_question"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"


class PlannerTool(str, Enum):
    CHAT = "chat"
    WEB = "web"
    FILE = "file"
    IMAGE = "image"
    MEMORY = "memory"


class PlannerDecision(str, Enum):
    ACT = "act"
    ASK = "ask"
    FALLBACK = "fallback"
    DEFER = "defer"


class ConversationMode(str, Enum):
    NORMAL_CHAT = "normal_chat"
    IMAGE_ITERATION = "image_iteration"
    LIVE_INFO = "live_info"
    FILE_ANALYSIS = "file_analysis"


class FollowupKind(str, Enum):
    IMAGE = "image"
    LIVE_INFO = "live_info"
    CHAT_STYLE = "chat_style"
    FILE = "file"


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str | None = None
    image_instruction: str | None = None
    analysis_target: str | None = None
    style_directive: str | None = None
    memory_operation: str | None = None
    memory_content: str | None = None
    artifact_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlannerStatePatch(BaseModel):
    active_mode: ConversationMode | None = None
    pending_intent: str | None = None
    pending_followup_kind: FollowupKind | None = None
    pending_followup_target: str | None = None
    allow_context_carryover: bool | None = None
    clear_pending_target: bool = False
    explained_flags: dict[str, bool] = Field(default_factory=dict)
    approval_flags: dict[str, bool] = Field(default_factory=dict)


class PlannerResolution(BaseModel):
    topic_switch: bool = False
    uses_last_artifact: bool = False
    uses_pending_target: bool = False
    clear_pending_target: bool = False
    notes: list[str] = Field(default_factory=list)


class PlannerAction(BaseModel):
    intent: PlannerIntent
    tool: PlannerTool
    decision: PlannerDecision = PlannerDecision.ACT
    tool_input: ToolInput = Field(default_factory=ToolInput)
    conversation_mode: ConversationMode = ConversationMode.NORMAL_CHAT
    state_patch: PlannerStatePatch = Field(default_factory=PlannerStatePatch)
    resolution: PlannerResolution = Field(default_factory=PlannerResolution)
    reply_text: str | None = None
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PlannerTraceEntry(BaseModel):
    stage: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class PlannerStep(BaseModel):
    intent: str
    tool: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    order: int = 0
    depends_on: list[int] | None = None


class PlannerResult(BaseModel):
    action: PlannerAction
    trace: list[PlannerTraceEntry] = Field(default_factory=list)
    steps: list[PlannerStep] = Field(default_factory=list)
    is_multi_intent: bool = False
