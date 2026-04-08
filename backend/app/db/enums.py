from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TurnStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolName(str, Enum):
    CHAT = "chat"
    WEB = "web"
    IMAGE = "image"
    FILE = "file"
    MEMORY = "memory"


class ToolStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class ArtifactType(str, Enum):
    IMAGE = "image"
    FILE = "file"
    WEB_RESULT = "web_result"
    TEXT = "text"


class MemoryType(str, Enum):
    PROFILE = "profile"
    EPISODIC = "episodic"
    WORKING = "working"


class ConversationMode(str, Enum):
    NORMAL_CHAT = "normal_chat"
    IMAGE_ITERATION = "image_iteration"
    LIVE_INFO = "live_info"
    FILE_ANALYSIS = "file_analysis"
