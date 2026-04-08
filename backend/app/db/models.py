from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.enums import (
    ArtifactType,
    ConversationMode,
    MemoryType,
    MessageRole,
    ToolName,
    ToolStatus,
    TurnStatus,
)


def utcnow() -> datetime:
    return datetime.utcnow()


# =========================
# User
# =========================
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        default=lambda: str(uuid4()),
        index=True,
    )
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(
        String(255), unique=True, index=True, nullable=True
    )
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")
    memories: Mapped[list["Memory"]] = relationship(back_populates="user")
    session_tokens: Mapped[list["SessionToken"]] = relationship(back_populates="user")
    uploads: Mapped[list["Upload"]] = relationship(back_populates="user")


# =========================
# Session Tokens (Auth)
# =========================
class SessionToken(Base):
    __tablename__ = "session_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36), unique=True, default=lambda: str(uuid4()), index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    access_token: Mapped[str] = mapped_column(Text, unique=True)
    refresh_token: Mapped[str] = mapped_column(Text, unique=True)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship(back_populates="session_tokens")


# =========================
# Conversation
# =========================
class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        default=lambda: str(uuid4()),
        index=True,
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    active_mode: Mapped[ConversationMode] = mapped_column(
        Enum(ConversationMode),
        default=ConversationMode.NORMAL_CHAT,
    )

    pending_followup_kind: Mapped[str | None] = mapped_column(String(50), nullable=True)
    pending_followup_target: Mapped[str | None] = mapped_column(Text, nullable=True)
    allow_context_carryover: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    user: Mapped["User"] = relationship(back_populates="conversations")
    turns: Mapped[list["Turn"]] = relationship(back_populates="conversation")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation")
    tool_events: Mapped[list["ToolEvent"]] = relationship(back_populates="conversation")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="conversation")
    uploads: Mapped[list["Upload"]] = relationship(back_populates="conversation")


# =========================
# Turn
# =========================
class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        default=lambda: str(uuid4()),
        index=True,
    )
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), index=True
    )
    sequence_number: Mapped[int] = mapped_column(Integer, index=True)

    user_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("messages.id"), nullable=True
    )
    assistant_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("messages.id"), nullable=True
    )

    planner_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    state_patch: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    status: Mapped[TurnStatus] = mapped_column(
        Enum(TurnStatus), default=TurnStatus.STARTED
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="turns")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="turn",
        foreign_keys="Message.turn_id",
    )
    tool_events: Mapped[list["ToolEvent"]] = relationship(back_populates="turn")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="turn")

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence_number",
            name="uq_turn_sequence_per_conversation",
        ),
    )


# =========================
# Message
# =========================
class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        default=lambda: str(uuid4()),
        index=True,
    )
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), index=True
    )
    turn_id: Mapped[int | None] = mapped_column(
        ForeignKey("turns.id"), nullable=True, index=True
    )

    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), index=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    turn: Mapped["Turn"] = relationship(back_populates="messages")


# =========================
# Tool Events
# =========================
class ToolEvent(Base):
    __tablename__ = "tool_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        default=lambda: str(uuid4()),
        index=True,
    )
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), index=True
    )
    turn_id: Mapped[int | None] = mapped_column(
        ForeignKey("turns.id"), nullable=True, index=True
    )

    tool_name: Mapped[ToolName] = mapped_column(Enum(ToolName), index=True)
    status: Mapped[ToolStatus] = mapped_column(
        Enum(ToolStatus), default=ToolStatus.PENDING, index=True
    )

    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="tool_events")
    turn: Mapped["Turn"] = relationship(back_populates="tool_events")


# =========================
# Artifact
# =========================
class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        default=lambda: str(uuid4()),
        index=True,
    )

    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), index=True
    )
    turn_id: Mapped[int | None] = mapped_column(
        ForeignKey("turns.id"), nullable=True, index=True
    )

    source_tool_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("tool_events.id"), nullable=True
    )
    parent_artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True
    )

    artifact_type: Mapped[ArtifactType] = mapped_column(
        Enum(ArtifactType), index=True
    )

    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    storage_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    inline_data: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    effective_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="artifacts")
    turn: Mapped["Turn"] = relationship(back_populates="artifacts")
    parent_artifact: Mapped["Artifact | None"] = relationship(remote_side=[id])


# =========================
# Uploads
# =========================
class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36), unique=True, default=lambda: str(uuid4()), index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True, index=True
    )
    artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True
    )

    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    storage_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped["User"] = relationship(back_populates="uploads")
    conversation: Mapped["Conversation"] = relationship(back_populates="uploads")


# =========================
# Memory
# =========================
class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    memory_type: Mapped[MemoryType] = mapped_column(
        Enum(MemoryType), index=True
    )

    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[str | None] = mapped_column(Text, nullable=True)

    salience_score: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    user: Mapped["User"] = relationship(back_populates="memories")
