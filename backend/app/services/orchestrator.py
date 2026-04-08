from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.enums import ArtifactType, MessageRole, ToolName, ToolStatus, TurnStatus
from app.db.models import Artifact, Conversation, Message, ToolEvent, Turn
from app.planner.contracts import PlannerTool
from app.planner.state_resolver import ResolvedConversationState
from app.services.model_service import ModelService
from app.services.planner_service import PlannerService
from app.tools.registry import ToolRegistry


@dataclass(slots=True)
class OrchestratorResult:
    ok: bool
    conversation_id: int
    turn_id: int | None
    assistant_text: str | None
    planner_action: dict[str, Any]
    planner_trace: list[dict[str, Any]]
    tool_result: dict[str, Any] | None = None
    error: str | None = None


class ConversationOrchestrator:
    def __init__(self, db: Session, tool_registry: ToolRegistry) -> None:
        self.db = db
        self.tool_registry = tool_registry
        self.planner_service = PlannerService()
        self.model_service = ModelService()

    def _load_recent_messages(self, conversation_id: int, limit: int = 12) -> list[Message]:
        rows = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))

    def _load_last_artifact(self, conversation_id: int) -> Artifact | None:
        return (
            self.db.query(Artifact)
            .filter(Artifact.conversation_id == conversation_id)
            .order_by(Artifact.created_at.desc(), Artifact.id.desc())
            .first()
        )

    def _resolve_state(self, conversation: Conversation) -> ResolvedConversationState:
        last_artifact = self._load_last_artifact(conversation.id)
        last_messages = self._load_recent_messages(conversation.id, limit=6)

        last_user_message = None
        last_assistant_message = None
        for msg in reversed(last_messages):
            if msg.role == MessageRole.USER and last_user_message is None:
                last_user_message = msg.content
            if msg.role == MessageRole.ASSISTANT and last_assistant_message is None:
                last_assistant_message = msg.content

        return ResolvedConversationState(
            active_mode=conversation.active_mode.value,
            pending_followup_kind=conversation.pending_followup_kind,
            pending_followup_target=conversation.pending_followup_target,
            allow_context_carryover=conversation.allow_context_carryover,
            last_artifact_type=last_artifact.artifact_type.value if last_artifact else None,
            last_artifact_id=last_artifact.public_id if last_artifact else None,
            last_artifact_prompt=last_artifact.effective_prompt if last_artifact else None,
            last_user_message=last_user_message,
            last_assistant_message=last_assistant_message,
            has_files=False,
            recent_turn_count=self.db.query(func.count(Turn.id)).filter(Turn.conversation_id == conversation.id).scalar() or 0,
        )

    def _create_turn(self, conversation_id: int) -> Turn:
        sequence_number = (
            self.db.query(func.coalesce(func.max(Turn.sequence_number), 0))
            .filter(Turn.conversation_id == conversation_id)
            .scalar()
            or 0
        ) + 1

        turn = Turn(
            conversation_id=conversation_id,
            sequence_number=sequence_number,
            status=TurnStatus.STARTED,
        )
        self.db.add(turn)
        self.db.flush()
        return turn

    def _save_message(
        self,
        conversation_id: int,
        turn_id: int,
        role: MessageRole,
        content: str | None,
        content_json: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=role,
            content=content,
            content_json=content_json,
        )
        self.db.add(message)
        self.db.flush()
        return message

    def _save_tool_event(
        self,
        conversation_id: int,
        turn_id: int,
        tool_name: ToolName,
        status: ToolStatus,
        input_text: str | None,
        output_text: str | None,
        payload_json: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> ToolEvent:
        event = ToolEvent(
            conversation_id=conversation_id,
            turn_id=turn_id,
            tool_name=tool_name,
            status=status,
            input_text=input_text,
            output_text=output_text,
            payload_json=payload_json,
            latency_ms=latency_ms,
        )
        self.db.add(event)
        self.db.flush()
        return event

    def _apply_state_patch(self, conversation: Conversation, state_patch: dict[str, Any]) -> None:
        if not state_patch:
            return

        if state_patch.get("clear_pending_target"):
            conversation.pending_followup_kind = None
            conversation.pending_followup_target = None
            conversation.allow_context_carryover = False

        active_mode = state_patch.get("active_mode")
        if active_mode:
            conversation.active_mode = active_mode

        if "pending_followup_kind" in state_patch:
            conversation.pending_followup_kind = state_patch.get("pending_followup_kind")

        if "pending_followup_target" in state_patch:
            conversation.pending_followup_target = state_patch.get("pending_followup_target")

        if "allow_context_carryover" in state_patch and state_patch.get("allow_context_carryover") is not None:
            conversation.allow_context_carryover = bool(state_patch.get("allow_context_carryover"))

    def _build_chat_messages(
        self,
        conversation: Conversation,
        recent_messages: list[Message],
        user_message: str,
        tool_result: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are Mohab AI, a capable conversational assistant. "
                    "Do not describe missing capabilities when tools were already used. "
                    "Answer naturally, directly, and consistently with the conversation."
                ),
            }
        ]

        for msg in recent_messages[-10:]:
            if msg.content:
                messages.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        if tool_result and tool_result.get("content"):
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "A tool has produced the following result. "
                        "Use it if relevant and answer naturally.\n\n"
                        f"{tool_result['content']}"
                    ),
                }
            )

        messages.append({"role": "user", "content": user_message})
        return messages

    async def _generate_chat_reply(
        self,
        conversation: Conversation,
        user_message: str,
        recent_messages: list[Message],
        tool_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        messages = self._build_chat_messages(
            conversation=conversation,
            recent_messages=recent_messages,
            user_message=user_message,
            tool_result=tool_result,
        )
        response = await self.model_service.complete(messages=messages, max_tokens=1400)
        return {
            "ok": True,
            "tool": "chat",
            "result_type": "tool_result",
            "content": response.content,
            "assistant_content_json": None,
            "artifacts": [],
            "tool_payload": {"raw": response.raw},
            "state_patch": {},
            "citations": tool_result.get("citations", []) if tool_result else [],
            "latency_ms": 0,
        }

    async def handle_turn(self, conversation: Conversation, user_message: str) -> OrchestratorResult:
        state = self._resolve_state(conversation)
        turn = self._create_turn(conversation.id)

        user_msg = self._save_message(
            conversation_id=conversation.id,
            turn_id=turn.id,
            role=MessageRole.USER,
            content=user_message,
        )
        turn.user_message_id = user_msg.id

        recent_messages = self._load_recent_messages(conversation.id, limit=12)
        planner_result = await self.planner_service.plan_turn(
            user_message=user_message,
            state=state,
            recent_messages=[
                {"role": msg.role.value, "content": msg.content or ""}
                for msg in recent_messages
            ],
        )

        planner_action = planner_result.action.model_dump(mode="json")
        planner_trace = [entry.model_dump(mode="json") for entry in planner_result.trace]
        turn.planner_trace = planner_trace
        turn.final_plan = planner_action

        if planner_result.action.decision.value == "ask" and planner_result.action.reply_text:
            assistant_msg = self._save_message(
                conversation_id=conversation.id,
                turn_id=turn.id,
                role=MessageRole.ASSISTANT,
                content=planner_result.action.reply_text,
            )
            turn.assistant_message_id = assistant_msg.id
            self._apply_state_patch(conversation, planner_result.action.state_patch.model_dump(mode="json"))
            turn.state_patch = planner_result.action.state_patch.model_dump(mode="json")
            turn.status = TurnStatus.COMPLETED
            self.db.commit()

            return OrchestratorResult(
                ok=True,
                conversation_id=conversation.id,
                turn_id=turn.id,
                assistant_text=planner_result.action.reply_text,
                planner_action=planner_action,
                planner_trace=planner_trace,
            )

        tool_result: dict[str, Any] | None = None

        if planner_result.action.tool == PlannerTool.CHAT:
            tool_result = await self._generate_chat_reply(
                conversation=conversation,
                user_message=user_message,
                recent_messages=recent_messages,
            )
        else:
            tool = self.tool_registry.get(planner_result.action.tool.value)
            tool_result = await tool.execute(
                planner_action=planner_result.action,
                conversation=conversation,
                turn=turn,
                db=self.db,
            )

            if tool_result.get("ok") and planner_result.action.tool == PlannerTool.WEB:
                tool_result = await self._generate_chat_reply(
                    conversation=conversation,
                    user_message=user_message,
                    recent_messages=recent_messages,
                    tool_result=tool_result,
                )

        tool_event = self._save_tool_event(
            conversation_id=conversation.id,
            turn_id=turn.id,
            tool_name=ToolName(planner_result.action.tool.value),
            status=ToolStatus.SUCCESS if tool_result.get("ok") else ToolStatus.FAILED,
            input_text=user_message,
            output_text=tool_result.get("content"),
            payload_json=tool_result,
            latency_ms=tool_result.get("latency_ms"),
        )

        assistant_text = tool_result.get("content")
        assistant_json = tool_result.get("assistant_content_json")

        assistant_msg = self._save_message(
            conversation_id=conversation.id,
            turn_id=turn.id,
            role=MessageRole.ASSISTANT,
            content=assistant_text,
            content_json=assistant_json,
        )
        turn.assistant_message_id = assistant_msg.id

        for artifact in tool_result.get("artifacts", []):
            self.db.add(
                Artifact(
                    conversation_id=conversation.id,
                    turn_id=turn.id,
                    source_tool_event_id=tool_event.id,
                    parent_artifact_id=artifact.get("parent_artifact_id"),
                    artifact_type=ArtifactType(artifact["artifact_type"]),
                    title=artifact.get("title"),
                    storage_url=artifact.get("storage_url"),
                    inline_data=artifact.get("inline_data"),
                    prompt=artifact.get("prompt"),
                    effective_prompt=artifact.get("effective_prompt"),
                    metadata_json=artifact.get("metadata_json"),
                )
            )

        merged_state_patch = planner_result.action.state_patch.model_dump(mode="json")
        merged_state_patch.update(tool_result.get("state_patch", {}))
        self._apply_state_patch(conversation, merged_state_patch)

        turn.state_patch = merged_state_patch
        turn.status = TurnStatus.COMPLETED if tool_result.get("ok") else TurnStatus.FAILED

        self.db.commit()

        return OrchestratorResult(
            ok=bool(tool_result.get("ok")),
            conversation_id=conversation.id,
            turn_id=turn.id,
            assistant_text=assistant_text,
            planner_action=planner_action,
            planner_trace=planner_trace,
            tool_result=tool_result,
            error=tool_result.get("error"),
        )
