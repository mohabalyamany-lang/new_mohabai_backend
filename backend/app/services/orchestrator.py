from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.planner.contracts import (
    PlannerIntent,
    PlannerTool,
    ToolInput,
)
from app.planner.semantic_planner import semantic_planner
from app.runtime.guards import tool_sandbox
from app.services.context.context_builder import context_builder
from app.services.execution_engine import ExecutionEngine
from app.services.memory_service import MemoryService
from app.services.model_service import ModelService
from app.tools.registry import ToolRegistry

model_service = ModelService()


@dataclass
class OrchestratorResult:
    ok: bool
    reply: str
    conversation_id: int | None = None
    turn_id: str | None = None
    planner_action: dict[str, Any] = field(default_factory=dict)
    planner_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def assistant_text(self) -> str | None:
        """Alias for reply — used by stream_chat endpoint."""
        return self.reply if self.ok else None


class LastUserIntent(BaseModel):
    tool: str | None = None
    intent_type: str | None = None
    entities: dict[str, Any] = {}
    timestamp: float | None = None


_intent_store: dict[int, LastUserIntent] = {}


def is_followup_question(user_message: str) -> bool:
    followup_markers = [
        "what about",
        "and ",
        "tomorrow",
        "yesterday",
        "next ",
        "then ",
        "also ",
    ]
    lower = user_message.lower().strip()
    return any(marker in lower for marker in followup_markers)


def extract_entities_from_action(action, original_message: str = "") -> dict[str, Any]:
    return {
        "query": action.tool_input.query or original_message,
        "tool": action.tool.value,
    }


def rebuild_query(last_intent: LastUserIntent, user_message: str) -> str:
    base_query = last_intent.entities.get("query", "")
    lower = user_message.lower()

    if ("tomorrow" in lower or "yesterday" in lower) and "time" in base_query:
        return base_query

    if ("tomorrow" in lower or "yesterday" in lower) and "weather" in base_query:
        return f"{base_query} {user_message}"

    if "in " in lower:
        base_without_location = (
            base_query.split(" in ")[0] if " in " in base_query else base_query
        )
        return f"{base_without_location} {user_message}"

    return f"{base_query} {user_message}"


def _extract_storage_url(result: dict[str, Any]) -> str | None:
    content_json = result.get("assistant_content_json") or {}
    url = content_json.get("storage_url") or result.get("storage_url")
    if not url:
        artifacts = result.get("artifacts", [])
        if artifacts:
            url = artifacts[0].get("storage_url")
    return url


class ConversationOrchestrator:

    def __init__(
        self,
        db: Session | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.db = db
        self.tool_registry = tool_registry or ToolRegistry()
        self.execution_engine = ExecutionEngine(self.tool_registry)

    async def _generate_chat_reply(
        self,
        msg_list: list[dict],
        user_message: str,
        tool_result: dict[str, Any] | None = None,
    ) -> str:
        messages = list(msg_list)

        if tool_result:
            storage_url = _extract_storage_url(tool_result)
            if storage_url:
                return f"Here is your image:\n{storage_url}"

            if tool_result.get("content"):
                messages.append({
                    "role": "system",
                    "content": (
                        "A tool has produced the following result. "
                        "Use it to answer naturally:\n\n"
                        f"{tool_result['content']}"
                    ),
                })

        messages.append({"role": "user", "content": user_message})
        return await model_service.chat(messages=messages)

    async def handle_turn(
        self,
        conversation: Any,
        user_message: str,
        user_id: int,
        db: Session | None = None,
    ) -> OrchestratorResult:
        """
        Main entry point for handling a user turn.

        Args:
            conversation: SQLAlchemy Conversation model (must have .id)
            user_message: Raw user message string
            user_id: Authenticated user ID
            db: Optional DB session override (uses self.db if not provided)
        """
        from app.planner.state_resolver import ResolvedConversationState

        effective_db = db or self.db
        turn_id = uuid.uuid4().hex[:12]

        # ---- Follow-up query reconstruction ----
        last_intent = _intent_store.get(user_id)
        if (
            is_followup_question(user_message)
            and last_intent
            and last_intent.entities.get("query")
        ):
            augmented_message = rebuild_query(last_intent, user_message)
        else:
            augmented_message = user_message

        # ---- Build context ----
        context_bundle = await context_builder.build(
            db=effective_db,
            conversation_id=conversation.id,
            user_message=user_message,
            user_id=user_id,
        )
        msg_list = [
            {"role": m.role, "content": m.content}
            for m in context_bundle.messages
        ]

        # ---- Plan ----
        state = ResolvedConversationState()
        planner_result = await semantic_planner.plan(
            user_message=augmented_message,
            state=state,
        )

        action = planner_result.action
        planner_action_dict = action.model_dump(mode="json")
        planner_trace = [e.model_dump(mode="json") for e in planner_result.trace]

        # Force web tool for follow-up of web intent
        if (
            is_followup_question(user_message)
            and last_intent
            and last_intent.intent_type == "web"
            and action.tool == PlannerTool.CHAT
        ):
            previous_query = last_intent.entities.get("query", "")
            combined_query = f"{previous_query} {user_message}".strip()
            action = action.model_copy()
            action.tool = PlannerTool.WEB
            action.intent = PlannerIntent.WEB_SEARCH
            action.tool_input = ToolInput(query=combined_query)
            planner_action_dict = action.model_dump(mode="json")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MEMORY WRITE — LLM suggests, system decides
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if action.intent.value == "memory_write":
            memory_content = getattr(action.tool_input, "memory_content", None)
            memory_operation = getattr(action.tool_input, "memory_operation", None)
            confidence = action.confidence or 0.0

            memory_service = MemoryService(effective_db)
            memory = memory_service.add_memory_from_planner(
                user_id=user_id,
                memory_content=memory_content,
                memory_operation=memory_operation,
                confidence=confidence,
            )

            if memory:
                reply = "Got it, I'll remember that."
                planner_trace.append({
                    "stage": "memory_stored",
                    "summary": "Memory validated and stored",
                    "details": {"memory_id": memory.id},
                })
            else:
                # Memory rejected by validation — fall through to normal chat
                planner_trace.append({
                    "stage": "memory_rejected",
                    "summary": "Memory validation rejected the write",
                    "details": {"reason": "validation_failed"},
                })
                reply = await self._generate_chat_reply(
                    msg_list=msg_list,
                    user_message=user_message,
                )

            return OrchestratorResult(
                ok=True,
                reply=reply,
                conversation_id=conversation.id,
                turn_id=turn_id,
                planner_action=planner_action_dict,
                planner_trace=planner_trace,
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MULTI-INTENT → Execution Engine
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if planner_result.is_multi_intent and planner_result.steps:
            # Validate all steps through sandbox before executing
            for step in planner_result.steps:
                is_safe, reason = tool_sandbox.validate_tool_call(
                    tool_name=step.tool,
                    args=step.tool_input or {},
                )
                if not is_safe:
                    return OrchestratorResult(
                        ok=False,
                        reply="I can't process that request.",
                        conversation_id=conversation.id,
                        turn_id=turn_id,
                        planner_action=planner_action_dict,
                        planner_trace=planner_trace,
                        error=f"tool_sandbox_rejected: {reason}",
                    )

            exec_result = await self.execution_engine.execute_plan(
                planner_result=planner_result,
                original_message=user_message,
                conversation=conversation,
                turn=None,
                db=effective_db,
            )

            exec_trace = [e.model_dump(mode="json") for e in exec_result.trace]
            combined_trace = planner_trace + exec_trace

            # Check if execution completely failed
            if exec_result.error and not exec_result.final_response:
                return OrchestratorResult(
                    ok=False,
                    reply="Something went wrong processing your request.",
                    conversation_id=conversation.id,
                    turn_id=turn_id,
                    planner_action=planner_action_dict,
                    planner_trace=combined_trace,
                    error=exec_result.error,
                )

            # Build aggregated tool_result for frontend (images, citations, etc.)
            aggregated_tool_result: dict[str, Any] = {
                "ok": True,
                "content": exec_result.final_response,
                "artifacts": exec_result.final_artifacts,
                "citations": exec_result.final_citations,
                "multi_intent": True,
                "step_count": len(planner_result.steps),
                "steps_succeeded": sum(1 for s in exec_result.step_results if s.ok),
            }

            reply = exec_result.final_response or "Done."

            return OrchestratorResult(
                ok=True,
                reply=reply,
                conversation_id=conversation.id,
                turn_id=turn_id,
                planner_action=planner_action_dict,
                planner_trace=combined_trace,
                tool_result=aggregated_tool_result,
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SINGLE-STEP EXECUTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        is_safe, reason = tool_sandbox.validate_tool_call(
            tool_name=action.tool.value,
            args=action.tool_input.model_dump(exclude_none=True),
        )
        if not is_safe:
            return OrchestratorResult(
                ok=False,
                reply="I can't process that request.",
                conversation_id=conversation.id,
                turn_id=turn_id,
                planner_action=planner_action_dict,
                planner_trace=planner_trace,
                error=f"tool_sandbox_rejected: {reason}",
            )

        # ---- Pure chat ----
        if action.tool == PlannerTool.CHAT:
            reply = await self._generate_chat_reply(
                msg_list=msg_list,
                user_message=user_message,
            )

            # Clear intent tracking on chat (no tool was used)
            if user_id in _intent_store:
                del _intent_store[user_id]

            return OrchestratorResult(
                ok=True,
                reply=reply,
                conversation_id=conversation.id,
                turn_id=turn_id,
                planner_action=planner_action_dict,
                planner_trace=planner_trace,
            )

        # ---- Tool execution ----
        tool_result: dict[str, Any] = {}
        try:
            tool = self.tool_registry.get(action.tool.value)
            tool_result = await tool.execute(
                planner_action=action,
                conversation=None,
                turn=None,
                db=effective_db,
            )
        except Exception as exc:
            tool_result = {"ok": False, "error": str(exc)}
            planner_trace.append({
                "stage": "tool_exception",
                "summary": f"Tool {action.tool.value} raised exception",
                "details": {"error": str(exc)},
            })

        # Track intent for follow-up query reconstruction
        _intent_store[user_id] = LastUserIntent(
            tool=action.tool.value,
            intent_type="web" if action.tool == PlannerTool.WEB else action.tool.value,
            entities=extract_entities_from_action(action, user_message),
            timestamp=time.time(),
        )

        # Generate reply using tool result as context
        reply = await self._generate_chat_reply(
            msg_list=msg_list,
            user_message=user_message,
            tool_result=tool_result if tool_result.get("ok") else None,
        )

        # Pass tool_result to frontend only if tool succeeded
        # (frontend uses this for images, citations, etc.)
        passthrough_tool_result = tool_result if tool_result.get("ok") else None

        return OrchestratorResult(
            ok=True,
            reply=reply,
            conversation_id=conversation.id,
            turn_id=turn_id,
            planner_action=planner_action_dict,
            planner_trace=planner_trace,
            tool_result=passthrough_tool_result,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LEGACY APIS — kept for backward compatibility
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def handle(
        self,
        db: Session,
        user_id: int,
        message: str,
        conversation_state: dict[str, Any],
    ) -> str:
        """Legacy API — returns reply string only."""
        class _MinimalConv:
            def __init__(self, cid: int):
                self.id = cid

        result = await self.handle_turn(
            conversation=_MinimalConv(conversation_state.get("conversation_id", 0)),
            user_message=message,
            user_id=user_id,
            db=db,
        )
        return result.reply

    async def stream(
        self,
        db: Session,
        user_id: int,
        message: str,
        conversation_state: dict[str, Any],
    ):
        """Legacy streaming API — direct model stream, no planner."""
        context_bundle = await context_builder.build(
            db=db,
            conversation_id=conversation_state.get("conversation_id"),
            user_message=message,
            user_id=user_id,
        )
        msg_list = [
            {"role": m.role, "content": m.content}
            for m in context_bundle.messages
        ]
        msg_list.append({"role": "user", "content": message})
        async for chunk in model_service.stream(messages=msg_list):
            yield chunk


conversation_orchestrator = ConversationOrchestrator()
