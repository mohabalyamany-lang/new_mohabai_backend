from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.planner.contracts import PlannerIntent, PlannerTool, ToolInput
from app.planner.semantic_planner import semantic_planner
from app.runtime.guards import tool_sandbox
from app.services.context.context_builder import context_builder
from app.services.model_service import ModelService
from app.tools.registry import ToolRegistry

model_service = ModelService()


@dataclass
class OrchestratorResult:
    ok: bool
    reply: str
    conversation_id: int | None = None
    turn_id: int | None = None
    planner_action: dict[str, Any] = field(default_factory=dict)
    planner_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


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

    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()

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
        db: Session,
        user_id: int,
        user_message: str,
        conversation_state: dict[str, Any],
    ) -> OrchestratorResult:
        from app.planner.state_resolver import ResolvedConversationState

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
            db=db,
            conversation_id=conversation_state.get("conversation_id"),
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

        planner_action = planner_result.action.model_dump(mode="json")
        planner_trace = [e.model_dump(mode="json") for e in planner_result.trace]

        # Force web tool for follow-up of web intent
        if (
            is_followup_question(user_message)
            and last_intent
            and last_intent.intent_type == "web"
            and planner_result.action.tool == PlannerTool.CHAT
        ):
            previous_query = last_intent.entities.get("query", "")
            combined_query = f"{previous_query} {user_message}".strip()
            planner_result.action.tool = PlannerTool.WEB
            planner_result.action.intent = PlannerIntent.WEB_SEARCH
            planner_result.action.tool_input = ToolInput(query=combined_query)

        action = planner_result.action

        # ---- MULTI-STEP EXECUTION (Phase 10) ----
        if planner_result.is_multi_intent and planner_result.steps:
            execution_context: dict[str, Any] = {}

            for step in sorted(planner_result.steps, key=lambda s: s.order):
                step_tool = step.tool
                step_input = step.tool_input or {}

                is_safe, reason = tool_sandbox.validate_tool_call(
                    tool_name=step_tool,
                    args=step_input,
                )
                if not is_safe:
                    return OrchestratorResult(
                        ok=False,
                        reply="I can't process that request.",
                        planner_action=planner_action,
                        planner_trace=planner_trace,
                        error=f"tool_sandbox_rejected: {reason}",
                    )

                if step_tool == "chat":
                    chat_reply = await model_service.chat(
                        messages=msg_list + [
                            {"role": "user", "content": step_input.get("query", user_message)}
                        ]
                    )
                    execution_context[f"step_{step.order}"] = chat_reply
                else:
                    try:
                        tool = self.tool_registry.get(step_tool)
                        step_action = action.model_copy(update={
                            "tool": PlannerTool(step_tool),
                            "tool_input": ToolInput.model_validate(step_input),
                        })
                        step_result = await tool.execute(
                            planner_action=step_action,
                            conversation=None,
                            turn=None,
                            db=db,
                        )
                        storage_url = _extract_storage_url(step_result)
                        if storage_url:
                            execution_context[f"step_{step.order}"] = f"IMAGE_URL:{storage_url}"
                        else:
                            execution_context[f"step_{step.order}"] = step_result.get(
                                "content", str(step_result)
                            )
                    except Exception as exc:
                        execution_context[f"step_{step.order}"] = f"[error: {exc}]"

            # Format execution context — surface image URLs directly
            formatted_results = []
            image_urls = []
            for key, value in execution_context.items():
                if isinstance(value, str) and value.startswith("IMAGE_URL:"):
                    url = value[len("IMAGE_URL:"):]
                    image_urls.append(url)
                    formatted_results.append(f"{key}: [Image generated: {url}]")
                else:
                    formatted_results.append(f"{key}: {value}")

            if image_urls and len(execution_context) == 1:
                return OrchestratorResult(
                    ok=True,
                    reply=f"Here is your image:\n{image_urls[0]}",
                    planner_action=planner_action,
                    planner_trace=planner_trace,
                    tool_results=execution_context,
                )

            image_prefix = ""
            if image_urls:
                image_prefix = "Generated image(s):\n" + "\n".join(image_urls) + "\n\n"

            reasoning_prompt = (
                "Use the following tool results to answer the user's question "
                "naturally and completely. If there are image URLs, present them "
                "to the user directly.\n\n"
                + "\n".join(formatted_results)
            )
            final_msg_list = msg_list + [
                {"role": "system", "content": reasoning_prompt},
                {"role": "user", "content": user_message},
            ]
            reply = await model_service.chat(messages=final_msg_list)
            if image_prefix:
                reply = image_prefix + reply

            return OrchestratorResult(
                ok=True,
                reply=reply,
                planner_action=planner_action,
                planner_trace=planner_trace,
                tool_results=execution_context,
            )

        # ---- SINGLE-STEP EXECUTION ----
        is_safe, reason = tool_sandbox.validate_tool_call(
            tool_name=action.tool.value,
            args=action.tool_input.model_dump(exclude_none=True),
        )
        if not is_safe:
            return OrchestratorResult(
                ok=False,
                reply="I can't process that request.",
                planner_action=planner_action,
                planner_trace=planner_trace,
                error=f"tool_sandbox_rejected: {reason}",
            )

        if action.tool == PlannerTool.CHAT:
            reply = await self._generate_chat_reply(
                msg_list=msg_list,
                user_message=user_message,
            )
        else:
            tool_result: dict[str, Any] = {}
            try:
                tool = self.tool_registry.get(action.tool.value)
                tool_result = await tool.execute(
                    planner_action=action,
                    conversation=None,
                    turn=None,
                    db=db,
                )
            except Exception:
                pass

            _intent_store[user_id] = LastUserIntent(
                tool=action.tool.value,
                intent_type="web" if action.tool == PlannerTool.WEB else action.tool.value,
                entities=extract_entities_from_action(action, user_message),
                timestamp=time.time(),
            )

            reply = await self._generate_chat_reply(
                msg_list=msg_list,
                user_message=user_message,
                tool_result=tool_result if tool_result.get("ok") else None,
            )

        if action.tool == PlannerTool.CHAT and user_id in _intent_store:
            del _intent_store[user_id]

        return OrchestratorResult(
            ok=True,
            reply=reply,
            planner_action=planner_action,
            planner_trace=planner_trace,
        )

    async def handle(
        self,
        db: Session,
        user_id: int,
        message: str,
        conversation_state: dict[str, Any],
    ) -> str:
        result = await self.handle_turn(
            db=db,
            user_id=user_id,
            user_message=message,
            conversation_state=conversation_state,
        )
        return result.reply

    async def stream(
        self,
        db: Session,
        user_id: int,
        message: str,
        conversation_state: dict[str, Any],
    ):
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