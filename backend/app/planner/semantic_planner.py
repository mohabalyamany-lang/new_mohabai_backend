from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.planner.contracts import (
    ConversationMode,
    PlannerAction,
    PlannerDecision,
    PlannerIntent,
    PlannerResolution,
    PlannerResult,
    PlannerStatePatch,
    PlannerTool,
    PlannerTraceEntry,
    ToolInput,
    PlannerStep,
)
from app.planner.state_resolver import (
    ResolvedConversationState,
    is_general_chat_switch,
    is_lookup_followup,
    is_social_feedback,
    is_style_request,
    looks_like_image_edit,
    looks_like_image_question,
    looks_like_image_request,
    needs_live_information,
    normalize_text,
)

settings = get_settings()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ZAI_URL = "https://api.z.ai/v1/chat/completions"


class SemanticPlannerOutput(BaseModel):
    intent: str
    tool: str
    decision: str = "act"
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_input: dict[str, Any] = Field(default_factory=dict)
    uses_last_artifact: bool = False
    uses_pending_target: bool = False
    clear_pending_target: bool = False
    conversation_mode: str = "normal_chat"
    pending_followup_kind: str | None = None
    pending_followup_target: str | None = None
    allow_context_carryover: bool = False
    reply_text: str | None = None


class PlannerContext(BaseModel):
    user_message: str
    normalized_message: str
    state: dict[str, Any]
    recent_messages: list[dict[str, str]] = Field(default_factory=list)


class SemanticPlanner:
    def __init__(self) -> None:
        self.providers = self._build_provider_configs()

    def _build_provider_configs(self) -> list[dict[str, Any]]:
        providers: list[dict[str, Any]] = []

# 1. GLM-4.5 (Primary Intelligence)
        if settings.zai_api_key:
            providers.append(
                {
                    "name": "zai_glm",
                    "url": ZAI_URL,
                    "headers": {
                        "Authorization": f"Bearer {settings.zai_api_key}",
                        "Content-Type": "application/json",
                    },
                    "model": "glm-4.5-flash",
                }
            )

        # 2. Groq (High-Speed Fallback)
        if settings.groq_api_key:
            providers.append(
                {
                    "name": "groq",
                    "url": GROQ_URL,
                    "headers": {
                        "Authorization": f"Bearer {settings.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    "model": "llama-3.1-8b-instant",
                }
            )

        # 3. OpenRouter (Backup)
        if settings.openrouter_api_key:
            providers.append(
                {
                    "name": "openrouter",
                    "url": OPENROUTER_URL,
                    "headers": {
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://mohabai.vercel.app",
                        "X-Title": "Mohab AI Planner",
                    },
                    "model": "google/gemma-3-12b-it:free",
                }
            )

        # 4. OpenAI (Premium Fallback)
        if settings.openai_api_key:
            providers.append(
                {
                    "name": "openai",
                    "url": OPENAI_URL,
                    "headers": {
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    "model": "gpt-4.1-mini",
                }
            )

        return providers

    def _system_prompt(self) -> str:
        return (
            "You are the semantic planner for a production conversational AI system.\n"
            "Return ONLY valid JSON.\n\n"
            "Your job is to determine the user's actual current goal, not just keywords.\n"
            "You must resolve follow-ups naturally using conversation state.\n\n"
            "Allowed intents:\n"
            "- chat\n"
            "- web_search\n"
            "- file_analysis\n"
            "- image_gen\n"
            "- image_edit\n"
            "- image_retry\n"
            "- image_question\n"
            "- memory_read\n"
            "- memory_write\n\n"
            "Allowed tools:\n"
            "- chat\n"
            "- web\n"
            "- file\n"
            "- image\n"
            "- memory\n\n"
            "Rules:\n"
            "1. Ordinary conversation or social feedback is chat, even after tool use.\n"
            "2. Style/control requests like 'be more talkative' are chat, not image/web.\n"
            "3. If the user refers to a recent image artifact for modification, use image_edit.\n"
            "4. If the user asks about the content of a recent image, use image_question.\n"
            "5. If the user needs live/current information, use web_search.\n"
            "6. If the user says 'look it up' after a live-info request, resolve that reference.\n"
            "7. When using tools, produce concrete tool_input, not placeholders.\n"
            "8. Do not narrate capability limitations. Plan the action.\n\n"
            "9. If the user shares personal details (name, job, preferences), use memory_write.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "intent": "chat",\n'
            '  "tool": "chat",\n'
            '  "decision": "act",\n'
            '  "reason": "brief reason",\n'
            '  "confidence": 0.95,\n'
            '  "tool_input": {\n'
            '    "query": null,\n'
            '    "image_instruction": null,\n'
            '    "analysis_target": null,\n'
            '    "style_directive": null,\n'
            '    "memory_operation": null,\n'
            '    "memory_content": null,\n'
            '    "artifact_id": null,\n'
            '    "metadata": {}\n'
            '  },\n'
            '  "uses_last_artifact": false,\n'
            '  "uses_pending_target": false,\n'
            '  "clear_pending_target": false,\n'
            '  "conversation_mode": "normal_chat",\n'
            '  "pending_followup_kind": null,\n'
            '  "pending_followup_target": null,\n'
            '  "allow_context_carryover": false,\n'
            '  "reply_text": null\n'
            "}\n"
        )

    async def _classify_sub_intent(self, text: str) -> tuple[str, str, dict]:
        # Fast deterministic overrides first
        if looks_like_image_request(text):
            return "image_gen", "image", {"image_instruction": text}

        if is_live_info_intent(text):
            return "web_search", "web", {"query": text}

        # fallback to semantic LLM (cheap model)
        for provider in self.providers:
            try:
                payload = {
                    "model": provider["model"],
                    "temperature": 0,
                    "max_tokens": 100,
                    "messages": [
                        {"role": "system", "content": "You are a sub-intent classifier. Respond ONLY with JSON: {\"intent\": \"chat\" | \"web_search\" | \"image_gen\"}. Choose web_search for facts/news, image_gen for creating art, and chat for conversation."},
                        {"role": "user", "content": text},
                    ],
                }

                async with httpx.AsyncClient(timeout=6) as client:
                    r = await client.post(provider["url"], headers=provider["headers"], json=payload)
                    if r.status_code != 200:
                        continue

                    data = r.json()["choices"][0]["message"]["content"]
                    # Handle possible markdown wrapping
                    if "```json" in data:
                        data = data.split("```json")[1].split("```")[0].strip()
                    
                    parsed = json.loads(data)
                    intent = parsed.get("intent", "chat")

                    if intent == "image_gen":
                        return "image_gen", "image", {"image_instruction": text}
                    if intent == "web_search":
                        return "web_search", "web", {"query": text}

                    return "chat", "chat", {}

            except Exception:
                continue

        return "chat", "chat", {}

    async def _provider_call(self, provider: dict[str, Any], planner_context: PlannerContext) -> SemanticPlannerOutput | None:
        payload = {
            "model": provider["model"],
            "temperature": 0,
            "max_tokens": 350,
            "stream": False,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": planner_context.model_dump_json()},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=4.0)) as client:
                response = await client.post(
                    provider["url"],
                    headers=provider["headers"],
                    json=payload,
                )
                if response.status_code != 200:
                    return None
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # Strip markdown code blocks if present
                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```"):
                    content = content.split("```")[1].split("```")[0].strip()
                    
                parsed = json.loads(content)
                return SemanticPlannerOutput.model_validate(parsed)
        except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValidationError):
            return None

    async def plan(
        self,
        user_message: str,
        state: ResolvedConversationState,
        recent_messages: list[dict[str, str]] | None = None,
    ) -> PlannerResult:
        normalized = normalize_text(user_message)
        trace: list[PlannerTraceEntry] = []

        # Multi-intent detection
        if detect_multi_intent(user_message):
            parts = split_intents(user_message)
            if len(parts) > 1:
                steps = []
                for i, part in enumerate(parts):
                    try:
                        intent, tool, tool_input = await self._classify_sub_intent(part)
                        steps.append(PlannerStep(
                            intent=intent,
                            tool=tool,
                            tool_input=tool_input,
                            order=i,
                        ))
                    except Exception:
                        continue
                if steps:
                    fallback_action = PlannerAction(
                        intent=PlannerIntent.CHAT,
                        tool=PlannerTool.CHAT,
                        decision=PlannerDecision.ACT,
                        conversation_mode=ConversationMode.NORMAL_CHAT,
                        reason="multi_intent_decomposed",
                        confidence=0.95,
                    )
                    return PlannerResult(
                        action=fallback_action,
                        steps=steps,
                        is_multi_intent=True,
                        trace=[PlannerTraceEntry(
                            stage="multi_intent",
                            summary=f"Decomposed into {len(steps)} steps",
                            details={"parts": parts},
                        )],
                    )

        # Hard exits from sticky tool modes
        if is_style_request(user_message):
            return PlannerResult(
                action=PlannerAction(
                    intent=PlannerIntent.CHAT,
                    tool=PlannerTool.CHAT,
                    decision=PlannerDecision.ACT,
                    conversation_mode=ConversationMode.NORMAL_CHAT,
                    tool_input=ToolInput(style_directive=user_message),
                    state_patch=PlannerStatePatch(
                        active_mode=ConversationMode.NORMAL_CHAT,
                        pending_intent=None,
                        clear_pending_target=True,
                    ),
                    resolution=PlannerResolution(
                        topic_switch=True,
                        clear_pending_target=True,
                        notes=["style_request_exits_tool_mode"],
                    ),
                    reason="user_changed_response_style",
                    confidence=0.99,
                ),
                trace=[
                    PlannerTraceEntry(
                        stage="guard",
                        summary="Style request resolved to chat",
                        details={"message": user_message},
                    )
                ],
            )

        if is_social_feedback(user_message):
            return PlannerResult(
                action=PlannerAction(
                    intent=PlannerIntent.CHAT,
                    tool=PlannerTool.CHAT,
                    decision=PlannerDecision.ACT,
                    conversation_mode=ConversationMode.NORMAL_CHAT,
                    state_patch=PlannerStatePatch(
                        active_mode=ConversationMode.NORMAL_CHAT,
                        clear_pending_target=True,
                    ),
                    resolution=PlannerResolution(
                        topic_switch=True,
                        clear_pending_target=True,
                        notes=["social_feedback_not_artifact_followup"],
                    ),
                    reason="social_feedback",
                    confidence=0.98,
                ),
                trace=[
                    PlannerTraceEntry(
                        stage="guard",
                        summary="Social feedback forced to chat",
                        details={"message": user_message},
                    )
                ],
            )

        if is_general_chat_switch(user_message):
            return PlannerResult(
                action=PlannerAction(
                    intent=PlannerIntent.CHAT,
                    tool=PlannerTool.CHAT,
                    decision=PlannerDecision.ACT,
                    conversation_mode=ConversationMode.NORMAL_CHAT,
                    state_patch=PlannerStatePatch(
                        active_mode=ConversationMode.NORMAL_CHAT,
                        clear_pending_target=True,
                    ),
                    resolution=PlannerResolution(
                        topic_switch=True,
                        clear_pending_target=True,
                        notes=["general_chat_switch_exits_tool_mode"],
                    ),
                    reason="general_chat_topic_switch",
                    confidence=0.98,
                ),
                trace=[
                    PlannerTraceEntry(
                        stage="guard",
                        summary="General chat switch resolved to chat",
                        details={"message": user_message},
                    )
                ],
            )

        if any([
            "who are you" in user_message.lower(),
            "what are you" in user_message.lower(),
            "how do you work" in user_message.lower(),
        ]):
            return PlannerResult(
                action=PlannerAction(
                    intent=PlannerIntent.CHAT,
                    tool=PlannerTool.CHAT,
                    decision=PlannerDecision.ACT,
                    conversation_mode=ConversationMode.NORMAL_CHAT,
                    state_patch=PlannerStatePatch(
                        active_mode=ConversationMode.NORMAL_CHAT,
                        clear_pending_target=True,
                    ),
                    resolution=PlannerResolution(
                        topic_switch=True,
                        clear_pending_target=True,
                        notes=["explicit_identity_or_meta_question"],
                    ),
                    reason="user_switched_to_meta_chat",
                    confidence=0.99,
                ),
                trace=[
                    PlannerTraceEntry(
                        stage="guard",
                        summary="Forced exit to normal chat",
                        details={"message": user_message},
                    )
                ],
            )

        # Clean deterministic image cases
        if state.last_artifact_type == "image":
            if looks_like_image_question(user_message):
                return PlannerResult(
                    action=PlannerAction(
                        intent=PlannerIntent.IMAGE_QUESTION,
                        tool=PlannerTool.IMAGE,
                        decision=PlannerDecision.ACT,
                        conversation_mode=ConversationMode.IMAGE_ITERATION,
                        tool_input=ToolInput(
                            analysis_target=user_message,
                            artifact_id=state.last_artifact_id,
                        ),
                        state_patch=PlannerStatePatch(
                            active_mode=ConversationMode.IMAGE_ITERATION,
                            pending_intent="image_question",
                            pending_followup_kind="image",
                            pending_followup_target=state.last_artifact_id,
                            allow_context_carryover=False,
                        ),
                        resolution=PlannerResolution(
                            uses_last_artifact=True,
                            notes=["resolved_against_last_image"],
                        ),
                        reason="image_question_against_recent_artifact",
                        confidence=0.96,
                    ),
                    trace=[
                        PlannerTraceEntry(
                            stage="deterministic",
                            summary="Image question resolved against last artifact",
                            details={"artifact_id": state.last_artifact_id},
                        )
                    ],
                )

            if looks_like_image_edit(user_message):
                return PlannerResult(
                    action=PlannerAction(
                        intent=PlannerIntent.IMAGE_EDIT,
                        tool=PlannerTool.IMAGE,
                        decision=PlannerDecision.ACT,
                        conversation_mode=ConversationMode.IMAGE_ITERATION,
                        tool_input=ToolInput(
                            image_instruction=user_message,
                            artifact_id=state.last_artifact_id,
                            metadata={
                                "parent_prompt": state.last_artifact_prompt,
                            },
                        ),
                        state_patch=PlannerStatePatch(
                            active_mode=ConversationMode.IMAGE_ITERATION,
                            pending_intent="image_edit",
                            pending_followup_kind="image",
                            pending_followup_target=state.last_artifact_id,
                            allow_context_carryover=False,
                        ),
                        resolution=PlannerResolution(
                            uses_last_artifact=True,
                            notes=["resolved_as_image_edit_against_last_artifact"],
                        ),
                        reason="image_edit_against_recent_artifact",
                        confidence=0.97,
                    ),
                    trace=[
                        PlannerTraceEntry(
                            stage="deterministic",
                            summary="Image edit resolved against last artifact",
                            details={"artifact_id": state.last_artifact_id},
                        )
                    ],
                )

        # Follow-up lookup against pending live target
        if (
            state.pending_followup_kind == "live_info"
            and state.pending_followup_target
            and len(user_message.split()) <= 5
        ):
            return PlannerResult(
                action=PlannerAction(
                    intent=PlannerIntent.WEB_SEARCH,
                    tool=PlannerTool.WEB,
                    decision=PlannerDecision.ACT,
                    conversation_mode=ConversationMode.LIVE_INFO,
                    tool_input=ToolInput(query=state.pending_followup_target),
                    state_patch=PlannerStatePatch(
                        active_mode=ConversationMode.LIVE_INFO,
                        pending_intent="web_search",
                        pending_followup_kind="live_info",
                        pending_followup_target=state.pending_followup_target,
                        allow_context_carryover=False,
                    ),
                    resolution=PlannerResolution(
                        uses_pending_target=True,
                        notes=["lookup_followup_resolved_against_pending_live_target"],
                    ),
                    reason="live_info_followup",
                    confidence=0.99,
                ),
                trace=[
                    PlannerTraceEntry(
                        stage="deterministic",
                        summary="Lookup follow-up resolved to pending live target",
                        details={"query": state.pending_followup_target},
                    )
                ],
            )

        # Simple deterministic live-info routing
        if needs_live_information(user_message) or is_live_info_intent(user_message):
            return PlannerResult(
                action=PlannerAction(
                    intent=PlannerIntent.WEB_SEARCH,
                    tool=PlannerTool.WEB,
                    decision=PlannerDecision.ACT,
                    conversation_mode=ConversationMode.LIVE_INFO,
                    tool_input=ToolInput(query=user_message),
                    state_patch=PlannerStatePatch(
                        active_mode=ConversationMode.LIVE_INFO,
                        pending_intent="web_search",
                        pending_followup_kind="live_info",
                        pending_followup_target=user_message,
                        allow_context_carryover=False,
                    ),
                    resolution=PlannerResolution(
                        notes=["live_information_required"],
                    ),
                    reason="user_needs_current_information",
                    confidence=0.90,
                ),
                trace=[
                    PlannerTraceEntry(
                        stage="deterministic",
                        summary="Current-information request routed to web",
                        details={"query": user_message},
                    )
                ],
            )

        # Controlled image generation
        if looks_like_image_request(user_message):
            if any([
                is_social_feedback(user_message),
                is_style_request(user_message),
                is_general_chat_switch(user_message),
            ]):
                pass
            else:
                return PlannerResult(
                    action=PlannerAction(
                        intent=PlannerIntent.IMAGE_GEN,
                        tool=PlannerTool.IMAGE,
                        decision=PlannerDecision.ACT,
                        conversation_mode=ConversationMode.IMAGE_ITERATION,
                        tool_input=ToolInput(image_instruction=user_message),
                        state_patch=PlannerStatePatch(
                            active_mode=ConversationMode.IMAGE_ITERATION,
                            pending_intent="image_gen",
                            pending_followup_kind="image",
                            allow_context_carryover=False,
                        ),
                        resolution=PlannerResolution(
                            notes=["explicit_image_request_with_guard"]
                        ),
                        reason="validated_image_request",
                        confidence=0.92,
                    ),
                    trace=[
                        PlannerTraceEntry(
                            stage="deterministic",
                            summary="Validated image request",
                            details={"instruction": user_message},
                        )
                    ],
                )

        # Ensure state is a dictionary for the LLM context
        state_dict = state.model_dump() if hasattr(state, 'model_dump') else vars(state)
        
        planner_context = PlannerContext(
            user_message=user_message,
            normalized_message=normalized,
            state=state_dict,
            recent_messages=recent_messages or [],
        )
        trace.append(
            PlannerTraceEntry(
                stage="semantic",
                summary="Escalating to semantic planner",
                details={"message": user_message},
            )
        )

        for provider in self.providers:
            semantic = await self._provider_call(provider, planner_context)
            if semantic is None:
                continue

            try:
                action = PlannerAction(
                    intent=PlannerIntent(semantic.intent),
                    tool=PlannerTool(semantic.tool),
                    decision=PlannerDecision(semantic.decision),
                    conversation_mode=ConversationMode(semantic.conversation_mode),
                    tool_input=ToolInput.model_validate(semantic.tool_input),
                    state_patch=PlannerStatePatch(
                        active_mode=ConversationMode(semantic.conversation_mode),
                        pending_intent=semantic.intent,
                        pending_followup_kind=semantic.pending_followup_kind,
                        pending_followup_target=semantic.pending_followup_target,
                        allow_context_carryover=semantic.allow_context_carryover,
                        clear_pending_target=semantic.clear_pending_target,
                    ),
                    resolution=PlannerResolution(
                        uses_last_artifact=semantic.uses_last_artifact,
                        uses_pending_target=semantic.uses_pending_target,
                        clear_pending_target=semantic.clear_pending_target,
                        notes=[semantic.reason] if semantic.reason else [],
                    ),
                    reply_text=semantic.reply_text,
                    reason=semantic.reason,
                    confidence=semantic.confidence,
                )
            except ValueError:
                continue

            trace.append(
                PlannerTraceEntry(
                    stage="semantic_result",
                    summary="Semantic planner returned action",
                    details={"provider": provider["name"], "intent": semantic.intent, "tool": semantic.tool},
                )
            )
            return PlannerResult(action=action, trace=trace)

        return PlannerResult(
            action=PlannerAction(
                intent=PlannerIntent.CHAT,
                tool=PlannerTool.CHAT,
                decision=PlannerDecision.ACT,
                conversation_mode=ConversationMode.NORMAL_CHAT,
                state_patch=PlannerStatePatch(
                    active_mode=ConversationMode.NORMAL_CHAT,
                    clear_pending_target=True,
                ),
                resolution=PlannerResolution(
                    topic_switch=True,
                    clear_pending_target=True,
                    notes=["semantic_planner_unavailable_default_chat"],
                ),
                reason="fallback_to_chat",
                confidence=0.55,
            ),
            trace=trace
            + [
                PlannerTraceEntry(
                    stage="fallback",
                    summary="Planner defaulted to chat",
                    details={},
                )
            ],
        )


def is_live_info_intent(text: str) -> bool:
    text = text.lower()
    keywords = [
        "weather", "price", "stock", "bitcoin", "time",
        "news", "score", "temperature", "today", "tomorrow"
    ]
    return any(k in text for k in keywords)


def detect_multi_intent(user_message: str) -> bool:
    text = user_message.lower()
    if " and " not in text:
        return False

    # must contain at least 2 verbs/actions to qualify as multiple tasks
    triggers = ["generate", "create", "tell", "explain", "search", "find", "show"]
    count = sum(1 for t in triggers if t in text)
    return count >= 2


def split_intents(user_message: str) -> list[str]:
    msg = user_message.replace(", and", " and").replace(", then", " then")
    parts = msg.split(" and ")
    return [p.strip() for p in parts if p.strip()]


semantic_planner = SemanticPlanner()
