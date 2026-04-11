"""
Execution Engine — Production-grade sequential step executor.

Responsibilities:
- Takes PlannerResult (single or multi-intent)
- Executes steps sequentially via ToolRegistry
- Passes tool outputs between steps
- Calls LLM for final synthesis when needed
- Streams results with trace observability

Usage:
    engine = ExecutionEngine(tool_registry)
    results = await engine.execute_plan(
        planner_result=result,
        original_message=user_message,
        conversation=conversation,
        turn=turn,
        db=db,
    )
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.planner.contracts import (
    PlannerAction,
    PlannerDecision,
    PlannerIntent,
    PlannerResult,
    PlannerTool,
    PlannerTraceEntry,
    ToolInput,
)
from app.tools.registry import ToolRegistry

settings = get_settings()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ZAI_URL = "https://api.z.ai/v1/chat/completions"


class ExecutionStepResult(BaseModel):
    """Result of executing a single step."""
    step_index: int
    tool_name: str
    ok: bool
    content: str | None = None
    artifacts: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0


class ExecutionResult(BaseModel):
    """Result of executing a full plan."""
    is_multi_intent: bool
    step_results: list[ExecutionStepResult] = Field(default_factory=list)
    final_response: str | None = None
    final_artifacts: list[dict] = Field(default_factory=list)
    final_citations: list[dict] = Field(default_factory=list)
    trace: list[PlannerTraceEntry] = Field(default_factory=list)
    error: str | None = None


def _extract_json(content: str) -> dict | None:
    """Reuse the same robust extraction from planner."""
    if not content:
        return None
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            for lang in ("json", "JSON", "Json"):
                if content.startswith(lang):
                    content = content[len(lang):]
                    break
            content = content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(content[start:end + 1])
    except Exception:
        return None


class ExecutionEngine:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._llm_providers = self._build_llm_providers()

    def _build_llm_providers(self) -> list[dict[str, Any]]:
        """Build LLM provider list for synthesis calls.
        Same priority as planner: OpenAI → Groq → OpenRouter → ZAI.
        """
        providers: list[dict[str, Any]] = []

        if settings.openai_api_key:
            providers.append({
                "name": "openai",
                "url": OPENAI_URL,
                "headers": {
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                "model": "gpt-4.1-mini",
            })

        if settings.groq_api_key:
            providers.append({
                "name": "groq",
                "url": GROQ_URL,
                "headers": {
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                "model": "llama-3.1-8b-instant",
            })

        if settings.openrouter_api_key:
            providers.append({
                "name": "openrouter",
                "url": OPENROUTER_URL,
                "headers": {
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://mohabai.vercel.app",
                    "X-Title": "Mohab AI Synthesis",
                },
                "model": "google/gemma-3-12b-it:free",
            })

        if settings.zai_api_key:
            providers.append({
                "name": "zai_glm",
                "url": ZAI_URL,
                "headers": {
                    "Authorization": f"Bearer {settings.zai_api_key}",
                    "Content-Type": "application/json",
                },
                "model": "glm-4.5-flash",
            })

        return providers

    async def _call_llm(
        self,
        user_message: str,
        context: str,
        system_prompt: str | None = None,
    ) -> str | None:
        """Call LLM to synthesize accumulated tool results."""
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. The user asked a multi-part question. "
                "You have been given the results of each part. "
                "Synthesize them into a clear, natural response. "
                "Do not mention 'Step 1' or 'Step 2' — just answer naturally.\n\n"
                f"User's original question:\n{user_message}\n\n"
                f"Information gathered:\n{context}"
            )

        for provider in self._llm_providers:
            try:
                payload = {
                    "model": provider["model"],
                    "temperature": 0.7,
                    "max_tokens": 800,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Using this information:\n\n{context}\n\nAnswer the user's question: {user_message}"},
                    ],
                }

                async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
                    response = await client.post(
                        provider["url"],
                        headers=provider["headers"],
                        json=payload,
                    )
                    if response.status_code != 200:
                        continue

                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        return content.strip()

            except Exception:
                continue

        return None

    async def execute_plan(
        self,
        planner_result: PlannerResult,
        original_message: str,
        conversation: Any,
        turn: Any,
        db: Any,
    ) -> ExecutionResult:
        """Execute a planner result (single or multi-intent).

        For single intent: execute the tool directly.
        For multi-intent: execute steps sequentially, pass outputs, synthesize.
        """
        trace: list[PlannerTraceEntry] = []

        # ━━━ Single intent ━━━
        if not planner_result.is_multi_intent or not planner_result.steps:
            trace.append(
                PlannerTraceEntry(
                    stage="execution",
                    summary="Single-intent execution",
                    details={"intent": planner_result.action.intent.value},
                )
            )

            tool_name = planner_result.action.tool.value
            try:
                tool = self.tool_registry.get(tool_name)
            except ValueError as exc:
                return ExecutionResult(
                    is_multi_intent=False,
                    trace=trace,
                    error=f"Tool not found: {exc}",
                )

            started = time.perf_counter()
            try:
                tool_result = await tool.execute(
                    planner_action=planner_result.action,
                    conversation=conversation,
                    turn=turn,
                    db=db,
                )
            except Exception as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                trace.append(
                    PlannerTraceEntry(
                        stage="execution_error",
                        summary=f"Tool {tool_name} raised exception",
                        details={"error": str(exc), "latency_ms": latency_ms},
                    )
                )
                return ExecutionResult(
                    is_multi_intent=False,
                    trace=trace,
                    error=str(exc),
                )

            latency_ms = int((time.perf_counter() - started) * 1000)
            step_result = ExecutionStepResult(
                step_index=0,
                tool_name=tool_name,
                ok=tool_result.get("ok", False),
                content=tool_result.get("content"),
                artifacts=tool_result.get("artifacts", []),
                citations=tool_result.get("citations", []),
                error=tool_result.get("error"),
                latency_ms=latency_ms,
            )

            trace.append(
                PlannerTraceEntry(
                    stage="execution_complete",
                    summary=f"Tool {tool_name} completed",
                    details={
                        "ok": step_result.ok,
                        "latency_ms": latency_ms,
                    },
                )
            )

            return ExecutionResult(
                is_multi_intent=False,
                step_results=[step_result],
                final_artifacts=step_result.artifacts,
                final_citations=step_result.citations,
                trace=trace,
            )

        # ━━━ Multi-intent execution ━━━
        steps = planner_result.steps
        trace.append(
            PlannerTraceEntry(
                stage="execution",
                summary=f"Multi-intent execution: {len(steps)} steps",
                details={"steps": [s.__dict__ for s in steps]},
            )
        )

        accumulated_context = ""
        step_results: list[ExecutionStepResult] = []
        all_artifacts: list[dict] = []
        all_citations: list[dict] = []
        has_chat_step = False
        chat_step_index = -1

        # Identify if there's a chat (synthesis) step
        for i, step in enumerate(steps):
            if step.tool == "chat":
                has_chat_step = True
                chat_step_index = i
                break

        # Execute non-chat steps sequentially
        for i, step in enumerate(steps):
            if step.tool == "chat":
                # Skip chat step — handle after collecting all tool results
                continue

            trace.append(
                PlannerTraceEntry(
                    stage="step_start",
                    summary=f"Step {i}: executing {step.tool}",
                    details={"intent": step.intent, "tool_input": step.tool_input},
                )
            )

            try:
                tool = self.tool_registry.get(step.tool)
            except ValueError as exc:
                step_results.append(
                    ExecutionStepResult(
                        step_index=i,
                        tool_name=step.tool,
                        ok=False,
                        error=f"Tool not found: {exc}",
                    )
                )
                trace.append(
                    PlannerTraceEntry(
                        stage="step_error",
                        summary=f"Step {i}: tool not found",
                        details={"error": str(exc)},
                    )
                )
                continue

            # Build a PlannerAction for this step
            step_action = PlannerAction(
                intent=PlannerIntent(step.intent),
                tool=PlannerTool(step.tool),
                decision=PlannerDecision.ACT,
                tool_input=ToolInput.model_validate(step.tool_input),
            )

            started = time.perf_counter()
            try:
                tool_result = await tool.execute(
                    planner_action=step_action,
                    conversation=conversation,
                    turn=turn,
                    db=db,
                )
            except Exception as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                step_results.append(
                    ExecutionStepResult(
                        step_index=i,
                        tool_name=step.tool,
                        ok=False,
                        error=str(exc),
                        latency_ms=latency_ms,
                    )
                )
                trace.append(
                    PlannerTraceEntry(
                        stage="step_error",
                        summary=f"Step {i}: execution exception",
                        details={"error": str(exc), "latency_ms": latency_ms},
                    )
                )
                continue

            latency_ms = int((time.perf_counter() - started) * 1000)

            step_ok = tool_result.get("ok", False)
            step_content = tool_result.get("content")
            step_artifacts = tool_result.get("artifacts", [])
            step_citations = tool_result.get("citations", [])

            step_results.append(
                ExecutionStepResult(
                    step_index=i,
                    tool_name=step.tool,
                    ok=step_ok,
                    content=step_content,
                    artifacts=step_artifacts,
                    citations=step_citations,
                    latency_ms=latency_ms,
                )
            )

            # Accumulate context for next steps
            if step_ok and step_content:
                accumulated_context += f"\n--- Result for \"{step.tool_input.get('query', step.tool_input.get('image_instruction', 'step ' + str(i)))}\":\n{step_content}\n"

            all_artifacts.extend(step_artifacts)
            all_citations.extend(step_citations)

            trace.append(
                PlannerTraceEntry(
                    stage="step_complete",
                    summary=f"Step {i}: {step.tool} completed",
                    details={"ok": step_ok, "latency_ms": latency_ms},
                )
            )

        # ━━━ Synthesize if chat step exists and we have context ━━━
        final_response = None

        if has_chat_step and accumulated_context.strip():
            trace.append(
                PlannerTraceEntry(
                    stage="synthesis_start",
                    summary="Starting LLM synthesis of accumulated results",
                    details={"context_length": len(accumulated_context)},
                )
            )

            started = time.perf_counter()
            final_response = await self._call_llm(
                user_message=original_message,
                context=accumulated_context,
            )
            synthesis_ms = int((time.perf_counter() - started) * 1000)

            if final_response:
                trace.append(
                    PlannerTraceEntry(
                        stage="synthesis_complete",
                        summary="LLM synthesis succeeded",
                        details={"latency_ms": synthesis_ms, "response_length": len(final_response)},
                    )
                )
            else:
                trace.append(
                    PlannerTraceEntry(
                        stage="synthesis_failed",
                        summary="All LLM providers failed for synthesis",
                        details={"latency_ms": synthesis_ms},
                    )
                )
                # Fallback: just return the accumulated context directly
                final_response = accumulated_context.strip()
        elif accumulated_context.strip() and not has_chat_step:
            # No chat step but we have results — return accumulated context
            final_response = accumulated_context.strip()

        trace.append(
            PlannerTraceEntry(
                stage="execution_complete",
                summary="Multi-intent execution finished",
                details={
                    "steps_total": len(steps),
                    "steps_succeeded": sum(1 for s in step_results if s.ok),
                    "has_final_response": final_response is not None,
                },
            )
        )

        return ExecutionResult(
            is_multi_intent=True,
            step_results=step_results,
            final_response=final_response,
            final_artifacts=all_artifacts,
            final_citations=all_citations,
            trace=trace,
        )
