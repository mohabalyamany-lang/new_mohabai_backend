"""
Execution Engine v3 — Production context control + data flow.

Fixes:
- Context explosion: MAX_CHARS budget, reverse chronological
- Type-aware filtering: only relevant steps feed into current step
- No unsafe web shortcut: requires explicit use_previous_output flag
- Step dependency model: depends_on support for future graph execution
- Memory injection: memories passed into synthesis
"""

from __future__ import annotations

import json
import time
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
    PlannerStep,
)
from app.tools.registry import ToolRegistry

settings = get_settings()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ZAI_URL = "https://api.z.ai/v1/chat/completions"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONTEXT CONTROL CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_PREVIOUS_OUTPUT_CHARS = 2000
MAX_SYNTHESIS_CONTEXT_CHARS = 4000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExecutionStepResult(BaseModel):
    step_index: int
    tool_name: str
    intent: str
    ok: bool
    content: str | None = None
    artifacts: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    structured: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: int = 0
    raw_output: dict[str, Any] = Field(default_factory=dict)


class AccumulatedStep(BaseModel):
    step_index: int
    intent: str
    tool: str
    output_type: str = "unknown"
    tool_input: dict[str, Any]
    output: ExecutionStepResult


class ExecutionResult(BaseModel):
    is_multi_intent: bool
    step_results: list[ExecutionStepResult] = Field(default_factory=list)
    execution_chain: list[AccumulatedStep] = Field(default_factory=list)
    final_response: str | None = None
    final_artifacts: list[dict] = Field(default_factory=list)
    final_citations: list[dict] = Field(default_factory=list)
    trace: list[PlannerTraceEntry] = Field(default_factory=list)
    error: str | None = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONTEXT CONTROL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def filter_relevant_context(
    chain: list[AccumulatedStep],
    current_step: PlannerStep,
) -> list[AccumulatedStep]:
    """Type-aware filtering: only pass context that is relevant to the current tool.

    Rules:
    - chat step → receives ALL previous outputs (it synthesizes)
    - web step → receives only previous web_result outputs
    - image step → receives only previous image_result outputs
    - unknown tool → receives nothing (safe default)
    """
    if not chain:
        return []

    current_tool = current_step.tool

    if current_tool == "chat":
        # Chat synthesizes everything
        return list(chain)

    if current_tool == "web":
        return [s for s in chain if s.output_type == "web_result"]

    if current_tool == "image":
        return [s for s in chain if s.output_type == "image_result"]

    # Unknown tool — safe default: no injection
    return []


def build_context_window(
    chain: list[AccumulatedStep],
    max_chars: int = MAX_PREVIOUS_OUTPUT_CHARS,
) -> str:
    """Build a bounded context window from execution chain.

    Rules:
    - Most recent steps first (reverse chronological)
    - Stop when budget exhausted
    - Never exceed max_chars
    """
    if not chain:
        return ""

    texts: list[tuple[str, int]] = []
    total = 0

    for step in reversed(chain):
        content = step.output.content
        if not content or not content.strip():
            continue

        content_len = len(content)

        if total + content_len > max_chars:
            # Take a truncated slice of this step's content
            remaining = max_chars - total
            if remaining > 50:  # Only include if we get meaningful content
                texts.append((content[:remaining] + "\n[...]", remaining))
                total += remaining
            break

        texts.append((content, content_len))
        total += content_len

    if not texts:
        return ""

    # Reverse back to chronological order
    texts.reverse()
    return "\n\n".join(t[0] for t in texts)


def build_step_context(
    chain: list[AccumulatedStep],
    current_step: PlannerStep,
    max_chars: int = MAX_PREVIOUS_OUTPUT_CHARS,
) -> tuple[str, list[dict], list[str]]:
    """Filter relevant context, then apply budget window.

    Returns:
        (context_text, citations, image_urls)
    """
    relevant = filter_relevant_context(chain, current_step)
    context_text = build_context_window(relevant, max_chars)

    # Collect citations and image URLs from relevant steps only
    citations: list[dict] = []
    image_urls: list[str] = []
    seen_cite_urls: set[str] = set()

    for step in relevant:
        if step.output.citations:
            for cite in step.output.citations:
                url = cite.get("url", "")
                if url and url not in seen_cite_urls:
                    seen_cite_urls.add(url)
                    citations.append(cite)

        for artifact in step.output.artifacts:
            if artifact.get("artifact_type") == "image" and artifact.get("storage_url"):
                image_urls.append(artifact["storage_url"])

    return context_text, citations, image_urls


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL OUTPUT NORMALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize_tool_output(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {"type": "error", "content": None, "structured": {}, "artifacts": [], "citations": []}

    output_type = raw.get("result_type") or raw.get("type") or "unknown"
    tool_name = raw.get("tool", "")

    if "web" in tool_name:
        output_type = "web_result"
    elif "image" in tool_name:
        output_type = "image_result"
    elif "chat" in tool_name:
        output_type = "chat_result"

    content = raw.get("content") or raw.get("text") or raw.get("result") or None
    structured = raw.get("assistant_content_json") or raw.get("structured") or raw.get("tool_payload") or {}
    if isinstance(structured, dict) and "raw" in structured:
        structured = {k: v for k, v in structured.items() if k != "raw"}

    artifacts = raw.get("artifacts") or []
    citations = raw.get("citations") or []

    return {
        "type": output_type,
        "content": content,
        "structured": structured,
        "artifacts": artifacts,
        "citations": citations,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL INPUT ENRICHMENT (SAFE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def enrich_tool_input(
    tool_input: dict[str, Any],
    chain: list[AccumulatedStep],
    current_step: PlannerStep,
) -> dict[str, Any]:
    """Enrich tool input with filtered, budget-controlled context.

    Unlike v2 which injected everything, this version:
    1. Filters by type relevance
    2. Applies character budget
    3. Sets use_previous_output flag explicitly (tools must opt in)
    """
    if not chain:
        return tool_input

    enriched = {**tool_input}
    context_text, citations, image_urls = build_step_context(chain, current_step)

    injected_keys: list[str] = []

    if context_text:
        enriched["previous_output"] = context_text
        injected_keys.append("previous_output")

        # Explicit flag — tools must check this to consume previous output
        enriched["use_previous_output"] = True
        injected_keys.append("use_previous_output")

    if citations:
        enriched["previous_citations"] = citations
        injected_keys.append("previous_citations")

    if image_urls:
        enriched["previous_image_urls"] = image_urls
        injected_keys.append("previous_image_urls")

    return enriched, injected_keys


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM PROVIDERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExecutionEngine:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._llm_providers = self._build_llm_providers()

    def _build_llm_providers(self) -> list[dict[str, Any]]:
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
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str | None:
        for provider in self._llm_providers:
            try:
                payload = {
                    "model": provider["model"],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
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

    def _build_synthesis_prompt(
        self,
        original_message: str,
        execution_chain: list[AccumulatedStep],
        memories: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        """Build synthesis prompt with budget-controlled context + memories."""
        step_descriptions: list[str] = []
        total_context_chars = 0

        for accumulated in execution_chain:
            if total_context_chars > MAX_SYNTHESIS_CONTEXT_CHARS:
                step_descriptions.append(f"Step {accumulated.step_index}: [context budget exhausted, remaining steps truncated]")
                break

            step = accumulated.output
            status = "succeeded" if step.ok else "failed"
            desc = f"Step {accumulated.step_index}: {accumulated.intent} ({accumulated.tool}) — {status}"

            if step.content:
                # Truncate per-step content to stay within budget
                remaining = MAX_SYNTHESIS_CONTEXT_CHARS - total_context_chars
                if remaining < 100:
                    desc += "\n  Output: [truncated]"
                else:
                    preview = step.content[:remaining]
                    if len(step.content) > remaining:
                        preview += "\n[...truncated...]"
                    desc += f"\n  Output: {preview}"
                    total_context_chars += len(step.content)

            if step.citations:
                cite_lines = [
                    f"- {c.get('title', 'Untitled')}: {c.get('url', '')}"
                    for c in step.citations[:5]
                ]
                desc += "\n  Sources:\n  " + "\n  ".join(cite_lines)

            if step.error:
                desc += f"\n  Error: {step.error}"

            step_descriptions.append(desc)

        system_prompt = (
            "You are a helpful assistant. The user asked a multi-part question.\n"
            "The system executed several steps to gather information.\n"
            "Your job is to synthesize ALL the step results into a single, clear, natural response.\n\n"
            "Rules:\n"
            "1. Do NOT mention 'Step 1', 'Step 2', or 'the system' — just answer naturally.\n"
            "2. Include relevant source URLs as inline references when citing facts.\n"
            "3. If any step failed, acknowledge it briefly but focus on what worked.\n"
            "4. If there are image URLs, present them (markdown: ![description](url)).\n"
            "5. Be complete but concise — don't repeat the same information.\n"
        )

        # Inject memories if available
        memory_block = ""
        if memories:
            memory_lines = [f"- {m['content']}" for m in memories if m.get("content")]
            if memory_lines:
                memory_block = (
                    "\n\nRelevant user context you may reference naturally "
                    "(do NOT say 'from your memory' or 'I remember'):\n"
                    + "\n".join(memory_lines)
                )

        user_prompt = (
            f"User's question:\n{original_message}\n\n"
            f"Information gathered:\n"
            + "\n\n".join(step_descriptions)
            + memory_block
            + "\n\nNow answer the user's question using all the information above."
        )

        return system_prompt, user_prompt

    async def _execute_single_step(
        self,
        step: PlannerStep,
        enriched_input: dict[str, Any],
        conversation: Any,
        turn: Any,
        db: Any,
    ) -> ExecutionStepResult:
        try:
            tool = self.tool_registry.get(step.tool)
        except ValueError as exc:
            return ExecutionStepResult(
                step_index=step.order,
                tool_name=step.tool,
                intent=step.intent,
                ok=False,
                error=f"Tool not found: {exc}",
            )

        step_action = PlannerAction(
            intent=PlannerIntent(step.intent),
            tool=PlannerTool(step.tool),
            decision=PlannerDecision.ACT,
            tool_input=ToolInput.model_validate(enriched_input),
        )

        started = time.perf_counter()
        try:
            raw_result = await tool.execute(
                planner_action=step_action,
                conversation=conversation,
                turn=turn,
                db=db,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return ExecutionStepResult(
                step_index=step.order,
                tool_name=step.tool,
                intent=step.intent,
                ok=False,
                error=str(exc),
                latency_ms=latency_ms,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        normalized = _normalize_tool_output(raw_result)
        step_ok = raw_result.get("ok", False)

        return ExecutionStepResult(
            step_index=step.order,
            tool_name=step.tool,
            intent=step.intent,
            ok=step_ok,
            content=normalized["content"],
            artifacts=normalized["artifacts"],
            citations=normalized["citations"],
            structured=normalized["structured"],
            error=raw_result.get("error"),
            latency_ms=latency_ms,
            raw_output=raw_result,
        )

    async def execute_plan(
        self,
        planner_result: PlannerResult,
        original_message: str,
        conversation: Any,
        turn: Any,
        db: Any,
        memories: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        trace: list[PlannerTraceEntry] = []

        # ━━━ Single intent ━━━
        if not planner_result.is_multi_intent or not planner_result.steps:
            trace.append(PlannerTraceEntry(
                stage="execution",
                summary="Single-intent execution",
                details={"intent": planner_result.action.intent.value},
            ))

            tool_name = planner_result.action.tool.value
            try:
                tool = self.tool_registry.get(tool_name)
            except ValueError as exc:
                return ExecutionResult(is_multi_intent=False, trace=trace, error=str(exc))

            step = PlannerStep(
                intent=planner_result.action.intent.value,
                tool=tool_name,
                tool_input=planner_result.action.tool_input.model_dump(exclude_none=True),
                order=0,
            )

            step_result = await self._execute_single_step(
                step=step,
                enriched_input=step.tool_input,
                conversation=conversation,
                turn=turn,
                db=db,
            )

            trace.append(PlannerTraceEntry(
                stage="execution_complete",
                summary=f"Tool {tool_name} completed",
                details={"ok": step_result.ok, "latency_ms": step_result.latency_ms},
            ))

            return ExecutionResult(
                is_multi_intent=False,
                step_results=[step_result],
                final_artifacts=step_result.artifacts,
                final_citations=step_result.citations,
                trace=trace,
            )

        # ━━━ Multi-intent with data flow + dependencies ━━━
        steps = sorted(planner_result.steps, key=lambda s: s.order)
        trace.append(PlannerTraceEntry(
            stage="execution",
            summary=f"Multi-intent execution: {len(steps)} steps",
            details={
                "steps": [{"index": s.order, "intent": s.intent, "tool": s.tool} for s in steps],
                "context_budget": MAX_PREVIOUS_OUTPUT_CHARS,
            },
        ))

        execution_chain: list[AccumulatedStep] = []
        step_results: list[ExecutionStepResult] = []
        all_artifacts: list[dict] = []
        all_citations: list[dict] = []
        has_chat_step = any(s.tool == "chat" for s in steps)

        for i, step in enumerate(steps):
            # ━── Resolve dependencies ━──
            depends_on = getattr(step, "depends_on", None) or []
            if depends_on:
                missing = [d for d in depends_on if d >= len(execution_chain)]
                if missing:
                    trace.append(PlannerTraceEntry(
                        stage="step_error",
                        summary=f"Step {i}: unmet dependencies {missing}",
                        details={},
                    ))
                    step_results.append(ExecutionStepResult(
                        step_index=i, tool_name=step.tool, intent=step.intent,
                        ok=False, error=f"unmet dependencies: {missing}",
                    ))
                    execution_chain.append(AccumulatedStep(
                        step_index=i, intent=step.intent, tool=step.tool,
                        output_type="error", tool_input=step.tool_input or {},
                        output=step_results[-1],
                    ))
                    continue

            trace.append(PlannerTraceEntry(
                stage="step_start",
                summary=f"Step {i}: {step.intent} via {step.tool}",
                details={
                    "depends_on": depends_on or None,
                    "chain_length": len(execution_chain),
                },
            ))

            # ━── Enrich with filtered, budget-controlled context ━──
            base_input = step.tool_input or {}
            enriched_input, injected_keys = enrich_tool_input(base_input, execution_chain, step)

            if injected_keys:
                trace.append(PlannerTraceEntry(
                    stage="step_enriched",
                    summary=f"Step {i}: injected {injected_keys}",
                    details={
                        "context_chars": len(enriched_input.get("previous_output", "")),
                        "filter_type": step.tool,
                    },
                ))

            # ━── Defer chat steps ━──
            if step.tool == "chat":
                step_results.append(ExecutionStepResult(
                    step_index=i, tool_name="chat", intent="chat", ok=True,
                ))
                execution_chain.append(AccumulatedStep(
                    step_index=i, intent="chat", tool="chat",
                    output_type="chat_result", tool_input=base_input,
                    output=step_results[-1],
                ))
                trace.append(PlannerTraceEntry(
                    stage="step_deferred",
                    summary=f"Step {i}: chat deferred to synthesis",
                    details={},
                ))
                continue

            # ━── Execute ━──
            step_result = await self._execute_single_step(
                step=step, enriched_input=enriched_input,
                conversation=conversation, turn=turn, db=db,
            )
            step_results.append(step_result)

            normalized = _normalize_tool_output(step_result.raw_output) if step_result.raw_output else {}
            execution_chain.append(AccumulatedStep(
                step_index=i, intent=step.intent, tool=step.tool,
                output_type=normalized.get("type", "unknown"),
                tool_input=base_input, output=step_result,
            ))

            all_artifacts.extend(step_result.artifacts)
            all_citations.extend(step_result.citations)

            trace.append(PlannerTraceEntry(
                stage="step_complete",
                summary=f"Step {i}: {step.tool} — {'ok' if step_result.ok else 'failed'}",
                details={
                    "ok": step_result.ok,
                    "latency_ms": step_result.latency_ms,
                    "content_chars": len(step_result.content) if step_result.content else 0,
                },
            ))

        # ━━━ Synthesis ━━━
        tool_steps = [c for c in execution_chain if c.tool != "chat"]
        has_tool_output = any(s.output.ok and s.output.content for s in tool_steps)
        final_response = None

        if (has_chat_step or has_tool_output) and has_tool_output:
            trace.append(PlannerTraceEntry(
                stage="synthesis_start",
                summary="LLM synthesis with controlled context",
                details={
                    "chain_length": len(execution_chain),
                    "memories_injected": len(memories) if memories else 0,
                },
            ))

            system_prompt, user_prompt = self._build_synthesis_prompt(
                original_message=original_message,
                execution_chain=execution_chain,
                memories=memories,
            )

            started = time.perf_counter()
            final_response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1000,
            )
            synthesis_ms = int((time.perf_counter() - started) * 1000)

            if final_response:
                trace.append(PlannerTraceEntry(
                    stage="synthesis_complete",
                    summary="Synthesis succeeded",
                    details={"latency_ms": synthesis_ms, "response_chars": len(final_response)},
                ))
            else:
                trace.append(PlannerTraceEntry(
                    stage="synthesis_failed",
                    summary="All LLM providers failed — raw fallback",
                    details={"latency_ms": synthesis_ms},
                ))
                parts = [c.output.content for c in tool_steps if c.output.ok and c.output.content]
                final_response = "\n\n".join(parts) if parts else None

        trace.append(PlannerTraceEntry(
            stage="execution_complete",
            summary="Multi-intent execution finished",
            details={
                "steps_total": len(steps),
                "steps_succeeded": sum(1 for s in step_results if s.ok),
                "has_response": final_response is not None,
            },
        ))

        return ExecutionResult(
            is_multi_intent=True,
            step_results=step_results,
            execution_chain=execution_chain,
            final_response=final_response,
            final_artifacts=all_artifacts,
            final_citations=all_citations,
            trace=trace,
        )
