"""
Execution Engine v2 — True multi-step reasoning with data flow.

Each step can consume previous step outputs.
Context accumulates through the chain.
Final synthesis receives full execution trace.
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
# DATA MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExecutionStepResult(BaseModel):
    """Result of executing a single step."""
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
    # The raw tool output — used for propagation to next step
    raw_output: dict[str, Any] = Field(default_factory=dict)


class AccumulatedStep(BaseModel):
    """One step in the execution chain — stored for next-step injection."""
    step_index: int
    intent: str
    tool: str
    tool_input: dict[str, Any]
    output: ExecutionStepResult


class ExecutionResult(BaseModel):
    """Result of executing a full plan."""
    is_multi_intent: bool
    step_results: list[ExecutionStepResult] = Field(default_factory=list)
    execution_chain: list[AccumulatedStep] = Field(default_factory=list)
    final_response: str | None = None
    final_artifacts: list[dict] = Field(default_factory=list)
    final_citations: list[dict] = Field(default_factory=list)
    trace: list[PlannerTraceEntry] = Field(default_factory=list)
    error: str | None = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL OUTPUT EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize_tool_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize any tool output into a standard shape.
    
    Tools may return different shapes. This extracts the important parts
    regardless of which tool produced the output.
    
    Standard output:
    {
        "type": "web_result" | "image_result" | "chat_result" | "error",
        "content": "...",           # main text content
        "structured": {...},        # optional parsed/typed data
        "artifacts": [...],         # images, files, etc
        "citations": [...],         # source references
    }
    """
    if not raw:
        return {"type": "error", "content": None, "structured": {}, "artifacts": [], "citations": []}

    output_type = raw.get("result_type") or raw.get("type") or "unknown"

    # Map tool names to standard types
    tool_name = raw.get("tool", "")
    if "web" in tool_name:
        output_type = "web_result"
    elif "image" in tool_name:
        output_type = "image_result"
    elif "chat" in tool_name:
        output_type = "chat_result"

    # Extract content — check multiple possible keys
    content = (
        raw.get("content")
        or raw.get("text")
        or raw.get("result")
        or None
    )

    # Extract structured data
    structured = (
        raw.get("assistant_content_json")
        or raw.get("structured")
        or raw.get("tool_payload")
        or {}
    )
    # Don't leak raw API responses into structured
    if isinstance(structured, dict) and "raw" in structured:
        structured = {k: v for k, v in structured.items() if k != "raw"}

    # Extract artifacts
    artifacts = raw.get("artifacts") or []

    # Extract citations
    citations = raw.get("citations") or []

    return {
        "type": output_type,
        "content": content,
        "structured": structured,
        "artifacts": artifacts,
        "citations": citations,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL INPUT ENRICHMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def enrich_tool_input(
    tool_input: dict[str, Any],
    execution_chain: list[AccumulatedStep],
) -> dict[str, Any]:
    """Inject previous step outputs into the current step's tool_input.
    
    Rules:
    - If the last step produced content, inject it as `previous_output`
    - If any previous step produced web results, inject relevant citations
    - If any previous step produced an image, inject the URL
    - Never mutate the original tool_input
    
    This is what makes chains like "search Tesla → summarize it" work.
    Step 2 receives step 1's search results in its tool_input.
    """
    if not execution_chain:
        return tool_input

    enriched = {**tool_input}

    # Collect all previous outputs
    all_content_parts: list[str] = []
    all_citations: list[dict] = []
    image_urls: list[str] = []

    for accumulated in execution_chain:
        step_output = accumulated.output
        if not step_output.ok:
            continue

        # Collect content
        if step_output.content:
            all_content_parts.append(step_output.content)

        # Collect citations
        if step_output.citations:
            all_citations.extend(step_output.citations)

        # Collect image URLs
        for artifact in step_output.artifacts:
            if artifact.get("artifact_type") == "image" and artifact.get("storage_url"):
                image_urls.append(artifact["storage_url"])

    # ━── Inject previous output (most recent first) ━──
    if all_content_parts:
        # Join all previous content, most recent last
        enriched["previous_output"] = "\n\n".join(all_content_parts)

    # ━── Inject previous citations if available ━──
    if all_citations:
        # Deduplicate by URL
        seen_urls = set()
        unique_citations = []
        for cite in all_citations:
            url = cite.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_citations.append(cite)
        if unique_citations:
            enriched["previous_citations"] = unique_citations

    # ━── Inject image URLs if available ━──
    if image_urls:
        enriched["previous_image_urls"] = image_urls

    return enriched


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM PROVIDERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_json(content: str) -> dict | None:
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXECUTION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExecutionEngine:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._llm_providers = self._build_llm_providers()

    def _build_llm_providers(self) -> list[dict[str, Any]]:
        """LLM providers for synthesis. Same priority as planner."""
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
        """Call LLM with fallback across providers."""
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
    ) -> tuple[str, str]:
        """Build synthesis system + user prompts from the full execution chain.
        
        Returns (system_prompt, user_prompt).
        """
        # Build step-by-step trace
        step_descriptions: list[str] = []

        for accumulated in execution_chain:
            step = accumulated.output
            status = "succeeded" if step.ok else "failed"

            desc = f"Step {accumulated.step_index}: {accumulated.intent} ({accumulated.tool}) — {status}"

            if step.content:
                # Truncate very long content to avoid token overflow
                content_preview = step.content[:1500]
                if len(step.content) > 1500:
                    content_preview += "\n[...truncated...]"
                desc += f"\n  Output: {content_preview}"

            if step.citations:
                cite_lines = [f"- {c.get('title', 'Untitled')}: {c.get('url', '')}" for c in step.citations[:5]]
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
            "4. If there are image URLs, present them to the user (markdown format: ![description](url)).\n"
            "5. Be complete but concise — don't repeat the same information.\n"
        )

        user_prompt = (
            f"User's question:\n{original_message}\n\n"
            f"Information gathered:\n"
            + "\n\n".join(step_descriptions)
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
        """Execute a single step and return normalized result."""
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

        # Build PlannerAction for this step
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

        # Normalize the output
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
    ) -> ExecutionResult:
        """Execute a planner result with full data flow between steps."""
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

            # For single intent, no enrichment needed
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

            trace.append(
                PlannerTraceEntry(
                    stage="execution_complete",
                    summary=f"Tool {tool_name} completed",
                    details={
                        "ok": step_result.ok,
                        "latency_ms": step_result.latency_ms,
                        "output_type": "unknown" if not step_result.content else "has_content",
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

        # ━━━ Multi-intent execution with data flow ━━━
        steps = sorted(planner_result.steps, key=lambda s: s.order)
        trace.append(
            PlannerTraceEntry(
                stage="execution",
                summary=f"Multi-intent execution: {len(steps)} steps with data flow",
                details={
                    "steps": [
                        {"index": s.order, "intent": s.intent, "tool": s.tool}
                        for s in steps
                    ],
                },
            )
        )

        execution_chain: list[AccumulatedStep] = []
        step_results: list[ExecutionStepResult] = []
        all_artifacts: list[dict] = []
        all_citations: list[dict] = []
        has_chat_step = False

        # Identify chat steps (used for final synthesis)
        for i, step in enumerate(steps):
            if step.tool == "chat":
                has_chat_step = True

        # ━━━ Execute each step with context injection ━━━
        for i, step in enumerate(steps):
            trace.append(
                PlannerTraceEntry(
                    stage="step_start",
                    summary=f"Step {i}: {step.intent} via {step.tool}",
                    details={
                        "tool_input": step.tool_input,
                        "chain_length": len(execution_chain),
                        "has_previous_output": (
                            execution_chain[-1].output.ok
                            and execution_chain[-1].output.content is not None
                            if execution_chain
                            else False
                        ),
                    },
                )
            )

            # ━── Enrich tool input with previous outputs ━──
            base_input = step.tool_input or {}
            enriched_input = enrich_tool_input(base_input, execution_chain)

            if enriched_input != base_input:
                trace.append(
                    PlannerTraceEntry(
                        stage="step_enriched",
                        summary=f"Step {i}: tool_input enriched with previous output",
                        details={
                            "injected_keys": [
                                k for k in enriched_input
                                if k not in base_input
                            ],
                            "previous_output_length": len(
                                enriched_input.get("previous_output", "")
                            ) if enriched_input.get("previous_output") else 0,
                        },
                    )
                )

            # Skip chat steps — handle after data collection
            if step.tool == "chat":
                step_results.append(
                    ExecutionStepResult(
                        step_index=i,
                        tool_name="chat",
                        intent="chat",
                        ok=True,
                        content=None,  # Will be filled by synthesis
                    )
                )
                execution_chain.append(
                    AccumulatedStep(
                        step_index=i,
                        intent="chat",
                        tool="chat",
                        tool_input=base_input,
                        output=step_results[-1],
                    )
                )
                trace.append(
                    PlannerTraceEntry(
                        stage="step_deferred",
                        summary=f"Step {i}: chat step deferred to synthesis phase",
                        details={},
                    )
                )
                continue

            # ━── Execute the step ━──
            step_result = await self._execute_single_step(
                step=step,
                enriched_input=enriched_input,
                conversation=conversation,
                turn=turn,
                db=db,
            )

            step_results.append(step_result)

            # Add to execution chain for next step's enrichment
            execution_chain.append(
                AccumulatedStep(
                    step_index=i,
                    intent=step.intent,
                    tool=step.tool,
                    tool_input=base_input,
                    output=step_result,
                )
            )

            # Collect artifacts and citations
            all_artifacts.extend(step_result.artifacts)
            all_citations.extend(step_result.citations)

            trace.append(
                PlannerTraceEntry(
                    stage="step_complete",
                    summary=f"Step {i}: {step.tool} — {'ok' if step_result.ok else 'failed'}",
                    details={
                        "ok": step_result.ok,
                        "latency_ms": step_result.latency_ms,
                        "content_length": len(step_result.content) if step_result.content else 0,
                        "artifacts_count": len(step_result.artifacts),
                        "citations_count": len(step_result.citations),
                    },
                )
            )

        # ━━━ Synthesis ━━━
        final_response = None

        # Check if we have any non-chat tool results
        tool_steps = [
            chain for chain in execution_chain
            if chain.tool != "chat"
        ]
        has_tool_output = any(s.output.ok and s.output.content for s in tool_steps)

        if has_chat_step and has_tool_output:
            trace.append(
                PlannerTraceEntry(
                    stage="synthesis_start",
                    summary="Starting LLM synthesis with full execution chain",
                    details={
                        "chain_length": len(execution_chain),
                        "tool_steps_with_output": sum(
                            1 for s in tool_steps if s.output.ok and s.output.content
                        ),
                    },
                )
            )

            system_prompt, user_prompt = self._build_synthesis_prompt(
                original_message=original_message,
                execution_chain=execution_chain,
            )

            started = time.perf_counter()
            final_response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1000,
                temperature=0.7,
            )
            synthesis_ms = int((time.perf_counter() - started) * 1000)

            if final_response:
                trace.append(
                    PlannerTraceEntry(
                        stage="synthesis_complete",
                        summary="LLM synthesis succeeded",
                        details={
                            "latency_ms": synthesis_ms,
                            "response_length": len(final_response),
                        },
                    )
                )
            else:
                trace.append(
                    PlannerTraceEntry(
                        stage="synthesis_failed",
                        summary="All LLM providers failed for synthesis — falling back to raw output",
                        details={"latency_ms": synthesis_ms},
                    )
                )
                # Fallback: concatenate tool outputs
                parts = []
                for chain in tool_steps:
                    if chain.output.ok and chain.output.content:
                        parts.append(chain.output.content)
                final_response = "\n\n".join(parts) if parts else None

        elif has_tool_output and not has_chat_step:
            # No chat step but we have results — synthesize anyway for natural output
            trace.append(
                PlannerTraceEntry(
                    stage="synthesis_start",
                    summary="No chat step but tool outputs exist — auto-synthesizing",
                    details={"tool_steps": len(tool_steps)},
                )
            )

            system_prompt, user_prompt = self._build_synthesis_prompt(
                original_message=original_message,
                execution_chain=execution_chain,
            )

            started = time.perf_counter()
            final_response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1000,
                temperature=0.7,
            )
            synthesis_ms = int((time.perf_counter() - started) * 1000)

            if not final_response:
                parts = []
                for chain in tool_steps:
                    if chain.output.ok and chain.output.content:
                        parts.append(chain.output.content)
                final_response = "\n\n".join(parts) if parts else None

            trace.append(
                PlannerTraceEntry(
                    stage="synthesis_complete" if final_response else "synthesis_failed",
                    summary="Auto-synthesis " + ("succeeded" if final_response else "fell back to raw"),
                    details={"latency_ms": synthesis_ms},
                )
            )

        trace.append(
            PlannerTraceEntry(
                stage="execution_complete",
                summary="Multi-intent execution finished",
                details={
                    "steps_total": len(steps),
                    "steps_succeeded": sum(1 for s in step_results if s.ok),
                    "steps_failed": sum(1 for s in step_results if not s.ok),
                    "has_final_response": final_response is not None,
                    "total_artifacts": len(all_artifacts),
                    "total_citations": len(all_citations),
                },
            )
        )

        return ExecutionResult(
            is_multi_intent=True,
            step_results=step_results,
            execution_chain=execution_chain,
            final_response=final_response,
            final_artifacts=all_artifacts,
            final_citations=all_citations,
            trace=trace,
        )
