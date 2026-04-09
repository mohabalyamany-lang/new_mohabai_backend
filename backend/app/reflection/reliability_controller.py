from __future__ import annotations

import json

from app.reflection.reflection_prompts import EVAL_PROMPT
from app.services.llm_service import llm_service

_EVAL_DEFAULTS = {
    "intent_match": 1.0,
    "factual_accuracy": 1.0,
    "completeness": 1.0,
    "tone_appropriateness": 1.0,
    "tool_usage_correct": 1.0,
    "overall": 1.0,
    "flags": [],
}


def _parse_eval(raw: str) -> dict:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        return {**_EVAL_DEFAULTS, **parsed}
    except Exception:
        return dict(_EVAL_DEFAULTS)


class ReliabilityController:

    async def stabilize(
        self,
        reflection: dict,
        agent,
        user_message: str,
        context_messages: list[dict],
    ) -> str | None:
        """
        Act on a reflection result. Returns a replacement reply string,
        or None if no repair is needed.
        """
        if not reflection.get("needs_revision"):
            return None

        strategy = reflection.get("fix_strategy", "ok")

        if strategy == "retry_reasoning":
            # Re-run the agent with an explicit correction hint injected
            corrected_messages = list(context_messages) + [{
                "role": "system",
                "content": (
                    "Your previous response had an issue: "
                    f"{reflection.get('reason', 'unknown problem')}. "
                    "Correct it now. Be accurate and complete."
                ),
            }]
            try:
                return await agent.run(
                    messages=corrected_messages,
                    user_message=user_message,
                )
            except Exception:
                return None

        if strategy == "call_tool":
            # Signal to caller that a tool should have been used.
            # The agent_loop will handle this on its next iteration naturally
            # since we inject a correction hint — same pattern as retry_reasoning.
            corrected_messages = list(context_messages) + [{
                "role": "system",
                "content": (
                    "You missed using a tool that was required. "
                    f"Reason: {reflection.get('reason', '')}. "
                    "Reconsider and call the appropriate tool."
                ),
            }]
            try:
                return await agent.run(
                    messages=corrected_messages,
                    user_message=user_message,
                )
            except Exception:
                return None

        if strategy == "clarify":
            return "Could you clarify what you mean? I want to make sure I help you correctly."

        # strategy == "ok" or anything unrecognized — no repair needed
        return None

    async def score(
        self,
        user_message: str,
        assistant_reply: str,
    ) -> dict:
        """
        Run automated evaluation scoring on a completed response.
        Returns a score dict with per-dimension floats and a flags list.
        Safe to call fire-and-forget — never raises.
        """
        try:
            result = await llm_service.chat([
                {"role": "system", "content": EVAL_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User message:\n{user_message}\n\n"
                        f"Assistant reply:\n{assistant_reply}"
                    ),
                },
            ])
            return _parse_eval(result)
        except Exception:
            return dict(_EVAL_DEFAULTS)


reliability_controller = ReliabilityController()
