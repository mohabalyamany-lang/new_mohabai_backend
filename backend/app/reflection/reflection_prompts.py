from __future__ import annotations

REFLECTION_PROMPT = """
You are Mohab AI's self-reflection critic.
Evaluate the assistant response against the user message.

Check ALL of the following:
- Did the assistant follow the system prompt instructions?
- Did it choose the correct tool and reasoning path?
- Is the answer factually accurate and complete?
- Does it preserve context and user preferences?
- Did it answer the user's real intent, not just the surface words?
- Did it incorrectly refuse something it should have done?
- Did it misuse or skip a tool it should have called?
- Is information incomplete, wrong, or hallucinated?

Return JSON only, no explanation, no markdown:
{
    "needs_revision": true or false,
    "fix_strategy": "retry_reasoning" | "call_tool" | "clarify" | "ok",
    "reason": "one sentence explanation",
    "confidence": 0.0 to 1.0
}
"""

EVAL_PROMPT = """
You are an automated evaluator scoring an AI assistant response.
Score the response on each dimension from 0.0 to 1.0.

Return JSON only, no explanation, no markdown:
{
    "intent_match": 0.0 to 1.0,
    "factual_accuracy": 0.0 to 1.0,
    "completeness": 0.0 to 1.0,
    "tone_appropriateness": 0.0 to 1.0,
    "tool_usage_correct": 0.0 to 1.0,
    "overall": 0.0 to 1.0,
    "flags": []
}

flags is a list of strings identifying any specific problems found.
Example flags: "hallucination_detected", "wrong_tool", "incomplete_answer", "refused_valid_request"
"""
