from typing import List, Dict, Any

from .reasoning_step import ReasoningStep
from .reflection import reflect_on_output


MAX_STEPS = 3


class CognitiveLoop:
    """
    Cognitive reasoning loop.

    This is the system's internal thinking engine.
    It allows the AI to:
        - think
        - decide actions
        - use tools
        - evaluate results
        - produce final answers

    No phrase triggers.
    Only reasoning → decisions.
    """

    def __init__(self, llm):
        self.llm = llm

    # ==========================================================
    # MAIN REASONING LOOP
    # ==========================================================
    def run(self, plan, context) -> Dict[str, Any]:
        steps: List[ReasoningStep] = []

        for _ in range(MAX_STEPS):

            # 1️⃣ THINK
            thought = self._think(plan, context, steps)

            step = ReasoningStep(thought=thought)
            steps.append(step)

            # 2️⃣ DECIDE ACTION
            decision = self._decide_action(thought)

            # --------------------------------------------------
            # FINAL ANSWER
            # --------------------------------------------------
            if decision["type"] == "final_answer":
                answer = decision["content"]

                reflection = reflect_on_output(
                    answer=answer,
                    intent=plan.intent,
                )

                if reflection.get("valid", True):
                    return {
                        "answer": answer,
                        "steps": steps,
                        "reflection": reflection,
                    }

                # if reflection fails → think again
                continue

            # --------------------------------------------------
            # TOOL EXECUTION
            # --------------------------------------------------
            if decision["type"] == "tool":

                observation = context.execute_tool(
                    decision["tool"],
                    decision.get("input", {}),
                )

                step.action = decision["tool"]
                step.action_input = decision.get("input", {})
                step.observation = observation

        # fallback if reasoning exhausted
        return {
            "answer": "I couldn't fully complete the task.",
            "steps": steps,
        }

    # ==========================================================
    # INTERNAL THINKING
    # ==========================================================
    def _think(self, plan, context, steps: List[ReasoningStep]) -> str:
        """
        Ask the LLM what should happen next.
        This replaces phrase-based routing.
        """

        history = "\n".join(
            [
                f"Thought: {s.thought}\n"
                f"Action: {s.action}\n"
                f"Observation: {s.observation}"
                for s in steps
            ]
        )

        prompt = f"""
You are an AI reasoning engine.

Intent: {plan.intent}

Previous reasoning:
{history if history else "None"}

Decide the next step.

Rules:
- If you need external information → write:
  TOOL: <tool_name>

- If you can answer → write:
  FINAL: <answer>

Think step-by-step before deciding.
"""

        return self.llm.generate(prompt).strip()

    # ==========================================================
    # DECISION PARSER
    # ==========================================================
    def _decide_action(self, thought: str) -> Dict[str, Any]:
        """
        Converts model reasoning into executable actions.
        """

        # ---------- FINAL ANSWER ----------
        if "FINAL:" in thought:
            return {
                "type": "final_answer",
                "content": thought.split("FINAL:", 1)[-1].strip(),
            }

        # ---------- TOOL CALL ----------
        if "TOOL:" in thought:
            tool_line = thought.split("TOOL:", 1)[1]
            tool = tool_line.split("\n")[0].strip()

            return {
                "type": "tool",
                "tool": tool,
                "input": {},
            }

        # ---------- DEFAULT ----------
        return {
            "type": "final_answer",
            "content": thought.strip(),
        }
