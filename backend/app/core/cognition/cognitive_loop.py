from typing import List
from .reasoning_step import ReasoningStep
from .reflection import reflect_on_output

MAX_STEPS = 3


class CognitiveLoop:

    def __init__(self, llm):
        self.llm = llm

    def run(self, plan, context):
        steps: List[ReasoningStep] = []

        for _ in range(MAX_STEPS):

            thought = self._think(plan, context, steps)

            step = ReasoningStep(thought=thought)
            steps.append(step)

            decision = self._decide_action(thought)

            if decision["type"] == "final_answer":
                answer = decision["content"]

                reflection = reflect_on_output(
                    answer,
                    plan.intent
                )

                if reflection["valid"]:
                    return {
                        "answer": answer,
                        "steps": steps,
                        "reflection": reflection,
                    }

                continue

            if decision["type"] == "tool":
                observation = context.execute_tool(
                    decision["tool"],
                    decision["input"]
                )

                step.action = decision["tool"]
                step.action_input = decision["input"]
                step.observation = observation

        return {
            "answer": "I couldn't fully complete the task.",
            "steps": steps,
        }

    def _think(self, plan, context, steps):
        prompt = f"""
Intent: {plan.intent}
History steps: {len(steps)}

What should be done next?
Think step-by-step.
"""
        return self.llm.generate(prompt)

    def _decide_action(self, thought: str):

        if "FINAL:" in thought:
            return {
                "type": "final_answer",
                "content": thought.split("FINAL:")[-1].strip(),
            }

        if "TOOL:" in thought:
            tool = thought.split("TOOL:")[1].split("\n")[0].strip()

            return {
                "type": "tool",
                "tool": tool,
                "input": {},
            }

        return {
            "type": "final_answer",
            "content": thought,
        }
