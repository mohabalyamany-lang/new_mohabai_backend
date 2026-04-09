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

                # retry thinking if failed
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
