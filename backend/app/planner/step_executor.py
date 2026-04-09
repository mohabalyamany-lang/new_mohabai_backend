import json


class StepExecutor:

    def next_step(self, task):

        plan = json.loads(task.plan_json)
        steps = plan["steps"]

        if task.current_step >= len(steps):
            task.status = "completed"
            return None

        return steps[task.current_step]


step_executor = StepExecutor()
