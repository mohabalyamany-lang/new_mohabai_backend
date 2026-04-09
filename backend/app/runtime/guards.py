class ExecutionGuards:

    MAX_REASONING_STEPS = 6
    MAX_TOOL_CALLS = 4

    def __init__(self):
        self.reason_steps = 0
        self.tool_calls = 0

    def allow_reason(self):
        self.reason_steps += 1
        return self.reason_steps <= self.MAX_REASONING_STEPS

    def allow_tool(self):
        self.tool_calls += 1
        return self.tool_calls <= self.MAX_TOOL_CALLS
