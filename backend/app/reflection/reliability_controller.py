class ReliabilityController:

    async def stabilize(
        self,
        reflection,
        agent,
        user_message,
        context_messages,
    ):

        if not reflection.get("needs_revision"):
            return None

        strategy = reflection.get("fix_strategy")

        if strategy == "retry_reasoning":
            return await agent.reason(
                user_message,
                context_messages,
                retry=True,
            )

        if strategy == "clarify":
            return "Could you clarify what you mean?"

        return None


reliability_controller = ReliabilityController()
