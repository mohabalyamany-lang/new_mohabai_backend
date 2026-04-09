from app.services.llm_service import llm_service


class ContextCompressor:

    async def compress(self, messages):

        if len(messages) < 20:
            return messages

        summary = await llm_service.chat([
            {
                "role": "system",
                "content": "Summarize conversation preserving goals and facts."
            },
            {
                "role": "user",
                "content": str(messages[:-10])
            }
        ])

        return [
            {"role": "system", "content": f"Conversation summary:\n{summary}"}
        ] + messages[-10:]
