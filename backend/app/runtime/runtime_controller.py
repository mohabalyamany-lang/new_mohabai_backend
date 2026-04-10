from app.runtime.guard_pipeline import guard_pipeline
from app.runtime.postprocessor import postprocessor
from app.runtime.reflection_pipeline import reflection_pipeline
from app.services.orchestrator import conversation_orchestrator


class RuntimeController:

    async def handle(
        self,
        db,
        user_id,
        message,
        conversation_state,
    ):
        """
        Standard non-streaming handler. 
        Best for short messages or background processing.
        """
        # ---- Pre-guards ----
        guard_result = await guard_pipeline.check(
            user_id=user_id,
            message=message,
        )

        if guard_result:
            return guard_result

        # ---- Core execution ----
        reply = await conversation_orchestrator.handle(
            db=db,
            user_id=user_id,
            message=message,
            conversation_state=conversation_state,
        )

        # ---- Reflection ----
        reply = await reflection_pipeline.run(
            user_message=message,
            reply=reply,
        )

        # ---- Post processing ----
        await postprocessor.run(
            db=db,
            user_id=user_id,
            user_message=message,
            reply=reply,
        )

        return reply

    async def stream_handle(
        self,
        db,
        user_id,
        message,
        conversation_state,
    ):
        """
        High-performance streaming handler.
        Allows the UI to display characters as they are generated.
        """
        # ---- Pre-guards ----
        guard_result = await guard_pipeline.check(
            user_id=user_id,
            message=message,
        )

        if guard_result:
            yield guard_result
            return

        # ---- Core execution (Streaming) ----
        # We collect the full reply in a variable to run post-processing at the end
        full_reply_content = ""

        async for chunk in conversation_orchestrator.stream(
            db=db,
            user_id=user_id,
            message=message,
            conversation_state=conversation_state,
        ):
            full_reply_content += chunk
            yield chunk

        # ---- Post processing (Background) ----
        # Note: Reflection is usually skipped or handled differently in streaming 
        # to prevent "rewriting" text the user has already seen.
        await postprocessor.run(
            db=db,
            user_id=user_id,
            user_message=message,
            reply=full_reply_content,
        )


runtime_controller = RuntimeController()
