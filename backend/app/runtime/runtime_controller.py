from app.runtime.guard_pipeline import guard_pipeline
from app.runtime.postprocessor import postprocessor
from app.runtime.reflection_pipeline import reflection_pipeline
from app.services.orchestrator import conversation_orchestrator
from app.state.state_middleware import state_middleware


class RuntimeController:

    async def handle(
        self,
        db,
        user_id,
        message,
        conversation_state,
    ):
        """
        Standard non-streaming handler with State Integration.
        """
        # ---- 1. Load Global State ----
        # This retrieves long-term user preferences/data from the DB
        global_state = await state_middleware.load(user_id)

        # ---- 2. Pre-guards ----
        guard_result = await guard_pipeline.check(
            user_id=user_id,
            message=message,
        )

        if guard_result:
            return guard_result

        # ---- 3. Core execution ----
        # We merge the current chat state with the global stored state
        reply = await conversation_orchestrator.handle(
            db=db,
            user_id=user_id,
            message=message,
            conversation_state={
                **conversation_state,
                **global_state,
            },
        )

        # ---- 4. Reflection ----
        reply = await reflection_pipeline.run(
            user_message=message,
            reply=reply,
        )

        # ---- 5. Post processing ----
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
        Streaming handler with State Integration.
        """
        # ---- 1. Load Global State ----
        global_state = await state_middleware.load(user_id)

        # ---- 2. Pre-guards ----
        guard_result = await guard_pipeline.check(
            user_id=user_id,
            message=message,
        )

        if guard_result:
            yield guard_result
            return

        # ---- 3. Core execution (Streaming) ----
        full_reply_content = ""

        async for chunk in conversation_orchestrator.stream(
            db=db,
            user_id=user_id,
            message=message,
            conversation_state={
                **conversation_state,
                **global_state,
            },
        ):
            full_reply_content += chunk
            yield chunk

        # ---- 4. Post processing ----
        await postprocessor.run(
            db=db,
            user_id=user_id,
            user_message=message,
            reply=full_reply_content,
        )


runtime_controller = RuntimeController()
