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


runtime_controller = RuntimeController()
