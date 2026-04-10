from app.runtime.runtime_controller import runtime_controller


class StreamingController:

    async def stream(
        self,
        db,
        user_id,
        message,
        conversation_state,
    ):
        # We reuse runtime but allow streaming
        async for chunk in runtime_controller.stream_handle(
            db=db,
            user_id=user_id,
            message=message,
            conversation_state=conversation_state,
        ):
            yield chunk


streaming_controller = StreamingController()
