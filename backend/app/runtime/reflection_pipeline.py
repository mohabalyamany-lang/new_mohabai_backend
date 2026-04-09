from app.reflection.reliability_controller import reliability_controller


class ReflectionPipeline:

    async def run(self, user_message, reply):

        scores = await reliability_controller.score(
            user_message,
            reply,
        )

        repaired = await reliability_controller.stabilize(
            scores,
            user_message,
            reply,
        )

        if repaired:
            return repaired

        return reply


reflection_pipeline = ReflectionPipeline()
