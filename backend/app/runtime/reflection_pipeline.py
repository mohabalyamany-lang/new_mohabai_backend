from app.reflection.reflection_engine import reflection_engine
from app.reflection.reliability_controller import reliability_controller


class ReflectionPipeline:

    async def run(self, user_message, reply):
        try:
            reflection = await reflection_engine.evaluate(
                user_message,
                reply,
            )
            repaired = await reliability_controller.stabilize(
                reflection,
                None,
                user_message,
                [],
            )
            if repaired:
                return repaired
        except Exception:
            pass
        return reply


reflection_pipeline = ReflectionPipeline()