from app.guards.rate_limiter import rate_limiter
from app.guards.input_guard import input_guard


class GuardPipeline:

    async def check(self, user_id, message):

        # Rate limit
        allowed = await rate_limiter.allow(user_id)
        if not allowed:
            return "You're sending messages too fast. Please slow down."

        # Input validation
        violation = input_guard.validate(message)
        if violation:
            return violation

        return None


guard_pipeline = GuardPipeline()
