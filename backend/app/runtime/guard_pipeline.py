from __future__ import annotations

from app.runtime.guards import input_guard, rate_limiter


class GuardPipeline:

    async def check(self, user_id: int, message: str) -> str | None:
        # Rate limiting — sliding window, 20 requests per 60 seconds
        if not rate_limiter.allow(user_id):
            return "You're sending messages too fast. Please slow down."

        # Input validation + injection detection
        is_valid, rejection_reason = input_guard.validate(message)
        if not is_valid:
            return rejection_reason

        return None


guard_pipeline = GuardPipeline()