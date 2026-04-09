from __future__ import annotations

from app.interaction.interaction_policy import InteractionPolicy


class PolicyResolver:
    def resolve(self, user_message: str, state: dict | None = None) -> InteractionPolicy:
        text = (user_message or "").lower()
        state = state or {}

        policy = InteractionPolicy()

        if "be more talkative" in text or "talk more" in text:
            policy.style = "talkative"

        elif "be brief" in text or "short answer" in text or "keep it short" in text:
            policy.style = "brief"
            policy.keep_context_tight = True

        elif "explain in detail" in text or "be detailed" in text:
            policy.style = "detailed"

        elif "be more technical" in text or "technical" in text:
            policy.style = "technical"

        elif "be casual" in text or "more casual" in text:
            policy.style = "casual"

        return policy


policy_resolver = PolicyResolver()
