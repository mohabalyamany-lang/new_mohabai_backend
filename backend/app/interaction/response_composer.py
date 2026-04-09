from __future__ import annotations

from app.interaction.interaction_policy import InteractionPolicy


class ResponseComposer:
    def compose(self, text: str, policy: InteractionPolicy) -> str:
        if not text:
            return text

        if policy.style == "brief":
            return self._make_brief(text)

        if policy.style == "talkative":
            return self._make_more_talkative(text)

        if policy.style == "technical":
            return self._make_more_technical(text)

        return text

    def _make_brief(self, text: str) -> str:
        lines = text.splitlines()
        return lines[0].strip() if lines else text[:300]

    def _make_more_talkative(self, text: str) -> str:
        if len(text) < 120:
            return text + "\n\nIf you want, I can also break this into steps or explain the reasoning behind it."
        return text

    def _make_more_technical(self, text: str) -> str:
        return text + "\n\nIf useful, I can also describe the implementation details and failure modes."
        

response_composer = ResponseComposer()
