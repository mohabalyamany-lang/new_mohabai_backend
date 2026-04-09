from __future__ import annotations


class ToneAdapter:
    def mirror(self, user_message: str, assistant_text: str) -> str:
        if not user_message or not assistant_text:
            return assistant_text

        user = user_message.strip()

        if user.endswith("!") and not assistant_text.endswith("!"):
            return assistant_text + "!"

        return assistant_text


tone_adapter = ToneAdapter()
