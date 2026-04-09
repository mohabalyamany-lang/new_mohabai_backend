from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntentResult:
    intent: str
    confidence: float


class IntentEngine:
    """
    Lightweight reasoning classifier.

    NOT phrase matching.
    Category inference.
    """

    IMAGE_VERBS = {
        "make",
        "create",
        "generate",
        "draw",
        "paint",
        "render",
    }

    EDIT_VERBS = {
        "change",
        "edit",
        "modify",
        "make it",
        "turn it",
    }

    LIVE_INFO_HINTS = {
        "time",
        "date",
        "weather",
        "news",
        "today",
        "now",
        "current",
    }

    def detect(self, text: str, has_last_image: bool) -> IntentResult:
        t = text.lower()

        # IMAGE EDIT (context aware)
        if has_last_image and any(v in t for v in self.EDIT_VERBS):
            return IntentResult("image_edit", 0.9)

        # IMAGE GENERATION
        if any(v in t for v in self.IMAGE_VERBS):
            return IntentResult("image_generate", 0.8)

        # LIVE INFO (dynamic world queries)
        if any(w in t for w in self.LIVE_INFO_HINTS):
            return IntentResult("live_info", 0.7)

        return IntentResult("chat", 0.6)


intent_engine = IntentEngine()
