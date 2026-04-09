from __future__ import annotations


class ClarificationEngine:
    def should_clarify(self, user_message: str, planner_action: dict) -> bool:
        decision = (planner_action.get("decision") or "").lower()
        intent = (planner_action.get("intent") or "").lower()

        if decision == "ask":
            return True

        if intent == "chat":
            return False

        if intent in {"image_edit", "image_question", "web_search"}:
            return False

        text = (user_message or "").strip().lower()
        if len(text) < 3:
            return True

        return False


clarification_engine = ClarificationEngine()
