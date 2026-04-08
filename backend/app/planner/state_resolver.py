from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ResolvedConversationState:
    active_mode: str = "normal_chat"
    pending_followup_kind: str | None = None
    pending_followup_target: str | None = None
    allow_context_carryover: bool = False
    last_artifact_type: str | None = None
    last_artifact_id: str | None = None
    last_artifact_prompt: str | None = None
    last_user_message: str | None = None
    last_assistant_message: str | None = None
    has_files: bool = False
    recent_turn_count: int = 0


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def is_short_followup(text: str) -> bool:
    return len(normalize_text(text).split()) <= 5


def is_affirmation(text: str) -> bool:
    t = normalize_text(text)
    return t in {
        "ok",
        "okay",
        "yes",
        "yeah",
        "sure",
        "go ahead",
        "do it",
        "continue",
        "again",
        "retry",
    }


def is_social_feedback(text: str) -> bool:
    t = normalize_text(text).rstrip(".!?")
    return t in {
        "thats nice",
        "that's nice",
        "nice",
        "cool",
        "great",
        "good",
        "love it",
        "perfect",
        "awesome",
        "amazing",
        "beautiful",
    }


def is_style_request(text: str) -> bool:
    t = normalize_text(text)
    prefixes = (
        "be more ",
        "be less ",
        "be a bit more ",
        "be a bit less ",
        "be concise",
        "be more concise",
        "be more talkative",
        "talk more",
        "be clearer",
        "sound more natural",
        "sound human",
    )
    return any(t.startswith(p) for p in prefixes)


def is_general_chat_switch(text: str) -> bool:
    t = normalize_text(text)
    prefixes = (
        "who are you",
        "who made you",
        "what are you",
        "tell me about yourself",
        "why are you",
        "can we talk",
        "lets talk",
        "let's talk",
        "i said ",
        "that is not what i meant",
        "thats not what i meant",
        "that's not what i meant",
    )
    return any(t.startswith(p) for p in prefixes)


def looks_like_image_request(text: str) -> bool:
    t = normalize_text(text)
    verbs = ("make ", "generate ", "create ", "draw ", "paint ", "render ", "sketch ")
    nouns = ("image", "picture", "photo", "pic", "drawing", "art", "illustration")
    if any(t.startswith(v) for v in verbs) and any(n in t for n in nouns):
        return True
    return False


def looks_like_image_edit(text: str) -> bool:
    t = normalize_text(text)
    edit_markers = (
        "make it ",
        "change it ",
        "turn it ",
        "edit it ",
        "modify it ",
        "same but ",
        "but make it ",
        "use the same ",
        "keep the same ",
        "remove ",
        "add ",
        "change ",
    )
    direct_terms = (
        "blue",
        "red",
        "green",
        "darker",
        "lighter",
        "background",
        "bigger",
        "smaller",
        "more realistic",
        "cartoon",
        "anime",
    )
    return any(t.startswith(m) for m in edit_markers) or any(term in t for term in direct_terms)


def looks_like_image_question(text: str) -> bool:
    t = normalize_text(text)
    markers = (
        "describe it",
        "analyze it",
        "what is in it",
        "what's in it",
        "what color",
        "what does it show",
        "describe the image",
        "describe this",
        "what do you see",
    )
    return any(m in t for m in markers)


def needs_live_information(text: str) -> bool:
    t = normalize_text(text)
    live_markers = (
        "today",
        "current",
        "now",
        "latest",
        "weather",
        "forecast",
        "price",
        "rate",
        "score",
        "time",
        "date",
        "news",
    )
    question_markers = ("what", "when", "how much", "how many", "is", "are")
    return any(m in t for m in live_markers) and any(q in t for q in question_markers)


def is_lookup_followup(text: str) -> bool:
    t = normalize_text(text)
    return t in {
        "look it up",
        "search it",
        "check it",
        "find it",
        "search that",
        "look that up",
        "check that",
        "find that",
    }
