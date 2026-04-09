from __future__ import annotations

from sqlalchemy.orm import Session

from app.memory.memory_retriever import memory_retriever
from app.planner.planning_prompts import PLANNER_PROMPT
from app.services.artifact_service import ArtifactService
from app.services.context.context_types import ContextBundle, ContextMessage
from app.services.context.conversation_memory_service import memory_service

SYSTEM_PROMPT = """
You are Mohab AI, an advanced conversational agent.
- Always infer user intent, do not rely on keywords.
- Use tools autonomously when needed: web_search, generate_image, edit_image.
- Maintain context across turns; remember ongoing goals.
- Self-correct if answers are wrong or incomplete.
- Use memory and past conversation to provide continuity.
- Respond naturally, in the tone preferred by the user.
- Follow interaction policy (brief, talkative, casual, technical).
- Never say you cannot do something if a tool exists.
- When uncertain, ask a clarifying question before assuming.
- Keep reasoning tight and actionable.
- Ensure responses are accurate, helpful, and user-focused.
"""

MEMORY_PROMPT_PREFIX = "Include the following user facts and past context to answer accurately and naturally:\n"

INTERACTION_PROMPT = """
Follow the interaction policy:
- Adjust tone according to user preferences
- Ask clarifying questions only when needed
- Be brief, detailed, casual, technical, or talkative as indicated
- Mirror user tone subtly for natural flow
"""


class ContextBuilder:
    async def build(
        self,
        db: Session,
        conversation_id: int,
        user_message: str,
        user_id: int | None = None,
    ) -> ContextBundle:

        recent = memory_service.load_recent_messages(db, conversation_id)
        messages = []

        # 1. SYSTEM PROMPT — global behavior rules
        messages.append(ContextMessage(role="system", content=SYSTEM_PROMPT))

        # 2. MEMORY PROMPT — known user facts from long-term memory
        if user_id is not None:
            memories = await memory_retriever.retrieve(
                db,
                user_id=user_id,
                query=user_message,
            )
            if memories:
                messages.append(
                    ContextMessage(
                        role="system",
                        content=(
                            MEMORY_PROMPT_PREFIX
                            + "\n".join(f"- {m}" for m in memories)
                        ),
                    )
                )

        # 3. PLANNER PROMPT — reasoning and tool decision instructions
        messages.append(ContextMessage(role="system", content=PLANNER_PROMPT))

        # 4. INTERACTION PROMPT — tone and style rules
        messages.append(ContextMessage(role="system", content=INTERACTION_PROMPT))

        # 5. CONVERSATION HISTORY
        for m in recent:
            messages.append(ContextMessage(role=m.role, content=m.content))

        # 6. CURRENT USER MESSAGE
        messages.append(ContextMessage(role="user", content=user_message))

        last_image = ArtifactService.get_last_image(db, conversation_id)
        return ContextBundle(
            messages=messages,
            last_image_prompt=(
                last_image.effective_prompt if last_image else None
            ),
        )


context_builder = ContextBuilder()
