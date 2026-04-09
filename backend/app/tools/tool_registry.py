from __future__ import annotations

from typing import Dict

from app.tools.image_provider_pollinations import PollinationsImageProvider


class ToolRegistry:
    """
    Central place where runtime discovers capabilities.
    """

    def __init__(self) -> None:
        self.image_provider = PollinationsImageProvider()

    # ---------- IMAGE ----------

    async def generate_image(self, prompt: str):
        return await self.image_provider.generate(prompt)

    async def edit_image(
        self,
        instruction: str,
        parent_prompt: str | None,
        parent_reference: str | None,
    ):
        return await self.image_provider.edit(
            instruction,
            parent_prompt=parent_prompt,
            parent_artifact_reference=parent_reference,
        )


tool_registry = ToolRegistry()
