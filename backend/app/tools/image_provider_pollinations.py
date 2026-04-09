from __future__ import annotations

import urllib.parse

from app.tools.image_provider_base import (
    BaseImageProvider,
    ImageGenerationResult,
)


class PollinationsImageProvider(BaseImageProvider):
    """
    Free image provider.
    Stateless — edits are prompt-based reconstruction.
    """

    name = "pollinations"

    def __init__(self, model: str = "flux") -> None:
        self.model = model

    async def generate(self, prompt: str) -> ImageGenerationResult:
        encoded = urllib.parse.quote(prompt[:400])

        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?model={self.model}&width=1024&height=1024"
            f"&nologo=true&enhance=false"
        )

        return ImageGenerationResult(
            ok=True,
            storage_url=url,
            provider=self.name,
            model=self.model,
            revised_prompt=prompt,
        )

    async def edit(
        self,
        instruction: str,
        parent_artifact_reference: str | None = None,
        parent_prompt: str | None = None,
    ) -> ImageGenerationResult:
        """
        Pollinations has no real edit API.
        We reconstruct prompt intelligently.
        """

        if parent_prompt:
            new_prompt = f"{parent_prompt}, {instruction}"
        else:
            new_prompt = instruction

        return await self.generate(new_prompt)
