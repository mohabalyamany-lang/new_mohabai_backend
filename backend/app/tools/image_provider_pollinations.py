from __future__ import annotations

import urllib.parse

from app.tools.image_provider_base import BaseImageProvider, ImageGenerationResult


class PollinationsImageProvider(BaseImageProvider):
    name = "pollinations"

    def __init__(self, model: str = "flux") -> None:
        self.model = model

    async def generate(self, prompt: str) -> ImageGenerationResult:
        encoded = urllib.parse.quote(prompt[:400])
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?model={self.model}&width=1024&height=1024&nologo=true&enhance=false"
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
        merged_prompt = instruction if not parent_prompt else f"{parent_prompt}. Edit instruction: {instruction}"
        return await self.generate(merged_prompt)
