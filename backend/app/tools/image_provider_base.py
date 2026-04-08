from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class ImageGenerationResult:
    ok: bool
    storage_url: str | None = None
    inline_data: str | None = None
    provider: str | None = None
    model: str | None = None
    revised_prompt: str | None = None
    error: str | None = None


class BaseImageProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, prompt: str) -> ImageGenerationResult:
        raise NotImplementedError

    @abstractmethod
    async def edit(
        self,
        instruction: str,
        parent_artifact_reference: str | None = None,
        parent_prompt: str | None = None,
    ) -> ImageGenerationResult:
        raise NotImplementedError
