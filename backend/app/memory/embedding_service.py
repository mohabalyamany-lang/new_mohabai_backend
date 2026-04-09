import json
from app.services.llm_service import llm_service


class EmbeddingService:

    async def embed(self, text: str):
        emb = await llm_service.embed(text)
        return json.dumps(emb)


embedding_service = EmbeddingService()
