import json
import hashlib


class EmbeddingService:

    async def embed(self, text: str) -> str:
        # Stub — returns a deterministic fake embedding
        # Replace with real OpenAI/Groq embeddings later
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        size = 1536
        vector = [(((hash_val >> i) & 0xFF) / 255.0) for i in range(size)]
        return json.dumps(vector)


embedding_service = EmbeddingService()