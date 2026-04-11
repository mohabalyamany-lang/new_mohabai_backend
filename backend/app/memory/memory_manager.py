from app.memory.memory_store import memory_store


class MemoryManager:

    async def store(self, db, user_id: int, memory: str):
        await memory_store.save_memories(
            db,
            user_id=user_id,
            memories=[memory],
        )

memory_manager = MemoryManager()