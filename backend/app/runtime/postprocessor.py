from app.memory.memory_extractor import memory_extractor
from app.memory.memory_manager import memory_manager
from app.evals.eval_logger import eval_logger


class PostProcessor:

    async def run(self, db, user_id, user_message, reply):
        # Memory learning — silent, never crashes main response
        try:
            mems = await memory_extractor.extract(
                f"User: {user_message}\nAssistant: {reply}",
            )
            for mem in mems:
                await memory_manager.store(db, user_id, mem)
        except Exception:
            pass

        # Eval logging — silent
        try:
            await eval_logger.log(
                user_message=user_message,
                reply=reply,
            )
        except Exception:
            pass


postprocessor = PostProcessor()