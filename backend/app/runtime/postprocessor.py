from app.memory.memory_extractor import memory_extractor
from app.memory.memory_manager import memory_manager
from app.evals.eval_logger import eval_logger


class PostProcessor:

    async def run(
        self,
        db,
        user_id,
        user_message,
        reply,
    ):
        # Memory learning
        mems = await memory_extractor.extract(
            user_message,
            reply,
        )

        for mem in mems:
            await memory_manager.store(
                db,
                user_id,
                mem,
            )

        # Eval logging
        await eval_logger.log(
            user_message=user_message,
            reply=reply,
        )


postprocessor = PostProcessor()
