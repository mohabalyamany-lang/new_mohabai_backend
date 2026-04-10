from app.state.state_manager import state_manager


class StateMiddleware:

    async def load(self, user_id):
        return await state_manager.load(user_id)

    async def save(self, user_id, state):
        await state_manager.save(user_id, state)


state_middleware = StateMiddleware()
