from app.state.global_state import global_state


class StateManager:

    async def load(self, user_id):
        return global_state.get(user_id)

    async def save(self, user_id, state):
        global_state.set(user_id, state)

    async def patch(self, user_id, patch):
        global_state.update(user_id, patch)


state_manager = StateManager()
