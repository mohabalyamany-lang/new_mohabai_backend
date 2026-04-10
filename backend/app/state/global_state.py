class GlobalState:

    def __init__(self):
        self._state = {}

    def get(self, user_id):
        return self._state.get(user_id, {})

    def set(self, user_id, value):
        self._state[user_id] = value

    def update(self, user_id, patch):
        current = self._state.get(user_id, {})
        current.update(patch)
        self._state[user_id] = current


global_state = GlobalState()
