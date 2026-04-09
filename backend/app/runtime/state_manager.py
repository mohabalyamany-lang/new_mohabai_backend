class StateManager:

    def apply_patch(self, state, patch):

        new_state = dict(state)

        for k, v in patch.items():
            new_state[k] = v

        return new_state
