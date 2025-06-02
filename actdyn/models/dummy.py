class DummyModel:
    def __init__(self, **kwargs):
        self.models = [self]  # For compatibility with save/load

    def initialize_belief(self, obs):
        return obs

    def update_belief(self, belief, obs, action):
        return obs

    def predict(self, state, action):
        return state

    def train(self, batch):
        return 0.0  # Dummy loss

    def save(self, path):
        pass  # Nothing to save

    def load(self, path):
        pass  # Nothing to load
