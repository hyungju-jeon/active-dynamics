import numpy as np


class RandomPolicy:
    def __init__(self, action_space, **kwargs):
        self.action_space = action_space

    def __call__(self, state):
        # Return random action from action space
        return self.action_space.sample()
