import gym
from actdyn.policy.base import BasePolicy


class LazyPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state):
        return self.action_space.sample() * 0.0

    def update(self, batch):
        pass

    def __call__(self, state):
        return self.get_action(state)
