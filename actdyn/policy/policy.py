import gymnasium as gym
import torch
from actdyn.policy.base import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state: torch.Tensor):
        return torch.FloatTensor(self.action_space.sample()).to(self.device)


class LazyPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state):
        return self.action_space.sample() * 0.0

    def update(self, batch):
        pass
