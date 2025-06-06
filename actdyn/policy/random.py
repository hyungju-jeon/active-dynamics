import gym
import torch
from actdyn.policy.base import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state: torch.Tensor):
        return self.action_space.sample()

    def __call__(self, state: torch.Tensor):
        return self.get_action(state)
