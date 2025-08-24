import gymnasium as gym
import torch
from actdyn.policy.base import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state: torch.Tensor):
        return (
            torch.FloatTensor(self.action_space.sample()).unsqueeze(0).unsqueeze(0).to(self.device)
        )


class OffPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state):
        return (
            torch.FloatTensor(self.action_space.sample() * 0.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

    def update(self, batch):
        pass
