import gymnasium as gym
import torch
from actdyn.policy.base import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state: torch.Tensor):
        return (
            torch.FloatTensor(self.action_space.sample()).unsqueeze(0).unsqueeze(0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )


class StepPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, step_size=1000, **kwargs):
        super().__init__(action_space, **kwargs)
        self.step_size = step_size
        self.current_step = 0
        self.current_action = torch.zeros(self.action_space.shape).to(self.device)

    def get_action(self, state: torch.Tensor):
        if self.current_step % self.step_size == 0:
            self.current_action = torch.FloatTensor(self.action_space.sample()).to(self.device)
        self.current_step += 1
        return self.current_action.unsqueeze(0).unsqueeze(0), torch.tensor(0.0).to(self.device)


class OffPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state):
        return (
            torch.FloatTensor(self.action_space.sample() * 0.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device),
            torch.tensor(0.0).to(self.device),
        )

    def update(self, batch):
        pass
