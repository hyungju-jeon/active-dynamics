import gymnasium as gym
import numpy as np
import torch
from actdyn.policy.base import BasePolicy


def sample_in_ball(rng: np.random.Generator, radius: float, dim: int) -> np.ndarray:
    """One input drawn uniformly from the ball ||u|| <= radius, shape (dim,)."""
    direction = rng.normal(size=dim)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return (radius * rng.random() ** (1.0 / dim) * direction).astype(np.float32)


class RandomPolicy(BasePolicy):
    """Independent random input at every step: uniform in the action-space box, or uniform in the
    ball ||u|| <= action_radius when ``action_radius`` is set (both drawn from the action space's
    seeded generator)."""

    def __init__(self, action_space: gym.Space, action_radius: float | None = None, **kwargs):
        super().__init__(action_space, **kwargs)
        self.action_radius = None if action_radius is None else float(action_radius)

    def get_action(self, state: torch.Tensor, **kwargs):
        if self.action_radius is None:
            action = self.action_space.sample()
        else:
            action = sample_in_ball(self.action_space.np_random, self.action_radius, int(np.prod(self.action_space.shape)))
        return (
            torch.FloatTensor(action).unsqueeze(0).unsqueeze(0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )


class StepPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, step_size=1000, **kwargs):
        super().__init__(action_space, **kwargs)
        self.step_size = step_size
        self.current_step = 0
        self.current_action = torch.zeros(self.action_space.shape).to(self.device)

    def get_action(self, state: torch.Tensor, **kwargs):
        if self.current_step % self.step_size == 0:
            self.current_action = torch.FloatTensor(self.action_space.sample()).to(self.device)
        self.current_step += 1
        return self.current_action.unsqueeze(0).unsqueeze(0), torch.tensor(0.0).to(self.device)


class OffPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space, **kwargs):
        super().__init__(action_space, **kwargs)

    def get_action(self, state, **kwargs):
        return (
            torch.FloatTensor(self.action_space.sample() * 0.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device),
            torch.tensor(0.0).to(self.device),
        )

    def update(self, batch):
        pass
