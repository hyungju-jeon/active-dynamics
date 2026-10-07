"""Random-action benchmark baseline policy."""

from __future__ import annotations

import gymnasium as gym
import torch

from .base import BasePolicy
from .policy import sample_in_ball


class BaselineRandomPolicy(BasePolicy):
    """Uniform random policy over the provided action space, or over the ball ||u|| <= action_radius
    when ``action_radius`` is set."""

    def __init__(self, action_space: gym.Space, seed: int | None = None, action_radius: float | None = None,
                 **kwargs):
        super().__init__(action_space=action_space, **kwargs)
        if seed is not None and hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        self.action_radius = None if action_radius is None else float(action_radius)

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        del state, kwargs
        if self.action_radius is None:
            sample = self.action_space.sample()
        else:
            sample = sample_in_ball(self.action_space.np_random, self.action_radius, int(self.action_space.shape[0]))
        action = torch.as_tensor(sample, dtype=torch.float32, device=self.device)
        return action.view(1, 1, -1), torch.zeros(1, device=self.device)
