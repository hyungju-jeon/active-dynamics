"""Random-action benchmark baseline policy."""

from __future__ import annotations

import gymnasium as gym
import torch

from .base import BasePolicy


class BaselineRandomPolicy(BasePolicy):
    """Uniform random policy over the provided action space."""

    def __init__(self, action_space: gym.Space, seed: int | None = None, **kwargs):
        super().__init__(action_space=action_space, **kwargs)
        if seed is not None and hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        del state, kwargs
        action = torch.as_tensor(self.action_space.sample(), dtype=torch.float32, device=self.device)
        return action.view(1, 1, -1), torch.zeros(1, device=self.device)
