"""Base policy classes for the active dynamics package."""

import gym
import torch
import numpy as np
from typing import Dict, Any, Callable
from actdyn.models.model import BaseModel


class BasePolicy:
    """Base class for all policies."""

    def __init__(self, action_space: gym.Space, device: str = "cpu"):
        self.action_space = action_space
        self.device = torch.device(device)

    def get_action(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get action for given state."""
        raise NotImplementedError

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy parameters."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save policy to disk."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load policy from disk."""
        raise NotImplementedError

    def to_device(self, device: str) -> None:
        """Move policy to specified device."""

        self.device = torch.device(device)

    def __call__(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get action for given state."""
        return self.get_action(state, **kwargs)


class BaseMPC(BasePolicy):
    """Base class for Model Predictive Control policies."""

    def __init__(
        self,
        action_dim: int,
        cost_fn: Callable,
        model: BaseModel,
        horizon: int,
        num_samples: int,
        num_elite: int,
        device: str = "cpu",
    ):
        super().__init__(action_dim, device)
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elite = num_elite
        self.cost_fn = cost_fn
        self.model = model
