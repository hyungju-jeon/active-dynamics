"""Base policy classes for the active dynamics package."""

import gym
import torch
import numpy as np
from typing import Dict, Any, Callable
from actdyn.models import BaseModel
import gymnasium


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

    def __call__(self, state, **kwargs) -> torch.Tensor:
        """Get action for given state."""
        return torch.FloatTensor(self.get_action(state))


class BaseMPC(BasePolicy):
    """Base class for Model Predictive Control policies.
    Currently, we only support continuous control with box action space."""

    def __init__(
        self,
        cost_fn: Callable,
        model: BaseModel,
        mpc_params: Dict[str, Any],
        verbose: bool = False,
        device: str = "cpu",
    ):
        super().__init__(model.action_space, device)
        # Accept both gym and gymnasium Box spaces
        assert isinstance(
            self.action_space, gymnasium.spaces.Box
        ), "Only box action space is supported"
        self.action_dim = self.action_space.shape[0]
        self.action_bounds = (
            (self.action_space.low, self.action_space.high)
            if self.action_space.is_bounded()
            else None
        )
        self.horizon = mpc_params["horizon"]
        self.num_samples = mpc_params["num_samples"]
        self.cost_fn = cost_fn
        self.model = model
        self.verbose = verbose

    def beginning_of_rollout(self, state: torch.Tensor):
        pass
