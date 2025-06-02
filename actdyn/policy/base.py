"""Base policy classes for the active dynamics package."""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any


class BasePolicy:
    """Base class for all policies."""

    def __init__(self, action_dim: int, device: str = "cpu"):
        self.action_dim = action_dim
        self.device = torch.device(device)

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False, **kwargs
    ) -> torch.Tensor:
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


class BaseMPC(BasePolicy):
    """Base class for Model Predictive Control policies."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        num_samples: int,
        num_elite: int,
        device: str = "cpu",
    ):
        super().__init__(action_dim, device)
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elite = num_elite

    def optimize(
        self, state: torch.Tensor, dynamics_model: Any, cost_fn: Any, **kwargs
    ) -> torch.Tensor:
        """Optimize action sequence."""
        raise NotImplementedError
