"""Base policy classes for the active dynamics package."""

import torch
from typing import Dict
from actdyn.metrics.base import BaseMetric
from actdyn.models import BaseModel
import gymnasium as gym


class BasePolicy:
    """Base class for all policies."""

    def __init__(self, action_space: gym.Space, chunk=1, device: str = "cpu"):
        self.action_space = action_space
        self.chunk = chunk
        self.device = torch.device(device)
        self.count = 0
        self.action_list = []
        self.cost = 0.0

    def get_action(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get action for given state."""
        raise NotImplementedError

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy parameters."""
        raise NotImplementedError

    def to_device(self, device: str) -> None:
        """Move policy to specified device."""

        self.device = torch.device(device)

    def __call__(self, state, **kwargs) -> torch.Tensor:
        """Get action for given state."""
        if self.chunk > 1:
            if self.count % self.chunk == 0:
                action, cost = self.get_action(state, **kwargs)
                if action.shape[-2] == 1:
                    self.action_list = [action] * self.chunk
                elif action.shape[-2] >= self.chunk:
                    self.action_list = [a.unsqueeze(0) for a in action[0, : self.chunk]]
                else:
                    raise ValueError(
                        f"Action sequence length {action.shape[-2]} is less than chunk size {self.chunk}"
                    )
                self.cost = cost.squeeze().item()

            action = self.action_list[self.count % self.chunk]
            self.count += 1
            return action
        else:
            action, cost = self.get_action(state, **kwargs)
            self.cost = cost.squeeze().item()

            return action[:, :1]


class BaseMPC(BasePolicy):
    """Base class for Model Predictive Control policies.
    Currently, we only support continuous control with box action space."""

    def __init__(
        self,
        metric: BaseMetric,
        model: BaseModel,
        horizon: int,
        num_samples: int,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(model.action_encoder.action_space, **kwargs)
        # Accept both gym and gymnasium Box spaces
        assert isinstance(self.action_space, gym.spaces.Box), "Only box action space is supported"
        self.action_dim = self.action_space.shape[0]
        self.action_bounds = (
            (self.action_space.low, self.action_space.high)
            if self.action_space.is_bounded()
            else None
        )
        self.horizon = horizon
        self.num_samples = num_samples
        self.metric = metric
        self.model = model
        self.verbose = verbose

    def beginning_of_rollout(self, state: torch.Tensor):
        pass

    def end_of_rollout(self, state: torch.Tensor):
        pass
