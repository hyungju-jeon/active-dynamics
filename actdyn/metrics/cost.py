from __future__ import annotations

import torch
from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric


class ActionCost(BaseMetric):
    """Cost based on action magnitude."""

    def __init__(self, compute_type="sum", device: str = "cpu", **kwargs):
        super().__init__(compute_type, device)

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer, **kwargs) -> torch.Tensor:
        """cost = ||action||_2"""
        self.current_cost = torch.sqrt((rollout["action"] ** 2).sum(dim=-1))
        return self.current_cost


class NormalizedActionCost(BaseMetric):
    """Dimensionless squared action cost normalized by per-dimension bounds."""

    def __init__(
        self,
        action_scale,
        compute_type="sum",
        device: str = "cpu",
        normalize_horizon: bool = True,
        **kwargs,
    ):
        super().__init__(compute_type, device)
        scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device).reshape(
            1, 1, -1
        )
        self.action_scale = scale.clamp_min(1e-8)
        self.normalize_horizon = bool(normalize_horizon)

    @classmethod
    def from_action_bounds(
        cls,
        action_bounds,
        compute_type="sum",
        device: str = "cpu",
        normalize_horizon: bool = True,
        **kwargs,
    ):
        low, high = action_bounds
        low_t = torch.as_tensor(low, dtype=torch.float32, device=device).reshape(-1)
        high_t = torch.as_tensor(high, dtype=torch.float32, device=device).reshape(-1)
        scale = torch.minimum(low_t.abs(), high_t.abs())
        fallback = torch.maximum(low_t.abs(), high_t.abs())
        scale = torch.where(scale > 1e-8, scale, fallback)
        return cls(
            action_scale=scale,
            compute_type=compute_type,
            device=device,
            normalize_horizon=normalize_horizon,
            **kwargs,
        )

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer, **kwargs) -> torch.Tensor:
        """cost_t = mean_d (u_t,d / u_max,d)^2, optionally divided by horizon."""
        action = rollout["action"].to(self.device).float()
        cost = ((action / self.action_scale.to(action.device, action.dtype)) ** 2).mean(dim=-1)
        if self.normalize_horizon and cost.shape[-1] > 0:
            cost = cost / float(cost.shape[-1])
        self.current_cost = cost
        return self.current_cost
