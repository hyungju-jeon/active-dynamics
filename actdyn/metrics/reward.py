from __future__ import annotations

import torch

from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric


class RewardMetric(BaseMetric):
    """Basic reward metric that sums rewards."""

    def __init__(self, compute_type: str = "sum", device: str = "cuda", **kwargs):
        super().__init__(compute_type=compute_type, device=device)

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer) -> torch.Tensor:
        self.current_cost = -rollout["reward"]
        return self.current_cost


class GoalDistanceMetric(BaseMetric):
    """Metric based on final distance to goal."""

    def __init__(
        self,
        goal: torch.Tensor,
        compute_type: str = "last",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(compute_type=compute_type, device=device)
        self.set_goal(goal)

    def set_goal(self, goal: torch.Tensor):
        """Set the goal for the metric."""
        self.goal = goal.to(self.device)

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer) -> torch.Tensor:
        self.current_metric = torch.norm(rollout["model_state"] - self.goal, dim=-1)
        return self.current_metric
