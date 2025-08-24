import torch

from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric


class RewardMetric(BaseMetric):
    """Basic reward metric that sums rewards."""

    def __init__(self, compute_type: str = "sum", device: str = "cuda", **kwargs):
        super().__init__(compute_type, device)

    def compute(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        self.metric = -rollout["reward"]
        return self.metric


class GoalDistanceMetric(BaseMetric):
    """Metric based on final distance to goal."""

    def __init__(
        self,
        goal: torch.Tensor,
        compute_type: str = "sum",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(compute_type, device)
        self.goal = goal.to(device)

    def set_goal(self, goal: torch.Tensor):
        """Set the goal for the metric."""
        self.goal = goal.to(self.device)

    def compute(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        self.metric = (
            torch.norm(rollout["model_state"] - self.goal, dim=-1).sum(dim=-1).unsqueeze(-1)
        )
        return self.metric
