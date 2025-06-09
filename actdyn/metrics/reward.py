import torch

from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric


class RewardMetric(BaseMetric):
    """Basic reward metric that sums rewards."""

    def compute(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        return rollout["reward"]


class DiscountedRewardMetric(BaseMetric):
    """Discounted reward metric."""

    def __init__(self, gamma: float = 0.99, device: str = "cuda"):
        super().__init__(device)
        self.gamma = gamma

    def compute(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        return (
            self.gamma ** torch.arange(rollout["reward"].shape[-2], device=self.device)
            * rollout["reward"]
        )


class GoalDistanceMetric(BaseMetric):
    """Metric based on final distance to goal."""

    def __init__(self, goal: torch.Tensor, device: str = "cuda"):
        super().__init__(device)
        self.goal = goal.to(device)

    def compute(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        return -torch.norm(rollout["state"][..., -1, :] - self.goal, dim=-1)
