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
