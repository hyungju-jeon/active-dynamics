from typing import Union
import torch
from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric


class ActionCost(BaseMetric):
    """Cost based on action magnitude."""

    def __init__(self, weight: float = 1.0, device: str = "cpu"):
        super().__init__(device)
        self.weight = weight

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        return self.weight * (rollout["action"] ** 2).sum(dim=-1)
