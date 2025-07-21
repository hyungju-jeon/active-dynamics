from typing import Union
import torch
from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric


class ActionCost(BaseMetric):
    """Cost based on action magnitude."""

    def __init__(self, compute_type="sum", device: str = "cpu", **kwargs):
        super().__init__(compute_type, device)

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        self.metric = (rollout["action"] ** 2).sum(dim=-1)
        return self.metric
