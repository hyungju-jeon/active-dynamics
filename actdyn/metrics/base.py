from typing import List, Optional
import torch
from actdyn.utils.rollout import RolloutBuffer, Rollout


class BaseMetric:
    """Base class for metrics that compute both point-wise and final values."""

    def __init__(self, compute_type: str = "sum", device: str = "cuda"):
        self.device = device
        self.metric = None
        self.compute_type = compute_type

    def compute(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        """Compute metric value along the trajectory."""
        raise NotImplementedError

    def compute_final(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        """Compute final metric value over entire trajectory."""
        if self.metric is None:
            self.metric = self.compute(rollout)

        if self.compute_type == "sum":
            return self.metric.sum(dim=-2)
        elif self.compute_type == "max":
            return self.metric.max(dim=-2)[0]
        elif self.compute_type == "last":
            return self.metric[..., -1, :]
        else:
            raise ValueError(f"Invalid compute type: {self.compute_type}")

    def __call__(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        return self.compute(rollout)


class CompositeSumCost(BaseMetric):
    """Composite cost combining multiple cost functions."""

    def __init__(
        self,
        cost_fns: List[BaseMetric],
        weights: Optional[List[float]] = None,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.cost_fns = cost_fns
        self.weights = weights if weights is not None else [1.0] * len(cost_fns)
        assert len(self.weights) == len(
            self.cost_fns
        ), "Number of weights must match number of cost functions"

    def compute(self, rollout: Rollout or RolloutBuffer) -> torch.Tensor:
        total_cost = 0.0
        for cost_fn, weight in zip(self.cost_fns, self.weights):
            total_cost += weight * cost_fn.compute(rollout)
        return total_cost
