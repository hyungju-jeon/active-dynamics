from __future__ import annotations

from typing import List, Optional
import torch
from actdyn.utils.rollout import RolloutBuffer, Rollout


class BaseMetric:
    """Base class for metrics/costs"""

    current_cost: Optional[torch.Tensor] = None

    def __init__(self, compute_type: str = "sum", device: str = "cuda"):
        self.device = torch.device(device)
        self.compute_type = compute_type
        self.metric_list = [self]

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer | dict) -> torch.Tensor:
        """Compute and update current metric value per step along the trajectory."""
        raise NotImplementedError

    def aggregate(self) -> torch.Tensor:
        """Compute final metric value over entire trajectory."""
        if self.current_cost is None:
            raise ValueError("No current cost to aggregate. Call compute_stepwise first.")

        if self.compute_type == "sum":
            return self.current_cost.sum(dim=-1)
        elif self.compute_type == "max":
            return self.current_cost.max(dim=-1)[0]
        elif self.compute_type == "last":
            return self.current_cost[..., -1]
        else:
            raise ValueError(f"Invalid compute type: {self.compute_type}")

    def update(self, rollout: Rollout) -> None:
        """Update internal state based on new transition data, if needed."""
        pass

    def __call__(self, rollout: Rollout | RolloutBuffer, **kwargs) -> torch.Tensor:
        self.compute_stepwise(rollout)
        return self.aggregate()


class DiscountedMetric(BaseMetric):
    """Wrapper for discounting a metric."""

    def __init__(self, compute_type: str = "sum", gamma: float = 0.99, device: str = "cpu"):
        super().__init__(compute_type=compute_type, device=device)
        self.gamma = gamma

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer | dict) -> torch.Tensor:
        super().compute_stepwise(rollout)
        T = self.current_cost.shape[-1]
        discounts = self.gamma ** torch.arange(T, device=self.device)
        self.current_cost *= discounts
        return self.current_cost


class CompositeMetric(BaseMetric):
    """Wrapper for composite cost/metric combining multiple functions."""

    def __init__(
        self,
        metrics: List[BaseMetric],
        compute_type: str = "sum",
        weights: Optional[List[float]] = None,
        device: str = "cuda",
    ):
        super().__init__(compute_type=compute_type, device=device)
        self.metric_list = metrics
        self.weights = weights if weights is not None else [1.0] * len(metrics)
        self.weights = torch.tensor(
            self.weights,
        ).to(device)
        assert len(self.weights) == len(
            self.metric_list
        ), "Number of weights must match number of cost functions"

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer | dict) -> torch.Tensor:
        current_cost_list = []
        for weight, metric in zip(self.weights, self.metric_list):
            metric_cost = metric.compute_stepwise(rollout)
            weighted_cost = weight * metric_cost
            current_cost_list.append(weighted_cost)

        acc = None
        for c in current_cost_list:
            acc = c if acc is None else acc + c
        self.current_cost = acc

        return self.current_cost
