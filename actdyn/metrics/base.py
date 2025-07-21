from typing import List, Optional, Union
import torch
from actdyn.utils.rollout import RolloutBuffer, Rollout


class BaseMetric:
    """Base class for metrics that compute both point-wise and final values."""

    def __init__(self, compute_type: str = "sum", device: str = "cpu"):
        self.device = device
        self.compute_type = compute_type

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        """Compute metric value along the trajectory."""
        raise NotImplementedError

    def compute_final(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
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

    def __call__(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        return self.compute(rollout)


class DiscountedMetric(BaseMetric):
    """Wrapper for discounting a metric."""

    def __init__(self, metric: BaseMetric, gamma: float = 0.99, device: str = "cpu"):
        super().__init__(device)
        self.metric = metric
        self.gamma = gamma

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        if isinstance(rollout, RolloutBuffer):
            reward_tensor = rollout.flat["reward"]
        else:
            if not getattr(rollout, "finalized", True):
                rollout.finalize()
            reward_tensor = rollout["reward"]
        if not isinstance(reward_tensor, torch.Tensor):
            reward_tensor = torch.as_tensor(reward_tensor, device=self.metric.device)
        return self.gamma ** torch.arange(
            reward_tensor.shape[-2], device=self.metric.device
        ).unsqueeze(-1) * self.metric.compute(rollout)


class CompositeMetric(BaseMetric):
    """Wrapper for composite cost/metric combining multiple functions."""

    def __init__(
        self,
        metrics: List[BaseMetric],
        weights: Optional[List[float]] = None,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.metrics = metrics
        self.weights = weights if weights is not None else [1.0] * len(metrics)
        assert len(self.weights) == len(
            self.metrics
        ), "Number of weights must match number of cost functions"

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        total_cost = torch.zeros(1, device=self.device)
        for metric, weight in zip(self.metrics, self.weights):
            total_cost = total_cost + weight * metric.compute(rollout)
        return total_cost
