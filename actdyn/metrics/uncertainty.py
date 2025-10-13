from ast import Dict
import torch
from actdyn.metrics.base import BaseMetric
from actdyn.models.base import BaseDynamicsEnsemble
from actdyn.utils.rollout import Rollout, RolloutBuffer
from typing import Union


class EnsembleDisagreement(BaseMetric):
    """Metric to compute the disagreement among ensemble members."""

    def __init__(
        self,
        ensemble_dynamics: BaseDynamicsEnsemble,
        compute_type="sum",
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(compute_type, device)
        self.ensemble_dyn = ensemble_dynamics

    def compute_uncertainty(self, x):
        """Compute uncertainty of ensemble predictions."""
        with torch.no_grad():
            preds = torch.stack(
                [dynamics(x) for dynamics in self.ensemble_dyn.ensemble], dim=0
            )  # [N, B, 2]
            var = preds.var(dim=0)  # [B, 2]
            return var

    def compute(self, rollout: Union[Rollout, RolloutBuffer, Dict]) -> torch.Tensor:
        """Compute the disagreement metric for the ensemble."""
        uncertainty = -self.compute_uncertainty(rollout["model_state"])
        self.metric = uncertainty.mean(dim=-1).sum(dim=-1).unsqueeze(-1)

        return self.metric


class RandomNetworkDistillation(BaseMetric):
    """Metric to compute the Random Network Distillation (RND) uncertainty."""

    def __init__(
        self,
        target_network: torch.nn.Module,
        predictor_network: torch.nn.Module,
        compute_type="sum",
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(compute_type, device)
        self.target_network = target_network.to(device).eval()
        self.predictor_network = predictor_network.to(device).train()

    def compute_uncertainty(self, x):
        """Compute uncertainty using RND."""
        with torch.no_grad():
            target_features = self.target_network(x)
        pred_features = self.predictor_network(x)
        uncertainty = ((pred_features - target_features) ** 2).mean(dim=-1)
        return uncertainty

    def compute(self, rollout: Union[Rollout, RolloutBuffer, Dict]) -> torch.Tensor:
        """Compute the RND uncertainty metric."""
        uncertainty = self.compute_uncertainty(rollout["model_state"])
        self.metric = uncertainty.unsqueeze(-1)

        return self.metric


if __name__ == "__main__":
    pass
