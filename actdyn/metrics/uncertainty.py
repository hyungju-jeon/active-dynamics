from ast import Dict
import torch
from actdyn.metrics.base import BaseMetric
from actdyn.models.base import BaseDynamicsEnsemble
from actdyn.utils.rollout import Rollout, RolloutBuffer
from typing import Union


class EnsembleDisagreement(BaseMetric):
    """Metric to compute the disagreement among ensemble members."""

    def __init__(
        self, ensemble: BaseDynamicsEnsemble, compute_type="sum", device: str = "cpu", **kwargs
    ):
        super().__init__(compute_type, device)
        self.ensemble = ensemble

    def compute_uncertainty(self, ensemble, x):
        """Compute uncertainty of ensemble predictions."""
        models = ensemble.models
        with torch.no_grad():
            preds = torch.stack([net(x) for net in models], dim=0)  # [N, B, 2]
            var = preds.var(dim=0)  # [B, 2]
            return var

    def compute(self, rollout: Union[Rollout, RolloutBuffer, Dict]) -> torch.Tensor:
        """Compute the disagreement metric for the ensemble."""
        uncertainty = self.compute_uncertainty(self.ensemble, rollout["model_state"])
        self.metric = uncertainty.mean(dim=-1).sum(dim=-1).unsqueeze(-1)

        return self.metric
