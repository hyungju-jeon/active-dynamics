from __future__ import annotations

import torch
import torch.nn as nn
from actdyn.metrics.base import BaseMetric
from actdyn.models.base import BaseDynamicsEnsemble
from actdyn.utils.rollout import Rollout, RolloutBuffer
import torch.nn.functional as F


class EnsembleDisagreement(BaseMetric):
    """Metric to compute the disagreement among ensemble members."""

    def __init__(
        self,
        ensemble_dynamics: BaseDynamicsEnsemble,
        compute_type="sum",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(compute_type=compute_type, device=device)
        _ = kwargs
        self.ensemble_dyn = ensemble_dynamics

    def compute_uncertainty(self, x):
        """Compute uncertainty of ensemble predictions."""
        with torch.no_grad():
            preds = torch.stack(
                [dynamics(x) for dynamics in self.ensemble_dyn.ensemble], dim=0
            )  # [N, B, T, dim]
            var = preds.var(dim=0)  # [B, T, dim]
            return var

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer | dict) -> torch.Tensor:
        """Compute the disagreement metric for the ensemble."""
        uncertainty = -self.compute_uncertainty(rollout["model_state"])
        self.current_cost = uncertainty.sum(dim=-1)  # [B, T]

        return self.current_cost


class RandomNetworkDistillation(BaseMetric):
    """Metric to compute the Random Network Distillation (RND) uncertainty."""

    def __init__(self, compute_type="sum", device: str = "cpu", **kwargs):
        super().__init__(compute_type=compute_type, device=device)
        _ = kwargs
        self.metric = torch.tensor(0.0, device=self.device)
        # define target and predictor MLP networks
        self.target_network = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128))
        self.predictor_network = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128))

        # freeze target params and initialize both nets for stable features
        for m in self.target_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
        for p in self.target_network.parameters():
            p.requires_grad_(False)
        self.target_network = self.target_network.to(device).eval()

        for m in self.predictor_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
        for p in self.predictor_network.parameters():
            p.requires_grad_(True)
        self.predictor_network = self.predictor_network.to(device).train()

        self.optimizer = torch.optim.SGD(self.predictor_network.parameters(), lr=1e-3)

    def compute_uncertainty(self, x):
        """Compute uncertainty using RND."""
        with torch.no_grad():
            target_features = self.target_network(x)
        pred_features = self.predictor_network(x)
        uncertainty = ((pred_features - target_features) ** 2).mean(dim=-1)
        return uncertainty

    def compute_stepwise(self, rollout: Rollout | RolloutBuffer | dict) -> torch.Tensor:
        """Compute the RND uncertainty metric."""
        next_state = torch.as_tensor(rollout["next_model_state"], device=self.device)
        uncertainty = self.compute_uncertainty(next_state)
        self.metric = uncertainty.unsqueeze(-1).sum(dim=1)

        self.current_cost = -self.metric
        assert self.current_cost is not None
        return self.current_cost

    def update(self, rollout: Rollout | RolloutBuffer | dict):
        """Update the predictor network using new transitions."""
        x = torch.as_tensor(rollout["next_model_state"], device=self.device)
        with torch.no_grad():
            target_features = self.target_network(x)
        pred_features = self.predictor_network(x)
        # detach target_features to be explicit that only predictor gets gradients
        loss = F.mse_loss(pred_features, target_features.detach())

        self.optimizer.zero_grad()
        # loss should require grad if predictor params are trainable
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    pass
