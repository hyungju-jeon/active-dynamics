import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus
from .base import BaseDynamics


# Small constant to prevent numerical instability
eps = 1e-6


class LinearDynamics(BaseDynamics):
    """
    Linear dynamics model using nn.Linear.
    """

    def __init__(self, state_dim, device="cpu"):
        super().__init__(state_dim, device=device)
        self.dynamics = nn.Linear(state_dim, state_dim)


class MLPDynamics(BaseDynamics):
    """
    MLP-based dynamics model.
    """

    def __init__(self, state_dim, hidden_dim=16, device="cpu"):
        super().__init__(state_dim, device=device)
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )


class RBFDynamics(BaseDynamics):
    """
    RBF-based dynamics model.
    """

    def __init__(self, centers, device="cpu"):
        super().__init__(state_dim=2, device=device)
        self.centers = centers.to(self.device)
        self.sigmas = 0.25
        self.weights = nn.Parameter(
            torch.randn(self.centers.shape),
            requires_grad=True,
        )

    def _rbf(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return torch.exp(-torch.cdist(state, self.centers, p=2) ** 2) * (self.sigmas**2)

    def dynamics(self, state):
        return torch.matmul(self._rbf(state), self.weights)


class EnsembleDynamics(BaseDynamics):
    """
    Generic ensemble wrapper for any dynamics model.
    Exposes .models (nn.ModuleList) and .n_models (int).
    """

    def __init__(
        self, state_dim, dynamics_class, n_models=5, dynamics_kwargs=None, device="cpu"
    ):
        super().__init__(state_dim, device=device)
        if dynamics_kwargs is None:
            dynamics_kwargs = {}
        self.models = nn.ModuleList(
            [dynamics_class(**dynamics_kwargs, device=device) for _ in range(n_models)]
        )
        self.n_models = n_models

    def sample_forward(self, state, action=None):
        all_means, all_variances = [], []
        for model in self.models:
            means, variances = model.sample_forward(state, action)
            all_means.append(means)
            all_variances.append(variances)
        means = torch.stack(all_means)
        variances = torch.stack(all_variances)
        mean_prediction = means.mean(dim=0)
        total_variance = variances.mean(dim=0) + means.var(dim=0)
        return mean_prediction, total_variance

    def forward(self, state):
        predictions = [model(state) for model in self.models]
        return torch.stack(predictions)
