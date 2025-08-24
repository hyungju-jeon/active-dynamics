"""Base model classes for the active dynamics package."""

import torch
import torch.nn as nn
from torch.nn.functional import softplus


# Small constant to prevent numerical instability
eps = 1e-6


# Encoder models
class BaseEncoder(nn.Module):
    """Base class for encoder models."""

    network: nn.Module

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.network = None

    def compute_param(self, z):
        raise NotImplementedError

    def forward(self, y, a, n_samples=1):
        raise NotImplementedError

    def to(self, device):
        self.device = torch.device(device)
        self.network.to(device)
        return self


# Observation mappings
class BaseMapping(nn.Module):

    network: nn.Module

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.network = None

    def forward(self, z):
        return self.network(z)

    def to(self, device):
        self.device = torch.device(device)
        self.network.to(device)
        return self

    def set_weights(self, weights):
        raise NotImplementedError

    def set_bias(self, bias):
        raise NotImplementedError

    def set_params(self, weights, bias):
        """Set both weights and bias of the linear mapping."""
        self.set_weights(weights)
        self.set_bias(bias)


# Noise models
class BaseNoise(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def log_prob(self, mean, x):
        raise NotImplementedError


# Base model class
class BaseModel(nn.Module):
    """Base class for all models in the package."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        self.device = torch.device(device)
        self.to(self.device)


# Dynamics models
class BaseDynamics(nn.Module):
    """Base class for all dynamics models with sampling utility."""

    def __init__(self, state_dim, dt=1, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.to(self.device)
        self.dt = dt
        self.log_var = nn.Parameter(
            -2 * torch.rand(1, state_dim, device=self.device), requires_grad=True
        )
        self.network = None

    def compute_param(self, state):
        mu = self.network(state)
        var = softplus(self.log_var) + eps
        return mu, var * self.dt**2

    def sample_forward(self, state, action=None, k_step=1, return_traj=False):
        """Generates samples from forward dynamics model."""
        if action is not None:
            if len(action.shape) == 2:
                action = action.unsqueeze(0)

        samples, mus = [state], []
        for k in range(k_step):
            mu, var = self.compute_param(samples[k])
            next_state = samples[k] + mu * self.dt
            if len(next_state.shape) == 2:
                next_state = mu.unsqueeze(0)
            if action is not None:
                if k > 0:
                    next_state[:, :-k, :] += action[:, k:, :] * self.dt
                else:
                    next_state += action * self.dt

            mus.append(next_state)
            samples.append(mus[k] + torch.sqrt(var) * torch.randn_like(mus[k], device=self.device))

        if return_traj:
            return mus, var
        else:
            return mus[-1], var

    def forward(self, state):
        return self.compute_param(state)[0]


class BaseDynamicsEnsemble(nn.Module):
    """
    Generic ensemble wrapper for any dynamics model.
    """

    def __init__(
        self,
        dynamics_cls,
        n_models=5,
        dynamics_kwargs=None,
    ):
        super().__init__()
        if dynamics_kwargs is None:
            dynamics_kwargs = {}
        self.models = nn.ModuleList([dynamics_cls(**dynamics_kwargs) for _ in range(n_models)])
        self.n_models = n_models

    def sample_forward(self, state, action=None, k_step=1, return_traj=False):
        all_means, all_variances = [], []
        for model in self.models:
            means, variances = model.sample_forward(
                state, action, k_step=k_step, return_traj=return_traj
            )
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
