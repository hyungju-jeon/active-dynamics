"""Base model classes for the active dynamics package."""

import torch
import torch.nn as nn
from torch.nn.functional import softplus


# Small constant to prevent numerical instability
eps = 1e-6


# Encoder models
class BaseEncoder(nn.Module):
    """Base class for encoder models."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.network = None

    def compute_param(self, x):
        raise NotImplementedError

    def sample(self, x, n_samples=1):
        raise NotImplementedError

    def forward(self, x, n_samples=1):
        raise NotImplementedError

    def to(self, device):
        self.device = torch.device(device)
        self.network.to(device)
        return self


# Observation mappings
class BaseMapping(nn.Module):
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

    def __init__(self, state_dim, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.to(self.device)
        self.log_var = nn.Parameter(
            -2 * torch.rand(1, state_dim, device=self.device), requires_grad=True
        )
        self.network = None

    def compute_param(self, state):
        mu = self.network(state)
        var = softplus(self.log_var) + eps
        return mu, var

    def sample_forward(self, state, action=None):
        """Generates samples from forward dynamics model."""

        mu, var = self.compute_param(state)
        next_state = mu + state

        if action is not None:
            next_state += action

        return next_state, var

    def forward(self, state):
        return self.network(state).view_as(state)
