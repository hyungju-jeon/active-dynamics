"""Observation models for gym environments."""

import torch
import torch.nn as nn
from typing import Optional

from actdyn.utils.helper import activation_from_str
from .base import BaseObservation


class Exp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class Scale(nn.Module):
    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale_factor


class IdentityObservation(BaseObservation):
    """Identity observation model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = nn.Identity().to(self.device)


class LinearObservation(BaseObservation):
    """Linear observation model: y = Cz + b."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = nn.Linear(self.d_latent, self.d_obs).to(self.device)


class LogLinearObservation(BaseObservation):
    """Log-linear observation model: y = exp(Cz + b)."""

    def __init__(self, dt: float = 1.0, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.dt = dt
        self.network = nn.Sequential(
            nn.Linear(self.d_latent, self.d_obs), Exp(), Scale(self.dt)
        ).to(self.device)

    def set_dt(self, dt: float):
        """Set the time step for scaling the output."""
        self.dt = dt
        self.network[2].scale_factor = self.dt


class NonlinearObservation(BaseObservation):
    """Nonlinear observation model with MLP: y = MLP(z)."""

    def __init__(
        self,
        hidden_dims: list = [64, 64],
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation_from_str(activation)

        layers = []
        prev_dim = self.d_latent
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(self.activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, self.d_obs))
        self.network = nn.Sequential(*layers).to(self.device)
