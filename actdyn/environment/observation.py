"""Observation models for gym environments."""

import torch
import torch.nn as nn
from typing import Optional

from actdyn.utils.torch_helper import activation_from_str
from .base import BaseObservation


class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class IdentityObservation(BaseObservation):
    """Identity observation model."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            device=device,
            noise_type=noise_type,
            noise_scale=noise_scale,
        )
        self.network = nn.Identity()
        if latent_dim != obs_dim:
            raise ValueError("Identity observation requires latent_dim == obs_dim")


class LinearObservation(BaseObservation):
    """Linear observation model: y = Cz + b."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            noise_type=noise_type,
            noise_scale=noise_scale,
            device=device,
        )
        self.network = nn.Linear(latent_dim, obs_dim).to(device)


class LogLinearObservation(BaseObservation):
    """Log-linear observation model: y = exp(Cz + b)."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            noise_type=noise_type,
            noise_scale=noise_scale,
            device=device,
        )
        self.network = nn.Sequential(
            nn.Linear(latent_dim, obs_dim),
            Exp(),
        ).to(device)


class NonlinearObservation(BaseObservation):
    """Nonlinear observation model with MLP: y = MLP(z)."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dims: Optional[list] = None,
        activation: str = "relu",
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            noise_type=noise_type,
            noise_scale=noise_scale,
            device=device,
        )
        self.activation = activation_from_str(activation)

        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(self.activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, obs_dim))
        self.network = nn.Sequential(*layers).to(device)
