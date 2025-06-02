"""Observation models for gym environments."""

import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
import numpy as np


class BaseObservation:
    """Base class for observation models."""

    def __init__(
        self,
        dz: int,
        dy: int,
        device: str = "cpu",
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
    ):
        self.dz = dz  # latent dimension
        self.dy = dy  # observation dimension
        self.device = torch.device(device)
        self.noise_type = noise_type
        self.noise_scale = noise_scale

    def _add_noise(self, y: torch.Tensor) -> torch.Tensor:
        if self.noise_type is None or self.noise_scale == 0.0:
            return y
        if self.noise_type == "gaussian":
            noise = torch.randn_like(y) * self.noise_scale
            return y + noise
        elif self.noise_type == "poisson":
            # For Poisson, the rate parameter (lambda) should be positive
            rate = torch.clamp(y, min=1e-6)
            noise = torch.poisson(rate) - rate
            return y + noise * self.noise_scale
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent state to observation (z → y)."""
        raise NotImplementedError


class IdentityObservation(BaseObservation):
    """Identity observation model."""

    def __init__(
        self,
        dz: int,
        dy: int,
        device: str = "cpu",
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
    ):
        super().__init__(
            dz=dz, dy=dy, device=device, noise_type=noise_type, noise_scale=noise_scale
        )
        if dz != dy:
            raise ValueError("Identity observation requires dz == dy")

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """Identity mapping (z → y)."""
        y = z
        return self._add_noise(y)


class LinearObservation(BaseObservation):
    """Linear observation model: y = Cz + b."""

    def __init__(
        self,
        dz: int,
        dy: int,
        device: str = "cpu",
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
    ):
        super().__init__(
            dz=dz, dy=dy, device=device, noise_type=noise_type, noise_scale=noise_scale
        )
        self.linear = nn.Linear(dz, dy).to(device)

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """Linear mapping (z → y)."""
        y = self.linear(z)
        return self._add_noise(y)


class LogLinearObservation(BaseObservation):
    """Log-linear observation model: y = exp(Cz + b)."""

    def __init__(
        self,
        dz: int,
        dy: int,
        device: str = "cpu",
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
    ):
        super().__init__(
            dz=dz, dy=dy, device=device, noise_type=noise_type, noise_scale=noise_scale
        )
        self.linear = nn.Linear(dz, dy).to(device)

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """Log-linear mapping (z → y): y = exp(Cz + b)."""
        y = torch.exp(self.linear(z))
        return self._add_noise(y)


class NonlinearObservation(BaseObservation):
    """Nonlinear observation model with MLP: y = MLP(z)."""

    def __init__(
        self,
        dz: int,
        dy: int,
        device: str = "cpu",
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
    ):
        super().__init__(
            dz=dz, dy=dy, device=device, noise_type=noise_type, noise_scale=noise_scale
        )
        self.mlp = nn.Sequential(
            nn.Linear(dz, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dy),
        ).to(device)

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """Nonlinear mapping (z → y)."""
        y = self.mlp(z)
        return self._add_noise(y)


# Dictionary mapping observation model names to their classes
OBSERVATION_MODELS = {
    "identity": IdentityObservation,
    "linear": LinearObservation,
    "loglinear": LogLinearObservation,
    "nonlinear": NonlinearObservation,
}
