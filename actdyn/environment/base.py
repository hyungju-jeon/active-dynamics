"""Base environment classes for the active dynamics package."""

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, Sequence
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseDynamicsEnv(gym.Env, ABC):
    """Base class for dynamics environments."""

    def __init__(
        self,
        state_dim: int,
        noise_scale: float = 0.1,
        dt: float = 0.1,
        device: str = "cpu",
        render_mode: Optional[str] = None,
        action_bounds: Sequence[float] = (-1.0, 1.0),
        state_bounds: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.noise_scale = noise_scale
        self.dt = dt
        self.device = torch.device(device)
        self.render_mode = render_mode

        # Initialize spaces with configurable bounds
        # Create action bounds arrays of shape (state_dim,)
        action_bounds = np.asarray(action_bounds, dtype=np.float32)
        if len(action_bounds) != state_dim:
            raise ValueError(f"Action bounds must have length {state_dim}")
        action_low = action_bounds
        action_high = action_bounds

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )

        if state_bounds is None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
            )
        else:
            state_bounds = np.asarray(state_bounds, dtype=np.float32)
            if len(state_bounds) != state_dim:
                raise ValueError(f"State bounds must have length {state_dim}")
            state_low = state_bounds
            state_high = state_bounds

            self.observation_space = spaces.Box(
                low=state_low,
                high=state_high,
                dtype=np.float32,
            )

        self.state = None

    @abstractmethod
    def _get_dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Compute vector field at given state."""
        pass

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        pass

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        pass

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            print(f"Current state: {self.state.cpu().numpy()}")

    def close(self):
        """Clean up resources."""
        pass


class BaseAction(nn.Module):
    """Base class for deterministic action encoder."""

    def __init__(self, input_dim, latent_dim, device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.network = None

    def forward(self, action):
        return self.network(action)

    def to(self, device):
        if self.network is not None:
            self.network.to(device)
        return self


class BaseObservation(nn.Module):
    """Base class for observation models."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        noise_type: Optional[str] = None,
        noise_scale: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.latent_dim = latent_dim  # latent dimension
        self.obs_dim = obs_dim  # observation dimension
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.network = None
        self.device = torch.device(device)

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

    def to(self, device: str):
        self.device = torch.device(device)
        if self.network is not None:
            self.network.to(device)
        return self

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """Nonlinear mapping (z → y)."""
        y = self.network(z)
        return self._add_noise(y)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.observe(z)
