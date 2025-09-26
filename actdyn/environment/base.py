"""Base environment classes for the active dynamics package."""

from einops import repeat
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
        if not (isinstance(action_bounds, (tuple, list)) and len(action_bounds) == 2):
            raise ValueError(
                f"action_bounds must be a tuple or list of (low, high), got {action_bounds}"
            )
        action_low = np.full((state_dim,), action_bounds[0], dtype=np.float32)
        action_high = np.full((state_dim,), action_bounds[1], dtype=np.float32)

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
            if not (isinstance(state_bounds, (tuple, list)) and len(state_bounds) == 2):
                raise ValueError(
                    f"state_bounds must be a tuple or list of (low, high), got {state_bounds}"
                )
            state_low = np.full((state_dim,), state_bounds[0], dtype=np.float32)
            state_high = np.full((state_dim,), state_bounds[1], dtype=np.float32)

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
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:  # Fixed return type
        """Reset the environment."""
        super().reset(seed=seed)

    @abstractmethod
    def step(
        self, action: torch.Tensor  # Changed from np.ndarray to torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        pass

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            if self.state is not None and hasattr(self.state, "cpu"):
                print(f"Current state: {self.state.cpu().numpy()}")
            else:
                print(f"Current state: {self.state}")

    def close(self):
        """Clean up resources."""
        pass


class BaseAction(nn.Module):
    """Base class for deterministic action encoder."""

    def __init__(self, action_dim, latent_dim, action_bounds, state_dependent=False, device="cpu"):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.action_space = spaces.Box(
            low=action_bounds[0],
            high=action_bounds[1],
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.state_dependent = state_dependent
        self.device = torch.device(device)
        self.network = None

    def forward(self, action, state=None):
        if self.network is not None:
            if self.state_dependent and state is not None:
                if state.ndim == 4:
                    action = repeat(action, "b t d-> s b t d", s=state.shape[0])
                z_u = torch.cat([action, state], dim=-1)
                return self.network(z_u)
            else:
                self.network(action)
            return self.network(action)
        raise NotImplementedError("Network is not defined in BaseAction.")

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
        self.device = torch.device(device)
        self.network: Optional[nn.Module] = None

    def _add_noise(self, y: torch.Tensor) -> torch.Tensor:
        if self.noise_type is None or self.noise_scale == 0.0:
            return y
        if self.noise_type == "gaussian":
            noise = torch.randn_like(y) * self.noise_scale
            return y + noise
        elif self.noise_type == "poisson":
            # For Poisson, the rate parameter (lambda) should be positive
            rate = torch.clamp(y, min=1e-6)
            noise = torch.poisson(rate)
            return y + noise
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def to(self, device: str):
        self.device = torch.device(device)
        if self.network is not None:
            self.network.to(device)
        return self

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """Nonlinear mapping (z → y)."""
        if self.network is not None:
            y = self.network(z)
            return self._add_noise(y)
        raise NotImplementedError("Network is not defined in BaseObservation.")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.observe(z)
