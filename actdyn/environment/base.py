from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from gymnasium import spaces


class BaseAction(nn.Module):
    """Base class for deterministic action encoder."""

    def __init__(self, d_action, d_latent, action_bounds, state_dependent=False, device="cpu"):
        super().__init__()
        self.d_action = d_action
        self.d_latent = d_latent
        self.action_space = spaces.Box(
            low=action_bounds[0],
            high=action_bounds[1],
            shape=(d_action,),
            dtype=np.float32,
        )
        self.state_dependent = state_dependent
        self.device = torch.device(device)
        self.network = None

    def forward(self, action: torch.Tensor, state: Optional[torch.Tensor] = None):
        if self.network is not None:
            if self.state_dependent and state is not None:
                _action, _state = self.validate_input(action, state)
                z_u = torch.cat([_action, _state], dim=-1)
                return self.network(z_u)
            else:
                self.network(action)
            return self.network(action)

    def validate_input(
        self, action: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            action.dim() == 3
        ), f"Input action must be of shape (batch, time, input_dim), got {action.shape}"
        assert (
            state.dim() == 3
        ), f"Input state must be of shape (batch, time, input_dim), got {state.shape}"

        # time dimension must match
        assert (
            action.shape[1] == state.shape[1]
        ), f"Time dimension of action and state must match, got {action.shape[1]} and {state.shape[1]}"

        # if the batch size does not match, repeat the smaller one
        if action.shape[0] != state.shape[0]:
            if action.shape[0] < state.shape[0]:
                state = state.repeat_interleave(action.shape[0] // state.shape[0], dim=0)
            else:
                action = action.repeat_interleave(state.shape[0] // action.shape[0], dim=0)

        return action, state

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
        self.d_latent = latent_dim  # latent dimension
        self.d_obs = obs_dim  # observation dimension
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
            return torch.poisson(rate)
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
