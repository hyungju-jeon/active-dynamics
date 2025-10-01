"""Base model classes for the active dynamics package."""

from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn
from torch.nn.functional import softplus

from actdyn.environment.base import BaseAction


# Small constant to prevent numerical instability
eps = 1e-6


# Encoder models
class BaseEncoder(nn.Module):
    """Base class for encoder models.
    
    Args:
        obs_dim: Dimension of observations
        action_dim: Dimension of actions
        latent_dim: Dimension of latent space
        device: Device to run computations on
    """

    network: Optional[nn.Module]

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, device: str = "cpu"):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.network: Optional[nn.Module] = None

    def compute_param(self) -> None:
        """Compute encoder parameters. To be implemented by subclasses."""
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """Forward pass through encoder. To be implemented by subclasses."""
        raise NotImplementedError

    def to(self, device: str) -> "BaseEncoder":
        """Move encoder to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        self.device = torch.device(device)
        if self.network is not None:
            self.network.to(device)
        return self

    def validate_input(self, y: torch.Tensor, u: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate and prepare input tensors.
        
        Args:
            y: Observation tensor of shape (batch, time, obs_dim)
            u: Optional action tensor of shape (batch, time, action_dim)
            
        Returns:
            Tuple of validated (y, u) tensors on correct device
        """
        assert y.dim() == 3, f"Input y must be of shape (batch, time, input_dim), got {y.shape}"
        if u is not None:
            assert u.dim() == 3
            assert (
                u.shape[0] == y.shape[0]
            ), f"Batch size of a {u.shape[0]} must match y {y.shape[0]}"
            assert (
                u.shape[1] == y.shape[1]
            ), f"Time dimension of a {u.shape[1]} must match y {y.shape[1]}"
            assert (
                u.shape[2] == self.action_dim
            ), f"Action dimension of a {u.shape[2]} must match action_dim {self.action_dim}"
        else:
            u = torch.zeros(
                (y.shape[0], y.shape[1], self.action_dim), device=y.device, dtype=y.dtype
            )

        return y.to(self.device), u.to(self.device)


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


# Dynamics models
class BaseDynamics(nn.Module):
    """Base class for all dynamics models with sampling utility."""

    network: nn.Module

    def __init__(self, state_dim, dt, is_residual: bool = False, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.dt = dt
        self.is_residual = is_residual
        self.logvar = nn.Parameter(
            -2 * torch.rand(1, state_dim, device=self.device), requires_grad=True
        )
        self.network = None
        self.state_dim = state_dim

    def to(self, device):
        self.device = torch.device(device)
        self.network.to(device)

    def compute_param(self, state):
        mu = self.network(state)
        var = softplus(self.logvar) + eps
        return mu, var

    def sample_forward(self, init_z, action=None, k_step=1, return_traj=False):
        """Generates samples from forward dynamics model."""
        T = init_z.shape[-2]
        if action is not None:
            if len(action.shape) == 2:
                action = action.unsqueeze(0)

        samples, mus, vars = [init_z], [], []
        for k in range(1, k_step + 1):
            mu, var = self.compute_param(samples[k - 1])
            if self.is_residual:
                z_pred = samples[k - 1] + mu * self.dt  # Residual connection
            else:
                z_pred = mu
            if len(z_pred.shape) == 2:
                z_pred = z_pred.unsqueeze(0)

            if action is not None:
                valid_T = min(z_pred.shape[-2], action.shape[-2])
                z_pred = z_pred[..., :valid_T, :]
                z_pred += action[..., :valid_T, :] * self.dt
                action = action[..., 1:, :]  # Shift action for next step

            mus.append(z_pred)
            vars.append(var)

            samples.append(
                z_pred + torch.sqrt(var * self.dt) * torch.randn_like(z_pred, device=self.device)
            )

        if return_traj:
            return samples, mus, vars
        else:
            return samples[-1], mus[-1], vars[-1]

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
        dynamics_config=None,
    ):
        super().__init__()
        if dynamics_config is None:
            dynamics_config = {}
        self.ensemble = nn.ModuleList([dynamics_cls(**dynamics_config) for _ in range(n_models)])
        self.n_models = n_models

    def sample_forward(self, init_z, action=None, k_step=1, return_traj=False):
        all_mus, all_vars = [], []
        for model in self.ensemble:
            samples, mus, vars = model.sample_forward(
                init_z, action, k_step=k_step, return_traj=return_traj
            )
            all_mus.append(mus)
            all_vars.append(vars)
        mus = torch.stack(all_mus)
        vars = torch.stack(all_vars)
        mean_prediction = mus.mean(dim=0)
        total_variance = vars.mean(dim=0) + mus.var(dim=0)
        return mean_prediction, total_variance

    def forward(self, state):
        predictions = [model(state) for model in self.ensemble]
        return torch.stack(predictions)


# Base model class
class BaseModel(nn.Module):
    """Base class for all models in the package."""

    from actdyn.models.decoder import Decoder

    encoder: BaseEncoder
    decoder: Decoder
    dynamics: BaseDynamics | BaseDynamicsEnsemble
    action_encoder: BaseAction

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
