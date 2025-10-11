import torch
import torch.nn as nn
from torch.distributions import Normal

from actdyn.utils.torch_helper import activation_from_str
from .base import BaseMapping, BaseNoise
from torch.nn.functional import softplus

eps = 1e-6


# --- Observation Mappings ---
class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class IdentityMapping(BaseMapping):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device)
        self.network = nn.Identity()

    @property
    def jacobian(self):
        def _jac(z=None):
            if z is None:
                raise ValueError("z must be provided to compute the Jacobian for IdentityMapping")
            dim = z.shape[-1]
            return torch.eye(dim, device=self.device)

        return _jac


class LinearMapping(BaseMapping):
    def __init__(self, latent_dim, obs_dim, device="cpu", **kwargs):
        super().__init__(device)
        self.network = nn.Linear(latent_dim, obs_dim).to(device)

    def set_weights(self, weights):
        """Set the weights of the linear mapping."""
        assert (
            weights.shape == self.network.weight.shape
        ), f"Expected weights shape {self.network.weight.shape}, got {weights.shape}"

        if isinstance(weights, torch.Tensor):
            self.network.weight.data = weights
        else:
            raise ValueError("Weights must be a torch.Tensor")

    def set_bias(self, bias):
        """Set the bias of the linear mapping."""
        assert (
            bias.shape == self.network.bias.shape
        ), f"Expected bias shape {self.network.bias.shape}, got {bias.shape}"

        if isinstance(bias, torch.Tensor):
            self.network.bias.data = bias
        else:
            raise ValueError("Bias must be a torch.Tensor")

    @property
    def jacobian(self):
        return self.network.weight


class LogLinearMapping(BaseMapping):
    network: nn.Sequential

    def __init__(self, latent_dim, obs_dim, device="cpu", **kwargs):
        super().__init__(device)
        self.network = nn.Sequential(
            nn.Linear(latent_dim, obs_dim),
            Exp(),
        ).to(device)

    def set_weights(self, weights):
        """Set the weights of the linear mapping."""
        assert (
            weights.shape == self.network[0].weight.shape
        ), f"Expected weights shape {self.network[0].weight.shape}, got {weights.shape}"

        if isinstance(weights, torch.Tensor):
            self.network[0].weight.data = weights
        else:
            raise ValueError("Weights must be a torch.Tensor")

    def set_bias(self, bias):
        """Set the bias of the linear mapping."""
        assert (
            bias.shape == self.network[0].bias.shape
        ), f"Expected bias shape {self.network[0].bias.shape}, got {bias.shape}"

        if isinstance(bias, torch.Tensor):
            self.network[0].bias.data = bias
        else:
            raise ValueError("Bias must be a torch.Tensor")

    @property
    def jacobian(self):
        def _jac(z):
            if z is None:
                raise ValueError("z must be provided to compute the Jacobian for LogLinearMapping")
            mean = self.network(z)  # this is exp(W z + b)
            # mean: (..., obs_dim), weight: (obs_dim, latent_dim)
            # diag(mean) @ W can be implemented via broadcasting
            return mean.unsqueeze(-1) * self.network[0].weight

        return _jac


class MLPMapping(BaseMapping):
    network: nn.Sequential

    def __init__(
        self,
        latent_dim,
        obs_dim,
        hidden_dim: int | list = [16],
        activation="relu",
        device="cpu",
    ):
        super().__init__(device)
        self.activation = activation_from_str(activation)
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = hidden_dim

        layers = []
        prev_dim = latent_dim
        for h in hidden_dims:
            if h > 0:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(self.activation)
                prev_dim = h
        layers.append(nn.Linear(prev_dim, obs_dim))
        self.network = nn.Sequential(*layers)

    @property
    def jacobian(self):
        # Jacobian for a general MLP is not implemented. Return a callable
        # that explicitly raises to make the API consistent.
        def _jac(z=None):
            raise NotImplementedError("Jacobian is not implemented for MLPMapping")

        return _jac


# --- Noise Models ---
class GaussianNoise(BaseNoise):
    def __init__(self, obs_dim, sigma=0.01, device="cpu"):
        super().__init__(device)
        self.logvar = nn.Parameter(
            -2 * torch.rand(1, obs_dim, device=self.device), requires_grad=True
        )

    def log_prob(self, mean, y):
        var = softplus(self.logvar) + eps
        return torch.sum(Normal(mean, torch.sqrt(var)).log_prob(y), dim=(-1, -2))

    def to(self, device):
        self.device = device
        self.logvar = self.logvar.to(device)
        return self


class PoissonNoise(BaseNoise):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device)

    def log_prob(self, rate, y):
        return torch.sum(y * torch.log(rate + 1e-8) - rate - torch.lgamma(y + 1), dim=(-1, -2))

    def to(self, device):
        self.device = device
        return self


# --- Generic Decoder ---
class Decoder(nn.Module):
    def __init__(self, mapping: BaseMapping, noise: BaseNoise, device: str = "cpu"):
        super().__init__()
        self.mapping = mapping.to(device)
        self.noise = noise.to(device)
        self.device = torch.device(device)

    def compute_log_prob(self, z, x):
        mean = self.mapping(z)
        return self.noise.log_prob(mean, x)

    def forward(self, z):
        return self.mapping(z)

    @property
    def jacobian(self):
        return self.mapping.jacobian

    @property
    def logvar(self):
        if isinstance(self.noise, GaussianNoise):
            return self.noise.logvar
        else:
            raise NotImplementedError("Log-variance is only implemented for Gaussian noise.")

    def to(self, device):
        self.device = torch.device(device)
        self.mapping.to(device)
        self.noise.to(device)
        return self
