import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from .base import BaseMapping, BaseNoise

eps = 1e-6


# --- Observation Mappings ---
class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class IdentityMapping(BaseMapping):
    def __init__(self, device="cpu"):
        super().__init__(device)
        self.network = nn.Identity()


class LinearMapping(BaseMapping):
    def __init__(self, latent_dim, output_dim, device="cpu"):
        super().__init__(device)
        self.network = nn.Linear(latent_dim, output_dim)


class LogLinearMapping(BaseMapping):
    def __init__(self, latent_dim, output_dim, device="cpu"):
        super().__init__(device)
        self.network = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            Exp(),
        )


class MLPMapping(BaseMapping):
    def __init__(
        self,
        latent_dim,
        output_dim,
        hidden_dims=[16, 16],
        activation=nn.ReLU(),
        device="cpu",
    ):
        super().__init__(device)
        layers = []
        prev_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)


# --- Noise Models ---


class GaussianNoise(BaseNoise):
    def __init__(self, output_dim, sigma=1.0, device="cpu"):
        super().__init__(device)
        self.logvar = nn.Parameter(
            torch.log(torch.ones(1, output_dim, device=device) * sigma),
            requires_grad=True,
        )

    def log_prob(self, mean, y):
        var = torch.exp(self.logvar)
        return torch.sum(Normal(mean, torch.sqrt(var)).log_prob(y), dim=(-1, -2))

    def to(self, device):
        self.device = device
        self.logvar = torch.nn.Parameter(
            self.logvar.data.to(device), requires_grad=True
        )
        return self


class PoissonNoise(BaseNoise):
    def __init__(self, device="cpu"):
        super().__init__(device)

    def log_prob(self, rate, y):
        return torch.sum(
            y * torch.log(rate + 1e-8) - rate - torch.lgamma(y + 1), dim=(-1, -2)
        )

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

    def compute_param(self, z):
        return self.mapping(z)

    def compute_log_prob(self, z, x):
        mean = self.compute_param(z)
        return self.noise.log_prob(mean, x)

    def forward(self, z):
        return self.compute_param(z)

    def to(self, device):
        self.device = torch.device(device)
        self.mapping.to(device)
        self.noise.to(device)
        return self
