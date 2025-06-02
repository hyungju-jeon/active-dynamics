import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from .base import BaseMapping, BaseNoise

eps = 1e-6


# --- Observation Mappings ---


class IdentityMapping(BaseMapping):
    def forward(self, z):
        return z


class LinearMapping(BaseMapping):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.linear(z)


class LogLinearMapping(BaseMapping):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return torch.exp(self.linear(z))


class MLPMapping(BaseMapping):
    def __init__(
        self, latent_dim, output_dim, hidden_dims=[16, 16], activation=nn.ReLU()
    ):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        return self.mlp(z)


# --- Noise Models ---


class GaussianNoise(BaseNoise):
    def __init__(self, output_dim, sigma=1.0):
        super().__init__()
        self.logvar = nn.Parameter(
            torch.log(torch.ones(1, output_dim) * sigma), requires_grad=True
        )

    def log_prob(self, mean, y):
        var = torch.exp(self.logvar)
        return torch.sum(Normal(mean, torch.sqrt(var)).log_prob(y), dim=(-1, -2))


class PoissonNoise(BaseNoise):
    def log_prob(self, rate, y):
        return torch.sum(
            y * torch.log(rate + 1e-8) - rate - torch.lgamma(y + 1), dim=(-1, -2)
        )


# --- Generic Decoder ---
class Decoder(nn.Module):
    def __init__(self, mapping: BaseMapping, noise: BaseNoise):
        super().__init__()
        self.mapping = mapping
        self.noise = noise

    def compute_param(self, z):
        return self.mapping(z)

    def compute_log_prob(self, z, x):
        mean = self.compute_param(z)
        return self.noise.log_prob(mean, x)

    def forward(self, z):
        return self.compute_param(z)


# --- Example Factory Functions ---
def make_identity_gaussian_decoder(latent_dim, sigma=1.0):
    return Decoder(IdentityMapping(), GaussianNoise(latent_dim, sigma))


def make_linear_gaussian_decoder(latent_dim, output_dim, sigma=1.0):
    return Decoder(
        LinearMapping(latent_dim, output_dim), GaussianNoise(output_dim, sigma)
    )


def make_loglinear_gaussian_decoder(latent_dim, output_dim, sigma=1.0):
    return Decoder(
        LogLinearMapping(latent_dim, output_dim), GaussianNoise(output_dim, sigma)
    )


def make_loglinear_poisson_decoder(latent_dim, output_dim):
    return Decoder(LogLinearMapping(latent_dim, output_dim), PoissonNoise())


def make_mlp_gaussian_decoder(latent_dim, output_dim, hidden_dims=[16, 16], sigma=1.0):
    return Decoder(
        MLPMapping(latent_dim, output_dim, hidden_dims),
        GaussianNoise(output_dim, sigma),
    )
