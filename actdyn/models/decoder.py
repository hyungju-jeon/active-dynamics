from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from actdyn.utils.torch_utils import activation_from_str
from .base import BaseMapping, BaseNoise
from torch.nn.functional import softplus
from actdyn.utils.torch_utils import eps, symmetrize


# --- Observation Mappings ---
class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class Scale(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x * self.scale_factor


class IdentityMapping(BaseMapping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = nn.Linear(self.latent_dim, self.obs_dim).to(self.device)

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
        def _jac(z=None):
            return self.network.weight.unsqueeze(0)

        return _jac


class LogLinearMapping(BaseMapping):
    network: nn.Sequential

    def __init__(self, dt: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, self.obs_dim), Exp(), Scale(self.dt)
        ).to(self.device)

    def set_weights(self, weights, requires_grad=False):
        """Set the weights of the linear mapping."""
        assert (
            weights.shape == self.network[0].weight.shape
        ), f"Expected weights shape {self.network[0].weight.shape}, got {weights.shape}"

        if isinstance(weights, torch.Tensor):
            self.network[0].weight.data = weights
            self.network[0].weight.requires_grad = requires_grad
        else:
            raise ValueError("Weights must be a torch.Tensor")

    def set_bias(self, bias, requires_grad=False):
        """Set the bias of the linear mapping."""
        assert (
            bias.shape == self.network[0].bias.shape
        ), f"Expected bias shape {self.network[0].bias.shape}, got {bias.shape}"

        if isinstance(bias, torch.Tensor):
            self.network[0].bias.data = bias
            self.network[0].bias.requires_grad = requires_grad
        else:
            raise ValueError("Bias must be a torch.Tensor")

    @property
    def jacobian(self):
        def _jac(z):
            if z is None:
                raise ValueError("z must be provided to compute the Jacobian for LogLinearMapping")
            mean = self.network(z)  # this is dt * exp(W z + b)
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
        super().__init__(latent_dim, obs_dim, device)
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
            torch.log(sigma * torch.ones(1, obs_dim, device=self.device)), requires_grad=True
        )

    def log_prob(self, mean, y):
        var = softplus(self.logvar) + eps
        return torch.sum(Normal(mean, torch.sqrt(var)).log_prob(y), dim=(-1, -2))


class PoissonNoise(BaseNoise):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device)

    def log_prob(self, rate, y):
        return torch.sum(y * torch.log(rate + 1e-8) - rate - torch.lgamma(y + 1), dim=(-1, -2))


# --- Generic Decoder ---
class Decoder(nn.Module):
    def __init__(self, mapping: BaseMapping, noise: BaseNoise, device: str = "cpu"):
        super().__init__()
        self.mapping = mapping.to(device)
        self.noise = noise.to(device)
        self.device = torch.device(device)
        self.obs_dim = mapping.obs_dim
        self.latent_dim = mapping.latent_dim

    def to(self, device):
        """Override to update device attribute."""
        super().to(device)
        self.device = torch.device(device)
        self.mapping = self.mapping.to(device)
        self.noise = self.noise.to(device)
        return self

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

    def var(self, z=None):
        if isinstance(self.noise, GaussianNoise):
            return (softplus(self.logvar) + eps).unsqueeze(0)
        else:
            return self.mapping(z)

    def set_params(self, obs_model):
        """Set the parameters of the decoder mapping from an observation model."""
        if isinstance(self.mapping, LogLinearMapping):
            self.mapping.set_weights(obs_model.network[0].weight.data.clone())
            self.mapping.set_bias(obs_model.network[0].bias.data.clone())
            # self.mapping.network[0].weight.requires_grad = False
            # self.mapping.network[0].bias.requires_grad = False
        else:
            self.mapping.set_weights(obs_model.network.weight.data.clone())
            self.mapping.set_bias(obs_model.network.bias.data.clone())
            # self.mapping.network.weight.requires_grad = False
            # self.mapping.network.bias.requires_grad = False


def diagonal_observation_information(
    decoder,
    z: torch.Tensor,
    observation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return diagonal-noise observation mean, Fisher, and optional score.

    Args:
        decoder: Decoder-like object with mapping/noise, ``jacobian``, ``var``,
            and ``__call__``. Fast paths cover log-linear Poisson and linear
            Gaussian decoders; other diagonal-noise decoders use the generic
            ``H.T @ diag(1 / R) @ H`` formula.
        z: Latent states with shape ``(..., latent_dim)``.
        observation: Optional observations with shape ``(..., obs_dim)``.

    Returns:
        ``(mean, I_z, score, r_Rinv_r)`` where ``I_z`` has shape
        ``(..., latent_dim, latent_dim)``. ``score`` and ``r_Rinv_r`` are
        ``None`` unless ``observation`` is provided.
    """
    mean = torch.nan_to_num(decoder(z), nan=0.0, posinf=1e6, neginf=-1e6)
    residual = None if observation is None else observation - mean
    mapping = getattr(decoder, "mapping", None)
    noise = getattr(decoder, "noise", None)

    if isinstance(mapping, LogLinearMapping) and isinstance(noise, PoissonNoise):
        rate = torch.nan_to_num(mean, nan=1e-6, posinf=1e6, neginf=1e-6).clamp_min(1e-6)
        weight = mapping.network[0].weight.to(device=z.device, dtype=z.dtype)
        I_z = symmetrize(torch.einsum("...o,od,oe->...de", rate, weight, weight))
        score = None if residual is None else torch.einsum("od,...o->...d", weight, residual)
        r_Rinv_r = None
        if residual is not None:
            r_Rinv_r = (residual.square() * rate.reciprocal()).sum(dim=-1)
        return mean, I_z, score, r_Rinv_r

    if isinstance(mapping, LinearMapping) and isinstance(noise, GaussianNoise):
        weight = mapping.network.weight.to(device=z.device, dtype=z.dtype)
        R_diag = decoder.var(z).to(device=z.device, dtype=z.dtype)
        R_inv = _expand_observation_prefix(R_diag, z).reciprocal()
        I_z = symmetrize(torch.einsum("od,...o,oe->...de", weight, R_inv, weight))
        score = None
        r_Rinv_r = None
        if residual is not None:
            score = torch.einsum("od,...o,...o->...d", weight, R_inv, residual)
            r_Rinv_r = (residual.square() * R_inv).sum(dim=-1)
        return mean, I_z, score, r_Rinv_r

    H = _expand_observation_prefix(decoder.jacobian(z).to(device=z.device, dtype=z.dtype), z, suffix_ndim=2)
    if isinstance(noise, PoissonNoise):
        R_diag = mean
    else:
        R_diag = decoder.var(z).to(device=z.device, dtype=z.dtype)
    R_inv = _expand_observation_prefix(R_diag, z).reciprocal()
    I_z = symmetrize(torch.einsum("...od,...o,...oe->...de", H, R_inv, H))
    score = None
    r_Rinv_r = None
    if residual is not None:
        score = torch.einsum("...od,...o,...o->...d", H, R_inv, residual)
        r_Rinv_r = (residual.square() * R_inv).sum(dim=-1)
    return mean, I_z, score, r_Rinv_r


def _expand_observation_prefix(
    value: torch.Tensor,
    z: torch.Tensor,
    *,
    suffix_ndim: int = 1,
) -> torch.Tensor:
    prefix = tuple(z.shape[:-1])
    value = torch.nan_to_num(value, nan=1e-6, posinf=1e6, neginf=1e-6)
    if suffix_ndim == 1:
        value = value.clamp_min(1e-6)
    while value.dim() < len(prefix) + suffix_ndim:
        value = value.unsqueeze(0)
    suffix = tuple(value.shape[-suffix_ndim:])
    return value.expand(*prefix, *suffix)
