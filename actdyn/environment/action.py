import torch.nn as nn
from .base import BaseAction


class IdentityActionEncoder(BaseAction):
    """Identity action encoder."""

    def __init__(self, input_dim, latent_dim, device="cpu", **kwargs):
        super().__init__(input_dim, latent_dim, device)
        self.network = nn.Identity()


class LinearActionEncoder(BaseAction):
    """Simpler action encoder: just a single linear layer."""

    def __init__(self, input_dim, latent_dim, device="cpu"):
        super().__init__(input_dim, latent_dim, device)
        self.network = nn.Linear(input_dim, latent_dim).to(device)


class MlpActionEncoder(BaseAction):
    """MLP-based action encoder."""

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dims=[16],
        activation=nn.ReLU(),
        device="cpu",
    ):
        super().__init__(input_dim, latent_dim, device)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers).to(device)
