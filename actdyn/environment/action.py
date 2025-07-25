import torch.nn as nn
from .base import BaseAction
from actdyn.utils.torch_helper import activation_from_str


class IdentityActionEncoder(BaseAction):
    """Identity action encoder."""

    def __init__(self, action_dim, latent_dim, action_bounds, device="cpu", **kwargs):
        super().__init__(action_dim, latent_dim, action_bounds, device)
        self.network = nn.Identity()


class LinearActionEncoder(BaseAction):
    """Simpler action encoder: just a single linear layer."""

    def __init__(self, action_dim, latent_dim, action_bounds, device="cpu", **kwargs):
        super().__init__(action_dim, latent_dim, action_bounds, device)
        self.network = nn.Linear(action_dim, latent_dim).to(device)


class MlpActionEncoder(BaseAction):
    """MLP-based action encoder."""

    def __init__(
        self,
        action_dim,
        latent_dim,
        action_bounds,
        hidden_dims=[16],
        activation="relu",
        device="cpu",
    ):
        super().__init__(action_dim, latent_dim, action_bounds, device)
        self.activation = activation_from_str(activation)

        layers = []
        prev_dim = action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(self.activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers).to(device)
