import torch.nn as nn
from .base import BaseAction
from actdyn.utils.helper import activation_from_str


class IdentityActionEncoder(BaseAction):
    """Identity action encoder."""

    def __init__(self, action_dim, latent_dim, action_bounds, device="cpu", **kwargs):
        self.state_dependent = kwargs.get("state_dependent", False)
        super().__init__(
            action_dim,
            latent_dim,
            action_bounds,
            state_dependent=self.state_dependent,
            device=device,
        )
        self.network = nn.Identity()


class LinearActionEncoder(BaseAction):
    """Simpler action encoder: just a single linear layer."""

    def __init__(self, action_dim, latent_dim, action_bounds, device="cpu", **kwargs):
        self.state_dependent = kwargs.get("state_dependent", False)
        super().__init__(
            action_dim,
            latent_dim,
            action_bounds,
            state_dependent=self.state_dependent,
            device=device,
        )
        self.network = nn.Linear(action_dim, latent_dim).to(device)


class MlpActionEncoder(BaseAction):
    """MLP-based action encoder."""

    def __init__(
        self,
        action_dim,
        latent_dim,
        action_bounds,
        hidden_dim=[16],
        state_dependent=False,
        activation="relu",
        device="cpu",
    ):
        super().__init__(action_dim, latent_dim, action_bounds, state_dependent, device)
        self.activation = activation_from_str(activation)

        layers = []
        prev_dim = action_dim if not state_dependent else action_dim + latent_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(self.activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers).to(device)
