import torch.nn as nn
from .base import BaseAction
from actdyn.utils.helper import activation_from_str


class IdentityActionEncoder(BaseAction):
    """Identity action encoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = nn.Identity()


class LinearActionEncoder(BaseAction):
    """Simpler action encoder: just a single linear layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = nn.Linear(self.d_action, self.d_latent).to(self.device)


class MlpActionEncoder(BaseAction):
    """MLP-based action encoder."""

    def __init__(
        self,
        hidden_dim=[16],
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation_from_str(activation)

        layers = []
        prev_dim = self.d_action if not self.state_dependent else self.d_action + self.d_latent
        for h in hidden_dim:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(self.activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, self.d_latent))
        self.network = nn.Sequential(*layers).to(self.device)
