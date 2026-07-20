import torch
import torch.nn as nn
from .base import BaseAction
from actdyn.utils.torch_utils import activation_from_str


class IdentityActionEncoder(BaseAction):
    """Identity action encoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = nn.Identity()


class PaddedIdentityActionEncoder(BaseAction):
    """Embed actions into the first latent coordinates and pad the rest with zero.

    Inputs have shape ``(..., d_action)`` and outputs have shape
    ``(..., d_latent)`` with the same dtype and device.  This keeps nuisance
    latent coordinates uncontrollable when ``d_action < d_latent``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.d_action > self.d_latent:
            raise ValueError(
                "PaddedIdentityActionEncoder requires d_action <= d_latent, "
                f"got {self.d_action} > {self.d_latent}."
            )

    def forward(
        self, action: torch.Tensor, state: torch.Tensor | None = None
    ) -> torch.Tensor:
        if action.shape[-1] != self.d_action:
            raise ValueError(
                f"Expected action dimension {self.d_action}, got {action.shape[-1]}."
            )
        padding = action.new_zeros(*action.shape[:-1], self.d_latent - self.d_action)
        return torch.cat((action, padding), dim=-1)


class LinearActionEncoder(BaseAction):
    """Simpler action encoder: just a single linear layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = nn.Linear(self.d_action, self.d_latent).to(self.device)


class MlpActionEncoder(BaseAction):
    """MLP-based action encoder."""

    def __init__(
        self,
        hidden_dims=[16],
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation_from_str(activation)

        layers = []
        prev_dim = self.d_action if not self.state_dependent else self.d_action + self.d_latent
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(self.activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, self.d_latent))
        self.network = nn.Sequential(*layers).to(self.device)
