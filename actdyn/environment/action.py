import torch.nn as nn


class BaseAction(nn.Module):
    """Base class for deterministic action encoder."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, action):
        return self.encoder(action)


class LinearActionEncoder(BaseAction):
    """Simpler action encoder: just a single linear layer."""

    def __init__(self, input_dim, latent_dim):
        super().__init__(input_dim, latent_dim)
        self.encoder = nn.Linear(input_dim, latent_dim)


class MlpActionEncoder(BaseAction):
    """MLP-based action encoder."""

    def __init__(
        self, input_dim, latent_dim, hidden_dims=[16, 16], activation=nn.ReLU()
    ):
        super().__init__(input_dim, latent_dim)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)


# Factory function for modularity
def make_action_encoder(encoder_type, input_dim, latent_dim, **kwargs):
    if encoder_type == "mlp":
        return MlpActionEncoder(input_dim, latent_dim, **kwargs)
    elif encoder_type == "linear":
        return LinearActionEncoder(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown action encoder type: {encoder_type}")
