import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus
from .base import BaseEncoder
from actdyn.utils.torch_helper import activation_from_str


# Small constant to prevent numerical instability
eps = 1e-6


class MLPEncoder(BaseEncoder):
    """MLP-based encoder model class (previously Encoder)"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 0,
        hidden_dims: list = [16],
        latent_dim: int = 2,
        activation: str = "relu",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.activation = activation_from_str(activation)

        # Build encoder layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

    def compute_param(self, y: torch.Tensor, u: torch.Tensor = None):
        """Computes the mean and variance of the latent distribution for each time step."""
        batch_dim, time_dim, _ = y.shape

        # Apply MLP
        mlp_out = self.network(y)  # (batch * time, hidden_dim)

        # Split into mu and logvar
        mu = self.fc_mu(mlp_out)
        log_var = self.fc_log_var(mlp_out)
        var = softplus(log_var) + eps
        return mu, var

    def forward(self, y, u=None, n_samples=1):
        """Computes samples, mean, variance, and log probability of the latent distribution."""

        # compute parameters and sample
        mu, var = self.compute_param(y)
        samples = mu + torch.sqrt(var) * torch.randn([n_samples] + list(mu.shape), device=y.device)

        # If n_samples is 1, remove the sample dimension
        if n_samples == 1:
            samples = samples.squeeze(0)

        return samples, mu, var


class RNNEncoder(BaseEncoder):
    """RNN-based encoder for a moving window of k time steps."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 0,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        rnn_type: str = "gru",  # or "lstm"
        num_layers: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.h = None
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers

        if self.rnn_type == "gru":
            self.network = nn.GRU(
                self.obs_dim + self.action_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                device=self.device,
            )
        elif self.rnn_type == "lstm":
            self.network = nn.LSTM(
                self.obs_dim + self.action_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                device=self.device,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.fc_mu = nn.Linear(hidden_dim, latent_dim, device=self.device)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim, device=self.device)

    def compute_param(
        self, y: torch.Tensor, u: torch.Tensor | None = None, h: torch.Tensor | None = None
    ):
        y, u = self.validate_input(y, u)
        # Concatenate y and u along the last dimension
        y_u = torch.cat((y, u), dim=-1)

        # Compute Hidden state and output
        rnn_out, _ = self.network(y_u, h)  # (batch, time, hidden_dim)
        mu = self.fc_mu(rnn_out)
        # mu = torch.tanh(mu)  # Ensure mu is in a reasonable range
        log_var = self.fc_log_var(rnn_out)
        var = softplus(log_var) + eps

        return mu, var

    def forward(
        self,
        y: torch.Tensor,
        u: torch.Tensor | None = None,
        n_samples=1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y, u = self.validate_input(y, u)
        # Compute parameters and sample
        mu, var = self.compute_param(y=y, u=u, h=self.h)
        samples = mu + torch.sqrt(var) * torch.randn(
            [n_samples] + list(mu.shape), device=y.device
        )  # [n_samples, batch, time, latent_dim]
        if n_samples == 1:
            samples = samples.squeeze(0)  # [batch, time, latent_dim]

        return samples, mu, var

    def validate_input(self, y, u):
        assert y.dim() == 3, f"Input y must be of shape (batch, time, input_dim), got {y.shape}"
        if u is not None:
            assert u.dim() == 3
            assert (
                u.shape[0] == y.shape[0]
            ), f"Batch size of a {u.shape[0]} must match y {y.shape[0]}"
            assert (
                u.shape[1] == y.shape[1]
            ), f"Time dimension of a {u.shape[1]} must match y {y.shape[1]}"
            assert (
                u.shape[2] == self.action_dim
            ), f"Action dimension of a {u.shape[2]} must match action_dim {self.action_dim}"
        else:
            u = torch.zeros(
                (y.shape[0], y.shape[1], self.action_dim), device=y.device, dtype=y.dtype
            )

        return y.to(self.device), u.to(self.device)
