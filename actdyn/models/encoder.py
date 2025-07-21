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
        input_dim: int,
        hidden_dims: list = [16],
        latent_dim: int = 2,
        activation: str = "relu",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.activation = activation_from_str(activation)

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

    def compute_param(self, x):
        """Computes the mean and variance of the latent distribution for each time step."""
        batch_dim, time_dim, _ = x.shape
        # Reshape input to process each time step independently: (Batch * Time, dy)
        x_flat = x.view(batch_dim * time_dim, -1)

        # Apply MLP
        out_flat = self.network(x_flat)  # Shape: (Batch * Time, hidden_dims[-1])

        # Reshape output back to (Batch, Time, hidden_dims[-1])
        out = out_flat.view(batch_dim, time_dim, -1)

        # Split into mu and logvar
        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)
        var = softplus(log_var) + eps
        return mu, var

    def sample(self, x, n_samples=1):
        """Samples from the latent distribution."""
        mu, var = self.compute_param(x)

        samples = mu + torch.sqrt(var) * torch.randn(
            [n_samples] + list(mu.shape), device=x.device
        )
        # If n_samples is 1, remove the sample dimension
        if n_samples == 1:
            samples = samples.squeeze(0)

        return samples, mu, var

    def forward(self, x, n_samples=1):
        """Computes samples, mean, variance, and log probability of the latent distribution."""
        # check dimension of x is (batch, time, input_dim) if not, add dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Invalid dimension of x: {x.dim()}")

        # compute parameters and sample
        samples, mu, var = self.sample(x, n_samples=n_samples)
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(samples), (-2, -1))
        return samples, mu, var, log_prob


class RNNEncoder(BaseEncoder):
    """RNN-based encoder for a moving window of k time steps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        rnn_type: str = "gru",  # or "lstm"
        num_layers: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers

        if self.rnn_type == "gru":
            self.network = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        elif self.rnn_type == "lstm":
            self.network = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def compute_param(self, x):
        # x: (batch, time, input_dim)
        if self.rnn_type == "lstm":
            rnn_out, _ = self.network(x)
        else:
            rnn_out, _ = self.network(x)
        # rnn_out: (batch, time, hidden_dim)
        mu = self.fc_mu(rnn_out)
        log_var = self.fc_log_var(rnn_out)
        var = softplus(log_var) + eps
        return mu, var

    def sample(self, x, n_samples=1):
        mu, var = self.compute_param(x)
        samples = mu + torch.sqrt(var) * torch.randn(
            [n_samples] + list(mu.shape), device=x.device
        )
        if n_samples == 1:
            samples = samples.squeeze(0)
        return samples, mu, var

    def forward(self, x, n_samples=1):
        # check dimension of x is (batch, time, input_dim) if not, add dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Invalid dimension of x: {x.dim()}")

        samples, mu, var = self.sample(x, n_samples=n_samples)
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(samples), (-2, -1))
        return samples, mu, var, log_prob
