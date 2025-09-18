import torch
import torch.nn as nn
from torch.nn.functional import softplus
from .base import BaseEncoder
from actdyn.utils.torch_helper import activation_from_str


# Small constant to prevent numerical instability
eps = 1e-6


class MLPEncoder(BaseEncoder):
    """MLP-based encoder model class"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 0,
        hidden_dim: int | list = [16],
        latent_dim: int = 2,
        activation: str = "relu",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim, device=device
        )
        self.activation = activation_from_str(activation)

        # Build encoder layers
        layers = []
        prev_dim = self.obs_dim + self.action_dim

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = hidden_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)

        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        # Initialize weights for fc_logvar to very small values to prevent large variances at the start. Note that the result will be log variance, so weight should return very negative values
        nn.init.constant_(self.fc_logvar.weight, 0.0)
        nn.init.constant_(self.fc_logvar.bias, -3.0)

    def compute_param(self, y: torch.Tensor, u: torch.Tensor | None = None):
        """Computes the mean and variance of the latent distribution for each time step."""
        y, u = self.validate_input(y, u)
        # Concatenate y and u along the last dimension
        y_u = torch.cat((y, u), dim=-1)  # (batch, time, obs_dim + action_dim)

        # Apply MLP
        mlp_out = self.network(y_u)  # (batch, time, hidden_dim)

        # Split into mu and logvar
        mu = self.fc_mu(mlp_out)
        log_var = self.fc_logvar(mlp_out)
        var = softplus(log_var) + eps
        return mu, var

    def forward(self, y: torch.Tensor, u: torch.Tensor | None = None, n_samples=1):
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
        rnn_hidden_dim: int | list = 16,
        hidden_dim: int | list = 32,
        latent_dim: int = 2,
        activation: str = "relu",
        rnn_type: str = "gru",  # or "lstm"
        device: str = "cpu",
        h_init: str = "reset",  # "reset", "carryover", "step"
        **kwargs,
    ):
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim, device=device
        )
        self.activation = activation_from_str(activation)
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.h_init = h_init.lower()
        self.h = None  # current hidden state

        if isinstance(rnn_hidden_dim, list):
            self.num_layers = len(rnn_hidden_dim)
            self.rnn_hidden_dim = rnn_hidden_dim[0]

        if self.rnn_type == "gru":
            self.network = nn.GRU(
                self.obs_dim + self.action_dim,
                self.rnn_hidden_dim,
                self.num_layers,
                batch_first=True,
                device=self.device,
            )
        elif self.rnn_type == "lstm":
            self.network = nn.LSTM(
                self.obs_dim + self.action_dim,
                self.rnn_hidden_dim,
                self.num_layers,
                batch_first=True,
                device=self.device,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = hidden_dim

        mu_layers = []
        logvar_layers = []
        prev_dim = self.rnn_hidden_dim
        for hidden_dim in hidden_dims:
            mu_layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
            logvar_layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
            prev_dim = hidden_dim

        mu_layers.append(nn.Linear(prev_dim, self.latent_dim))
        logvar_layers.append(nn.Linear(prev_dim, self.latent_dim))
        logvar_layers[-1].weight.data.fill_(0.0)
        logvar_layers[-1].bias.data.fill_(-3.0)

        self.fc_mu = nn.Sequential(*mu_layers)
        self.fc_logvar = nn.Sequential(*logvar_layers)

    def compute_param(
        self, y: torch.Tensor, u: torch.Tensor | None = None, h: torch.Tensor | None = None
    ):
        y, u = self.validate_input(y, u)
        # Concatenate y and u along the last dimension
        y_u = torch.cat((y, u), dim=-1)

        # Hidden state carryover strategy
        if h is None:
            if self.h_init == "reset":
                h = None  # Let GRU initialize to zero
            else:
                h = self.h
                if h is not None:
                    h = h.detach()  # carry over state but not gradients

        # Compute Hidden state and output
        rnn_out, h_final = self.network(y_u, h)

        # Store the next hidden state for carry_over/hybrid strategies
        if self.h_init == "carryover":
            self.h = h_final
        elif self.h_init == "step":
            self.h = self.network(y_u[:, :1, :], h)[1]

        # Decode the output
        mu = self.fc_mu(rnn_out)
        log_var = self.fc_logvar(rnn_out)
        var = softplus(log_var) + eps
        return mu, var

    def sample(self, y, u, n_samples=1):
        """Samples from the latent distribution."""
        mu, var = self.compute_param(y, u)
        samples = mu + torch.sqrt(var) * torch.randn([n_samples] + list(mu.shape), device=y.device)

        return samples.squeeze(0), mu, var

    def forward(self, y: torch.Tensor, u: torch.Tensor | None = None, n_samples=1):
        """Computes samples, mean, variance, and log probability of the latent distribution."""

        # compute parameters and sample
        samples, mu, var = self.sample(y, u, n_samples)
        return samples, mu, var
