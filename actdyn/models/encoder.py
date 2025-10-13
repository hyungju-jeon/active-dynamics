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
        else:
            self.num_layers = 1
            self.rnn_hidden_dim = rnn_hidden_dim

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
            if hidden_dim > 0:
                mu_layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
                logvar_layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
                prev_dim = hidden_dim

        mu_layers.append(nn.Linear(prev_dim, self.latent_dim))
        logvar_layers.append(nn.Linear(prev_dim, self.latent_dim))
        logvar_layers[-1].weight.data.fill_(0.0)
        logvar_layers[-1].bias.data.fill_(-3.0)

        self.fc_mu = nn.Sequential(*mu_layers).to(self.device)
        self.fc_logvar = nn.Sequential(*logvar_layers).to(self.device)

    def compute_param(
        self, y: torch.Tensor, u: torch.Tensor | None = None, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def sample(self, y, u, n_samples=1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples from the latent distribution."""
        mu, var = self.compute_param(y, u)
        samples = mu + torch.sqrt(var) * torch.randn([n_samples] + list(mu.shape), device=y.device)

        return samples.squeeze(0), mu, var

    def forward(
        self, y: torch.Tensor, u: torch.Tensor | None = None, n_samples=1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes samples, mean, variance, and log probability of the latent distribution."""
        # compute parameters and sample
        samples, mu, var = self.sample(y, u, n_samples)  # (n_samples, batch, time, latent_dim)
        return samples, mu, var


class RNNEmbeddingEncoder(BaseEncoder):
    """RNN-based encoder for a moving window of k time steps."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 0,
        embedding_dim: int = 2,
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
        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type.lower()
        self.h_init = h_init.lower()
        self.h = None  # current hidden state

        if isinstance(rnn_hidden_dim, list):
            self.num_layers = len(rnn_hidden_dim)
            self.rnn_hidden_dim = rnn_hidden_dim[0]
        else:
            self.num_layers = 1
            self.rnn_hidden_dim = rnn_hidden_dim

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

        self.e_gamma = nn.Linear(self.embedding_dim, self.rnn_hidden_dim)
        self.e_beta = nn.Linear(self.embedding_dim, self.rnn_hidden_dim)

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = hidden_dim

        mu_layers = []
        logvar_layers = []
        prev_dim = self.rnn_hidden_dim
        for hidden_dim in hidden_dims:
            if hidden_dim > 0:
                mu_layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
                logvar_layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
                prev_dim = hidden_dim

        mu_layers.append(nn.Linear(prev_dim, self.latent_dim))
        logvar_layers.append(nn.Linear(prev_dim, self.latent_dim))
        logvar_layers[-1].weight.data.fill_(0.0)
        logvar_layers[-1].bias.data.fill_(-3.0)

        self.fc_mu = nn.Sequential(*mu_layers).to(self.device)
        self.fc_logvar = nn.Sequential(*logvar_layers).to(self.device)

    def compute_param(
        self,
        y: torch.Tensor,
        e: torch.Tensor,
        u: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        gamma: float | None = None,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, u = self.validate_input(y, u)
        # Concatenate y and u along the last dimension
        y_u = torch.cat((y, u), dim=-1)

        # Match the dimension of embedding to the batch size and time steps
        y_ue = torch.cat((y_u, e), dim=-1)

        # Hidden state carryover strategy
        if h is None:
            if self.h_init == "reset":
                h = None  # Let GRU initialize to zero
            else:
                h = self.h

        # Compute Hidden state and output
        rnn_out, h_final = self.network(y_ue, h)

        # Store the next hidden state for carry_over/hybrid strategies
        if self.h_init == "carryover":
            self.h = h_final.detach()
        elif self.h_init == "step":
            self.h = self.network(y_u[:, :1, :], h)[1]

        # Apply FiLM conditioning
        gamma = self.e_gamma(e) if gamma is None else gamma
        beta = self.e_beta(e) if beta is None else beta
        rnn_out = gamma * rnn_out + beta

        # Decode the output
        mu = self.fc_mu(rnn_out)
        log_var = self.fc_logvar(rnn_out)
        var = softplus(log_var) + eps
        return mu, var

    def sample(self, y, e, u, n_samples=1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples from the latent distribution."""
        mu, var = self.compute_param(y, e, u)
        samples = mu + torch.sqrt(var) * torch.randn([n_samples] + list(mu.shape), device=y.device)

        return samples.squeeze(0), mu, var

    def forward(
        self, y: torch.Tensor, e: torch.Tensor, u: torch.Tensor | None = None, n_samples=1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes samples, mean, variance, and log probability of the latent distribution."""
        # compute parameters and sample
        samples, mu, var = self.sample(y, e, u, n_samples)  # (n_samples, batch, time, latent_dim)
        return samples, mu, var


class HybridEncoder(BaseEncoder):
    """Hybrid RNN-MLP encoder for state estimation and correction."""

    Belief = dict[str, torch.Tensor]

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        embedding_dim: int,
        action_dim: int = 0,
        hidden_dim: int | list = 128,
        activation: str = "relu",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim, device=device
        )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.activation = activation_from_str(activation)
        self.embedding_dim = embedding_dim

        # if isinstance(hidden_dim, int):
        #     hidden_dims = [hidden_dim]
        # else:
        #     hidden_dims = hidden_dim

        # # Amortized residual gain correction network
        # corrector_layers = []
        # prev_dim = latent_dim + obs_dim + embedding_dim + action_dim
        # for hidden in hidden_dims:
        #     corrector_layers.extend([nn.Linear(prev_dim, hidden), nn.SiLU()])
        #     prev_dim = hidden
        # corrector_layers.append(nn.Linear(prev_dim, 2 * latent_dim))

        # self.correction = nn.Sequential(*corrector_layers)

    def forward(
        self,
        r: torch.Tensor,
        H: torch.Tensor = None,
        R: torch.Tensor = None,
        z_pred: Belief = None,
        e_mu: torch.Tensor = None,
    ) -> Belief:
        if z_pred is None:
            raise ValueError("z_pred (prior belief) must be provided for HybridEncoder")
        if e_mu is None:
            raise ValueError("e_mu (embedding mean) must be provided for HybridEncoder")

        m, P = z_pred["m"], z_pred["P"]

        # analytic Kalman gain (no learning)
        S = H @ P @ H.transpose(-1, -2) + R
        K = P @ H.transpose(-1, -2) @ torch.linalg.inv(S)

        m_upd = m + (K @ r.unsqueeze(-1)).squeeze(-1)
        P_upd = (torch.eye(self.latent_dim, device=P.device) - K @ H) @ P
        P_upd = 0.5 * (P_upd + P_upd.transpose(-1, -2))
        return {"m": m_upd, "P": P_upd}


# class NeuralGainPosterior(nn.Module):
#     """
#     q(z_t | ·) = N(m_pr + K_t * r_t, Σ_po)
#     with r_t = x_t - x_hat(z=m_pr), K_t = KNN(features).

#     Requires an adapter `obs_pred_mean(z)` that returns decoder mean at z.
#     """

#     def __init__(
#         self,
#         cfg: NeuralGainCfg,
#         obs_encoder: Optional[nn.Module] = None,
#         obs_pred_mean: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
#     ):
#         super().__init__()
#         self.cfg = cfg
#         Dz, Dx, Du, De = cfg.Dz, cfg.Dx, cfg.Du, cfg.De
#         self.obs_pred_mean = obs_pred_mean  # must be provided by trainer

#         # Optional x encoder (for residual pre-processing); small MLP
#         self.obs_enc = obs_encoder or nn.Sequential(
#             nn.Linear(Dx, cfg.Denc), nn.GELU(), nn.Linear(cfg.Denc, cfg.Denc), nn.GELU()
#         )

#         # Features for gain net: [m_pr, z_prev, u_prev, e, r_t_enc]
#         fin = Dz + Dz + Du + De + cfg.Denc
#         self.gain_backbone = nn.GRU(input_size=fin, hidden_size=cfg.Dh, batch_first=True)
#         self.h0 = None  # initialized per batch

#         # Parameterizations of K_t
#         if cfg.diagonal_gain:
#             self.k_head = nn.Linear(cfg.Dh, Dz)  # elementwise scaling after residual projection
#             self.res_proj = nn.Linear(cfg.Dx, Dz, bias=False)  # maps residual into latent space
#         elif cfg.gain_low_rank is not None:
#             r = cfg.gain_low_rank
#             self.U_head = nn.Linear(cfg.Dh, Dz * r)
#             self.V_head = nn.Linear(cfg.Dh, Dx * r)
#         else:
#             # Full K_t as (Dz x Dx). For stability, keep size modest or apply low-rank
#             self.k_head = nn.Linear(cfg.Dh, Dz * Dx)

#         # Optional covariance delta head
#         if cfg.cov_mode == "diag_delta":
#             self.dlogvar_head = nn.Linear(cfg.Dh, Dz)

#     def init_state(self, B: int, device: torch.device):
#         return torch.zeros(1, B, self.cfg.Dh, device=device)

#     def forward(
#         self,
#         x_t: torch.Tensor,  # (B,Dx)
#         z_prev: torch.Tensor,  # (B,Dz)
#         u_prev: torch.Tensor,  # (B,Du)
#         e_t: torch.Tensor,  # (B,De)
#         m_pr: torch.Tensor,  # (B,Dz)
#         logvar_pr: torch.Tensor,  # (B,Dz) diag prior
#         h_prev: torch.Tensor,  # (1,B,Dh)
#     ) -> Dict[str, Any]:
#         B, Dz, Dx = x_t.size(0), self.cfg.Dz, self.cfg.Dx

#         # Predicted observation from decoder at the PRIOR mean
#         assert self.obs_pred_mean is not None, "obs_pred_mean(z) must be provided."
#         x_hat = self.obs_pred_mean(m_pr)  # (B,Dx)

#         # Innovation / residual
#         r_raw = x_t - x_hat  # (B,Dx)
#         r_enc = self.obs_enc(r_raw)  # (B,Denc)

#         feats = torch.cat([m_pr, z_prev, u_prev, e_t, r_enc], dim=-1).unsqueeze(1)  # (B,1,fin)
#         out, h_t = self.gain_backbone(feats, h_prev)
#         h = out.squeeze(1)  # (B,Dh)

#         # Build K_t
#         clamp = self.cfg.clamp_gain
#         if self.cfg.diagonal_gain:
#             k_vec = torch.tanh(self.k_head(h)) * clamp  # (B,Dz)
#             r_lat = self.res_proj(r_raw)  # (B,Dz)
#             delta_m = k_vec * r_lat  # (B,Dz)
#         elif self.cfg.gain_low_rank is not None:
#             r = self.cfg.gain_low_rank
#             U = self.U_head(h).view(B, Dz, r)  # (B,Dz,r)
#             V = self.V_head(h).view(B, Dx, r)  # (B,Dx,r)
#             # K = U @ V^T with bounded magnitude
#             U = torch.tanh(U) * clamp
#             V = torch.tanh(V) * clamp
#             delta_m = torch.einsum("bdr, bxr, bx -> bd", U, V, r_raw)  # K * r
#         else:
#             K = self.k_head(h).view(B, Dz, Dx)  # (B,Dz,Dx)
#             K = torch.tanh(K) * clamp
#             delta_m = torch.einsum("bdx, bx -> bd", K, r_raw)

#         m_po = m_pr + delta_m

#         if self.cfg.cov_mode == "prior":
#             logvar_po = logvar_pr
#         else:
#             dlog = torch.tanh(self.dlogvar_head(h))  # bounded change
#             logvar_po = torch.clamp(logvar_pr + dlog, self.cfg.min_logvar, self.cfg.max_logvar)

#         q = diag_gaussian(m_po, logvar_po)
#         z_sample = m_po + (0.5 * logvar_po).exp() * torch.randn_like(m_po)

#         return {
#             "q": q,
#             "z": z_sample,
#             "m_po": m_po,
#             "logvar_po": logvar_po,
#             "x_hat": x_hat,
#             "residual": r_raw,
#             "h": h_t,
#         }
