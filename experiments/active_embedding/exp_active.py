# %%
import os
import shutil
from collections import deque
from functools import partial
from turtle import color
from typing import Callable, Sequence

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from torch.nn.functional import softplus
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# from external.integrative_inference.src.utils import save_model, load_model
import actdyn
import actdyn.core.experiment
import actdyn.environment
import actdyn.environment.action
import actdyn.environment.observation
import actdyn.environment.vectorfield
import actdyn.metrics
import actdyn.models
import actdyn.models.dynamics
import actdyn.models.encoder
import actdyn.policy
import actdyn.policy.mpc
import external.integrative_inference.src.modules as metadyn
from actdyn.config import ExperimentConfig
from actdyn.utils import save_load
from actdyn.utils.experiment_helpers import setup_experiment
from actdyn.utils.rollout import RecentRollout, Rollout, RolloutBuffer
from actdyn.utils.helper import to_np
from actdyn.utils.visualize import plot_vector_field, set_matplotlib_style
from external.integrative_inference.experiments.model_utils import build_hypernetwork

# Small constant to prevent numerical instability
eps = 1e-6

# Configure matplotlib for better aesthetics
set_matplotlib_style()

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# %% Dataset and DataLoader Classes
class zeDataset(Dataset):
    def __init__(
        self,
        N: int,
        z_sampler: Callable[[int], torch.Tensor],
        e_sampler: Callable[[int], torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.N = N
        self.zs = z_sampler(N).to(device)
        self.es = e_sampler(N).to(device)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        z = self.zs[idx : idx + 1]
        e = self.es[idx : idx + 1]
        return z.squeeze(0), e.squeeze(0)


class FeDataset(Dataset):
    def __init__(
        self,
        fn: Callable,
        N: int,
        z_sampler: Callable[[int], torch.Tensor],
        e_sampler: Callable[[int], torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.N = N
        self.zs = z_sampler(N).to(device)
        self.es = e_sampler(N).to(device)
        self.fn = fn

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        z = self.zs[idx : idx + 1]
        e = self.es[idx : idx + 1]
        Fe = jacobian_wrt_param(self.fn, [z, e], 1)  # [1, nz, ne]
        return z.squeeze(0), e.squeeze(0), Fe.squeeze(0)


class FzDataset(Dataset):
    def __init__(
        self,
        fn: Callable,
        N: int,
        z_sampler: Callable[[int], torch.Tensor],
        e_sampler: Callable[[int], torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.N = N
        self.zs = z_sampler(N).to(device)
        self.es = e_sampler(N).to(device)
        self.fn = fn

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        z = self.zs[idx : idx + 1]
        e = self.es[idx : idx + 1]
        Fz = jacobian_wrt_param(self.fn, [z, e], 0)  # [1, nz, ne]
        return z.squeeze(0), e.squeeze(0), Fz.squeeze(0)


class Amortized_Jacobian(nn.Module):
    def __init__(self, d_latent, d_embed, d_hidden: int = 32, n_hidden: int = 2, device="cpu"):
        super().__init__()
        self.d_latent = d_latent
        self.d_embed = d_embed
        self.device = device

        layers = [nn.Linear(d_latent + d_embed, d_hidden, device=device), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(d_hidden, d_hidden, device=device), nn.ReLU()]
        layers += [nn.Linear(d_hidden, d_latent * d_embed, device=device)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # Make z and e in batch x time x dim
        if z.ndim == 2:
            z = z.unsqueeze(1)
        B, T, _nz = z.shape

        if e.ndim == 2:
            e = e.unsqueeze(1)
        if e.shape[1] == 1:
            e = e.repeat(1, z.shape[1], 1)  # [B, T, ne]

        x = torch.cat((z, e), dim=-1)  # [B, T, nz+ne]
        Fe_hat = self.net(x).view(B, T, self.d_latent, self.d_embed)
        return Fe_hat


class MetaDynamics:
    def __init__(self, hypernet: metadyn.LowRankHypernet, mean_dynamics: metadyn.HyperMlpDynamics):
        self.hypernet = hypernet
        self.mean_dynamics = mean_dynamics
        self.e = None
        self.out = None

    def set_params(self, e):
        e = torch.tensor(e, device=device, dtype=torch.float32).unsqueeze(0)
        self.e = e[..., :2]
        self.out, _ = self.hypernet(self.e)

    def __call__(self, x, e=None):
        if e is None:
            if self.e is None or self.out is None:
                raise ValueError("Embedding e is not set. Please set e using set_embedding method.")
            out = self.out
        else:
            out, _ = self.hypernet(e)
        return self.mean_dynamics(x, out) * 10


def make_uniform_sampler(low: list[float] | float, high: list[float] | float, dim: int):
    if isinstance(low, float):
        low = [low] * dim
    if isinstance(high, float):
        high = [high] * dim

    def _sampler(N: int):
        return torch.stack(
            [low[i] + (high[i] - low[i]) * torch.rand(N) for i in range(dim)], dim=-1
        )

    return _sampler


def curvature_penalty(model: nn.Module, z: torch.Tensor, e: torch.Tensor, eps: float = 1e-2):
    """Finite-difference smoothness of F̂_e w.r.t. (z,e)."""
    B = z.size(0)
    # random unit perturbations
    dz = F.normalize(torch.randn_like(z), dim=-1) * eps
    de = F.normalize(torch.randn_like(e), dim=-1) * eps
    J = model(z, e)
    J_e = model(z + dz, e + de)
    return ((J_e - J) ** 2).mean()


def jacobian_wrt_param(fn: Callable, inputs: Sequence[torch.Tensor], argnum: int) -> torch.Tensor:
    """
    Compute Jacobian of `fn(*inputs)` w.r.t. the input indexed by `argnum` using vjp.

    Args:
        fn: callable that accepts the full inputs tuple and returns tensor of shape [batch, time, out_dim]
        inputs: tuple of input tensors (e.g., (z, e))
        argnum: which argument to differentiate wrt (0-based)

    Returns:
        Jacobian tensor of shape [batch, time, out_dim, in_dim]
    """
    has_time = inputs[0].ndim == 3
    if has_time:
        batch, T, in_dim = inputs[0].shape
    else:
        batch, in_dim = inputs[0].shape
        T = 1

    # Work on a local list copy of inputs so we can set requires_grad
    inputs_list = [
        t.reshape(batch * T, -1).requires_grad_(True) if not t.requires_grad else t for t in inputs
    ]

    out = fn(*inputs_list)
    if out.ndim == 1:
        out = out.unsqueeze(0)
    _, out_dim = out.shape

    in_dim = inputs_list[argnum].shape[-1]
    J = torch.zeros(batch, T, out_dim, in_dim, device=out.device, dtype=out.dtype)

    # Compute row-wise grads: for each output dim, grad wrt z_flat
    for i in range(out_dim):
        grad_outputs = torch.zeros_like(out)
        grad_outputs[:, i] = 1.0
        (gi,) = torch.autograd.grad(
            out,
            inputs_list[argnum],
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
        )
        J[..., i, :] = gi.reshape(batch, T, in_dim)

    return J.reshape(batch, T, out_dim, in_dim)


def train_jacobian(
    dataset, d_latent=2, d_embed=2, d_hidden=64, n_hidden=1, curv_loss=0.0, device="cpu"
):
    net = Amortized_Jacobian(
        d_latent=d_latent, d_embed=d_embed, d_hidden=d_hidden, n_hidden=n_hidden, device=device
    )
    dl = DataLoader(
        dataset, batch_size=500, shuffle=True, num_workers=0, pin_memory=False, drop_last=True
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n_epochs = 500
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    pbar = tqdm(range(n_epochs))
    for ep in pbar:
        net.train()
        total, n = 0.0, 0
        for z, e, J in dl:
            z, e, J = z.to(device), e.to(device), J.to(device)
            J_hat = net(z, e)
            loss = F.mse_loss(J_hat, J)  # Frobenius MSE
            if curv_loss > 0.0:
                loss_curv = curvature_penalty(J, z, e)
                loss += curv_loss * loss_curv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()
            total += loss.item() * z.size(0)
            n += z.size(0)
        sched.step()
        pbar.set_postfix(loss=total / n)
    return net


def Fe_true(z, e):
    if z.ndim == 2:
        z = z.unsqueeze(0)
    B, T, d = z.shape
    Fe = torch.zeros(B, T, 2, 2, device=z.device)
    Fe[..., 1, 0] = z[..., 1]
    Fe[..., 1, 1] = -z[..., 0]
    return Fe


def Fz_true(z, e):
    if e.ndim == 2:
        e = e.unsqueeze(0)
    if z.ndim == 2:
        z = z.unsqueeze(0)
    B, T, d = e.shape
    Fz = torch.zeros(B, T, 2, 2, device=z.device)
    Fz[..., 0, 0] = 0
    Fz[..., 0, 1] = 1
    Fz[..., 1, 0] = -e[..., 1] - 0.3 * z[..., 0] ** 2
    Fz[..., 1, 1] = e[..., 0]
    return Fz


def safe_cholesky(M, jitter=1e-6, max_tries=5, growth=10.0):
    I = torch.eye(M.size(-1), device=M.device).expand_as(M)
    j = 0.0
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(M + j * I)
        except RuntimeError:
            j = jitter if j == 0.0 else j * growth
    return torch.linalg.cholesky(M + j * I)


def symmetrize(M):
    return 0.5 * (M + M.transpose(-1, -2))


def debug_fix_decoder(
    decoder: actdyn.models.Decoder, obs_model: actdyn.environment.base.BaseObservation
):
    decoder.mapping.network.weight.data = obs_model.network.weight.data.clone()
    decoder.mapping.network.bias.data = obs_model.network.bias.data.clone()
    decoder.mapping.network.weight.requires_grad = False
    decoder.mapping.network.bias.requires_grad = False


# %% (Pretrain) Pretrain context dependent dynamics model
z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-3.0, -2.0], [-0.1, 2.0], 2)
ds = zeDataset(100000, z_sampler, e_sampler, device)

duffing_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)

cfg = {
    "d_latent": 2,
    "d_embed": 2,
    "du": 0,
    "d_hidden_embed": 16,
    "d_context": 2,
    "d_hidden_dynamics": 32,
    "d_hidden_hypernet_dynamics": 16,
    "n_hidden": 1,
    "likelihood": "gaussian",  # 'gaussian' or 'poisson'
    "l2_c": 1e-4,
    "l2_dw_dynamics": 1e-4,
    "rank_dynamics": 2,
    "update_input": True,  # Whether to update input weights in dynamics
    "update_hidden": True,  # Whether to update hidden weights in dynamics
    "update_output": False,  # Whether to update output weights in dynamics
    "linear_hypernetwork": False,  # Whether to use linear hypernetwork (no hidden layer)
}

hypernet_dynamics = build_hypernetwork(cfg, device)

mean_dynamics = metadyn.HyperMlpDynamics(
    d_latent=cfg["d_latent"],
    d_hidden=cfg["d_hidden_dynamics"],
    n_hidden=cfg["n_hidden"],
    update_input=cfg["update_input"],
    update_output=cfg["update_output"],
    update_hidden=cfg["update_hidden"],
    du=0,
    device=device,
)

hypernet_model_path = os.path.join(
    os.path.dirname(__file__), "models", "duffing_hypernet_dynamics.pth"
)
mean_dynamics_model_path = os.path.join(
    os.path.dirname(__file__), "models", "duffing_mean_dynamics.pth"
)
if os.path.exists(hypernet_model_path):
    hypernet_dynamics.load_state_dict(
        torch.load(hypernet_model_path, map_location=device, weights_only=True)
    )
    mean_dynamics.load_state_dict(
        torch.load(mean_dynamics_model_path, map_location=device, weights_only=True)
    )
    print("Loaded pretrained meta-dynamics model")

else:
    # Train with True embedding value and  RMSE loss
    optimizer = torch.optim.Adam(
        list(hypernet_dynamics.parameters()) + list(mean_dynamics.parameters()), lr=1e-3
    )
    n_epochs = 500
    batch_size = 10000
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        for z, e in dl:
            duffing_env.dynamics.a = e[:, 0]
            duffing_env.dynamics.b = e[:, 1]
            fx_true = duffing_env.compute_dynamics(z).to(device)

            out, _ = hypernet_dynamics(e)
            fx_pred = mean_dynamics.compute_param(z, out)

            loss = F.mse_loss(fx_pred, fx_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)

# %% (Pretrain) Check learned meta-dynamical model (Checked)
duffing_env = actdyn.VectorFieldEnv("duffing", x_range=5, dt=0.1, Q=0.0)

if False:
    for i in range(10):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs = axs.flatten()
        i = torch.randint(0, 100, (1,))
        e = e_sampler(1).to(device)
        duffing_env.set_params(e)
        plot_vector_field(
            duffing_env.dynamics,
            ax=axs[0],
            x_range=5,
            is_residual=True,
        )
        axs[0].set_title(
            f"True Vector Field of Duffing System for a={e[..., 0].item():.2f}, b={e[..., 1].item():.2f}"
        )
        plot_vector_field(
            lambda x: meta_dynamics(
                x.to(device),
                e=e,
            ),
            ax=axs[1],
            x_range=5,
            is_residual=True,
        )
        axs[1].set_title("Meta Learned Vector Field")
        plt.show()


# %% (Pretrain) Pretrain Jacobian Networks
z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-2.0, -1.0], [-0.1, 1.0], 2)

Fe_net = Amortized_Jacobian(d_latent=2, d_embed=2, d_hidden=64, n_hidden=1, device=device)
Fe_model_path = os.path.join(os.path.dirname(__file__), "models", "duffing_amortized_Fe.pth")
if os.path.exists(Fe_model_path):
    Fe_net.load_state_dict(torch.load(Fe_model_path, map_location=device))
    print("Loaded pretrained Fe model from", Fe_model_path)
else:
    fe_ds = FeDataset(meta_dynamics, 1000, z_sampler, e_sampler, device)
    Fe_net = train_jacobian(
        fe_ds, d_latent=2, d_embed=2, d_hidden=64, n_hidden=1, curv_loss=0.0, device="cpu"
    )
    torch.save(Fe_net.state_dict(), Fe_model_path)
Fe_net.eval()

Fz_net = Amortized_Jacobian(d_latent=2, d_embed=2, d_hidden=64, n_hidden=1, device=device)
Fz_model_path = os.path.join(os.path.dirname(__file__), "models", "duffing_amortized_Fz.pth")
if os.path.exists(Fz_model_path):
    Fz_net.load_state_dict(torch.load(Fz_model_path, map_location=device))
    print("Loaded pretrained Fz model from", Fz_model_path)
else:
    fz_ds = FzDataset(meta_dynamics, 1000, z_sampler, e_sampler, device)
    Fz_net = train_jacobian(
        fz_ds, d_latent=2, d_embed=2, d_hidden=64, n_hidden=1, curv_loss=0.0, device="cpu"
    )
    torch.save(Fz_net.state_dict(), Fz_model_path)
Fz_net.eval()

# %% (Pretrain) Test Amortized Jacobian Network (Tested)
if False:
    for i in range(1):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs = axs.flatten()
        z = z_sampler(1).to(device)
        e = e_sampler(1).to(device)
        Fe_meta = jacobian_wrt_param(meta_dynamics, [z, e], 1).cpu().detach().squeeze()
        Fe_hat = Fe_net(z, e).cpu().detach().squeeze() * 10
        Fe_star = Fe_true(z, e).cpu().detach().squeeze() * 10

        args = {"head_width": 0.7, "width": 0.3}
        axs[0].axis("equal")
        axs[0].arrow(0, 0, Fe_meta[0, 0], Fe_meta[1, 0], color="r", label="Meta", **args)
        axs[0].arrow(0, 0, Fe_meta[0, 1], Fe_meta[1, 1], color="r", **args, ls="--")
        axs[0].arrow(0, 0, Fe_hat[0, 0], Fe_hat[1, 0], color="g", label="Amort.", **args)
        axs[0].arrow(0, 0, Fe_hat[0, 1], Fe_hat[1, 1], color="g", **args, ls="--")
        axs[0].arrow(0, 0, Fe_star[0, 0], Fe_star[1, 0], color="b", label="True", **args)
        axs[0].arrow(0, 0, Fe_star[0, 1], Fe_star[1, 1], color="b", **args, ls="--")
        axs[0].legend()
        axs[0].set_title("Fe Comparison")
        axs[1].axis("equal")

        Fz_meta = jacobian_wrt_param(meta_dynamics, [z, e], 0).cpu().detach().squeeze()
        Fz_hat = Fz_net(z, e).cpu().detach().squeeze() * 10
        Fz_star = Fz_true(z, e).cpu().detach().squeeze() * 10
        axs[1].arrow(0, 0, Fz_meta[0, 0], Fz_meta[1, 0], color="r", label="Meta", **args)
        axs[1].arrow(0, 0, Fz_meta[0, 1], Fz_meta[1, 1], color="r", **args, ls="--")
        axs[1].arrow(0, 0, Fz_hat[0, 0], Fz_hat[1, 0], color="g", label="Amort.", **args)
        axs[1].arrow(0, 0, Fz_hat[0, 1], Fz_hat[1, 1], color="g", **args, ls="--")
        axs[1].arrow(0, 0, Fz_star[0, 0], Fz_star[1, 0], color="b", label="True", **args)
        axs[1].arrow(0, 0, Fz_star[0, 1], Fz_star[1, 1], color="b", **args, ls="--")
        axs[1].legend()
        axs[1].set_title("Fz Comparison")
        plt.show()


# %% (Experiment) Create an environment for active learning
latent_dim = 2
embedding_dim = 2
action_dim = 2
observation_dim = 50
data_idx = 55

# a, b = -1, 0.5
a, b = e_sampler(1).squeeze(0)
a, b = -1.372, 0.977
a, b = -0.5, -0.4
action_model = actdyn.environment.action.IdentityActionEncoder(
    action_dim=action_dim, latent_dim=latent_dim, action_bounds=[-3.0, 3.0], device=device
)
vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([a, b, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    action_bounds=[action_model.action_space.low, action_model.action_space.high],
    device=device,
)

env = actdyn.environment.EnvWrapper(vec_env, obs_model, action_model, dt=0.01, device=device)
base_dir = os.path.join(os.path.dirname(__file__), "../../results", "active_embedding")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# %% 1-1. ✅ EKF/EKF + Laplace
torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
params = list(decoder.parameters())
# debug_fix_decoder(decoder, obs_model)
warmup_step = 0

plt.close("all")
z_bel = {
    "m": torch.zeros(1, latent_dim, device=device),
    "P": 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    # "m": torch.zeros(1, embedding_dim, device=device),
    "m": torch.tensor([[-1.17, +0.11]], device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)

pbar = tqdm(range(100000))
for env_step in pbar:
    # 1) Random action sampling
    if env_step % 50 == 0:
        u_t = torch.tensor(env.action_space.sample(), device=device, dtype=torch.float32)
        prev_action = u_t
    u_t = prev_action

    # 2-1) Predict latent
    dfde = Fe_net(z_bel["m"], e_bel["m"]).detach() * env.dt  # (1, Dz, De)
    Fz = Fz_net(z_bel["m"], e_bel["m"]).detach()
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    z_pred = {
        "m": z_bel["m"] + meta_dynamics_fn(z_bel["m"], e_bel["m"]) * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
        + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
    }

    # 2-2) Predict observation
    y_pred = decoder(z_pred["m"])
    R = softplus(decoder.noise.logvar).diag_embed() + eps

    # 3) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  # (1, Do)
    r = y_true - y_pred

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)

    # 4) Embedding update (Laplace)
    S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
    S = symmetrize(S)

    chol_S = torch.linalg.cholesky(S)
    X = torch.cholesky_solve(HzFe, chol_S)
    curb_ll = einsum(HzFe, X, "b y d, b y e->b d e")  # (1, De, De)
    curv_ll = symmetrize(curv_ll)  # ensure symmetry
    if env_step > warmup_step:
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)
        for _ in range(1):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.unsqueeze(-1), chol_S)
            grad_ll = einsum(HzFe, invS_r, "b y d, b y k->b d")  # (1, De)

            Prec_old = e_bel["Prec"]
            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e, "P": Sigma_e, "Prec": Prec_new}
            Prec, eta = Prec_new, eta_new

    # Detach after all refinements
    e_bel = {k: v.detach() for k, v in e_bel.items()}

    # 5) EKF Update Posterior
    K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
    I = torch.eye(latent_dim, device=device).unsqueeze(0)
    KH = K @ dhdz

    P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    z_post = {
        "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
        "P": symmetrize(P_upd),
    }

    # 6) Roll updated z posterior as new prior
    z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

    # 7) Optimize Likelihood
    opt.zero_grad(set_to_none=True)

    # Single-sample NLL
    ll = decoder.compute_log_prob(z_bel["m"], y_true)
    loss = -ll
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 5.0)
    opt.step()

    # 7) Log (optional)
    if (env_step % 1000) == 0 and env_step > 0:
        plot_vector_field(
            lambda x: meta_dynamics_fn(
                x.to(device),
                e_bel["m"],
            ),
            x_range=5,
            is_residual=True,
        )
        z_np = np.stack(z)
        z_hat_np = np.stack(z_hat)
        plt.plot(z_np[:, 0, 0], z_np[:, 0, 1], label="true", alpha=0.5)
        plt.plot(z_hat_np[:, 0], z_hat_np[:, 1], label="inferred", alpha=0.5)
        plt.legend()
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()
        z, z_hat = [], []
    rows.append(
        {
            "t": env_step,
            "e_norm": float(e_bel["m"].norm()),
            "z_norm": float(z_bel["m"].norm()),
            "r_norm": float(r.norm()),
        }
    )
    z.append(info["latent_state"].squeeze(0).cpu())
    z_hat.append(z_bel["m"].squeeze(0).cpu())

    if env_step % 100 == 0:
        pbar.set_postfix(
            LL=f"{ll.item():.3f}",
            e_hat=f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
            e_true=f"({a:.2f},{b:.2f})",
        )
        pbar.update(100)


# %% 1-2. ✅ (True value) EKF/EKF + Laplace + Meta
meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)
e_dict = {
    "active (k=20)": [],
    "active (k=5)": [],
    "random": [],
    "step": [],
    "active chunk(k=20)": [],
}
exp_name = ["active (k=5)", "step", "active chunk(k=20)"]
for exp_id in exp_name:
    torch.manual_seed(1)
    for i in range(3):
        e_norm = []
        e = e_sampler(1)
        a, b = e.squeeze(0)
        a, b = -0.5, -0.4
        vec_env = actdyn.VectorFieldEnv(
            "duffing",
            x_range=5,
            dyn_params=torch.tensor([a, b, 0.1]),
            dt=0.01,
            alpha=10,
            Q=0.01,
            action_bounds=[action_model.action_space.low, action_model.action_space.high],
            device=device,
        )
        obs_model = actdyn.environment.observation.LinearObservation(
            obs_dim=observation_dim,
            latent_dim=latent_dim,
            noise_scale=0.1,
            noise_type="gaussian",
            device=device,
        )
        action_model = actdyn.environment.action.IdentityActionEncoder(
            action_dim=action_dim, latent_dim=latent_dim, action_bounds=[-5.0, 5.0], device=device
        )
        env = actdyn.environment.EnvWrapper(
            vec_env, obs_model, action_model, dt=0.01, device=device
        )

        mapping = actdyn.models.decoder.LinearMapping(
            latent_dim=latent_dim, obs_dim=observation_dim, device=device
        )
        noise = actdyn.models.decoder.GaussianNoise(
            obs_dim=observation_dim, sigma=0.01, device=device
        )
        decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
        params = list(decoder.parameters())
        # debug_fix_decoder(decoder, obs_model)
        warmup_step = 0

        plt.close("all")
        z_bel = {
            "m": torch.ones(1, latent_dim, device=device),
            "P": torch.eye(latent_dim, device=device).unsqueeze(0),
        }

        sigma_0 = 0.01
        e_bel = {
            "m": torch.ones(1, embedding_dim, device=device),
            "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
            "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
        }
        meta_dynamics.set_params(e_bel["m"])
        emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
            Fe_net=Fe_true, Fz_net=Fz_true, decoder=decoder
        )

        action_metric = actdyn.metrics.cost.ActionCost()
        composite_metric = actdyn.metrics.CompositeMetric(
            [emb_metric, action_metric], weights=[1.0, 0.001]
        )
        sim_vec_env = actdyn.VectorFieldEnv(
            "duffing",
            x_range=5,
            dyn_params=torch.tensor([0, 0, 0.1]),
            dt=0.01,
            alpha=10,
            Q=0.01,
            device=device,
        )
        dynamics = actdyn.models.dynamics.FunctionDynamics(
            state_dim=latent_dim, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
        )

        model = actdyn.models.BaseModel(
            action_encoder=action_model,
            dynamics=dynamics,
            device=device,
        )
        mpc_policy = actdyn.policy.mpc.MpcICem(
            metric=emb_metric,
            model=model,
            device=device,
            horizon=20 if exp_id == "active chunk(k=20)" or exp_id == "active (k=20)" else 5,
            num_iterations=20,
            num_samples=20,
            num_elite=5,
            verbose=False,
        )
        step_policy = actdyn.policy.StepPolicy(
            action_space=env.action_space, step_size=100, device=device
        )
        random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)

        opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
        rows = []
        obs, info = env.reset()
        z = []
        z_hat = []
        prev_action = torch.zeros(action_dim, device=device)
        results_dir = os.path.join(base_dir, "EKF_fixed_active_chunck")
        ro_path = os.path.join(results_dir, "rollout.pkl")
        writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))

        ro = Rollout()
        pbar = tqdm(range(1000))
        for env_step in pbar:
            e_norm.append(torch.norm(e_bel["m"].cpu() - e).numpy())
            current_obs = obs
            current_state = info["latent_state"]
            current_model_state = z_bel["m"]
            # Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
            Q = 1e-2 * torch.eye(latent_dim, device=device).unsqueeze(0)

            # 1) Random action sampling
            meta_dynamics.set_params(e_bel["m"])
            sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
            model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

            if exp_id == "random":
                u_t = random_policy(z_bel["m"].unsqueeze(0)).detach()
            elif exp_id == "step":
                u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
            elif exp_id == "active (k=5)":
                u_t, _, _ = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
            elif exp_id == "active (k=20)":
                u_t, _, _ = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
            elif exp_id == "active chunk(k=20)":
                if env_step % 5 == 0:
                    u_t, u_ts, cost = mpc_policy(
                        z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q
                    )
                else:
                    u_t = u_ts[:, env_step % 5].unsqueeze(1).detach()

            # 2-1) Predict latent
            dfde = Fe_true(z_bel["m"], e_bel["m"]) * env.dt
            Fz = Fz_true(z_bel["m"], e_bel["m"])
            dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
            dhdz = decoder.jacobian.unsqueeze(0)
            HzFe = dhdz @ dfde  # (1, Do, De)

            f_true = sim_vec_env.compute_dynamics(z_bel["m"]).to(device)  # For debug

            z_pred = {
                "m": z_bel["m"] + f_true * env.dt + u_t * env.dt,
                "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
                + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
            }

            # 2-2) Predict observation
            y_pred = decoder(z_pred["m"])
            R = softplus(decoder.noise.logvar).diag_embed() + eps

            # 3) Get true observation from env
            obs, reward, _, _, info = env.step(u_t)
            y_true = obs.squeeze(0)  # (1, Do)
            r = y_true - y_pred

            dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)

            # 4) Embedding update (Laplace)
            S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
            S = symmetrize(S)

            chol_S = torch.linalg.cholesky(S)
            X = torch.cholesky_solve(HzFe, chol_S)
            curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
            curv_ll = symmetrize(curv_ll)  # ensure symmetry
            if env_step > warmup_step:
                # predictive covariance and Cholesky solve (as fixed earlier)
                Prec = e_bel["Prec"]
                eta = Prec @ e_bel["m"].unsqueeze(-1)
                for _ in range(10):
                    y_hat = decoder(z_pred["m"])
                    r_t = y_true - y_hat

                    invS_r = torch.cholesky_solve(r_t.mT, chol_S)
                    grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

                    Prec_old = e_bel["Prec"]
                    Prec_new = Prec_old + curv_ll
                    eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
                    eta_new = eta_old + grad_ll.unsqueeze(-1)

                    chol_Prec_new = safe_cholesky(Prec_new)
                    Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
                    mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

                    # Update belief for next refinement
                    e_bel = {
                        "m": mu_e.squeeze(0),
                        "P": Sigma_e.squeeze(0),
                        "Prec": Prec_new.squeeze(0),
                    }
                    Prec, eta = Prec_new, eta_new

            # Detach after all refinements
            e_bel = {k: v.detach() for k, v in e_bel.items()}

            # 5) EKF Update Posterior
            # z_post = encoder(r=r, H=dhdz, R=R, z_pred=z_pred, e_mu=e_bel["m"])
            K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
            I = torch.eye(latent_dim, device=device).unsqueeze(0)
            KH = K @ dhdz

            P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(
                -1, -2
            )
            z_post = {
                "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
                "P": symmetrize(P_upd),
            }

            # 6) Roll updated z posterior as new prior
            z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

            # 7) Optimize Likelihood
            opt.zero_grad(set_to_none=True)

            # Single-sample NLL
            ll = decoder.compute_log_prob(z_bel["m"], y_true)
            loss = -ll
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 5.0)
            opt.step()
            writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
            writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
            writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
            writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

            transition = {
                "obs": current_obs,  # Observation  y_t
                "next_obs": obs,  # New Observation y_{t+1}
                "action": u_t,  # Action a_t
                "env_state": current_state,  # Environment state z_t
                "next_env_state": info["latent_state"],  # Next environment state z_{t+1}
                "model_state": current_model_state,  # Current belief state z'_t
                "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
            }
            ro.add(**transition)
            current_obs = obs
            current_state = info["latent_state"]
            current_model_state = z_bel["m"]

            if env_step % 100 == 0:
                pbar.set_postfix(
                    LL=f"{ll.item():.3f}",
                    e_hat=f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
                    e_true=f"({a:.2f},{b:.2f})",
                )
                pbar.update(100)
        writer.close()
        ro_path = os.path.join(results_dir, "rollout.pkl")
        save_load.save_rollout(ro, ro_path)
        e_norm = np.array(e_norm)
        e_dict[exp_id].append(e_norm)

# save final e_dict
import pickle

with open(os.path.join(base_dir, "unknown_comparison.pkl"), "wb") as f:
    pickle.dump(e_dict, f)

active_dict = pickle.load(open(os.path.join(base_dir, "active_comparison.pkl"), "rb"))
# plot std, mean of each method
plt.close("all")
plt.figure(figsize=(8, 4))
# use seaborn color palette
import seaborn as sns

plt.close("all")
plt.figure(figsize=(8, 4))
# use seaborn color palette
import seaborn as sns

sns.set_palette("Set1")
colorset = sns.color_palette("Set1", n_colors=5)
i = 0
for k, v in e_dict.items():
    if k not in ["active (k=5)", "step", "active chunk(k=20)"]:
        continue
    v = np.array(v)
    v2 = np.array(active_dict[k])
    mean = v.mean(0)
    mean2 = v2.mean(0)
    std = v.std(0)
    std2 = v2.std(0)
    # same color for each method but different line style
    plt.plot(mean, label=k + " (unknown obs.)", linestyle="--", color=colorset[i])
    plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.1, color=colorset[i])
    plt.plot(mean2, label=k + " (known obn)", linestyle="-", color=colorset[i])
    plt.fill_between(
        np.arange(len(mean2)), mean2 - std2, mean2 + std2, alpha=0.1, color=colorset[i]
    )
    i += 1
plt.xlim(0, 500)
plt.xlabel("Environment Steps")
plt.ylabel("Embedding Error Norm")
plt.legend()
plt.title("Embedding Error Norm over Environment Steps")
plt.tight_layout()


# %% 1-2-1. ✅ True EKF with Meta
meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)

torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
params = list(decoder.parameters())
debug_fix_decoder(decoder, obs_model)
warmup_step = 0

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=partial(jacobian_wrt_e, meta_dynamics),
    Fz_net=partial(jacobian_wrt_z, meta_dynamics),
    decoder=decoder,
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=meta_dynamics, device=device
)

model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=20,
    num_iterations=20,
    num_samples=20,
    num_elite=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)
random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "EKF_fixed_active_meta_invariant_discount")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))

ro = Rollout()
pbar = tqdm(range(2000))
for env_step in pbar:
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    # Q = 1e-2 * torch.eye(latent_dim, device=device).unsqueeze(0)

    # 1) Random action sampling
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

    u_t, _, _ = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
    # u_t = random_policy(z_bel["m"].unsqueeze(0)).detach()
    # u_t = step_policy(z_bel["m"].unsqueeze(0)).to(device)

    # 2-1) Predict latent
    dfde = partial(jacobian_wrt_e, meta_dynamics)(z_bel["m"], e_bel["m"]) * env.dt
    Fz = partial(jacobian_wrt_z, meta_dynamics)(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    z_pred = {
        "m": z_bel["m"] + model.dynamics(z_bel["m"]) * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
        + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
    }

    # 2-2) Predict observation
    y_pred = decoder(z_pred["m"])
    R = softplus(decoder.noise.logvar).diag_embed() + eps

    # 3) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  # (1, Do)
    r = y_true - y_pred

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)

    # 4) Embedding update (Laplace)
    S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
    S = symmetrize(S)

    chol_S = torch.linalg.cholesky(S)
    X = torch.cholesky_solve(HzFe, chol_S)
    curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
    curv_ll = symmetrize(curv_ll)  # ensure symmetry
    if env_step > warmup_step:
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)
        for _ in range(1):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.mT, chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_old = e_bel["Prec"]
            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e.squeeze(0), "P": Sigma_e.squeeze(0), "Prec": Prec_new.squeeze(0)}
            Prec, eta = Prec_new, eta_new

    # Detach after all refinements
    e_bel = {k: v.detach() for k, v in e_bel.items()}

    # 5) EKF Update Posterior
    # z_post = encoder(r=r, H=dhdz, R=R, z_pred=z_pred, e_mu=e_bel["m"])
    K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
    I = torch.eye(latent_dim, device=device).unsqueeze(0)
    KH = K @ dhdz

    P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    z_post = {
        "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
        "P": symmetrize(P_upd),
    }

    # 6) Roll updated z posterior as new prior
    z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

    # 7) Optimize Likelihood
    opt.zero_grad(set_to_none=True)

    # Single-sample NLL
    ll = decoder.compute_log_prob(z_bel["m"], y_true)
    loss = -ll
    loss.backward()

    # torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 5.0)
    opt.step()
    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "obs": current_obs,  # Observation  y_t
        "next_obs": obs,  # New Observation y_{t+1}
        "action": u_t,  # Action a_t
        "env_state": current_state,  # Environment state z_t
        "next_env_state": info["latent_state"],  # Next environment state z_{t+1}
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            LL=f"{ll.item():.3f}",
            e_hat=f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
            e_true=f"({a:.2f},{b:.2f})",
        )
        pbar.update(100)
writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)

# %% 1-2-1-1. Meta EKF + Action chunking
meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)

torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
params = list(decoder.parameters())
debug_fix_decoder(decoder, obs_model)
warmup_step = 0

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=partial(jacobian_wrt_e, meta_dynamics),
    Fz_net=partial(jacobian_wrt_z, meta_dynamics),
    decoder=decoder,
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=meta_dynamics, device=device
)

model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=10,
    num_iterations=20,
    num_samples=40,
    num_elite=10,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)
random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "EKF_fixed_active_meta_chunk")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))

ro = Rollout()
pbar = tqdm(range(2000))
for env_step in pbar:
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    # Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    Q = 1e-2 * torch.eye(latent_dim, device=device).unsqueeze(0)

    # 1) Random action sampling
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    if env_step % 5 == 0:
        u_t, u_ts, cost = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
        # print(f"Cost: {cost.mean().item():.3f}")
    else:
        u_t = u_ts[:, env_step % 5].unsqueeze(1).detach()
    # u_t, _, _ = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)

    # u_t = random_policy(z_bel["m"].unsqueeze(0)).detach()
    # u_t = step_policy(z_bel["m"].unsqueeze(0)).to(device)

    # 2-1) Predict latent
    dfde = partial(jacobian_wrt_e, meta_dynamics)(z_bel["m"], e_bel["m"]) * env.dt
    Fz = partial(jacobian_wrt_z, meta_dynamics)(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    z_pred = {
        "m": z_bel["m"] + model.dynamics(z_bel["m"]) * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
        + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
    }

    # 2-2) Predict observation
    y_pred = decoder(z_pred["m"])
    R = softplus(decoder.noise.logvar).diag_embed() + eps

    # 3) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  # (1, Do)
    r = y_true - y_pred

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)

    # 4) Embedding update (Laplace)
    S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
    S = symmetrize(S)

    chol_S = torch.linalg.cholesky(S)
    X = torch.cholesky_solve(HzFe, chol_S)
    curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
    curv_ll = symmetrize(curv_ll)  # ensure symmetry
    if env_step > warmup_step:
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)
        for _ in range(1):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.mT, chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_old = e_bel["Prec"]
            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e.squeeze(0), "P": Sigma_e.squeeze(0), "Prec": Prec_new.squeeze(0)}
            Prec, eta = Prec_new, eta_new

    # Detach after all refinements
    e_bel = {k: v.detach() for k, v in e_bel.items()}

    # 5) EKF Update Posterior
    # z_post = encoder(r=r, H=dhdz, R=R, z_pred=z_pred, e_mu=e_bel["m"])
    K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
    I = torch.eye(latent_dim, device=device).unsqueeze(0)
    KH = K @ dhdz

    P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    z_post = {
        "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
        "P": symmetrize(P_upd),
    }

    # 6) Roll updated z posterior as new prior
    z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

    # 7) Optimize Likelihood
    opt.zero_grad(set_to_none=True)

    # Single-sample NLL
    ll = decoder.compute_log_prob(z_bel["m"], y_true)
    loss = -ll
    loss.backward()

    # torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 5.0)
    opt.step()
    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "obs": current_obs,  # Observation  y_t
        "next_obs": obs,  # New Observation y_{t+1}
        "action": u_t,  # Action a_t
        "env_state": current_state,  # Environment state z_t
        "next_env_state": info["latent_state"],  # Next environment state z_{t+1}
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            LL=f"{ll.item():.3f}",
            e_hat=f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
            e_true=f"({a:.2f},{b:.2f})",
        )
        pbar.update(100)
writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)

# %% 1-2-2. ❌ Full EKF with no likelihood
meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)

torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
params = list(decoder.parameters())
warmup_step = 2500

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}
meta_dynamics.set_embedding(e_bel["m"])
emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=Fe_true, Fz_net=Fz_true, decoder=decoder
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
)

model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=10,
    num_iterations=5,
    num_samples=20,
    num_elite=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "EKF_active_warmup")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
# clear existing log
shutil.rmtree(writer.log_dir, ignore_errors=True)
os.makedirs(writer.log_dir, exist_ok=True)

ro = Rollout()
pbar = tqdm(range(5000))
for env_step in pbar:
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    # Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    Q = 1e-2 * torch.eye(latent_dim, device=device).unsqueeze(0)

    # 1) Random action sampling
    meta_dynamics.set_embedding(e_bel["m"])
    sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

    if env_step < warmup_step:
        u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
    else:
        u_t = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q).detach()

    # 2-1) Predict latent
    dfde = Fe_true(z_bel["m"], e_bel["m"]) * env.dt
    Fz = Fz_true(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    f_true = sim_vec_env.compute_dynamics(z_bel["m"]).to(device)  # For debug
    z_pred = {
        "m": z_bel["m"] + f_true * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
        + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
    }

    # 2-2) Predict observation
    y_pred = decoder(z_pred["m"])
    R = softplus(decoder.noise.logvar).diag_embed() + eps

    # 3) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  # (1, Do)
    r = y_true - y_pred

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)

    # 4) Embedding update (Laplace)

    S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
    S = symmetrize(S)
    chol_S = torch.linalg.cholesky(S)
    if env_step > warmup_step:
        X = torch.cholesky_solve(HzFe, chol_S)
        curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
        curv_ll = symmetrize(curv_ll)  # ensure symmetry
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)
        for _ in range(10):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.mT, chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_old = e_bel["Prec"]
            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e.squeeze(0), "P": Sigma_e.squeeze(0), "Prec": Prec_new.squeeze(0)}
            Prec, eta = Prec_new, eta_new

        # Detach after all refinements
        e_bel = {k: v.detach() for k, v in e_bel.items()}

    # 5) EKF Update Posterior
    # z_post = encoder(r=r, H=dhdz, R=R, z_pred=z_pred, e_mu=e_bel["m"])
    K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
    I = torch.eye(latent_dim, device=device).unsqueeze(0)
    KH = K @ dhdz

    P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    z_post = {
        "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
        "P": symmetrize(P_upd),
    }
    # 6) Roll updated z posterior as new prior
    z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

    # 7) Optimize Likelihood
    opt.zero_grad(set_to_none=True)

    # Single-sample NLL
    ll = decoder.compute_log_prob(z_bel["m"], y_true)
    loss = -ll
    loss.backward()

    # torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 5.0)
    opt.step()
    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "obs": current_obs,  # Observation  y_t
        "next_obs": obs,  # New Observation y_{t+1}
        "action": u_t,  # Action a_t
        "env_state": current_state,  # Environment state z_t
        "next_env_state": info["latent_state"],  # Next environment state z_{t+1}
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            LL=f"{ll.item():.3f}",
            e_hat=f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
            e_true=f"({a:.2f},{b:.2f})",
        )
        pbar.update(100)
writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)

# %% 1-2-3. ❌ EKF with no likelihood with alternate policy (random/step)
eta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)

torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
params = list(decoder.parameters())
warmup_step = 500

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}
meta_dynamics.set_embedding(e_bel["m"])
emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=Fe_true, Fz_net=Fz_true, decoder=decoder
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
)

model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=10,
    num_iterations=5,
    num_samples=20,
    num_elite=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "EKF_random_alternate")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
# clear existing log
shutil.rmtree(writer.log_dir, ignore_errors=True)
os.makedirs(writer.log_dir, exist_ok=True)

ro = Rollout()
pbar = tqdm(range(5000))
for env_step in pbar:
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    # Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    Q = 1e-2 * torch.eye(latent_dim, device=device).unsqueeze(0)

    # 1) Random action sampling
    meta_dynamics.set_embedding(e_bel["m"])
    sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

    u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
    # u_t = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q).detach()

    # 2-1) Predict latent
    dfde = Fe_true(z_bel["m"], e_bel["m"]) * env.dt
    Fz = Fz_true(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    f_true = sim_vec_env.compute_dynamics(z_bel["m"]).to(device)  # For debug
    z_pred = {
        "m": z_bel["m"] + f_true * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
        + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
    }

    # 2-2) Predict observation
    y_pred = decoder(z_pred["m"])
    R = softplus(decoder.noise.logvar).diag_embed() + eps

    # 3) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  # (1, Do)
    r = y_true - y_pred

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)

    # 4) Embedding update (Laplace)
    S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
    S = symmetrize(S)
    chol_S = torch.linalg.cholesky(S)
    if (env_step // warmup_step) % 2 == 0:
        e_bel.update(
            {
                "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
                "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
            }
        )
    if (env_step // warmup_step) % 2 == 1:
        X = torch.cholesky_solve(HzFe, chol_S)
        curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
        curv_ll = symmetrize(curv_ll)  # ensure symmetry
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec_old = e_bel["Prec"]
        eta = Prec_old @ e_bel["m"].unsqueeze(-1)
        for _ in range(10):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.mT, chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e.squeeze(0), "P": Sigma_e.squeeze(0), "Prec": Prec_new.squeeze(0)}
            Prec_old, eta = Prec_new, eta_new

        # Detach after all refinements
        e_bel = {k: v.detach() for k, v in e_bel.items()}

    # 5) EKF Update Posterior
    # z_post = encoder(r=r, H=dhdz, R=R, z_pred=z_pred, e_mu=e_bel["m"])
    K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
    I = torch.eye(latent_dim, device=device).unsqueeze(0)
    KH = K @ dhdz

    P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    z_post = {
        "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
        "P": symmetrize(P_upd),
    }
    # 6) Roll updated z posterior as new prior
    z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

    # 7) Optimize Likelihood
    opt.zero_grad(set_to_none=True)

    # Single-sample NLL
    if (env_step // warmup_step) % 2 == 0:
        ll = decoder.compute_log_prob(z_bel["m"], y_true)
        loss = -ll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 5.0)
        opt.step()

    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "obs": current_obs,  # Observation  y_t
        "next_obs": obs,  # New Observation y_{t+1}
        "action": u_t,  # Action a_t
        "env_state": current_state,  # Environment state z_t
        "next_env_state": info["latent_state"],  # Next environment state z_{t+1}
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            LL=f"{ll.item():.3f}",
            e_hat=f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
            e_true=f"({a:.2f},{b:.2f})",
        )
        pbar.update(100)
writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)

# %% 1-3. ✅ (True value) Amortized Latent with window + EKF/Laplace (with Embedding)
torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
encoder = actdyn.models.encoder.RNNEmbeddingEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    embedding_dim=embedding_dim,
    hidden_dim=64,
    device=device,
)
emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=Fe_true, Fz_net=Fz_true, decoder=decoder
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
)
model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=30,
    num_iterations=10,
    num_samples=20,
    num_elite=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)

params = list(decoder.parameters()) + list(encoder.parameters()) + list(dynamics.parameters())

warmup_step = 500

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "RNN_fixed_active_invariant_chunk")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
# clear existing log
shutil.rmtree(writer.log_dir, ignore_errors=True)
os.makedirs(writer.log_dir, exist_ok=True)

ro = Rollout()
windows_length = 100
rb = RecentRollout(max_len=windows_length)
pbar = tqdm(range(5000))
n_samples = 10
debug_fix_decoder(decoder, obs_model)

for env_step in pbar:
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    R = softplus(decoder.noise.logvar).diag_embed()

    # 1) Random action sampling
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    # u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
    # u_t = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q).detach()
    if env_step % 5 == 0:
        u_t, u_ts, _ = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
    else:
        u_t = u_ts[:, env_step % 5].unsqueeze(1).detach()

    # -----------------------------------
    # 2) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  #  (1, Do)
    y_pred = decoder(z_pred["m"])
    r = y_true - y_pred
    ro.add(
        **{
            "obs": current_obs,
            "next_obs": obs.detach(),
            "action": u_t.detach(),
            "env_state": current_state.detach(),
            "next_env_state": info["latent_state"].detach(),
        }
    )
    rb.add(
        **{
            "obs": current_obs,
            "next_obs": obs.detach(),
            "action": u_t.detach(),
            "env_state": current_state.detach(),
            "next_env_state": info["latent_state"].detach(),
        }
    )

    # -----------------------------------
    # 3) Update Latent Posterior
    e_rep = repeat(e_bel["m"], "b d -> b t d", t=len(rb)).to(device)
    if env_step < warmup_step:
        gamma, beta = 1.0, 0.0
    else:
        gamma, beta = None, None
    z_samples, mu_q, var_q = encoder(
        y=rb["next_obs"], u=rb["action"], e=e_rep, n_samples=n_samples, gamma=1, beta=0
    )  # (S, 1, T, Dz), (1, T, Dz), (1, T, Dz)

    if env_step == 0:
        z_bel = {"m": mu_q[:, -1].detach(), "P": var_q[0, -1].diag_embed().detach()}
    else:
        z_bel = {"m": mu_q[:, -2].detach(), "P": var_q[0, -2].diag_embed().detach()}

    # -----------------------------------
    # 4) Compute Predictive latent distribution
    dfde = Fe_true(z_bel["m"], e_bel["m"]) * env.dt
    Fz = Fz_true(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    z_pred = {
        "m": z_bel["m"] + model.dynamics(z_bel["m"]) * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
        + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
    }

    # -----------------------------------
    # 5) Embedding update (Laplace)
    if env_step > warmup_step:
        S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
        S = symmetrize(S)
        chol_S = torch.linalg.cholesky(S)
        X = torch.cholesky_solve(HzFe, chol_S)
        curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
        curv_ll = symmetrize(curv_ll)  # ensure symmetry
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec_old = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)

        for _ in range(10):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.transpose(1, 2), chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e.squeeze(0), "P": Sigma_e.squeeze(0), "Prec": Prec_new.squeeze(0)}
            Prec_old, eta = Prec_new, eta_new

        # EMA update for mu_e
        # mu_e = e_bel["m"] * 0.5 + mu_e * 0.5
        # mu_e.clamp_(-5.0, 5.0)
        # # e_bel["Prec"] *= 0.9999  # Forgetting
        # e_bel["Prec"].clamp_(-1e3, 1e3)
        e_bel = {k: v.detach() for k, v in e_bel.items()}
    # -----------------------------------
    # 7) Optimize likelihood and encoder
    opt.zero_grad(set_to_none=True)

    # Prior Gating
    z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
    T = z_flat.size(-2)
    t_mask = None

    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

    if env_step > warmup_step:
        z_p = (z_flat + model.dynamics(z_bel["m"]) * env.dt)[..., :-1, :]
        z_p += rb["action"][..., 1:, :] * env.dt
        z_p += torch.randn_like(z_p) * (Q[0].diag()).sqrt()
        z_p = torch.cat([z_flat[..., :1, :], z_p], dim=-2)  # ((S B),T,D)
        mu_p = (mu_q + model.dynamics(mu_q) * env.dt)[..., :-1, :]
        mu_p += rb["action"][..., 1:, :] * env.dt
        mu_p = torch.cat([mu_q[..., :1, :], mu_p], dim=-2)

    p_mask = 0.0
    if env_step > warmup_step and p_mask > 0:
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=device))
        z_flat = z_flat * t_mask + z_p * (1 - t_mask)

    # Compute log likelihood
    y_rep = repeat(rb["next_obs"], "b t d -> (s b) t d", s=n_samples)
    ll_sb = decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
    ll_b = ll_sb.view(n_samples, -1).mean(dim=0)  # (B,)

    # Compute KL
    kl_b = torch.zeros(1, device=device)
    if env_step > warmup_step:
        kl_d = 0.5 * (
            torch.log(Q[0].diag() / var_q)
            + ((mu_q - mu_p) ** 2) / Q[0].diag()
            + (var_q / Q[0].diag())
            - 1
        )
        kl_sb = kl_d.sum(dim=-1)  # (S*B, T)
        if t_mask is not None:
            kl_sb = kl_sb * t_mask[1:, :]  # (S*B,T)

        kl_b = kl_sb.view(n_samples, -1).mean(dim=0).sum(-1)  # (B,)

    beta = torch.min(
        torch.tensor(env_step / (warmup_step + 1), device=device), torch.tensor(1.0, device=device)
    )
    elbo = ll_b.mean() - kl_b.mean() * beta
    loss = -elbo
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()

    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            {
                "ELBO": f"{elbo/windows_length:.4f}",
                "e_hat": f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
                "e_true": f"({a:.2f},{b:.2f})",
            }
        )
        pbar.update(100)

writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)


# %% 1-4. Amortized with metadyn
torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
encoder = actdyn.models.encoder.RNNEmbeddingEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    embedding_dim=embedding_dim,
    hidden_dim=64,
    device=device,
)

meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)
emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=partial(jacobian_wrt_e, meta_dynamics),
    Fz_net=partial(jacobian_wrt_z, meta_dynamics),
    decoder=decoder,
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=meta_dynamics, device=device
)
model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=10,
    num_iterations=10,
    num_samples=20,
    num_elite=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)

params = list(decoder.parameters()) + list(encoder.parameters()) + list(dynamics.parameters())

warmup_step = 500

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "RNN_fixed_active_meta_invariant")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
# clear existing log
shutil.rmtree(writer.log_dir, ignore_errors=True)
os.makedirs(writer.log_dir, exist_ok=True)

ro = Rollout()
windows_length = 100
rb = RecentRollout(max_len=windows_length)
pbar = tqdm(range(10000))
n_samples = 10
debug_fix_decoder(decoder, obs_model)

for env_step in pbar:
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    R = softplus(decoder.noise.logvar).diag_embed()

    # 1) Random action sampling
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    if env_step < warmup_step:
        u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
    else:
        # u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
        chunk = 5
        if env_step % chunk == 0:
            u_t, u_ts, _ = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
        else:
            u_t = u_ts[:, env_step % chunk].unsqueeze(1).detach()

    # -----------------------------------
    # 2) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  #  (1, Do)

    ro.add(
        **{
            "obs": current_obs,
            "next_obs": obs.detach(),
            "action": u_t.detach(),
            "env_state": current_state.detach(),
            "next_env_state": info["latent_state"].detach(),
        }
    )
    rb.add(
        **{
            "obs": current_obs,
            "next_obs": obs.detach(),
            "action": u_t.detach(),
            "env_state": current_state.detach(),
            "next_env_state": info["latent_state"].detach(),
        }
    )

    # -----------------------------------
    # 3) Update Latent Posterior
    e_rep = repeat(e_bel["m"], "b d -> b t d", t=len(rb)).to(device)
    if env_step < warmup_step:
        gamma, beta = 1.0, 0.0
    else:
        gamma, beta = None, None
    z_samples, mu_q, var_q = encoder(
        y=rb["next_obs"], u=rb["action"], e=e_rep, n_samples=n_samples, gamma=1, beta=0
    )  # (S, 1, T, Dz), (1, T, Dz), (1, T, Dz)

    if env_step == 0:
        z_bel = {"m": mu_q[:, -1].detach(), "P": var_q[0, -1].diag_embed().detach()}
    else:
        z_bel = {"m": mu_q[:, -2].detach(), "P": var_q[0, -2].diag_embed().detach()}

    # -----------------------------------
    # 4) Compute Predictive latent distribution
    dfde = partial(jacobian_wrt_e, meta_dynamics)(z_bel["m"], e_bel["m"]) * env.dt
    Fz = partial(jacobian_wrt_z, meta_dynamics)(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    z_pred = {
        "m": z_bel["m"] + model.dynamics(z_bel["m"]) * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2) + Q,
    }

    # -----------------------------------
    # 5) Embedding update (Laplace)
    if env_step > warmup_step:
        S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
        S = symmetrize(S)
        chol_S = torch.linalg.cholesky(S)
        X = torch.cholesky_solve(HzFe, chol_S)
        curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
        curv_ll = symmetrize(curv_ll)  # ensure symmetry
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec_old = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)

        for _ in range(10):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.transpose(1, 2), chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e.squeeze(0), "P": Sigma_e.squeeze(0), "Prec": Prec_new.squeeze(0)}
            Prec_old, eta = Prec_new, eta_new

        # EMA update for mu_e
        # mu_e = e_bel["m"] * 0.5 + mu_e * 0.5
        # mu_e.clamp_(-5.0, 5.0)
        # e_bel["Prec"] *= 0.999  # Forgetting
        # e_bel["Prec"].clamp_(-1e3, 1e3)
        e_bel = {k: v.detach() for k, v in e_bel.items()}
    # -----------------------------------
    # 7) Optimize likelihood and encoder
    opt.zero_grad(set_to_none=True)

    # Prior Gating
    z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
    T = z_flat.size(-2)
    t_mask = None

    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

    if env_step > warmup_step:
        z_p = (z_flat + model.dynamics(z_bel["m"]) * env.dt)[..., :-1, :]
        z_p += rb["action"][..., 1:, :] * env.dt
        z_p += torch.randn_like(z_p) * (Q[0].diag()).sqrt()
        z_p = torch.cat([z_flat[..., :1, :], z_p], dim=-2)  # ((S B),T,D)
        mu_p = (mu_q + model.dynamics(mu_q) * env.dt)[..., :-1, :]
        mu_p += rb["action"][..., 1:, :] * env.dt
        mu_p = torch.cat([mu_q[..., :1, :], mu_p], dim=-2)

    p_mask = 0.3
    if env_step > warmup_step and p_mask > 0:
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=device))
        z_flat = z_flat * t_mask + z_p * (1 - t_mask)

    # Compute log likelihood
    y_rep = repeat(rb["next_obs"], "b t d -> (s b) t d", s=n_samples)
    ll_sb = decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
    ll_b = ll_sb.view(n_samples, -1).mean(dim=0)  # (B,)

    # Compute KL
    kl_b = torch.zeros(1, device=device)
    if env_step > warmup_step:
        kl_d = 0.5 * (
            torch.log(Q[0].diag() / var_q)
            + ((mu_q - mu_p) ** 2) / Q[0].diag()
            + (var_q / Q[0].diag())
            - 1
        )
        kl_sb = kl_d.sum(dim=-1)  # (S*B, T)
        if t_mask is not None:
            kl_sb = kl_sb * t_mask[1:, :]  # (S*B,T)

        kl_b = kl_sb.view(n_samples, -1).mean(dim=0).sum(-1)  # (B,)

    beta = torch.min(
        torch.tensor(env_step / (warmup_step + 1), device=device), torch.tensor(1.0, device=device)
    )
    elbo = ll_b.mean() - kl_b.mean() * beta
    loss = -elbo
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()

    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            {
                "ELBO": f"{elbo/windows_length:.4f}",
                "e_hat": f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
                "e_true": f"({a:.2f},{b:.2f})",
            }
        )
        pbar.update(100)

writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)

# %% 1-3. ❌ Amortized Latent with window + EKF/Laplace (with Embedding)
torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
encoder = actdyn.models.encoder.RNNEmbeddingEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    embedding_dim=embedding_dim,
    hidden_dim=64,
    device=device,
)
emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=Fe_true, Fz_net=Fz_true, decoder=decoder
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
)
model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=10,
    num_iterations=5,
    num_samples=20,
    num_elite=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)

params = list(decoder.parameters()) + list(encoder.parameters()) + list(dynamics.parameters())

warmup_step = 500

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-2, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "RNN_random_alternate")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
# clear existing log
shutil.rmtree(writer.log_dir, ignore_errors=True)
os.makedirs(writer.log_dir, exist_ok=True)

ro = Rollout()
windows_length = 100
rb = RecentRollout(max_len=windows_length)
pbar = tqdm(range(10000))
n_samples = 10
# debug_fix_decoder(decoder, obs_model)

for env_step in pbar:
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    R = softplus(decoder.noise.logvar).diag_embed()

    # 1) Random action sampling
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
    # u_t = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q).detach()

    # -----------------------------------
    # 2) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  #  (1, Do)
    r = y_true - y_pred
    ro.add(
        **{
            "obs": current_obs,
            "next_obs": obs.detach(),
            "action": u_t.detach(),
            "env_state": current_state.detach(),
            "next_env_state": info["latent_state"].detach(),
        }
    )
    rb.add(
        **{
            "obs": current_obs,
            "next_obs": obs.detach(),
            "action": u_t.detach(),
            "env_state": current_state.detach(),
            "next_env_state": info["latent_state"].detach(),
        }
    )

    # -----------------------------------
    # 3) Update Latent Posterior
    e_rep = repeat(e_bel["m"], "b d -> b t d", t=len(rb)).to(device)
    if env_step < warmup_step:
        gamma, beta = 1.0, 0.0
    else:
        gamma, beta = None, None
    z_samples, mu_q, var_q = encoder(
        y=rb["next_obs"], u=rb["action"], e=e_rep, n_samples=n_samples, gamma=1, beta=0
    )  # (S, 1, T, Dz), (1, T, Dz), (1, T, Dz)

    if env_step == 0:
        z_bel = {"m": mu_q[:, -1].detach(), "P": var_q[0, -1].diag_embed().detach()}
    else:
        z_bel = {"m": mu_q[:, -2].detach(), "P": var_q[0, -2].diag_embed().detach()}

    # -----------------------------------
    # 4) Compute Predictive latent distribution
    dfde = Fe_true(z_bel["m"], e_bel["m"]) * env.dt
    Fz = Fz_true(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    z_pred = {
        "m": z_bel["m"] + model.dynamics(z_bel["m"]) * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
        + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
    }

    if (env_step // warmup_step) % 2 == 0:
        e_bel.update(
            {
                "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
                "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
            }
        )
    if (env_step // warmup_step) % 2 == 1:
        S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
        S = symmetrize(S)
        chol_S = torch.linalg.cholesky(S)
        X = torch.cholesky_solve(HzFe, chol_S)
        curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
        curv_ll = symmetrize(curv_ll)  # ensure symmetry
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec_old = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)

        for _ in range(10):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.transpose(1, 2), chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e.squeeze(0), "P": Sigma_e.squeeze(0), "Prec": Prec_new.squeeze(0)}
            Prec_old, eta = Prec_new, eta_new

        # EMA update for mu_e
        # mu_e = e_bel["m"] * 0.5 + mu_e * 0.5
        # mu_e.clamp_(-5.0, 5.0)
        # # e_bel["Prec"] *= 0.9999  # Forgetting
        # e_bel["Prec"].clamp_(-1e3, 1e3)
        e_bel = {k: v.detach() for k, v in e_bel.items()}
    # -----------------------------------
    # 7) Optimize likelihood and encoder
    opt.zero_grad(set_to_none=True)

    # Prior Gating
    z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
    T = z_flat.size(-2)
    t_mask = None

    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

    if (env_step // warmup_step) % 2 == 0:
        z_p = (z_flat + model.dynamics(z_bel["m"]) * env.dt)[..., :-1, :]
        z_p += rb["action"][..., 1:, :] * env.dt
        z_p += torch.randn_like(z_p) * (Q[0].diag()).sqrt()
        z_p = torch.cat([z_flat[..., :1, :], z_p], dim=-2)  # ((S B),T,D)
        mu_p = (mu_q + model.dynamics(mu_q) * env.dt)[..., :-1, :]
        mu_p += rb["action"][..., 1:, :] * env.dt
        mu_p = torch.cat([mu_q[..., :1, :], mu_p], dim=-2)

    p_mask = 0.5
    if (env_step // warmup_step) % 2 == 0 and p_mask > 0:
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=device))
        z_flat = z_flat * t_mask + z_p * (1 - t_mask)

    # Compute log likelihood
    y_rep = repeat(rb["next_obs"], "b t d -> (s b) t d", s=n_samples)
    ll_sb = decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
    ll_b = ll_sb.view(n_samples, -1).mean(dim=0)  # (B,)

    # Compute KL
    kl_b = torch.zeros(1, device=device)
    if env_step > warmup_step:
        kl_d = 0.5 * (
            torch.log(Q[0].diag() / var_q)
            + ((mu_q - mu_p) ** 2) / Q[0].diag()
            + (var_q / Q[0].diag())
            - 1
        )
        kl_sb = kl_d.sum(dim=-1)  # (B, T)
        if t_mask is not None:
            kl_sb = kl_sb * t_mask[1:, :]  # (B,T)

        kl_b = kl_sb.mean(dim=0).sum(-1)  # (B,)

    beta = torch.min(
        torch.tensor(env_step / (warmup_step + 1), device=device), torch.tensor(1.0, device=device)
    )
    elbo = ll_b.mean() - kl_b.mean() * beta
    if (env_step // warmup_step) % 2 == 0:
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            {
                "ELBO": f"{elbo/windows_length:.4f}",
                "e_hat": f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
                "e_true": f"({a:.2f},{b:.2f})",
            }
        )
        pbar.update(100)

writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)

# %% 1-3. ❌ Amortized Latent with window + EKF/Laplace Alternating
torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
encoder = actdyn.models.encoder.RNNEmbeddingEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    embedding_dim=embedding_dim,
    hidden_dim=64,
    device=device,
)
emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=Fe_true, Fz_net=Fz_true, decoder=decoder
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
)
mapping_tf = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=latent_dim, device=device
)

model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=10,
    num_iterations=5,
    num_samples=20,
    num_elite=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)

params = list(decoder.parameters()) + list(encoder.parameters()) + list(dynamics.parameters())

warmup_step = 2000

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-2, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "RNN_random_alternate")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
# clear existing log
shutil.rmtree(writer.log_dir, ignore_errors=True)
os.makedirs(writer.log_dir, exist_ok=True)

ro = Rollout()
windows_length = 100
rb = RecentRollout(max_len=windows_length)
pbar = tqdm(range(50000))
n_samples = 10
# debug_fix_decoder(decoder, obs_model)

for env_step in pbar:
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    R = softplus(decoder.noise.logvar).diag_embed()

    # 1) Random action sampling
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
    # u_t = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q).detach()

    # -----------------------------------
    # 2) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  #  (1, Do)
    ro.add(
        **{
            "obs": current_obs,
            "next_obs": obs.detach(),
            "action": u_t.detach(),
            "env_state": current_state.detach(),
            "next_env_state": info["latent_state"].detach(),
        }
    )
    rb.add(
        **{
            "obs": current_obs,
            "next_obs": obs.detach(),
            "action": u_t.detach(),
            "env_state": current_state.detach(),
            "next_env_state": info["latent_state"].detach(),
        }
    )

    if (env_step // warmup_step) % 2 == 0:
        # -----------------------------------
        # 3) Update Latent Posterior
        e_rep = repeat(e_bel["m"], "b d -> b t d", t=len(rb)).to(device)
        if env_step < warmup_step:
            gamma, beta = 1.0, 0.0
        else:
            gamma, beta = None, None
        z_samples, mu_q, var_q = encoder(
            y=rb["next_obs"], u=rb["action"], e=e_rep, n_samples=n_samples, gamma=1.0, beta=0.0
        )  # (S, 1, T, Dz), (1, T, Dz), (1, T, Dz)

        if env_step == 0:
            z_bel = {"m": mu_q[:, -1].detach(), "P": var_q[0, -1].diag_embed().detach()}
        else:
            z_bel = {"m": mu_q[:, -2].detach(), "P": var_q[0, -2].diag_embed().detach()}

        # -----------------------------------
        # 4) Compute Predictive latent distribution
        opt.zero_grad(set_to_none=True)

        # Prior Gating
        z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
        T = z_flat.size(-2)
        t_mask = None

        model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

        z_p = (z_flat + model.dynamics(z_bel["m"]) * env.dt)[..., :-1, :]
        z_p += rb["action"][..., 1:, :] * env.dt
        z_p += torch.randn_like(z_p) * (Q[0].diag()).sqrt()
        z_p = torch.cat([z_flat[..., :1, :], z_p], dim=-2)  # ((S B),T,D)
        mu_p = (mu_q + model.dynamics(mu_q) * env.dt)[..., :-1, :]
        mu_p += rb["action"][..., 1:, :] * env.dt
        mu_p = torch.cat([mu_q[..., :1, :], mu_p], dim=-2)

        p_mask = 0.5
        if p_mask > 0:
            t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=device))
            z_flat = z_flat * t_mask + z_p * (1 - t_mask)

        # Compute log likelihood
        y_rep = repeat(rb["next_obs"], "b t d -> (s b) t d", s=n_samples)
        ll_sb = decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
        ll_b = ll_sb.view(n_samples, -1).mean(dim=0)  # (B,)

        # Compute KL
        kl_b = torch.zeros(1, device=device)
        kl_d = 0.5 * (
            torch.log(Q[0].diag() / var_q)
            + ((mu_q - mu_p) ** 2) / Q[0].diag()
            + (var_q / Q[0].diag())
            - 1
        )
        kl_sb = kl_d.sum(dim=-1)  # (B, T)
        if t_mask is not None:
            kl_sb = kl_sb * t_mask[1:, :]  # (B,T)

        kl_b = kl_sb.mean(dim=0).sum(-1)  # (B,)

        beta = torch.min(
            torch.tensor(env_step / (warmup_step + 1), device=device),
            torch.tensor(1.0, device=device),
        )
        elbo = ll_b.mean() - kl_b.mean() * beta

        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    else:

        z_pred = {
            "m": z_bel["m"] + model.dynamics(z_bel["m"]) * env.dt + u_t * env.dt,
            "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
            + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
        }
        y_true = obs.squeeze(0)  #  (1, Do)
        y_pred = decoder(z_pred["m"])
        r = y_true - y_pred
        # 2-1) Predict latent
        dfde = Fe_true(z_bel["m"], e_bel["m"]) * env.dt
        Fz = Fz_true(z_bel["m"], e_bel["m"])
        dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
        dhdz = decoder.jacobian.unsqueeze(0)
        HzFe = dhdz @ dfde  # (1, Do, De)

        # 2-2) Predict observation
        y_pred = decoder(z_pred["m"])
        R = softplus(decoder.noise.logvar).diag_embed() + eps
        S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
        S = symmetrize(S)
        chol_S = torch.linalg.cholesky(S)

        X = torch.cholesky_solve(HzFe, chol_S)
        curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
        curv_ll = symmetrize(curv_ll)  # ensure symmetry
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)
        for _ in range(10):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.mT, chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_old = e_bel["Prec"]
            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {"m": mu_e.squeeze(0), "P": Sigma_e.squeeze(0), "Prec": Prec_new.squeeze(0)}
            Prec, eta = Prec_new, eta_new

        # Detach after all refinements
        e_bel = {k: v.detach() for k, v in e_bel.items()}
        # 5) EKF Update Posterior
        # z_post = encoder(r=r, H=dhdz, R=R, z_pred=z_pred, e_mu=e_bel["m"])
        K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
        I = torch.eye(latent_dim, device=device).unsqueeze(0)
        KH = K @ dhdz

        P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
        z_post = {
            "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
            "P": symmetrize(P_upd),
        }
        # 6) Roll updated z posterior as new prior
        z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

        e_rep = repeat(e_bel["m"], "b d -> b t d", t=len(rb)).to(device)
        z_samples, mu_q, var_q = encoder(
            y=rb["next_obs"],
            u=rb["action"],
            e=e_rep,
            n_samples=n_samples,
            gamma=1.0,
            beta=0.0,
        )
        z_kf = torch.cat([rb["next_model_state"], z_bel["m"].unsqueeze(0)], dim=-2)
        z_tf = mapping_tf(z_kf)
        save_mu = mu_q
        save_z = rb["next_env_state"]

        # mse loss between mu_q and z_tf
        opt2 = torch.optim.SGD(mapping_tf.parameters(), lr=1e-4, weight_decay=1e-4)

        opt2.zero_grad(set_to_none=True)
        loss = F.mse_loss(mu_q.detach(), z_tf)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapping_tf.parameters(), 5.0)
        opt2.step()

    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    rb.add(**transition)

    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            {
                "ELBO": f"{elbo/windows_length:.4f}",
                "e_hat": f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
                "e_true": f"({a:.2f},{b:.2f})",
            }
        )
        pbar.update(100)

writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)


# %% 1-2. ? Amortized Latent with window + EKF/Laplace (with Embedding)
# Use amortized latent encoder with small trailing window to infer latent posterior
# Use EKF to get predictive latent covariance
# Use Laplace to refine embedding posterior

torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
encoder = actdyn.models.encoder.RNNEmbeddingEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    embedding_dim=embedding_dim,
    hidden_dim=64,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=meta_dynamics_fn, device=device
)
policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)
params = list(decoder.parameters()) + list(encoder.parameters()) + list(dynamics.parameters())
# debug_fix_decoder(decoder=decoder, obs_model=obs_model)
# debug_fix_decoder(decoder, obs_model)
frames = []
plt.close("all")
z_bel = {
    "m": torch.zeros(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z, z_hat = [], []  # For debugging purpose

total_steps = 100000
pbar = tqdm(range(1, total_steps))
windows_length = 1000
warmup_step = 1000
n_samples = 5
rb = RecentRollout(max_len=1000, device=device)
plot_rollout = RecentRollout(max_len=500, device=device)
rb.add(
    **{
        "obs": obs,
        "next_obs": obs,
        "action": torch.zeros(1, action_dim, device=device),
        "next_env_state": info["latent_state"],
        "next_model_state": z_bel["m"],
    }
)
plot_rollout.add(
    **{
        "next_env_state": info["latent_state"],
        "next_model_state": z_bel["m"],
    }
)
prev_action = torch.zeros(1, action_dim, device=device)

results_dir = os.path.join(base_dir, "ekf_laplace_amortized")
for subdir in ["rollouts", "logs", "model", "video", "video/images"]:
    p = os.path.join(results_dir, subdir)
    # Clean up previous results
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
video_path = os.path.join(results_dir, f"video/vecfield.mp4")


for env_step in pbar:
    # 1) Random action sampling
    u_t = policy(z_bel["m"]).detach()

    # -----------------------------------
    # 2) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    rb.add(
        **{
            "obs": rb["next_obs"][:, -1, :].unsqueeze(1).detach(),
            "next_obs": obs.detach(),
            "action": u_t.detach(),
        }
    )

    # -----------------------------------
    # 3) Update Latent Posterior
    e_rep = repeat(e_bel["m"], "b d -> b t d", t=len(rb)).to(device)
    if env_step < warmup_step:
        gamma, beta = 1.0, 0.0
    else:
        gamma, beta = None, None
    z_samples, mu_q, var_q = encoder(
        y=rb["next_obs"], u=rb["action"], e=e_rep, n_samples=n_samples, gamma=gamma, beta=beta
    )  # (S, 1, T, Dz), (1, T, Dz), (1, T, Dz)
    R = softplus(decoder.noise.logvar).diag_embed() + eps
    if env_step < warmup_step:
        R = R.detach()
    Q = softplus(dynamics.logvar).diag_embed().squeeze(0) * env.dt + eps

    z_bel = {"m": mu_q[:, -2].detach(), "P": var_q[0, -2].diag_embed().detach()}

    # -----------------------------------
    # 4) Compute Predictive latent distribution
    Fz = Fz_net(z_bel["m"], e_bel["m"]).detach().squeeze(0)  # (1, Dz, Dz)
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    Fe = Fe_net(z_bel["m"], e_bel["m"]).detach().squeeze(0)  # (1, Dz, De)
    dfde = Fe * env.dt

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)
    HzFe = dhdz @ dfde

    z_pred = {
        "m": z_bel["m"]
        + meta_dynamics_fn(z_bel["m"], e_bel["m"]) * env.dt
        + u_t * env.dt,  # (1, Dz)
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2) + Q,
    }

    # -----------------------------------
    # 5) Embedding update (Laplace)
    if env_step > warmup_step:
        with torch.no_grad():
            Prec = e_bel["Prec"]
            eta = Prec @ e_bel["m"].unsqueeze(-1)

            # predictive covariance and Cholesky solve
            S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
            S = symmetrize(S)
            chol_S = torch.linalg.cholesky(S)

            mu_e = e_bel["m"]
            Prec_old = e_bel["Prec"]
            for _ in range(5):
                m_pred = z_bel["m"] + meta_dynamics_fn(z_bel["m"], mu_e) * env.dt + u_t * env.dt
                y_hat_pred = decoder(m_pred).detach()
                r_t = obs - y_hat_pred

                invS_r = torch.cholesky_solve(r_t.transpose(1, 2), chol_S)
                grad_ll = einsum(HzFe, invS_r, "b y e, b y ... -> b e")
                X = torch.cholesky_solve(HzFe, chol_S)  # (1, Dy, De)
                curv_ll = einsum(HzFe, X, "b y d, b y e -> b d e")
                curv_ll = symmetrize(curv_ll)  # ensure symmetry

                Prec_new = Prec_old + curv_ll
                eta_old = Prec_old @ mu_e.unsqueeze(-1)
                eta_new = eta_old + grad_ll.unsqueeze(-1)
                Prec_old = Prec_new

                chol_Prec_new = safe_cholesky(Prec_new)
                Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
                mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

                # Update belief for next refinement
                Prec_old, eta = Prec_new, eta_new

        # EMA update for mu_e
        mu_e = e_bel["m"] * 0.5 + mu_e * 0.5
        mu_e.clamp_(-5.0, 5.0)
        e_bel = {"m": mu_e, "P": Sigma_e, "Prec": Prec_new}
        # e_bel["Prec"] *= 0.9999  # Forgetting
        e_bel["Prec"].clamp_(-1e3, 1e3)
        e_bel = {k: v.detach() for k, v in e_bel.items()}

    # -----------------------------------
    # 7) Optimize likelihood and encoder
    opt.zero_grad(set_to_none=True)

    # Prior Gating
    z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
    T = z_flat.size(-2)
    t_mask = None

    if env_step > warmup_step:
        z_p = (z_flat + meta_dynamics_fn(z_flat, e_bel["m"]) * env.dt)[..., :-1, :]
        z_p += rb["action"][..., 1:, :] * env.dt
        z_p += torch.randn_like(z_p) * (Q.diag()).sqrt()
        z_p = torch.cat([z_flat[..., :1, :], z_p], dim=-2)  # ((S B),T,D)
        mu_p = (mu_q + meta_dynamics_fn(mu_q, e_bel["m"]) * env.dt)[..., :-1, :]
        mu_p += rb["action"][..., 1:, :] * env.dt
        mu_p = torch.cat([mu_q[..., :1, :], mu_p], dim=-2)

    p_mask = 0.5
    if env_step > warmup_step and p_mask > 0:
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=device))
        z_flat = z_flat * t_mask + z_p * (1 - t_mask)

    # Compute log likelihood
    y_rep = repeat(rb["next_obs"], "b t d -> (s b) t d", s=n_samples)
    ll_sb = decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
    ll_b = ll_sb.view(n_samples, -1).mean(dim=0)  # (B,)

    # Compute KL
    kl_b = torch.zeros(1, device=device)
    if env_step > warmup_step:
        kl_d = 0.5 * (
            torch.log(Q.diag() / var_q) + ((mu_q - mu_p) ** 2) / Q.diag() + (var_q / Q.diag()) - 1
        )
        kl_sb = kl_d.sum(dim=-1)  # (S*B, T)
        if t_mask is not None:
            kl_sb = kl_sb * t_mask[1:, :]  # (S*B,T)

        kl_b = kl_sb.view(n_samples, -1).mean(dim=0).sum(-1)  # (B,)

    beta = torch.min(
        torch.tensor(env_step / (warmup_step + 1), device=device), torch.tensor(1.0, device=device)
    )
    elbo = ll_b.mean() - kl_b.mean() * beta
    loss = -elbo
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()

    # Predictive loss
    z_env = rb["next_env_state"][:, -1:]
    z_mod = rb["next_model_state"][:, -1:]
    z_future = env.env.generate_trajectory(z_env, 50)
    y_future = obs_model(z_future)
    z_pred = dynamics.sample_forward(z_mod, e_bel["m"], k_step=50, return_traj=True)[1]
    z_pred = [z_mod] + z_pred
    z_pred = torch.cat(z_pred, -2)
    y_pred = decoder(z_pred)

    ss_res = ((y_future - y_pred) ** 2).squeeze()
    ss_tot = ((y_future - y_future.mean(dim=1)) ** 2).squeeze()

    r2_mean = []
    r2_mean.append((1 - ss_res[:10].sum(dim=0) / (ss_tot[:10].sum(dim=0) + 1e-6)).mean())
    r2_mean.append((1 - ss_res[:25].sum(dim=0) / (ss_tot[:25].sum(dim=0) + 1e-6)).mean())
    r2_mean.append((1 - ss_res[:50].sum(dim=0) / (ss_tot[:50].sum(dim=0) + 1e-6)).mean())

    # Plotting
    if env_step % 50 == 0:
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        plot_vector_field(
            dynamics,
            x_range=5,
            ax=axs,
            is_residual=True,
        )
        data = to_np(plot_rollout["next_env_state"])
        create_gradient_line(axs, data, "royalblue", label="Env Traj")
        data = to_np(z_future)[0]
        axs.plot(data[:, 0], data[:, 1], "--", color="royalblue", lw=1, label="Env Future")

        data = to_np(plot_rollout["next_model_state"])
        create_gradient_line(axs, data, "crimson", label="Model Traj")
        data = to_np(z_pred)[0]
        axs.plot(data[:, 0], data[:, 1], "--", color="crimson", lw=1, label="Model Pred")
        plt.legend(loc="upper right")
        axs.set_title(f"Step {env_step}")
        plt.colorbar(label="Speed", aspect=20)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"video/images/vecfield_{env_step:05d}.png"))
        # Write video
        frames.append(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

    writer.add_scalar("train/ELBO", elbo / windows_length, env_step)
    writer.add_scalar(
        "train/log_like", ll_b.mean().item() / windows_length / observation_dim, env_step
    )
    writer.add_scalar("train/kl_d", kl_b.mean().item() / windows_length / latent_dim, env_step)
    writer.add_scalar("train/r2_10", r2_mean[0], env_step)
    writer.add_scalar("train/r2_25", r2_mean[1], env_step)
    writer.add_scalar("train/r2_50", r2_mean[2], env_step)
    writer.add_scalar("train/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("train/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("train/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("train/e2_true", env.env.dynamics.b.item(), env_step)

    if env_step % 100 == 0:
        pbar.set_postfix({"ELBO": f"{elbo/windows_length:.4f}"})
        pbar.update(100)

    if env_step % 1000 == 0:
        save_load.save_rollout(
            rb,
            os.path.join(results_dir, f"rollouts/rollout_{env_step}.pkl"),
        )

    rb.add(
        **{"next_model_state": z_bel["m"].detach(), "next_env_state": info["latent_state"].detach()}
    )
    plot_rollout.add(
        **{"next_model_state": z_bel["m"].detach(), "next_env_state": info["latent_state"].detach()}
    )
pbar.close()
writer.close()
imageio.mimsave(video_path, frames, fps=5)


# %% 1-2. ? (Debug) Amortized Latent with window + EKF/Laplace (with Embedding)
# Use amortized latent encoder with small trailing window to infer latent posterior
# Use EKF to get predictive latent covariance
# Use Laplace to refine embedding posterior

torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
encoder = actdyn.models.encoder.RNNEmbeddingEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    embedding_dim=embedding_dim,
    hidden_dim=64,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=meta_dynamics_fn, device=device
)
debug_fix_decoder(decoder, obs_model)
policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)
params = list(decoder.parameters()) + list(encoder.parameters()) + list(dynamics.parameters())

debug_fix_decoder(decoder, obs_model)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=1,
    Q=0.01,
    device=device,
)

frames = []
plt.close("all")
z_bel = {
    "m": torch.zeros(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z, z_hat = [], []  # For debugging purpose

total_steps = 10000
pbar = tqdm(range(1, total_steps))
windows_length = 1000
warmup_step = 1000
n_samples = 5
rb = RecentRollout(max_len=1000, device=device)
plot_rollout = RecentRollout(max_len=500, device=device)
rb.add(
    **{
        "obs": obs,
        "next_obs": obs,
        "action": torch.zeros(1, action_dim, device=device),
        "next_env_state": info["latent_state"],
        "next_model_state": z_bel["m"],
    }
)
plot_rollout.add(
    **{
        "next_env_state": info["latent_state"],
        "next_model_state": z_bel["m"],
    }
)
prev_action = torch.zeros(1, action_dim, device=device)

results_dir = os.path.join(base_dir, "debug_ekf_laplace_amortized")
for subdir in ["rollouts", "logs", "model", "video", "video/images"]:
    p = os.path.join(results_dir, subdir)
    # Clean up previous results
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
video_path = os.path.join(results_dir, f"video/vecfield.mp4")


for env_step in pbar:
    # 1) Random action sampling
    u_t = policy(z_bel["m"]).detach()
    # every 1000 steps silent action for 100 steps

    # -----------------------------------
    # 2) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    rb.add(
        **{
            "obs": rb["next_obs"][:, -1, :].unsqueeze(1).detach(),
            "next_obs": obs.detach(),
            "action": u_t.detach(),
        }
    )

    # -----------------------------------
    # 3) Update Latent Posterior
    e_rep = repeat(e_bel["m"], "b d -> b t d", t=len(rb)).to(device)
    if env_step < warmup_step:
        gamma, beta = 1.0, 0.0
    else:
        gamma, beta = None, None
    z_samples, mu_q, var_q = encoder(
        y=rb["next_obs"], u=rb["action"], e=e_rep, n_samples=n_samples, gamma=gamma, beta=beta
    )  # (S, 1, T, Dz), (1, T, Dz), (1, T, Dz)
    R = softplus(decoder.noise.logvar).diag_embed() + eps
    if env_step < warmup_step:
        R = R.detach()
    Q = softplus(dynamics.logvar).diag_embed().squeeze(0) * env.dt + eps

    z_bel = {"m": mu_q[:, -2].detach(), "P": var_q[0, -2].diag_embed().detach()}

    # -----------------------------------
    # 4) Compute Predictive latent distribution
    Fz = Fz_true(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    Fe = Fe_true(z_bel["m"], e_bel["m"])
    dfde = Fe * env.dt

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)
    HzFe = dhdz @ dfde

    sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    f_true = sim_vec_env.compute_dynamics(z_bel["m"]).to(device)  # For debugging

    z_pred = {
        "m": z_bel["m"] + f_true * env.dt + u_t * env.dt,  # (1, Dz)
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2) + Q,
    }

    # -----------------------------------
    # 5) Embedding update (Laplace)
    if env_step > warmup_step:
        with torch.no_grad():
            Prec = e_bel["Prec"]
            eta = Prec @ e_bel["m"].unsqueeze(-1)

            # predictive covariance and Cholesky solve
            S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
            S = symmetrize(S)
            chol_S = torch.linalg.cholesky(S)

            mu_e = e_bel["m"]
            Prec_old = e_bel["Prec"]
            for _ in range(5):
                sim_vec_env.dynamics.set_params([mu_e[0, 0], mu_e[0, 1], 0.1])
                f_true = sim_vec_env.compute_dynamics(z_bel["m"]).to(device)  # For debugging
                m_pred = z_bel["m"] + f_true * env.dt + u_t * env.dt
                y_hat_pred = decoder(m_pred).detach()
                r_t = obs - y_hat_pred

                invS_r = torch.cholesky_solve(r_t.transpose(1, 2), chol_S)
                grad_ll = einsum(HzFe, invS_r, "b y e, b y ... -> b e")
                X = torch.cholesky_solve(HzFe, chol_S)  # (1, Dy, De)
                curv_ll = einsum(HzFe, X, "b y d, b y e -> b d e")
                curv_ll = symmetrize(curv_ll)  # ensure symmetry

                Prec_new = Prec_old + curv_ll
                eta_old = Prec_old @ mu_e.unsqueeze(-1)
                eta_new = eta_old + grad_ll.unsqueeze(-1)
                Prec_old = Prec_new

                chol_Prec_new = safe_cholesky(Prec_new)
                Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
                mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

                # Update belief for next refinement
                Prec_old, eta = Prec_new, eta_new

        # EMA update for mu_e
        mu_e = e_bel["m"] * 0.5 + mu_e * 0.5
        mu_e.clamp_(-5.0, 5.0)
        e_bel = {"m": mu_e, "P": Sigma_e, "Prec": Prec_new}
        # e_bel["Prec"] *= 0.99  # Forgetting
        e_bel["Prec"].clamp_(-1e3, 1e3)
        e_bel = {k: v.detach() for k, v in e_bel.items()}

    # -----------------------------------
    # 7) Optimize likelihood and encoder
    opt.zero_grad(set_to_none=True)

    # Prior Gating
    z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
    T = z_flat.size(-2)
    t_mask = None

    if env_step > warmup_step:
        sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
        f_true = sim_vec_env.compute_dynamics(z_flat).to(device)  # For debugging
        z_p = (z_flat + f_true * env.dt)[..., :-1, :]
        z_p += rb["action"][..., 1:, :] * env.dt
        z_p += torch.randn_like(z_p) * (Q.diag()).sqrt()
        z_p = torch.cat([z_flat[..., :1, :], z_p], dim=-2)  # ((S B),T,D)

        f_true = sim_vec_env.compute_dynamics(mu_q).to(device)  # For debugging
        mu_p = (mu_q + f_true * env.dt)[..., :-1, :]
        mu_p += rb["action"][..., 1:, :] * env.dt
        mu_p = torch.cat([mu_q[..., :1, :], mu_p], dim=-2)

    p_mask = 0
    if env_step > warmup_step and p_mask > 0:
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=device))
        z_flat = z_flat * t_mask + z_p * (1 - t_mask)

    # Compute log likelihood
    y_rep = repeat(rb["next_obs"], "b t d -> (s b) t d", s=n_samples)
    ll_sb = decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
    ll_b = ll_sb.view(n_samples, -1).mean(dim=0)  # (B,)

    # Compute KL
    kl_b = torch.zeros(1, device=device)
    if env_step > warmup_step:
        kl_d = 0.5 * (
            torch.log(Q.diag() / var_q) + ((mu_q - mu_p) ** 2) / Q.diag() + (var_q / Q.diag()) - 1
        )
        kl_sb = kl_d.sum(dim=-1)  # (S*B, T)
        if t_mask is not None:
            kl_sb = kl_sb * t_mask[1:, :]  # (S*B,T)

        kl_b = kl_sb.view(n_samples, -1).mean(dim=0).sum(-1)  # (B,)

    beta = torch.min(
        torch.tensor(env_step / (warmup_step + 1), device=device), torch.tensor(1.0, device=device)
    )
    elbo = ll_b.mean() - kl_b.mean() * beta
    loss = -elbo
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()
    # Predictive loss
    z_env = rb["next_env_state"][:, -1:]
    z_mod = rb["next_model_state"][:, -1:]
    z_future = env.env.generate_trajectory(z_env, 50)
    y_future = obs_model(z_future)
    sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    z_pred = sim_vec_env.generate_trajectory(z_mod, 50)
    y_pred = decoder(z_pred)

    ss_res = ((y_future - y_pred) ** 2).squeeze()
    ss_tot = ((y_future - y_future.mean(dim=1)) ** 2).squeeze()

    r2_mean = []
    r2_mean.append((1 - ss_res[:10].sum(dim=0) / (ss_tot[:10].sum(dim=0) + 1e-6)).mean())
    r2_mean.append((1 - ss_res[:25].sum(dim=0) / (ss_tot[:25].sum(dim=0) + 1e-6)).mean())
    r2_mean.append((1 - ss_res[:50].sum(dim=0) / (ss_tot[:50].sum(dim=0) + 1e-6)).mean())

    # Plotting
    if env_step % 50 == 0:
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        plot_vector_field(
            sim_vec_env.dynamics,
            x_range=5,
            ax=axs,
            is_residual=True,
        )
        data = to_np(plot_rollout["next_env_state"])
        create_gradient_line(axs, data, "royalblue", label="Env Traj")
        data = to_np(z_future)[0]
        axs.plot(data[:, 0], data[:, 1], "--", color="royalblue", lw=1, label="Env Future")

        data = to_np(plot_rollout["next_model_state"])
        create_gradient_line(axs, data, "crimson", label="Model Traj")
        data = to_np(z_pred)[0]
        axs.plot(data[:, 0], data[:, 1], "--", color="crimson", lw=1, label="Model Pred")
        plt.legend(loc="upper right")
        axs.set_title(f"Step {env_step}")
        plt.colorbar(label="Speed", aspect=20)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"video/images/vecfield_{env_step:05d}.png"))
        # Write video
        frames.append(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

    writer.add_scalar("train/ELBO", elbo / windows_length, env_step)
    writer.add_scalar(
        "train/log_like", ll_b.mean().item() / windows_length / observation_dim, env_step
    )
    writer.add_scalar("train/kl_d", kl_b.mean().item() / windows_length / latent_dim, env_step)
    writer.add_scalar("train/r2_10", r2_mean[0], env_step)
    writer.add_scalar("train/r2_25", r2_mean[1], env_step)
    writer.add_scalar("train/r2_50", r2_mean[2], env_step)
    writer.add_scalar("train/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("train/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("train/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("train/e2_true", env.env.dynamics.b.item(), env_step)

    if env_step % 100 == 0:
        pbar.set_postfix({"ELBO": f"{elbo/windows_length:.4f}"})
        pbar.update(100)

    if env_step % 1000 == 0:
        save_load.save_rollout(
            rb,
            os.path.join(results_dir, f"rollouts/rollout_{env_step}.pkl"),
        )

    rb.add(
        **{"next_model_state": z_bel["m"].detach(), "next_env_state": info["latent_state"].detach()}
    )
    plot_rollout.add(
        **{"next_model_state": z_bel["m"].detach(), "next_env_state": info["latent_state"].detach()}
    )
pbar.close()
writer.close()
imageio.mimsave(video_path, frames, fps=5)
# %% 1-3.  DKF (without Embedding)
torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)

encoder = actdyn.models.encoder.RNNEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    hidden_dim=128,
    device=device,
)
action_encoder = actdyn.environment.action.IdentityActionEncoder(
    action_dim=action_dim, latent_dim=latent_dim, action_bounds=[-20.0, 20.0], device=device
)
dynamics = actdyn.models.dynamics.MLPDynamics(
    state_dim=latent_dim, hidden_dims=32, dt=env.dt, device=device, is_residual=True
)
seqVae = actdyn.models.SeqVae(
    encoder=encoder,
    decoder=decoder,
    dynamics=dynamics,
    action_encoder=action_encoder,
    device=device,
)
model_env = actdyn.models.model_wrapper.ModelWrapper(
    model=seqVae,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)
policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)
config_path = os.path.join(os.path.dirname(__file__), "conf/config.yaml")
exp_config = ExperimentConfig.from_yaml(config_path)
# result in actdyn module folder

exp_config.results_dir = os.path.join(base_dir, "dkf")

exp_config.training.total_steps = 100000
exp_config.training.warmup = 10000

experiment, agent, _, _ = setup_experiment(exp_config)
agent.env = env
agent.policy = policy
agent.model = model_env


for subdir in ["rollouts", "logs", "model", "video", "video/images"]:
    p = os.path.join(exp_config.results_dir, subdir)
    # Clean up previous results
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)

writer = SummaryWriter(log_dir=os.path.join(exp_config.results_dir, "logs"))
train_cfg = exp_config.training

agent.reset(seed=int(experiment.cfg.seed))
experiment.env_step = 0
experiment.rollout.clear()

# Setup progress bar

plot_rollout = RecentRollout(max_len=500, device=device)
video_path = os.path.join(experiment.results_path, f"video/vecfield.mp4")
frames = []
pbar = tqdm(total=train_cfg.total_steps, desc="Training")
while experiment.env_step < train_cfg.total_steps:
    experiment.env_step += 1

    with torch.no_grad():
        # 1. Plan
        action = agent.plan()
        # 2. Execute
        transition, done = agent.step(action)

    # Append transition to rollout
    experiment.rollout.add(**transition)
    plot_rollout.add(**transition)

    # Predictive loss
    z_env = experiment.rollout["next_env_state"][:, -1:]
    z_mod = experiment.rollout["next_model_state"][:, -1:]
    z_future = agent.env.env.generate_trajectory(z_env, 50)
    y_future = agent.env.obs_model(z_future)
    z_pred = agent.model.model.dynamics.sample_forward(z_mod, k_step=50, return_traj=True)[1]
    z_pred = [z_mod] + z_pred
    z_pred = torch.cat(z_pred, -2)
    y_pred = agent.model.model.decoder(z_pred)

    ss_res = ((y_future - y_pred) ** 2).squeeze()
    ss_tot = ((y_future - y_future.mean(dim=1)) ** 2).squeeze()

    r2_mean = []
    r2_mean.append((1 - ss_res[:10].sum(dim=0) / (ss_tot[:10].sum(dim=0) + 1e-6)).mean())
    r2_mean.append((1 - ss_res[:25].sum(dim=0) / (ss_tot[:25].sum(dim=0) + 1e-6)).mean())
    r2_mean.append((1 - ss_res[:50].sum(dim=0) / (ss_tot[:50].sum(dim=0) + 1e-6)).mean())

    # Plotting
    if experiment.env_step % 50 == 0:
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        plot_vector_field(
            agent.model.model.dynamics,
            x_range=5,
            ax=axs,
            is_residual=True,
        )
        data = to_np(plot_rollout["next_env_state"])
        create_gradient_line(axs, data, "royalblue", label="Env Traj")
        data = to_np(z_future)[0]
        axs.plot(data[:, 0], data[:, 1], "--", color="royalblue", lw=1, label="Env Future")

        data = to_np(plot_rollout["next_model_state"])
        create_gradient_line(axs, data, "crimson", label="Model Traj")
        data = to_np(z_pred)[0]
        axs.plot(data[:, 0], data[:, 1], "--", color="crimson", lw=1, label="Model Pred")
        plt.legend(loc="upper right")
        axs.set_title(f"Step {experiment.env_step}")
        plt.colorbar(label="Speed", aspect=20)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                experiment.results_path, f"video/images/vecfield_{experiment.env_step:05d}.png"
            )
        )
        # Write video

        frames.append(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

    if isinstance(experiment.training_loss, list):
        writer.add_scalar("train/ELBO", -experiment.training_loss[0][0], experiment.env_step)
        writer.add_scalar("train/log_like", experiment.training_loss[0][1], experiment.env_step)
        writer.add_scalar("train/kl_d", experiment.training_loss[0][2], experiment.env_step)
        writer.add_scalar("train/r2_10", r2_mean[0], experiment.env_step)
        writer.add_scalar("train/r2_25", r2_mean[1], experiment.env_step)
        writer.add_scalar("train/r2_50", r2_mean[1], experiment.env_step)
    else:
        writer.add_scalar("train/ELBO", 0, -experiment.env_step)
        writer.add_scalar("train/log_like", 0, experiment.env_step)
        writer.add_scalar("train/kl_d", 0, experiment.env_step)
        writer.add_scalar("train/r2_10", 0, experiment.env_step)
        writer.add_scalar("train/r2_25", 0, experiment.env_step)
        writer.add_scalar("train/r2_50", 0, experiment.env_step)

    agent.update_policy(transition)

    if experiment.env_step % 100 == 0:
        if isinstance(experiment.training_loss, list) and len(experiment.training_loss) > 0:
            elbo_loss = -experiment.training_loss[0][0]
            pbar.set_postfix({"ELBO": f"{elbo_loss:.4f}, beta: {agent.model.model.beta:.4f}"})
        else:
            pbar.set_postfix({"ELBO": "N/A"})
        pbar.update(100)

    # Train model periodically
    if experiment.env_step > train_cfg.rollout_horizon:
        sampling_ratio = agent.model.model.dynamics.dt / agent.env.dt
        experiment.training_loss = agent.train_model(
            **train_cfg.get_optim_cfg(), sampling_ratio=sampling_ratio
        )

    # Periodic rollout saving for crash recovery and memory management
    if experiment.env_step % experiment.cfg.logging.save_every == 0:
        save_load.save_rollout(
            experiment.rollout,
            os.path.join(experiment.results_path, f"rollouts/rollout_{experiment.env_step}.pkl"),
        )
        if experiment.env_step < train_cfg.total_steps:
            experiment.rollout.clear()

    # Clean up tensors to prevent memory accumulation
    if "cuda" in str(experiment.agent.device):
        del transition, action
        torch.cuda.empty_cache()

    if done:
        break
pbar.close()
experiment.rollout.finalize()
experiment.agent.model.save_model(os.path.join(experiment.results_path, f"model/model_final.pth"))
imageio.mimsave(video_path, frames, fps=5)
writer.close()


# %% 1-3.  (Debug) DKF (without Embedding)
torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)

encoder = actdyn.models.encoder.RNNEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    hidden_dim=128,
    device=device,
)
action_encoder = actdyn.environment.action.IdentityActionEncoder(
    action_dim=action_dim, latent_dim=latent_dim, action_bounds=[-2.0, 2.0], device=device
)
dynamics = actdyn.models.dynamics.MLPDynamics(
    state_dim=latent_dim, hidden_dims=32, dt=env.dt, device=device, is_residual=True
)
seqVae = actdyn.models.SeqVae(
    encoder=encoder,
    decoder=decoder,
    dynamics=dynamics,
    action_encoder=action_encoder,
    device=device,
)
model_env = actdyn.models.model_wrapper.ModelWrapper(
    model=seqVae,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)
policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)
config_path = os.path.join(os.path.dirname(__file__), "conf/config.yaml")
exp_config = ExperimentConfig.from_yaml(config_path)
exp_config.results_dir = os.path.join(os.path.dirname(__file__), "results", "active_embedding")


experiment, _, _, _ = setup_experiment(exp_config)
experiment.agent.env = env
experiment.agent.policy.action_space = env.action_space
experiment.agent.model = model_env
experiment.agent.policy = policy


experiment.run()


# %% 1-4. (TODO) VJF like filtering (post mean/cov as input) + Laplace embedding Inference

# %% 1-5. (TODO) DVBF (embedding as variational parameter)
# %% 1-5. (TODO) Amortized Gain + Laplace embedding Inference
# %% 2-1. Train with active learning (myopic)
# %% 2-2. Active Planning Amortized Latent with window + EKF/Laplace (with Embedding)
# Use amortized latent encoder with small trailing window to infer latent posterior
# Use EKF to get predictive latent covariance
# Use Laplace to refine embedding posterior

torch.manual_seed(1)
mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
encoder = actdyn.models.encoder.RNNEmbeddingEncoder(
    obs_dim=observation_dim,
    action_dim=action_dim,
    latent_dim=latent_dim,
    embedding_dim=embedding_dim,
    hidden_dim=64,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=meta_dynamics_fn, device=device
)
params = list(decoder.parameters()) + list(encoder.parameters()) + list(dynamics.parameters())
emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=Fe_net, Fz_net=Fz_net, decoder=decoder
)
model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)

mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=20,
    num_iterations=5,
    num_samples=16,
    num_elite=8,
    verbose=False,
)


# debug_fix_decoder(decoder, obs_model)
frames = []
plt.close("all")
z_bel = {
    "m": torch.zeros(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z, z_hat = [], []  # For debugging purpose

total_steps = 100000
pbar = tqdm(range(1, total_steps))
windows_length = 100
warmup_step = 1000
n_samples = 5
rb = RecentRollout(max_len=1000, device=device)
plot_rollout = RecentRollout(max_len=500, device=device)
rb.add(
    **{
        "obs": obs,
        "next_obs": obs,
        "action": torch.zeros(1, action_dim, device=device),
        "next_env_state": info["latent_state"],
        "next_model_state": z_bel["m"],
    }
)
plot_rollout.add(
    **{
        "next_env_state": info["latent_state"],
        "next_model_state": z_bel["m"],
    }
)
prev_action = torch.zeros(1, action_dim, device=device)

results_dir = os.path.join(base_dir, "n_active_planning_ekf_laplace_amortized")
for subdir in ["rollouts", "logs", "model", "video", "video/images"]:
    p = os.path.join(results_dir, subdir)
    # Clean up previous results
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))
video_path = os.path.join(results_dir, f"video/vecfield.mp4")


for env_step in pbar:
    Q = softplus(dynamics.logvar).diag_embed().squeeze(0) * env.dt
    # 1) Random action sampling
    u_t = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q).detach()
    # -----------------------------------
    # 2) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    rb.add(
        **{
            "obs": rb["next_obs"][:, -1, :].unsqueeze(1).detach(),
            "next_obs": obs.detach(),
            "action": u_t.detach(),
        }
    )

    # -----------------------------------
    # 3) Update Latent Posterior
    e_rep = repeat(e_bel["m"], "b d -> b t d", t=len(rb)).to(device)
    if env_step < warmup_step:
        gamma, beta = 1.0, 0.0
    else:
        gamma, beta = None, None
    z_samples, mu_q, var_q = encoder(
        y=rb["next_obs"], u=rb["action"], e=e_rep, n_samples=n_samples, gamma=gamma, beta=beta
    )  # (S, 1, T, Dz), (1, T, Dz), (1, T, Dz)
    R = softplus(decoder.noise.logvar).diag_embed() + eps
    if env_step < warmup_step:
        R = R.detach()

    z_bel = {"m": mu_q[:, -2].detach(), "P": var_q[0, -2].diag_embed().detach()}

    # -----------------------------------
    # 4) Compute Predictive latent distribution
    Fz = Fz_net(z_bel["m"], e_bel["m"]).detach().squeeze(0)  # (1, Dz, Dz)
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    Fe = Fe_net(z_bel["m"], e_bel["m"]).detach().squeeze(0)  # (1, Dz, De)
    dfde = Fe * env.dt

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)
    HzFe = dhdz @ dfde

    z_pred = {
        "m": z_bel["m"]
        + meta_dynamics_fn(z_bel["m"], e_bel["m"]) * env.dt
        + u_t * env.dt,  # (1, Dz)
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2) + Q,
    }

    # -----------------------------------
    # 5) Embedding update (Laplace)
    if env_step > warmup_step:
        with torch.no_grad():
            Prec = e_bel["Prec"]
            eta = Prec @ e_bel["m"].unsqueeze(-1)

            # predictive covariance and Cholesky solve
            S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
            S = symmetrize(S)
            chol_S = torch.linalg.cholesky(S)

            mu_e = e_bel["m"]
            Prec_old = e_bel["Prec"]
            for _ in range(5):
                m_pred = z_bel["m"] + meta_dynamics_fn(z_bel["m"], mu_e) * env.dt + u_t * env.dt
                y_hat_pred = decoder(m_pred).detach()
                r_t = obs - y_hat_pred

                invS_r = torch.cholesky_solve(r_t.transpose(1, 2), chol_S)
                grad_ll = einsum(HzFe, invS_r, "b y e, b y ... -> b e")
                X = torch.cholesky_solve(HzFe, chol_S)  # (1, Dy, De)
                curv_ll = einsum(HzFe, X, "b y d, b y e -> b d e")
                curv_ll = symmetrize(curv_ll)  # ensure symmetry

                Prec_new = Prec_old + curv_ll
                eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
                eta_new = eta_old + grad_ll.unsqueeze(-1)
                Prec_old = Prec_new

                chol_Prec_new = safe_cholesky(Prec_new)
                Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
                mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

                # Update belief for next refinement
                Prec_old, eta = Prec_new, eta_new

        # EMA update for mu_e
        mu_e = e_bel["m"] * 0.5 + mu_e * 0.5
        mu_e.clamp_(-5.0, 5.0)
        e_bel = {"m": mu_e, "P": Sigma_e, "Prec": Prec_new}
        # e_bel["Prec"] *= 0.9999  # Forgetting
        e_bel["Prec"].clamp_(-1e3, 1e3)
        e_bel = {k: v.detach() for k, v in e_bel.items()}

    # -----------------------------------
    # 7) Optimize likelihood and encoder
    opt.zero_grad(set_to_none=True)

    # Prior Gating
    z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
    T = z_flat.size(-2)
    t_mask = None

    if env_step > warmup_step:
        z_p = (z_flat + meta_dynamics_fn(z_flat, e_bel["m"]) * env.dt)[..., :-1, :]
        z_p += rb["action"][..., 1:, :] * env.dt
        z_p += torch.randn_like(z_p) * (Q.diag()).sqrt()
        z_p = torch.cat([z_flat[..., :1, :], z_p], dim=-2)  # ((S B),T,D)
        mu_p = (mu_q + meta_dynamics_fn(mu_q, e_bel["m"]) * env.dt)[..., :-1, :]
        mu_p += rb["action"][..., 1:, :] * env.dt
        mu_p = torch.cat([mu_q[..., :1, :], mu_p], dim=-2)

    p_mask = 0.5
    if env_step > warmup_step and p_mask > 0:
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=device))
        z_flat = z_flat * t_mask + z_p * (1 - t_mask)

    # Compute log likelihood
    y_rep = repeat(rb["next_obs"], "b t d -> (s b) t d", s=n_samples)
    ll_sb = decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
    ll_b = ll_sb.view(n_samples, -1).mean(dim=0)  # (B,)

    # Compute KL
    kl_b = torch.zeros(1, device=device)
    if env_step > warmup_step:
        kl_d = 0.5 * (
            torch.log(Q.diag() / var_q) + ((mu_q - mu_p) ** 2) / Q.diag() + (var_q / Q.diag()) - 1
        )
        kl_sb = kl_d.sum(dim=-1)  # (S*B, T)
        if t_mask is not None:
            kl_sb = kl_sb * t_mask[1:, :]  # (S*B,T)

        kl_b = kl_sb.view(n_samples, -1).mean(dim=0).sum(-1)  # (B,)

    beta = torch.min(
        torch.tensor(env_step / (warmup_step + 1), device=device), torch.tensor(1.0, device=device)
    )
    elbo = ll_b.mean() - kl_b.mean() * beta
    loss = -elbo
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()

    # Predictive loss
    z_env = rb["next_env_state"][:, -1:]
    z_mod = rb["next_model_state"][:, -1:]
    z_future = env.env.generate_trajectory(z_env, 50)
    y_future = obs_model(z_future)
    z_pred = dynamics.sample_forward(z_mod, e_bel["m"], k_step=50, return_traj=True)[1]
    z_pred = [z_mod] + z_pred
    z_pred = torch.cat(z_pred, -2)
    y_pred = decoder(z_pred)

    ss_res = ((y_future - y_pred) ** 2).squeeze()
    ss_tot = ((y_future - y_future.mean(dim=1)) ** 2).squeeze()

    r2_mean = []
    r2_mean.append((1 - ss_res[:10].sum(dim=0) / (ss_tot[:10].sum(dim=0) + 1e-6)).mean())
    r2_mean.append((1 - ss_res[:25].sum(dim=0) / (ss_tot[:25].sum(dim=0) + 1e-6)).mean())
    r2_mean.append((1 - ss_res[:50].sum(dim=0) / (ss_tot[:50].sum(dim=0) + 1e-6)).mean())

    # Plotting
    if env_step % 50 == 0:
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        plot_vector_field(
            dynamics,
            x_range=5,
            ax=axs,
            is_residual=True,
        )
        data = to_np(plot_rollout["next_env_state"])
        create_gradient_line(axs, data, "royalblue", label="Env Traj")
        data = to_np(z_future)[0]
        axs.plot(data[:, 0], data[:, 1], "--", color="royalblue", lw=1, label="Env Future")

        data = to_np(plot_rollout["next_model_state"])
        create_gradient_line(axs, data, "crimson", label="Model Traj")
        data = to_np(z_pred)[0]
        axs.plot(data[:, 0], data[:, 1], "--", color="crimson", lw=1, label="Model Pred")
        plt.legend(loc="upper right")
        axs.set_title(f"Step {env_step}")
        plt.colorbar(label="Speed", aspect=20)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"video/images/vecfield_{env_step:05d}.png"))
        # Write video
        frames.append(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

    writer.add_scalar("train/ELBO", elbo / windows_length, env_step)
    writer.add_scalar(
        "train/log_like", ll_b.mean().item() / windows_length / observation_dim, env_step
    )
    writer.add_scalar("train/kl_d", kl_b.mean().item() / windows_length / latent_dim, env_step)
    writer.add_scalar("train/r2_10", r2_mean[0], env_step)
    writer.add_scalar("train/r2_25", r2_mean[1], env_step)
    writer.add_scalar("train/r2_50", r2_mean[2], env_step)
    writer.add_scalar("train/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("train/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("train/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("train/e2_true", env.env.dynamics.b.item(), env_step)

    if env_step % 100 == 0:
        pbar.set_postfix({"ELBO": f"{elbo / windows_length:.4f}"})
        pbar.update(100)

    if env_step % 1000 == 0:
        save_load.save_rollout(
            rb,
            os.path.join(results_dir, f"rollouts/rollout_{env_step}.pkl"),
        )

    rb.add(
        **{"next_model_state": z_bel["m"].detach(), "next_env_state": info["latent_state"].detach()}
    )
    plot_rollout.add(
        **{"next_model_state": z_bel["m"].detach(), "next_env_state": info["latent_state"].detach()}
    )
pbar.close()
writer.close()
imageio.mimsave(video_path, frames, fps=5)


# %% EKF Test with Experiment Config
meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)
latent_dim = 2
embedding_dim = 2
action_dim = 2
observation_dim = 50
torch.manual_seed(1)
e = e_sampler(1)
a, b = e.squeeze(0)

action_model = actdyn.environment.action.IdentityActionEncoder(
    action_dim=action_dim, latent_dim=latent_dim, action_bounds=[-5.0, 5.0], device=device
)
obs_model = actdyn.environment.observation.LinearObservation(
    obs_dim=observation_dim,
    latent_dim=latent_dim,
    noise_scale=0.1,
    noise_type="gaussian",
    device=device,
)
duffing_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([a, b, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    action_bounds=[action_model.action_space.low, action_model.action_space.high],
    device=device,
)
env = actdyn.environment.EnvWrapper(duffing_env, obs_model, action_model, dt=0.01, device=device)

mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)

emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    Fe_net=Fe_true, Fz_net=Fz_true, decoder=decoder
)


sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    Q=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=latent_dim, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
)

model = actdyn.models.BaseModel(
    action_encoder=action_model,
    dynamics=dynamics,
    device=device,
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=20,
    num_iterations=20,
    num_samples=20,
    num_elite=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=100, device=device)
random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)


model = actdyn.models.FilteringEmbedding(dynamics=dynamics, decoder=decoder, device=device)
model_env = actdyn.models.model_wrapper.ModelWrapper(
    model=model,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)
agent = actdyn.Agent(
    env=env,
    model=model_env,
    policy=mpc_policy,
    device=device,
)

experiment = actdyn.core.experiment.MetaLearningExperiment(
    agent=agent,
    config=exp_config,
)

meta_dynamics.set_params(e_bel["m"])
params = list(decoder.parameters())
debug_fix_decoder(decoder, obs_model)


warmup_step = 0

plt.close("all")
z_bel = {
    "m": torch.ones(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.ones(1, embedding_dim, device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
obs, info = env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "EKF_fixed_active_chunck")
ro_path = os.path.join(results_dir, "rollout.pkl")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))

ro = Rollout()
pbar = tqdm(range(1000))
for env_step in pbar:
    e_norm.append(torch.norm(e_bel["m"].cpu() - e).numpy())
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]
    # Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    Q = 1e-2 * torch.eye(latent_dim, device=device).unsqueeze(0)

    # 1) Random action sampling
    meta_dynamics.set_params(e_bel["m"])
    sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

    if exp_id == "random":
        u_t = random_policy(z_bel["m"].unsqueeze(0)).detach()
    elif exp_id == "step":
        u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()
    elif exp_id == "active (k=5)":
        u_t, _, _ = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
    elif exp_id == "active (k=20)":
        u_t, _, _ = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
    elif exp_id == "active chunk(k=20)":
        if env_step % 5 == 0:
            u_t, u_ts, cost = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q)
        else:
            u_t = u_ts[:, env_step % 5].unsqueeze(1).detach()

    # 2-1) Predict latent
    dfde = Fe_true(z_bel["m"], e_bel["m"]) * env.dt
    Fz = Fz_true(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    f_true = sim_vec_env.compute_dynamics(z_bel["m"]).to(device)  # For debug

    z_pred = {
        "m": z_bel["m"] + f_true * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2)
        + 1e-4 * torch.eye(latent_dim, device=device).unsqueeze(0),
    }

    # 2-2) Predict observation
    y_pred = decoder(z_pred["m"])
    R = softplus(decoder.noise.logvar).diag_embed() + eps

    # 3) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  # (1, Do)
    r = y_true - y_pred

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)

    # 4) Embedding update (Laplace)
    S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
    S = symmetrize(S)

    chol_S = torch.linalg.cholesky(S)
    X = torch.cholesky_solve(HzFe, chol_S)
    curv_ll = einsum(HzFe, X, "b t y d, b t y e->b t d e")  # (1, De, De)
    curv_ll = symmetrize(curv_ll)  # ensure symmetry
    if env_step > warmup_step:
        # predictive covariance and Cholesky solve (as fixed earlier)
        Prec = e_bel["Prec"]
        eta = Prec @ e_bel["m"].unsqueeze(-1)
        for _ in range(10):
            y_hat = decoder(z_pred["m"])
            r_t = y_true - y_hat

            invS_r = torch.cholesky_solve(r_t.mT, chol_S)
            grad_ll = einsum(HzFe, invS_r, "b t y d, b t y k->b t d")  # (1, De)

            Prec_old = e_bel["Prec"]
            Prec_new = Prec_old + curv_ll
            eta_old = Prec_old @ e_bel["m"].unsqueeze(-1)
            eta_new = eta_old + grad_ll.unsqueeze(-1)

            chol_Prec_new = safe_cholesky(Prec_new)
            Sigma_e = torch.cholesky_inverse(chol_Prec_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            # Update belief for next refinement
            e_bel = {
                "m": mu_e.squeeze(0),
                "P": Sigma_e.squeeze(0),
                "Prec": Prec_new.squeeze(0),
            }
            Prec, eta = Prec_new, eta_new

    # Detach after all refinements
    e_bel = {k: v.detach() for k, v in e_bel.items()}

    # 5) EKF Update Posterior
    # z_post = encoder(r=r, H=dhdz, R=R, z_pred=z_pred, e_mu=e_bel["m"])
    K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
    I = torch.eye(latent_dim, device=device).unsqueeze(0)
    KH = K @ dhdz

    P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    z_post = {
        "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
        "P": symmetrize(P_upd),
    }

    # 6) Roll updated z posterior as new prior
    z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

    # 7) Optimize Likelihood
    opt.zero_grad(set_to_none=True)

    # Single-sample NLL
    ll = decoder.compute_log_prob(z_bel["m"], y_true)
    loss = -ll
    loss.backward()

    # torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 5.0)
    opt.step()
    writer.add_scalar("embedding/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("embedding/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("embedding/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("embedding/e2_true", env.env.dynamics.b.item(), env_step)

    transition = {
        "obs": current_obs,  # Observation  y_t
        "next_obs": obs,  # New Observation y_{t+1}
        "action": u_t,  # Action a_t
        "env_state": current_state,  # Environment state z_t
        "next_env_state": info["latent_state"],  # Next environment state z_{t+1}
        "model_state": current_model_state,  # Current belief state z'_t
        "next_model_state": z_bel["m"],  # Next belief state z'_{t+1}
    }
    ro.add(**transition)
    current_obs = obs
    current_state = info["latent_state"]
    current_model_state = z_bel["m"]

    if env_step % 100 == 0:
        pbar.set_postfix(
            LL=f"{ll.item():.3f}",
            e_hat=f"({e_bel['m'][..., 0].item():.2f},{e_bel['m'][..., 1].item():.2f})",
            e_true=f"({a:.2f},{b:.2f})",
        )
        pbar.update(100)
writer.close()
ro_path = os.path.join(results_dir, "rollout.pkl")
save_load.save_rollout(ro, ro_path)
e_norm = np.array(e_norm)
e_dict[exp_id].append(e_norm)
