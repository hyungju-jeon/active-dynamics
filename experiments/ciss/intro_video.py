# %%

import os
import shutil
from collections import deque
from functools import partial
from turtle import color
from typing import Callable, Sequence
from unittest import result

from matplotlib.pylab import True_
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from sympy import EX, E
from torch.nn.functional import softplus
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import pickle

# from external.integrative_inference.src.utils import save_model, load_model
import actdyn
import actdyn.core
import actdyn.core.experiment
import actdyn.environment
import actdyn.environment.action
import actdyn.environment.observation
import actdyn.environment.vectorfield
import actdyn.metrics
import actdyn.metrics.cost
import actdyn.metrics.uncertainty
import actdyn.models
import actdyn.models.dynamics
import actdyn.models.encoder
import actdyn.policy
import actdyn.policy.mpc
import external.integrative_inference.src.modules as metadyn
from actdyn.config import ExperimentConfig
from actdyn.utils import save_load
from actdyn.utils.experiment_helpers import setup_environment, setup_experiment
from actdyn.utils.helper import *
from actdyn.utils.rollout import RecentRollout, Rollout, RolloutBuffer
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

    def set_params(self, *args):
        self.e = torch.tensor(args, device=device, dtype=torch.float32).unsqueeze(0)
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


z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-3.0, -2.0], [-0.1, 2.0], 2)
ds = zeDataset(100000, z_sampler, e_sampler, device)

# %%
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


# %% EKF Test with Experiment Config
plt.rcParams.update(
    {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#FFFFFF",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.labelcolor": "#000000",
        "text.color": "#000000",
        "xtick.color": "#888888",
        "ytick.color": "#888888",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    }
)

base_dir = os.path.join(os.path.dirname(__file__), "../../results", "CISS")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

dz, de, du, dy = 2, 2, 2, 50
dt = 0.01
alpha = 10
total_t = 1000
action_strength = 0.5
noise_scale = 0.1

torch.manual_seed(7)
e = e_sampler(1)
a, b = e.reshape(-1)
# ------------------------------------------------------------------------------
# Action Model
# ------------------------------------------------------------------------------
action_model = actdyn.environment.action.IdentityActionEncoder(
    action_dim=du,
    latent_dim=dz,
    action_bounds=[-action_strength * alpha, action_strength * alpha],
    device=device,
)
# ------------------------------------------------------------------------------
# Observation Model
# ------------------------------------------------------------------------------
obs_model = actdyn.environment.observation.IdentityObservation(
    obs_dim=dy, latent_dim=dz, device=device
)
obs_model = actdyn.environment.observation.LinearObservation(
    obs_dim=dy,
    latent_dim=dz,
    noise_scale=noise_scale,
    noise_type="gaussian",
    device=device,
)
obs_model = actdyn.environment.observation.LogLinearObservation(
    obs_dim=dy,
    latent_dim=dz,
    noise_scale=0.1,
    noise_type="poisson",
    dt=dt,
    device=device,
)

C = obs_model.network[0].weight.detach()
C[:, 0] = torch.abs(C[:, 0])
# C[:, 0] = C[:, 0] * 3
# C[:, 1] = torch.abs(C[:, 1])
C[:, 1] = C[:, 1] * 2
# C = C / torch.norm(C, dim=1, keepdim=True)  # Normalize rows of C
C *= 1
mean_firing = 25
bias = torch.log(mean_firing * torch.ones(dy, device=device)) - 1 / 2 * torch.diag(C @ C.T)

obs_model.network[0].bias = nn.Parameter(bias)
obs_model.network[0].weight = nn.Parameter(C)


# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
duffing_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([a, b, 0.1]),
    dt=dt,
    alpha=alpha,
    Q=noise_scale,
    action_bounds=[action_model.action_space.low, action_model.action_space.high],
    device=device,
)
env = actdyn.environment.EnvWrapper(duffing_env, obs_model, action_model, dt=dt, device=device)
# ------------------------------------------------------------------------------
# Decoder with Gaussian Noise
# ------------------------------------------------------------------------------
mapping = actdyn.models.decoder.LinearMapping(latent_dim=dz, obs_dim=dy, device=device)
mapping = actdyn.models.decoder.LogLinearMapping(latent_dim=dz, obs_dim=dy, dt=dt, device=device)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=dy, sigma=0.01, device=device)
noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
# ------------------------------------------------------------------------------
# Model Components - Dynamics and model
# ------------------------------------------------------------------------------
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([0, 0, 0.1]),
    dt=dt,
    alpha=alpha,
    Q=noise_scale,
    device=device,
)

dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=dz, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
)
# dynamics = actdyn.models.dynamics.FunctionDynamics(
#     state_dim=dz, dt=env.dt, dynamics_fn=meta_dynamics, device=device
# )
dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz) * noise_scale).to(device))

sigma_0 = 1e-2
e_bel = {
    "m": torch.ones(1, de, device=device),
    "P": sigma_0 * torch.eye(de, device=device).unsqueeze(0),
    "L": 1 / sigma_0 * torch.eye(de, device=device).unsqueeze(0),
}


model = actdyn.models.FilteringEmbedding(
    dynamics=dynamics,
    decoder=decoder,
    e=e_bel,
    action_encoder=action_model,
    Fe=Fe_true,
    Fz=Fz_true,
    device=device,
)
model.set_params(e_bel["m"])
# model.set_params(e)

# ------------------------------------------------------------------------------
# Model Components - Policy
# ------------------------------------------------------------------------------
emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    model=model, Fe_net=Fe_true, Fz_net=Fz_true
)
rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)
action_metric = actdyn.metrics.cost.ActionCost()
composite_metric = actdyn.metrics.CompositeMetric(
    [rnd_metric, emb_metric, action_metric], weights=[0.0, 1.0, 0.00]
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=composite_metric,
    model=model,
    device=device,
    horizon=50,
    num_iterations=20,
    num_samples=40,
    num_elite=10,
    chunk=10,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=50, device=device)
random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
off_policy = actdyn.policy.OffPolicy(action_space=env.action_space, device=device)

# ------------------------------------------------------------------------------
# Model Components - Agent and Experiment
# ------------------------------------------------------------------------------
# agent = actdyn.AsyncAgent(env=env, model=model, policy=mpc_policy, device=device, buffer_length=10)
agent = actdyn.Agent(env=env, model=model, buffer_length=10, policy=mpc_policy, device=device)

exp_config = ExperimentConfig.from_yaml(
    os.path.join(os.path.dirname(__file__), "conf/intro_video.yaml")
)
exp_config.results_dir = base_dir
exp_config.training.total_steps = total_t
experiment = actdyn.core.experiment.MetaEmbeddingExperiment(
    agent=agent,
    config=exp_config,
)

decoder.set_params(obs_model)
experiment.run()

print(f"True embedding: {e}, Learned embedding: {model.e['m']}")


# %% Run Experiment
plt.rcParams.update(
    {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#FFFFFF",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.labelcolor": "#000000",
        "text.color": "#000000",
        "xtick.color": "#888888",
        "ytick.color": "#888888",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    }
)

meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)
e_dict = {
    "active_long": [],
    "active_short": [],
    "RND": [],
    "random": [],
    "step": [],
    "off_policy": [],
}
exp_name = ["active_short", "step", "active_long", "RND", "random", "off_policy"]
for exp_id in exp_name:
    base_dir = os.path.join(os.path.dirname(__file__), "../../results", "CISS", exp_id)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    torch.manual_seed(7)
    for i in range(1):
        dz, de, du, dy = 2, 2, 2, 100
        dt = 0.01
        alpha = 10
        total_t = 1000
        action_strength = 0.5
        noise_scale = 0.11

        e = e_sampler(1)
        a, b = e.reshape(-1)
        e_norm = []

        # ------------------------------------------------------------------------------
        # Action Model
        # ------------------------------------------------------------------------------
        action_model = actdyn.environment.action.IdentityActionEncoder(
            action_dim=du,
            latent_dim=dz,
            action_bounds=[-action_strength * alpha, action_strength * alpha],
            device=device,
        )

        # ------------------------------------------------------------------------------
        # Observation Model
        # ------------------------------------------------------------------------------
        obs_model = actdyn.environment.observation.IdentityObservation(
            obs_dim=dy, latent_dim=dz, device=device
        )
        obs_model = actdyn.environment.observation.LogLinearObservation(
            obs_dim=dy,
            latent_dim=dz,
            noise_scale=0.1,
            noise_type="poisson",
            dt=dt,
            device=device,
        )

        C = obs_model.network[0].weight.detach()
        C[:, 0] = torch.abs(C[:, 0])
        # C[:, 0] = C[:, 0] * 3
        # C[:, 1] = torch.abs(C[:, 1])
        C[:, 1] = C[:, 1] * 2
        # C = C / torch.norm(C, dim=1, keepdim=True)  # Normalize rows of C
        C *= 1
        mean_firing = 10
        bias = torch.log(mean_firing * torch.ones(dy, device=device)) - 1 / 2 * torch.diag(C @ C.T)

        obs_model.network[0].bias = nn.Parameter(bias)
        obs_model.network[0].weight = nn.Parameter(C)

        # ------------------------------------------------------------------------------
        # Environment
        # ------------------------------------------------------------------------------
        duffing_env = actdyn.VectorFieldEnv(
            "duffing",
            x_range=5,
            dyn_params=torch.tensor([a, b, 0.1]),
            dt=dt,
            alpha=alpha,
            Q=noise_scale,
            action_bounds=[action_model.action_space.low, action_model.action_space.high],
            device=device,
        )
        env = actdyn.environment.EnvWrapper(
            duffing_env, obs_model, action_model, dt=dt, device=device
        )

        # ------------------------------------------------------------------------------
        # Decoder with Gaussian Noise
        # ------------------------------------------------------------------------------
        mapping = actdyn.models.decoder.LinearMapping(latent_dim=dz, obs_dim=dy, device=device)
        noise = actdyn.models.decoder.GaussianNoise(obs_dim=dy, sigma=0.01, device=device)
        mapping = actdyn.models.decoder.LogLinearMapping(
            latent_dim=dz, obs_dim=dy, dt=dt, device=device
        )
        noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
        decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
        # ------------------------------------------------------------------------------
        # Model Components - Dynamics and model
        # ------------------------------------------------------------------------------
        sim_vec_env = actdyn.VectorFieldEnv(
            "duffing",
            x_range=5,
            dyn_params=torch.tensor([0, 0, 0.1]),
            dt=dt,
            alpha=alpha,
            Q=noise_scale,
            device=device,
        )

        dynamics = actdyn.models.dynamics.FunctionDynamics(
            state_dim=dz, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
        )
        dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz) * noise_scale).to(device))

        sigma_0 = 1e-2
        e_bel = {
            "m": torch.ones(1, de, device=device),
            "P": sigma_0 * torch.eye(de, device=device).unsqueeze(0),
            "L": 1 / sigma_0 * torch.eye(de, device=device).unsqueeze(0),
        }

        model = actdyn.models.FilteringEmbedding(
            dynamics=dynamics,
            decoder=decoder,
            e=e_bel,
            action_encoder=action_model,
            Fe=Fe_true,
            Fz=Fz_true,
            device=device,
        )
        model.set_params(e_bel["m"])

        # ------------------------------------------------------------------------------
        # Model Components - Policy
        # ------------------------------------------------------------------------------
        emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
            model=model, Fe_net=Fe_true, Fz_net=Fz_true
        )
        rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)
        action_metric = actdyn.metrics.cost.ActionCost()
        composite_metric = actdyn.metrics.CompositeMetric(
            [rnd_metric, emb_metric, action_metric], weights=[2.0, 1.0, 0.005]
        )
        mpc_policy = actdyn.policy.mpc.MpcICem(
            metric=rnd_metric if exp_id == "RND" else emb_metric,
            model=model,
            device=device,
            horizon=10 if exp_id == "active_long" else 3,
            num_iterations=10,
            num_samples=40,
            num_elite=10,
            chunk=5 if exp_id == "active_long" else 3,
            verbose=False,
        )
        step_policy = actdyn.policy.StepPolicy(
            action_space=env.action_space, step_size=50, device=device
        )
        random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
        off_policy = actdyn.policy.OffPolicy(action_space=env.action_space, device=device)

        # ------------------------------------------------------------------------------
        # Model Components - Agent and Experiment
        # ------------------------------------------------------------------------------
        if exp_id == "RND":
            current_policy = mpc_policy
        elif exp_id == "active_long":
            current_policy = mpc_policy
        elif exp_id == "active_short":
            current_policy = mpc_policy
        elif exp_id == "step":
            current_policy = step_policy
        elif exp_id == "random":
            current_policy = random_policy
        elif exp_id == "off_policy":
            current_policy = off_policy
        else:
            raise NotImplementedError

        agent = actdyn.Agent(
            env=env, model=model, buffer_length=10, policy=current_policy, device=device
        )

        exp_config = ExperimentConfig.from_yaml(
            os.path.join(os.path.dirname(__file__), "conf/intro_video.yaml")
        )
        exp_config.results_dir = base_dir
        exp_config.training.total_steps = total_t
        experiment = actdyn.core.experiment.MetaEmbeddingExperiment(
            agent=agent,
            config=exp_config,
        )

        train_cfg = exp_config.training
        # Initialize environment
        experiment.init_experiment(reset=True_)

        # Setup progress bar
        experiment.pbar = tqdm(total=train_cfg.total_steps - experiment.env_step, desc="Training")
        while experiment.env_step < train_cfg.total_steps:
            experiment.env_step += 1

            # 1. Plan
            action = experiment.agent.plan()
            # 2. Execute
            transition, done = experiment.agent.step(action)
            e_bel = experiment.agent.model.embedding.reshape(-1)
            e_norm.append(torch.norm(model.e["m"].cpu() - e).numpy())

            # Append transition to rollout
            experiment.rollout.add(**transition)

            # Update policy
            experiment.agent.update_policy(transition)

            # Update logs
            experiment.training_info["e"] = e_bel
            experiment.update_writer(experiment.training_info)
            experiment.writer.add_scalars(
                "e",
                {
                    "true_0": experiment.agent.env.env.dyn_param[0],
                    "true_1": experiment.agent.env.env.dyn_param[1],
                },
                experiment.env_step,
            )
            experiment.update_pbar(experiment.pbar)

            # Periodic rollout saving for crash recovery and memory management
            if experiment.is_save_step:
                save_load.save_rollout(
                    experiment.rollout,
                    os.path.join(
                        experiment.results_path, f"rollouts/rollout_{experiment.env_step}.pkl"
                    ),
                )
                experiment.rollout.clear()

            # Clean up tensors to prevent memory accumulation
            if "cuda" in str(experiment.agent.device):
                del transition, action
                torch.cuda.empty_cache()

            if done:
                break

        experiment.pbar.close()
        experiment.agent.model.save(os.path.join(experiment.results_path, f"model/model_final.pth"))

        e_norm = np.array(e_norm)
        e_dict[exp_id].append(e_norm)
        print(f"Exp: {exp_id}, True embedding: {e}, Learned embedding: {model.e['m']}")

with open(os.path.join(base_dir, "known_lg_comparison.pkl"), "wb") as f:
    pickle.dump(e_dict, f)


# active_dict = pickle.load(open(os.path.join(base_dir, "active_comparison.pkl"), "rb"))
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
colorset = sns.color_palette("tab10", n_colors=8)
i = 0
for k, v in e_dict.items():
    if len(v) == 0:
        i += 1
        continue

    v = np.array(v)
    mean = v.mean(0)
    std = v.std(0)
    # same color for each method but different line style
    plt.plot(mean, label=k + " (meta)", linestyle="-", color=colorset[i])
    plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.1, color=colorset[i])
    i += 1
plt.xlim(0, 500)
plt.xlabel("Environment Steps")
plt.ylabel("Error Norm")
plt.legend()
plt.title("Error Norm over Environment Steps")
plt.tight_layout()


# %%  Visualize iCEM
# Make even grid [-5,5]x[-5,5]
x = torch.linspace(-5, 5, 50)
y = torch.linspace(-5, 5, 50)
X, Y = torch.meshgrid(x, y)
Z = torch.zeros_like(X)
rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)

# Train on random points
zs = z_sampler(100).to(device)
ro = Rollout()
ro.add(**{"next_model_state": zs})
rnd_metric.update(ro)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = torch.tensor([[X[i, j], Y[i, j]]], device=device)
        Z[i, j] = rnd_metric.compute_uncertainty(pos).item()

plt.figure(figsize=(6, 5))
plt.contourf(X.cpu(), Y.cpu(), np.log(Z).cpu(), levels=50, cmap="viridis")
plt.colorbar(label="Random Network Distillation")
plt.title("Random Network Distillation Heatmap")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

z = torch.zeros(1, 1, 2, device=device)
model._state = z
actions = mpc_policy.beginning_of_rollout(z)
actions = mpc_policy.sample_action_sequences(20)
simulated_paths = torch.cat([model._state.repeat(20, 1, 1), model.predict(actions)], dim=-2)
costs = -rnd_metric.compute_uncertainty(simulated_paths).sum(-1)
elite_idxs = torch.topk(-costs, 5, dim=0)[1]

plt.figure(figsize=(6, 5))
plt.contourf(X.cpu(), Y.cpu(), np.log(Z).cpu(), levels=50, cmap="viridis")
for idx in range(simulated_paths.shape[0]):
    if idx in elite_idxs:
        plt.plot(
            to_np(simulated_paths[idx, :, 0]),
            to_np(simulated_paths[idx, :, 1]),
            color="red",
            linewidth=1,
        )
    else:
        plt.plot(
            to_np(simulated_paths[idx, :, 0]),
            to_np(simulated_paths[idx, :, 1]),
            color="#F79C63",
            alpha=0.7,
            linewidth=1,
        )
plt.colorbar(label="Random Network Distillation")
plt.title("iCEM Action Samples and Elites (Iter 1)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig(os.path.join(base_dir, "icem_iter1.svg"))

for i in range(2, 11):
    elite_actions = actions[elite_idxs]
    elite_costs_traj = costs[elite_idxs]

    new_mean = elite_actions.mean(dim=0).to(device)
    new_std = elite_actions.std(dim=0).to(device)

    mpc_policy.mean = (1 - mpc_policy.alpha) * new_mean + mpc_policy.alpha * mpc_policy.mean
    mpc_policy.std = (1 - mpc_policy.alpha) * new_std + mpc_policy.alpha * mpc_policy.std

    actions = mpc_policy.sample_action_sequences(20)
    actions = torch.cat([actions, elite_actions], dim=0)
    simulated_paths = torch.cat(
        [model._state.repeat(actions.shape[0], 1, 1), model.predict(actions)], dim=-2
    )
    costs = -rnd_metric.compute_uncertainty(simulated_paths).sum(-1)
    elite_idxs = torch.topk(-costs, 5, dim=0)[1]

    plt.figure(figsize=(6, 5))
    plt.contourf(X.cpu(), Y.cpu(), np.log(Z).cpu(), levels=50, cmap="viridis")
    for idx in range(simulated_paths.shape[0]):
        if idx in elite_idxs:
            plt.plot(
                to_np(simulated_paths[idx, :, 0]),
                to_np(simulated_paths[idx, :, 1]),
                color="red",
                linewidth=1,
            )
        else:
            plt.plot(
                to_np(simulated_paths[idx, :, 0]),
                to_np(simulated_paths[idx, :, 1]),
                color="#F79C63",
                alpha=0.7,
                linewidth=1,
            )
    plt.colorbar(label="Random Network Distillation")
    plt.title(f"iCEM Action Samples and Elites (Iter {i})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(os.path.join(base_dir, f"icem_iter{i}.svg"))


# %%
ro = save_load.load_and_concatenate_rollouts(os.path.join(base_dir, "rollouts"))
# ro = experiment.rollout
z = ro["env_state"]
z_hat = ro["model_state"]
sim_vec_env.set_params(model.embedding.squeeze())
# plot_vector_field(duffing_env.dynamics, x_range=10)
plot_vector_field(sim_vec_env.dynamics, x_range=3)
plt.plot(to_np(z[0])[:, 0], to_np(z[0])[:, 1], alpha=0.7, label="true")
plt.plot(to_np(z_hat[0])[:, 0], to_np(z_hat[0])[:, 1], alpha=0.7, label="model")
plt.legend()
plt.show()
y = ro["obs"]
plt.plot(to_np(y[0])[:, :5])
plt.show()

J = (
    lambda x: ((decoder.jacobian(x) @ Fe_true(x, e)).mT @ (decoder.jacobian(x) @ Fe_true(x, e)))[
        0, 0
    ]
    .diag()
    .sum()
)

# Create heatmap of Fisher Information
grid_size = 20
x = torch.linspace(-2, 2, grid_size)
y = torch.linspace(-2, 2, grid_size)
X, Y = torch.meshgrid(x, y)
Z = torch.zeros_like(X)
for i in range(grid_size):
    for j in range(grid_size):
        pos = torch.tensor([[X[i, j], Y[i, j]]], device=device)
        Z[i, j] = torch.log(J(pos)).item()
plt.figure(figsize=(6, 5))
plt.contourf(X.cpu(), Y.cpu(), Z.cpu(), levels=50, cmap="viridis")
plt.colorbar(label="Fisher Information")
plt.title("Fisher Information Heatmap")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


import matplotlib.animation as animation

from actdyn.utils.visualize import plot_spike_train

resample_rate = 10
obs_model.set_dt(dt / resample_rate)

resampled_obs = torch.zeros(total_t * resample_rate, dy, device=device)
for i in range(total_t):
    for j in range(resample_rate):
        resampled_obs[i * resample_rate + j] = obs_model(ro["env_state"][0][i].to(device))

plot_spike_train(ro["env_state"], resampled_obs, dt)
# %% Generate Intro Video
# Generate high sampling rate latent + spikes for visualization
fast_dt = 0.001
dy = 200
dz = 2
du = 2
alpha = 10
noise_scale = 0.1
action_strength = 0.5

e = e_sampler(1)
a, b = e.reshape(-1)
# ------------------------------------------------------------------------------
# Action Model
# ------------------------------------------------------------------------------
obs_model = actdyn.environment.observation.LogLinearObservation(
    obs_dim=dy,
    latent_dim=dz,
    noise_type="poisson",
    dt=fast_dt,
    device=device,
)

C = obs_model.network[0].weight.detach()
C[:, 0] = torch.abs(C[:, 0])
C[:, 1] = C[:, 1] * 2
C *= 1
mean_firing = 10
bias = torch.log(mean_firing * torch.ones(dy, device=device)) - 1 / 2 * torch.diag(C @ C.T)

obs_model.network[0].bias = nn.Parameter(bias)
obs_model.network[0].weight = nn.Parameter(C)
action_model = actdyn.environment.action.IdentityActionEncoder(
    action_dim=du,
    latent_dim=dz,
    action_bounds=[-action_strength * alpha, action_strength * alpha],
    device=device,
)
duffing_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=torch.tensor([a, b, 0.1]),
    dt=fast_dt,
    alpha=alpha,
    Q=noise_scale,
    action_bounds=[action_model.action_space.low, action_model.action_space.high],
    device=device,
)
env = actdyn.environment.EnvWrapper(duffing_env, obs_model, action_model, dt=fast_dt, device=device)
T = int(60 / fast_dt)

z_vis = torch.zeros(T, dz, device=device)
spikes_vis = torch.zeros(T, dy, device=device)

step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=1000, device=device)

for t in range(T):
    if t == 0:
        obs, info = env.reset()

    action = step_policy(None)
    obs, reward, _, done, info = env.step(action)
    z_vis[t] = info["latent_state"]
spikes_vis = to_np(torch.poisson(obs_model(z_vis)))
spikes = (spikes_vis > 0).astype(np.int8)

# %% Video of Spike Train
anim = animate_spikes(
    spikes=spikes[:, :],
    dt=fast_dt,
    window=1000,
    fps=60,
    prores=False,
    save_path=os.path.join(base_dir, "intro_white.mp4"),
)

ro = save_load.load_and_concatenate_rollouts(os.path.join(base_dir, "rollouts"))
z = to_np(ro["model_state"][0])
anim = animate_latent_trajectory(
    z,
    dt=0.01,
    fps=30,
    trail=100,
    out_path=os.path.join(base_dir, "intro_latent.mp4"),
    prores=False,
)

# %% Downsample

# %% Video


# =========================
# 2. Helper to build fading trail segments
# =========================
def build_trail_segments(z_xy, trail_len):
    """
    Given last K points of z_xy (K,2), return line segments (K-1,2,2)
    for LineCollection.
    """
    pts = z_xy[-trail_len:]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    return segs


# =========================
# 3. Animate
# =========================
def animate_latent_and_spikes(
    z, spikes, dt, trail_len=100, raster_window=200, fps=30, save_path=None
):
    """
    z: (T,2)
    spikes: (T,N) in {0,1}
    dt: float
    trail_len: how many past latent points to draw with fading line
    raster_window: how many recent time bins to show in raster
    fps: animation fps for saving (not for live plt.show())
    save_path: optional .mp4 path
    """

    T, latent_dim = z.shape
    _, N = spikes.shape
    assert latent_dim == 2, "Only 2D latent supported in this viz."

    # figure style
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f0f10",
            "axes.facecolor": "#0f0f10",
            "axes.edgecolor": "#cccccc",
            "text.color": "#cccccc",
            "axes.labelcolor": "#cccccc",
            "xtick.color": "#888888",
            "ytick.color": "#888888",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(8, 4), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.3)

    # ------ Left: latent space ------
    ax_latent = fig.add_subplot(gs[0, 0])
    ax_latent.set_title("Latent state", color="#cccccc", pad=6)
    ax_latent.set_xlabel("z1")
    ax_latent.set_ylabel("z2")

    # fix axis limits from whole trajectory for stability
    margin = 0.5
    x_min, x_max = z[:, 0].min() - margin, z[:, 0].max() + margin
    y_min, y_max = z[:, 1].min() - margin, z[:, 1].max() + margin
    ax_latent.set_xlim(x_min, x_max)
    ax_latent.set_ylim(y_min, y_max)

    # background full trajectory (very faint)
    ax_latent.plot(
        z[:, 0],
        z[:, 1],
        lw=0.5,
        alpha=0.15,
        color="#bbbbbb",
    )

    # dynamic trail: LineCollection for past ~trail_len points
    init_trail = build_trail_segments(z[:trail_len], trail_len)
    trail_lc = LineCollection(
        init_trail,
        linewidths=2.0,
        cmap="viridis",
        array=np.linspace(0.2, 1.0, trail_len - 1),
        alpha=0.9,
    )
    ax_latent.add_collection(trail_lc)

    # current point
    (head_dot,) = ax_latent.plot(
        z[0, 0],
        z[0, 1],
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markerfacecolor="none",
        markeredgewidth=0.8,
    )

    # time text
    time_text = ax_latent.text(
        0.02,
        0.95,
        "",
        transform=ax_latent.transAxes,
        ha="left",
        va="top",
        color="#cccccc",
    )

    # ------ Right: spike raster ------
    ax_raster = fig.add_subplot(gs[0, 1])
    ax_raster.set_title("Spikes (recent)", color="#cccccc", pad=6)
    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_ylabel("Neuron")

    ax_raster.set_ylim(-0.5, N - 0.5)
    ax_raster.set_yticks([0, N - 1])
    ax_raster.set_yticklabels(["0", f"{N-1}"])

    # x-limit is rolling window of raster_window steps
    win_dur_ms = raster_window * dt * 1000.0
    ax_raster.set_xlim(-win_dur_ms, 0.0)

    # pre-allocate scatter. we'll update offsets instead of replot.
    spike_offsets = np.zeros((1, 2))
    raster_scatter = ax_raster.scatter(
        spike_offsets[:, 0],
        spike_offsets[:, 1],
        s=6,
        linewidths=0,
        c="#39ff14",  # neon green-ish for contrast on dark bg
        alpha=0.8,
    )

    # =========================
    # update function
    # =========================
    def update(frame_idx):
        # --- latent panel ---
        start_idx = max(0, frame_idx - trail_len + 1)
        trail_slice = z[start_idx : frame_idx + 1]

        if trail_slice.shape[0] >= 2:
            segs = np.stack([trail_slice[:-1], trail_slice[1:]], axis=1)
            trail_lc.set_segments(segs)
            # refresh colormap alpha gradient length match
            L = segs.shape[0]
            trail_lc.set_array(np.linspace(0.2, 1.0, L))

        # head_dot.set_data(z[frame_idx, 0], z[frame_idx, 1])
        head_dot.set_data([z[frame_idx, 0]], [z[frame_idx, 1]])
        # time display in ms
        t_ms = frame_idx * dt * 1000.0
        time_text.set_text(f"t = {t_ms:.1f} ms")

        # --- spike raster panel ---
        # window [frame_idx - raster_window + 1, frame_idx]
        r_start = max(0, frame_idx - raster_window + 1)
        r_spk = spikes[r_start : frame_idx + 1]  # (W, N)
        W = r_spk.shape[0]

        # build offsets: each spike -> (time_rel_ms, neuron_id)
        if W > 0:
            # time relative to now, in ms, shape (W,)
            t_rel = np.arange(-W + 1, 1) * dt * 1000.0
            # repeat for each neuron
            tt = np.repeat(t_rel[:, None], N, axis=1)  # (W,N)
            nn = np.repeat(np.arange(N)[None, :], W, axis=0)  # (W,N)

            mask = r_spk > 0
            xs = tt[mask].reshape(-1)
            ys = nn[mask].reshape(-1)

            if xs.size == 0:
                # no spikes in window
                raster_scatter.set_offsets(np.zeros((0, 2)))
            else:
                raster_scatter.set_offsets(np.stack([xs, ys], axis=1))

        # ensure limits consistent (avoid autoscale each frame)
        ax_raster.set_xlim(-win_dur_ms, 0.0)

        return trail_lc, head_dot, time_text, raster_scatter

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000.0 / fps,
        blit=False,
    )

    # Optional save (fast-ish with ffmpeg if available)
    if save_path is not None:
        anim.save(save_path, fps=fps, dpi=150, codec="h264", bitrate=-1)

    return anim


def animate_latent_trajectory(
    z,
    dt=0.01,
    fps=60,
    skip=1,
    trail=100,
    out_path="latent_trajectory.mov",
    prores=True,
):
    """
    Animate a 2D latent trajectory with fading trail.

    Args:
        z (ndarray): latent trajectory, shape (T, 2)
        dt (float): time step (s)
        fps (int): playback frame rate
        skip (int): draw every Nth frame (for real-time scaling)
        trail (int): length of fading tail
        out_path (str): output file path (.mov or .mp4)
        prores (bool): if True, exports Keynote-safe ProRes (.mov)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    z = np.asarray(z)
    T = len(z)
    frames = range(0, T, skip)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    # grid on
    ax.grid(True, color="#333333", lw=0.5, ls="--", alpha=0.5)
    plt.tight_layout()

    # trail and head
    init_trail = build_trail_segments(z[:trail], trail)
    trail_line = LineCollection(
        init_trail,
        linewidths=2.0,
        cmap="magma",
        array=np.linspace(0.2, 1.0, trail - 1),
        alpha=0.9,
    )
    ax.add_collection(trail_line)
    (head_dot,) = ax.plot(z[0, 0], z[0, 1], "o", color="#000000", markersize=6)

    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top", color="white", fontsize=8
    )

    # --- initialization ---
    def init():
        trail_line.set_segments([])
        head_dot.set_data([], [])
        time_text.set_text("")
        return trail_line, head_dot, time_text

    # --- update function ---
    def update(i):
        print(f"[DEBUG] Frame {i}/{T}")
        i = min(i, T - 1)
        start = max(0, i - trail)
        if i - start > 1:
            seg1 = z[start:i, :]
            seg2 = z[start + 1 : i + 1, :]
            seg = np.stack([seg1, seg2], axis=1)
            trail_line.set_segments(seg)
            trail_line.set_array(np.linspace(0, 1, len(seg)))
        else:
            trail_line.set_segments([])  # nothing to draw yet
        head_dot.set_data([z[i, 0]], [z[i, 1]])
        time_text.set_text(f"t = {i * dt:.2f} s")
        return trail_line, head_dot, time_text

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    # --- writer setup ---
    if prores:
        out_path = out_path.replace(".mp4", ".mov")
        writer = FFMpegWriter(
            fps=fps,
            codec="prores_ks",
            bitrate=-1,
            extra_args=[
                "-profile:v",
                "3",  # ProRes 422 HQ
                "-pix_fmt",
                "yuv422p10le",
            ],
        )
    else:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=8000,
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-crf",
                "12",
                "-movflags",
                "+faststart",
            ],
        )

    print(f"[INFO] Saving {len(frames)} frames → {out_path} (fps={fps}, skip={skip})")
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    print("[INFO] Done.")


# %%


def animate_spikes(
    spikes,
    dt,
    window=200,
    fps=60,
    skip=None,
    save_path="spikes.mp4",
    prores=False,
):
    T, N = spikes.shape
    win_ms = window * dt * 1000.0

    # --- auto-select frame skip to approximate real-time ---
    if skip is None:
        skip = max(1, int(1.0 / (fps * dt)))  # ensures sim_time/playback_time ≈ 1

    # --- dark theme styling ---
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#FFFFFF",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "axes.labelcolor": "#000000",
            "text.color": "#000000",
        }
    )

    fig, ax = plt.subplots(figsize=(15, 5), dpi=150)
    ax.set_xlim(-win_ms, 0)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.set_title("Spike Raster", color="#ccc", pad=8)
    plt.tight_layout()

    scatter = ax.scatter([], [], s=8, marker="o", c="#313131", lw=0, alpha=0.9)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, color="#313131", ha="left", va="top"
    )
    # turn off axis remove gridlines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.grid(False)

    def init():
        scatter.set_offsets(np.zeros((0, 2)))
        time_text.set_text("")
        return scatter, time_text

    def update(frame):
        print(f"[DEBUG] Frame {frame}/{T}")
        start = max(0, frame - window)
        seg = spikes[start : frame + 1]
        W = seg.shape[0]
        if W == 0:
            scatter.set_offsets(np.zeros((0, 2)))
            return scatter, time_text

        t_rel = np.arange(-W + 1, 1) * dt * 1000.0
        tt = np.repeat(t_rel[:, None], N, axis=1)
        nn = np.repeat(np.arange(N)[None, :], W, axis=0)
        mask = seg > 0

        if np.any(mask):
            xs = tt[mask]
            ys = nn[mask]
            scatter.set_offsets(np.c_[xs, ys])
        else:
            scatter.set_offsets(np.zeros((0, 2)))

        time_text.set_text(f"")
        return scatter, time_text

    frames = range(0, T, skip)
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000.0 / fps,
        blit=True,
        repeat=False,
    )

    # --- choose codec for Keynote-safe export ---
    if prores:
        # --- Apple ProRes 422 HQ (native, high quality) ---
        save_path = save_path.replace(".mp4", ".mov")
        writer = FFMpegWriter(
            fps=fps,
            codec="prores_ks",
            bitrate=-1,
            extra_args=[
                "-profile:v",
                "3",  # 3 = ProRes 422 HQ
                "-pix_fmt",
                "yuv422p10le",  # 10-bit precision, no color banding
            ],
        )
    else:
        # --- High-bitrate H.264 baseline (Keynote-safe) ---
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=15000,  # ~15 Mbps
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-crf",
                "12",  # low CRF = high quality
                "-movflags",
                "+faststart",
            ],
        )

    print(f"[INFO] Saving {len(frames)} frames → {save_path} (fps={fps}, skip={skip})")
    anim.save(save_path, writer=writer, dpi=150)
    plt.close(fig)
    print("[INFO] Done.")


# %% Meta
plt.rcParams.update(
    {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#FFFFFF",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.labelcolor": "#000000",
        "text.color": "#000000",
        "xtick.color": "#888888",
        "ytick.color": "#888888",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    }
)


e_dict = {
    "active_long": [],
    "active_short": [],
    "RND": [],
    "random": [],
    "off_policy": [],
}
exp_name = ["active_short", "active_long", "RND", "random"]

meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)

for exp_id in exp_name:
    base_dir = os.path.join(os.path.dirname(__file__), "../../results", "CISS", exp_id)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    torch.manual_seed(10)
    for _ in range(5):
        dz, de, du, dy = 2, 2, 2, 50
        dt = 0.01
        alpha = 10
        total_t = 1000
        action_strength = 0.5
        noise_scale = 0.1
        # torch.manual_seed(7)
        e = e_sampler(1)
        a, b = e.reshape(-1)
        # ------------------------------------------------------------------------------
        # Action Model
        # ------------------------------------------------------------------------------
        action_model = actdyn.environment.action.IdentityActionEncoder(
            action_dim=du,
            latent_dim=dz,
            action_bounds=[-action_strength * alpha, action_strength * alpha],
            device=device,
        )
        # ------------------------------------------------------------------------------
        # Observation Model
        # ------------------------------------------------------------------------------
        obs_model = actdyn.environment.observation.IdentityObservation(
            obs_dim=dy, latent_dim=dz, device=device
        )
        obs_model = actdyn.environment.observation.LinearObservation(
            obs_dim=dy,
            latent_dim=dz,
            noise_scale=noise_scale,
            noise_type="gaussian",
            device=device,
        )
        obs_model = actdyn.environment.observation.LogLinearObservation(
            obs_dim=dy,
            latent_dim=dz,
            noise_scale=0.1,
            noise_type="poisson",
            dt=dt,
            device=device,
        )

        C = obs_model.network[0].weight.detach()
        C[:, 0] = torch.abs(C[:, 0])
        # C[:, 0] = C[:, 0] * 3
        # C[:, 1] = torch.abs(C[:, 1])
        C[:, 1] = C[:, 1] * 2
        # C = C / torch.norm(C, dim=1, keepdim=True)  # Normalize rows of C
        C *= 1
        mean_firing = 50
        bias = torch.log(mean_firing * torch.ones(dy, device=device)) - 1 / 2 * torch.diag(C @ C.T)

        obs_model.network[0].bias = nn.Parameter(bias)
        obs_model.network[0].weight = nn.Parameter(C)

        # ------------------------------------------------------------------------------
        # Environment
        # ------------------------------------------------------------------------------
        duffing_env = actdyn.VectorFieldEnv(
            "duffing",
            x_range=5,
            dyn_params=torch.tensor([a, b, 0.1]),
            dt=dt,
            alpha=alpha,
            Q=noise_scale,
            action_bounds=[action_model.action_space.low, action_model.action_space.high],
            device=device,
        )
        env = actdyn.environment.EnvWrapper(
            duffing_env, obs_model, action_model, dt=dt, device=device
        )
        # ------------------------------------------------------------------------------
        # Decoder with Gaussian Noise
        # ------------------------------------------------------------------------------
        mapping = actdyn.models.decoder.LinearMapping(latent_dim=dz, obs_dim=dy, device=device)
        mapping = actdyn.models.decoder.LogLinearMapping(
            latent_dim=dz, obs_dim=dy, dt=dt, device=device
        )
        noise = actdyn.models.decoder.GaussianNoise(obs_dim=dy, sigma=0.01, device=device)
        noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
        decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
        # ------------------------------------------------------------------------------
        # Model Components - Dynamics and model
        # ------------------------------------------------------------------------------
        sim_vec_env = actdyn.VectorFieldEnv(
            "duffing",
            x_range=5,
            dyn_params=torch.tensor([0, 0, 0.1]),
            dt=dt,
            alpha=alpha,
            Q=noise_scale,
            device=device,
        )

        # dynamics = actdyn.models.dynamics.FunctionDynamics(
        #     state_dim=dz, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
        # )
        dynamics = actdyn.models.dynamics.FunctionDynamics(
            state_dim=dz, dt=env.dt, dynamics_fn=meta_dynamics, device=device
        )
        dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz) * noise_scale).to(device))

        sigma_0 = 1e-2
        e_bel = {
            "m": torch.ones(1, de, device=device),
            "P": sigma_0 * torch.eye(de, device=device).unsqueeze(0),
            "L": 1 / sigma_0 * torch.eye(de, device=device).unsqueeze(0),
        }

        model = actdyn.models.FilteringEmbedding(
            dynamics=dynamics,
            decoder=decoder,
            e=e_bel,
            action_encoder=action_model,
            Fe=Fe_net,
            Fz=Fz_net,
            device=device,
        )
        model.set_params(e_bel["m"])
        # model.set_params(e)

        # ------------------------------------------------------------------------------
        # Model Components - Policy
        # ------------------------------------------------------------------------------
        emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
            model=model, Fe_net=Fe_net, Fz_net=Fz_net
        )
        rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)
        action_metric = actdyn.metrics.cost.ActionCost()
        composite_metric = actdyn.metrics.CompositeMetric(
            [rnd_metric, emb_metric, action_metric], weights=[2.0, 1.0, 0.005]
        )
        mpc_policy = actdyn.policy.mpc.MpcICem(
            metric=rnd_metric if exp_id == "RND" else emb_metric,
            model=model,
            device=device,
            horizon=10 if exp_id == "active_long" else 5,
            num_iterations=10,
            num_samples=40,
            num_elite=10,
            chunk=5 if exp_id == "active_long" else 3,
            verbose=False,
        )
        step_policy = actdyn.policy.StepPolicy(
            action_space=env.action_space, step_size=50, device=device
        )
        random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
        off_policy = actdyn.policy.OffPolicy(action_space=env.action_space, device=device)

        # ------------------------------------------------------------------------------
        # Model Components - Agent and Experiment
        # ------------------------------------------------------------------------------
        # agent = actdyn.AsyncAgent(env=env, model=model, policy=mpc_policy, device=device, buffer_length=10)

        if exp_id == "RND":
            current_policy = mpc_policy
        elif exp_id == "active_long":
            current_policy = mpc_policy
        elif exp_id == "active_short":
            current_policy = mpc_policy
        elif exp_id == "step":
            current_policy = step_policy
        elif exp_id == "random":
            current_policy = random_policy
        elif exp_id == "off_policy":
            current_policy = off_policy
        else:
            raise NotImplementedError

        agent = actdyn.Agent(
            env=env, model=model, buffer_length=10, policy=current_policy, device=device
        )
        exp_config = ExperimentConfig.from_yaml(
            os.path.join(os.path.dirname(__file__), "conf/config.yaml")
        )
        exp_config.results_dir = base_dir
        exp_config.training.total_steps = 1000
        experiment = actdyn.core.experiment.MetaEmbeddingExperiment(
            agent=agent,
            config=exp_config,
        )

        decoder.set_params(obs_model)
        experiment.run()
        e_norm = np.array(experiment.e_norm)
        e_dict[exp_id].append(e_norm)
        print(f"exp_id: {exp_id}, True embedding: {e}, Learned embedding: {model.e['m']}")
