# %%
import os
import shutil
from collections import deque
from functools import partial
from turtle import color
from typing import Callable, Sequence
from unittest import result

import matplotlib.pyplot as plt
import numpy as np
from sympy import E, EX
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch.nn.functional import softplus
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# from external.integrative_inference.src.utils import save_model, load_model
import actdyn
import actdyn.core
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
from actdyn.utils.experiment_helpers import setup_environment, setup_experiment
from actdyn.utils.rollout import RecentRollout, Rollout, RolloutBuffer
from actdyn.utils.helper import *
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


# %% (Pretrain) Pretrain context dependent dynamics model
z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-3.0, -2.0], [-0.1, 2.0], 2)
ds = zeDataset(100000, z_sampler, e_sampler, device)

duffing_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dt=0.01,
    alpha=10,
    noise_scale=0.01,
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
            fx_true = duffing_env._get_dynamics(z).to(device)

            out, _ = hypernet_dynamics(e)
            fx_pred = mean_dynamics.compute_param(z, out)

            loss = F.mse_loss(fx_pred, fx_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)

# %% (Pretrain) Check learned meta-dynamical model (Checked)
duffing_env = actdyn.VectorFieldEnv("duffing", x_range=5, dt=0.1, noise_scale=0.0)

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
base_dir = os.path.join(os.path.dirname(__file__), "../../results", "active_filtering_embedding")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

meta_dynamics = MetaDynamics(hypernet_dynamics, mean_dynamics)
dz, de, du, dy = 2, 2, 2, 50
dt = 0.01
torch.manual_seed(10)
e = e_sampler(1)
a, b = e.reshape(-1)
# ------------------------------------------------------------------------------
# Action Model
# ------------------------------------------------------------------------------

action_model = actdyn.environment.action.IdentityActionEncoder(
    action_dim=du, latent_dim=dz, action_bounds=[-5.0, 5.0], device=device
)
# ------------------------------------------------------------------------------
# Observation Model
# ------------------------------------------------------------------------------

obs_model = actdyn.environment.observation.LinearObservation(
    obs_dim=dy,
    latent_dim=dz,
    noise_scale=0.1,
    noise_type="gaussian",
    device=device,
)
# C = obs_model.network[0].weight.detach()
# C[:, 0] = torch.abs(C[:, 0])
# C[:, 1] = torch.abs(C[:, 1])
# C = C / torch.norm(C, dim=1, keepdim=True)  # Normalize rows of C
# C *= 1

# obs_model.network[0].weight = nn.Parameter(C)


# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

duffing_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_param=torch.tensor([a, b, 0.1]),
    dt=dt,
    alpha=10,
    noise_scale=0.01,
    action_bounds=[action_model.action_space.low, action_model.action_space.high],
    device=device,
)
env = actdyn.environment.GymObservationWrapper(
    duffing_env, obs_model, action_model, dt=dt, device=device
)
# ------------------------------------------------------------------------------
# Decoder with Gaussian Noise
# ------------------------------------------------------------------------------

mapping = actdyn.models.decoder.LinearMapping(latent_dim=dz, obs_dim=dy, device=device)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=dy, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
# ------------------------------------------------------------------------------
# Model Components - Dynamics and model
# ------------------------------------------------------------------------------

sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_param=torch.tensor([0, 0, 0.1]),
    dt=dt,
    alpha=10,
    noise_scale=0.01,
    device=device,
)
dynamics = actdyn.models.dynamics.FunctionDynamics(
    state_dim=dz, dt=env.dt, dynamics_fn=meta_dynamics, device=device
)
# dynamics = actdyn.models.dynamics.FunctionDynamics(
#     state_dim=dz, dt=env.dt, dynamics_fn=sim_vec_env.dynamics, device=device
# )
dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz) * 0.01 * dt).to(device))

sigma_0 = 0.01
e_bel = {
    "m": torch.ones(1, de, device=device),
    "P": sigma_0 * torch.eye(de, device=device).unsqueeze(0),
    "L": 1 / sigma_0 * torch.eye(de, device=device).unsqueeze(0),
}

dynamics.set_params(e_bel["m"])
model = actdyn.models.FilteringEmbedding(
    dynamics=dynamics,
    decoder=decoder,
    e=e_bel,
    action_encoder=action_model,
    Fe=Fe_net,
    Fz=Fz_net,
    device=device,
)

# ------------------------------------------------------------------------------
# Model Components - Policy
# ------------------------------------------------------------------------------

emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
    model=model, Fe_net=Fe_net, Fz_net=Fz_net
)
mpc_policy = actdyn.policy.mpc.MpcICem(
    metric=emb_metric,
    model=model,
    device=device,
    horizon=20,
    num_iterations=20,
    num_samples=20,
    num_elite=5,
    chunk=5,
    verbose=False,
)
step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=20, device=device)
random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
# ------------------------------------------------------------------------------
# Model Components - Agent and Experiment
# ------------------------------------------------------------------------------

agent = actdyn.Agent(
    env=env,
    model=model,
    policy=mpc_policy,
    device=device,
)

exp_config = ExperimentConfig.from_yaml(os.path.join(os.path.dirname(__file__), "conf/config.yaml"))
exp_config.results_dir = base_dir
experiment = actdyn.core.experiment.MetaEmbeddingExperiment(
    agent=agent,
    config=exp_config,
)

decoder.set_params(obs_model)

# %%
experiment.run()


# %%
e = e_sampler(1).to(device)
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
