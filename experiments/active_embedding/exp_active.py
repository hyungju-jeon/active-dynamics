# %%
import os
from collections import deque
import shutil
from typing import Callable
from functools import partial
import imageio
from matplotlib import colors
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from actdyn.config import ExperimentConfig
import actdyn.environment
import actdyn.environment.action
import actdyn.environment.observation
import actdyn.models
import actdyn.models.dynamics
import actdyn.models.encoder
import actdyn.policy
from actdyn.utils import save_load
from actdyn.utils.helpers import setup_experiment
from actdyn.utils.rollout import Rollout, RolloutBuffer, RecentRollout
from actdyn.utils.torch_helper import to_np
import external.integrative_inference.src.modules as metadyn
from external.integrative_inference.experiments.model_utils import build_hypernetwork
from actdyn.utils.visualize import plot_vector_field, set_matplotlib_style
from einops import rearrange, repeat, einsum
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softplus

from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

# from external.integrative_inference.src.utils import save_model, load_model
import actdyn

# Small constant to prevent numerical instability
eps = 1e-6

# Configure matplotlib for better aesthetics
set_matplotlib_style()

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% (Generate) Generate diverse environment and create trajectories
observation_dim = 50
# Linear gaussian observation model
obs_model = actdyn.environment.observation.LinearObservation(
    obs_dim=observation_dim, latent_dim=2, noise_scale=0.1, noise_type="gaussian", device=device
)

# data_path under current file directory
data_path = os.path.join(os.path.dirname(__file__), "full_data.pkl")
if os.path.exists(data_path):
    data = torch.load(data_path)
    x_list = data["x"]
    y_list = data["y"]
else:
    a_list = np.linspace(-0.1, -0.1, 10)
    b_list = np.linspace(-2, 2, 10)
    x_list, y_list = [], []
    # make all possible combinations of a and b
    param_list = [(a, b) for a in a_list for b in b_list]

    for a, b in param_list:
        print("Generating data for a={:.2f}, b={:.2f}".format(a, b))
        vec_env = actdyn.VectorFieldEnv(
            "duffing", x_range=5, dyn_param=[a, b, 0.1], dt=0.01, noise_scale=0.0
        )
        x0 = torch.randn(100, 1, 2) * 2
        T = 1000
        x = vec_env.generate_trajectory(x0, T).to(device)
        y = obs_model(x).detach()
        x_list.append(x)
        y_list.append(y)
    data = {"x": x_list, "y": y_list, "e": param_list}
    # save data for later use
    # torch.save(data, data_path)

    # Clean up memory and cuda cache
    del x, y, vec_env, x0, x_list, y_list
    torch.cuda.empty_cache()

    # Sanity check: plot some trajectories
    for j in range(100):
        for i in range(10):
            ax = plt.subplot(111)
            ax.plot(to_np(data["x"][j][i, :, 0]), to_np(data["x"][j][i, :, 1]))


# %% (Generate) Generate evenly sampled latent space for training
vec_env = actdyn.VectorFieldEnv("duffing", x_range=5, dt=0.1, noise_scale=0.0)


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


z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-3.0, -2.0], [-0.1, 2.0], 2)
ds = zeDataset(100000, z_sampler, e_sampler, device)

# %% (Pretrain) Pretrain context dependent dynamics model
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
            vec_env.dynamics.a = e[:, 0]
            vec_env.dynamics.b = e[:, 1]
            fx_true = vec_env._get_dynamics(z).to(device)

            out, _ = hypernet_dynamics(e)
            fx_pred = mean_dynamics.compute_param(z, out)

            loss = F.mse_loss(fx_pred, fx_true)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")


class MetaDynamics:
    def __init__(self, hypernet, mean_dynamics):
        self.hypernet = hypernet
        self.mean_dynamics = mean_dynamics
        self.e = None
        self.out = None

    def set_embedding(self, e):
        self.e = e
        self.out, _ = self.hypernet(self.e)

    def __call__(self, x, e=None):
        if e is None:
            if self.e is None or self.out is None:
                raise ValueError("Embedding e is not set. Please set e using set_embedding method.")
            out = self.out
        else:
            out, _ = self.hypernet(e)
        return self.mean_dynamics(x, out)


meta_dynamics_fn = MetaDynamics(hypernet_dynamics, mean_dynamics)
# %% (Pretrain) Check learned meta-dynamical model (Checked)
if False:
    for i in range(10):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs = axs.flatten()
        i = torch.randint(0, 100, (1,))
        e = e_sampler(1).squeeze(0)
        plot_vector_field(
            actdyn.VectorFieldEnv(
                "duffing",
                x_range=5,
                dyn_param=[e[0], e[1], 0.1],
                dt=0.1,
                noise_scale=0.0,
            ).dynamics,
            ax=axs[0],
            x_range=5,
            is_residual=True,
        )
        axs[0].set_title(
            "True Vector Field of Duffing System for a={:.2f}, b={:.2f}".format(e[0], e[1])
        )
        plot_vector_field(
            lambda x: meta_dynamics_fn(
                x.to(device),
                torch.tensor(e, device=device, dtype=torch.float32).unsqueeze(0),
            ),
            ax=axs[1],
            x_range=5,
            is_residual=True,
        )
        axs[1].set_title("Meta Learned Vector Field")
        plt.show()


# %% (Pretrain) Train Amortized Embedding Jacobian (Fe) Network
def curvature_penalty(model: nn.Module, z: torch.Tensor, e: torch.Tensor, eps: float = 1e-2):
    """Finite-difference smoothness of F̂_e w.r.t. (z,e)."""
    B = z.size(0)
    # random unit perturbations
    dz = F.normalize(torch.randn_like(z), dim=-1) * eps
    de = F.normalize(torch.randn_like(e), dim=-1) * eps
    J = model(z, e)
    J_e = model(z + dz, e + de)
    return ((J_e - J) ** 2).mean()


def jacobian_wrt_z(f_psi: Callable, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """
    Compute F_z = ∂f_psi/∂z at batch of points.
    Returns: [B, nz, nz]
    """
    B, d_latent, ne = z.shape[0], z.shape[-1], e.shape[-1]
    z = z.detach().requires_grad_(True)
    e = e.detach().requires_grad_(True)
    y = f_psi(z, e)  # [B, nz]
    Fz = []
    # Compute row-wise grads: for each output dim, grad wrt z
    for i in range(d_latent):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, i] = 1.0
        (gi,) = torch.autograd.grad(
            y, z, grad_outputs=grad_outputs, retain_graph=True, create_graph=False
        )
        Fz.append(gi.unsqueeze(1))  # [B,1,nz]
    Fz = torch.cat(Fz, dim=1)  # [B, nz, nz]
    return Fz.detach()


def jacobian_wrt_e(f_psi: Callable, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """
    Compute F_e = ∂f_psi/∂e at batch of points.
    Returns: [B, nz, ne]
    """
    B, d_latent, ne = z.shape[0], z.shape[-1], e.shape[-1]
    z = z.detach().requires_grad_(True)
    e = e.detach().requires_grad_(True)
    y = f_psi(z, e)  # [B, nz]
    Fe = []
    # Compute row-wise grads: for each output dim, grad wrt e
    for i in range(d_latent):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, i] = 1.0
        (gi,) = torch.autograd.grad(
            y, e, grad_outputs=grad_outputs, retain_graph=True, create_graph=False
        )
        Fe.append(gi.unsqueeze(1))  # [B,1,ne]
    Fe = torch.cat(Fe, dim=1)  # [B, nz, ne]
    return Fe.detach()


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
        Fe = jacobian_wrt_e(self.fn, z, e)  # [1, nz, ne]
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
        Fz = jacobian_wrt_z(self.fn, z, e)  # [1, nz, nz]
        return z.squeeze(0), e.squeeze(0), Fz.squeeze(0)


z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-2.0, -1.0], [-0.1, 1.0], 2)

ds = FeDataset(meta_dynamics_fn, 10000, z_sampler, e_sampler, device)
dl = DataLoader(ds, batch_size=500, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

Fe_net = Amortized_Jacobian(d_latent=2, d_embed=2, d_hidden=64, n_hidden=1, device=device)

Fe_model_path = os.path.join(os.path.dirname(__file__), "models", "duffing_amortized_Fe.pth")
if os.path.exists(Fe_model_path):
    Fe_net.load_state_dict(torch.load(Fe_model_path, map_location=device))
    print("Loaded pretrained Fe model from", Fe_model_path)
else:
    opt = torch.optim.AdamW(Fe_net.parameters(), lr=1e-3, weight_decay=1e-4)
    n_epochs = 500
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    pbar = tqdm(range(n_epochs))
    for ep in pbar:
        Fe_net.train()
        total, n = 0.0, 0
        for z, e, Fe_star in dl:
            z, e, Fe_star = z.to(device), e.to(device), Fe_star.to(device)
            Fe_hat = Fe_net(z, e)
            loss_fit = F.mse_loss(Fe_hat, Fe_star)  # Frobenius MSE
            # loss_curv = curvature_penalty(Fe, z, e)
            loss = loss_fit  # + 0.1 * loss_curv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Fe_net.parameters(), max_norm=5.0)
            opt.step()
            total += loss.item() * z.size(0)
            n += z.size(0)
        sched.step()
        pbar.set_postfix(loss=total / n)
        # print(f"[Epoch {ep:02d}] loss={total/n:.6f}")
Fe_net.eval()
# %% (Pretrain) Train Amortized State Jacobian (Fz) Network
z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-2.0, -1.0], [-0.1, 1.0], 2)

ds = FzDataset(meta_dynamics_fn, 10000, z_sampler, e_sampler, device)
dl = DataLoader(ds, batch_size=500, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

Fz_net = Amortized_Jacobian(d_latent=2, d_embed=2, d_hidden=64, n_hidden=1, device=device)

Fz_model_path = os.path.join(os.path.dirname(__file__), "models", "duffing_amortized_Fz.pth")
if os.path.exists(Fz_model_path):
    Fz_net.load_state_dict(torch.load(Fz_model_path, map_location=device))
    print("Loaded pretrained Fz model from", Fz_model_path)
else:
    opt = torch.optim.AdamW(Fz_net.parameters(), lr=1e-3, weight_decay=1e-4)
    n_epochs = 500
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    pbar = tqdm(range(n_epochs))
    for ep in pbar:
        Fz_net.train()
        total, n = 0.0, 0
        for z, e, Fz_star in dl:
            z, e, Fz_star = z.to(device), e.to(device), Fz_star.to(device)
            Fz_hat = Fz_net(z, e)
            loss_fit = F.mse_loss(Fz_hat, Fz_star)  # Frobenius MSE
            # loss_curv = curvature_penalty(Fz, z, e)
            loss = loss_fit  # + 0.1 * loss_curv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Fz_net.parameters(), max_norm=5.0)
            opt.step()
            total += loss.item() * z.size(0)
            n += z.size(0)
        sched.step()
        pbar.set_postfix(loss=total / n)
        # print(f"[Epoch {ep:02d}] loss={total/n:.6f}")
Fz_net.eval()
# %% (Pretrain) Test Amortized Jacobian Network (Tested)
if False:
    for i in range(10):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs = axs.flatten()
        i = torch.randint(0, 100, (1,))
        z = data["x"][i][:1, 0, :].to(device)
        e = torch.tensor(data["e"][i], device=device, dtype=torch.float32).unsqueeze(0)
        e = e.repeat(z.shape[0], 1)
        Fe_star = jacobian_wrt_e(meta_dynamics_fn, z, e).squeeze(0).cpu()
        Fe_hat = Fe_net(z, e).squeeze(0).cpu()

        vmax = max(torch.abs(Fe_star).max(), torch.abs(Fe_hat).max())
        im = axs[0].imshow(
            to_np(Fe_star), vmin=-vmax, vmax=vmax, cmap="bwr", aspect="auto", origin="lower"
        )
        axs[0].set_title("True Jacobian F_e")
        fig.colorbar(im, ax=axs[0])
        im = axs[1].imshow(
            to_np(Fe_hat), vmin=-vmax, vmax=vmax, cmap="bwr", aspect="auto", origin="lower"
        )
        axs[1].set_title("Amortized Jacobian F_e")
        fig.colorbar(im, ax=axs[1])
        plt.show()

# %% (Pretrain) Save trained models for later use
save_path = os.path.join(os.path.dirname(__file__), "models")
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(os.path.join(save_path, "duffing_hypernet_dynamics.pth")):
    torch.save(
        hypernet_dynamics.state_dict(), os.path.join(save_path, "duffing_hypernet_dynamics.pth")
    )
    torch.save(mean_dynamics.state_dict(), os.path.join(save_path, "duffing_mean_dynamics.pth"))
    torch.save(Fe_net.state_dict(), os.path.join(save_path, "duffing_amortized_Fe.pth"))
    torch.save(Fz_net.state_dict(), os.path.join(save_path, "duffing_amortized_Fz.pth"))

# %% (Experiment) Create an environment for active learning
latent_dim = 2
embedding_dim = 2
action_dim = 2

data_idx = 55

# a, b = -1, 0.5
a, b = e_sampler(1).squeeze(0)

action_model = actdyn.environment.action.IdentityActionEncoder(
    action_dim=action_dim, latent_dim=latent_dim, action_bounds=[-10.0, 10.0], device=device
)
vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_param=torch.tensor([a, b, 0.1]),
    dt=0.01,
    alpha=10,
    noise_scale=0.01,
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
env = actdyn.environment.GymObservationWrapper(
    vec_env, obs_model, action_model, dt=0.01, device=device
)
base_dir = os.path.join(os.path.dirname(__file__), "../../results", "active_embedding")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


# %% (ETC) Some helper functions
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


def rk4_step(
    dynamics: callable, dt: float, state: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Perform a single RK4 integration step."""
    k1 = dynamics(state) + action
    k2 = dynamics(state + dt / 2 * k1) + action
    k3 = dynamics(state + dt / 2 * k2) + action
    k4 = dynamics(state + dt * k3) + action
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


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


def debug_fix_decoder(
    decoder: actdyn.models.Decoder, obs_model: actdyn.environment.base.BaseObservation
):
    decoder.mapping.network.weight.data = obs_model.network.weight.data.clone()
    decoder.mapping.network.bias.data = obs_model.network.bias.data.clone()
    decoder.mapping.network.weight.requires_grad = False
    decoder.mapping.network.bias.requires_grad = False


def create_gradient_line(ax, data, c, label=None):
    points = np.array(data).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    alpha = np.linspace(0.1, 1.0, len(segments))
    base_color = np.array(to_rgba(c))
    colors = np.tile(base_color, (len(segments), 1))
    colors[:, -1] = alpha  # assign per-segment alpha

    # Build LineCollection
    lc = LineCollection(segments, colors=colors, linewidth=2, linestyle="solid", label=label)

    # Plot
    ax.add_collection(lc)


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


# %% 1-2. ✅ (Debug) EKF/EKF + Laplace
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
    "m": torch.ones(1, embedding_dim, device=device),
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
    dyn_param=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=10,
    noise_scale=0.01,
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
env.reset()
z = []
z_hat = []
prev_action = torch.zeros(action_dim, device=device)
results_dir = os.path.join(base_dir, "debug_active_ekf_laplace_amortized")
writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))

pbar = tqdm(range(3000))
for env_step in pbar:
    # Q = softplus(dynamics.logvar).diag_embed()  # (1, Dz, Dz)
    Q = 1e-2 * torch.eye(latent_dim, device=device).unsqueeze(0)
    # 1) Random action sampling
    meta_dynamics.set_embedding(e_bel["m"])
    sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    model.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])

    u_t = mpc_policy(z_bel["m"].unsqueeze(0), e_bel=e_bel, z_bel=z_bel, Q=Q).detach()

    # u_t = step_policy(z_bel["m"].unsqueeze(0)).detach()

    # 2-1) Predict latent
    dfde = Fe_true(z_bel["m"], e_bel["m"]) * env.dt
    Fz = Fz_true(z_bel["m"], e_bel["m"])
    dfdz = Fz * env.dt + torch.eye(latent_dim, device=device).unsqueeze(0)
    dhdz = decoder.jacobian.unsqueeze(0)
    HzFe = dhdz @ dfde  # (1, Do, De)

    sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    f_true = sim_vec_env._get_dynamics(z_bel["m"]).to(device)  # For debug

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
        for _ in range(5):
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
    writer.add_scalar("train/e1", e_bel["m"][0, 0].item(), env_step)
    writer.add_scalar("train/e2", e_bel["m"][0, 1].item(), env_step)
    writer.add_scalar("train/e1_true", env.env.dynamics.a.item(), env_step)
    writer.add_scalar("train/e2_true", env.env.dynamics.b.item(), env_step)
    # 7) Log (optional)
    # if (env_step % 1000) == 0 and env_step > 0:
    #     sim_vec_env.dynamics.set_params([e_bel["m"][0, 0], e_bel["m"][0, 1], 0.1])
    #     plot_vector_field(
    #         sim_vec_env.dynamics,
    #         x_range=5,
    #         is_residual=True,
    #     )
    #     z_np = np.stack(z)
    #     z_hat_np = np.stack(z_hat, axis=0).reshape(-1, 2)
    #     plt.plot(z_np[:, 0, 0], z_np[:, 0, 1], label="true", alpha=0.5)
    #     plt.plot(z_hat_np[:, 0], z_hat_np[:, 1], label="inferred", alpha=0.5)
    #     plt.legend()
    #     plt.xlim(-5, 5)
    #     plt.ylim(-5, 5)
    #     plt.show()
    #     z, z_hat = [], []
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
writer.close()


# %% 1-2. ✅✅ Amortized Latent with window + EKF/Laplace (with Embedding)
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


# %% 1-2. ✅✅ (Debug) Amortized Latent with window + EKF/Laplace (with Embedding)
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
    dyn_param=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=1,
    noise_scale=0.01,
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
    f_true = sim_vec_env._get_dynamics(z_bel["m"]).to(device)  # For debugging

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
                f_true = sim_vec_env._get_dynamics(z_bel["m"]).to(device)  # For debugging
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
        f_true = sim_vec_env._get_dynamics(z_flat).to(device)  # For debugging
        z_p = (z_flat + f_true * env.dt)[..., :-1, :]
        z_p += rb["action"][..., 1:, :] * env.dt
        z_p += torch.randn_like(z_p) * (Q.diag()).sqrt()
        z_p = torch.cat([z_flat[..., :1, :], z_p], dim=-2)  # ((S B),T,D)

        f_true = sim_vec_env._get_dynamics(mu_q).to(device)  # For debugging
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
# %% 1-3. ✅✅ DKF (without Embedding)
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
model_env = actdyn.models.model_wrapper.VAEWrapper(
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
agent.model_env = model_env


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
video_path = os.path.join(experiment.results_dir, f"video/vecfield.mp4")
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
    z_pred = agent.model_env.model.dynamics.sample_forward(z_mod, k_step=50, return_traj=True)[1]
    z_pred = [z_mod] + z_pred
    z_pred = torch.cat(z_pred, -2)
    y_pred = agent.model_env.model.decoder(z_pred)

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
            agent.model_env.model.dynamics,
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
                experiment.results_dir, f"video/images/vecfield_{experiment.env_step:05d}.png"
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
            pbar.set_postfix({"ELBO": f"{elbo_loss:.4f}, beta: {agent.model_env.model.beta:.4f}"})
        else:
            pbar.set_postfix({"ELBO": "N/A"})
        pbar.update(100)

    # Train model periodically
    if experiment.env_step > train_cfg.rollout_horizon:
        sampling_ratio = agent.model_env.model.dynamics.dt / agent.env.dt
        experiment.training_loss = agent.train_model(
            **train_cfg.get_optim_cfg(), sampling_ratio=sampling_ratio
        )

    # Periodic rollout saving for crash recovery and memory management
    if experiment.env_step % experiment.cfg.logging.save_every == 0:
        save_load.save_rollout(
            experiment.rollout,
            os.path.join(experiment.results_dir, f"rollouts/rollout_{experiment.env_step}.pkl"),
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
experiment.agent.model_env.save_model(
    os.path.join(experiment.results_dir, f"model/model_final.pth")
)
imageio.mimsave(video_path, frames, fps=5)
writer.close()


# %% 1-3. ✅ (Debug) DKF (without Embedding)
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
model_env = actdyn.models.model_wrapper.VAEWrapper(
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
debug_fix_decoder(decoder, obs_model)
experiment.agent.env = env
experiment.agent.policy.action_space = env.action_space
experiment.agent.model_env = model_env
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


# %% 2-2. Train with active learning + planning


# %% Test model

dynamics.dt = env.dt
obs = experiment.agent.reset()
rb = Rollout(device=device)
rb.add(
    **{
        "obs": obs,
        "next_obs": obs,
        "action": torch.zeros(1, action_dim, device=device),
        "env_state": experiment.agent._env_state,
        "model_state": experiment.agent._model_state,
    }
)
for i in range(1000):
    if env_step % 10 == 0:
        u_t = torch.tensor(env.action_space.sample(), device=device, dtype=torch.float32)
        prev_action = u_t
    u_t = prev_action
    transition, done = experiment.agent.step(u_t)
    rb.add(**transition)
rb.finalize()

plt.plot(to_np(rb["env_state"])[0, :, 0], to_np(rb["env_state"])[0, :, 1])
z_encode = encoder(rb["next_obs"], rb["action"], n_samples=1)[1]
z_pred = dynamics.sample_forward(
    rb["model_state"][:, :1, :], rb["action"], k_step=1000, return_traj=True
)[1]
z_pred = torch.cat(z_pred, 1)

pred_interval = 20
for i in range(pred_interval, 1000, pred_interval):
    z_pred[0, i : i + pred_interval] = torch.cat(
        dynamics.sample_forward(
            rb["model_state"][:, i : i + 1, :],
            rb["action"][:, i:],
            k_step=pred_interval,
            return_traj=True,
        )[1],
        1,
    )
plt.plot(to_np(z_pred)[0, :, 0], to_np(z_pred)[0, :, 1])
plt.plot(to_np(rb["model_state"])[0, :, 0], to_np(rb["model_state"])[0, :, 1])
plt.show()

for i in range(2):
    y_decode = experiment.agent.model_env.model.decoder(rb["model_state"])
    y_pred = experiment.agent.model_env.model.decoder(z_pred)
    plt.plot(to_np(rb["next_obs"])[0, :1000, i], label="obs1")
    plt.plot(to_np(y_pred)[0, :1000, i], label="pred")
    plt.plot(to_np(y_decode)[0, :1000, i], label="dec")
    plt.legend()
    plt.show()

# %% Debug test if we can identify embedding from sequential input
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 2
embedding_dim = 2
T = 1000
dt = 0.01

# --- True dynamics & true embedding (for generating data) ---

vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_param=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=1,
    noise_scale=0.1,
    device=device,
)
sim_vec_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_param=torch.tensor([0, 0, 0.1]),
    dt=0.01,
    alpha=1,
    noise_scale=0.1,
    device=device,
)

# e_true = e_sampler(1).to(device)  # (1, De)
e_true = torch.tensor([[-1.2, 0.1]], device=device)  # (1, De)
vec_env.dynamics.set_params([e_true[0, 0].item(), e_true[0, 1].item(), 0.1])
z0 = torch.randn(1, latent_dim, device=device)  # (1, Dz)
# z0 = torch.tensor([[2.0, 10.0]], device=device)
# Generate smooth cosine action

action = torch.randn(1, T // 10, 2, device=device)
action = action.repeat_interleave(10, dim=1) * 3


z = vec_env.generate_trajectory(z0, n_steps=T, action=action)
plot_vector_field(vec_env.dynamics, x_range=5, is_residual=True)
plt.plot(to_np(z[0])[:, 0], to_np(z[0])[:, 1])
plt.show()


# --- Initialize embedding belief ---
e_bel = {
    # "m": torch.ones(1, 1, embedding_dim, device=device),
    "m": torch.tensor([[[-0.2, -2]]], device=device),
    "P": 0.1 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 10.0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

# --- Laplace embedding update loop ---
loss_hist = []
e_trace = []

for env_step in range(T - 1):
    z_t, z_tp1 = z[:, env_step : env_step + 1], z[0, env_step + 1 : env_step + 2]
    # f_pred = meta_dynamics_fn(z_t, e_bel["m"])

    sim_vec_env.dynamics.set_params([e_bel["m"][0, 0, 0], e_bel["m"][0, 0, 1], 0.1])
    f_true = sim_vec_env._get_dynamics(z_t).to(device)
    r = z_tp1 - (z_t + f_true * dt + action[:, env_step : env_step + 1] * dt)  # (1, 1, Dz)

    # Jacobian wrt e
    # Fe = jacobian_wrt_e(meta_dynamics_fn, z_t.squeeze(0), e_bel["m"].squeeze(0)) * dt
    # Fe = Fe_net(z_t, e_bel["m"]) * dt  # (1, Dz, De)
    Fe = Fe_true(z_t, e_bel["m"]) * dt  # (1, Dz, De)
    # Fz = Fz_net(z_t, e_bel["m"]) * dt
    Fz = Fz_true(z_t, e_bel["m"]) * dt  # (1, Dz, Dz)
    Fz = Fz + torch.eye(latent_dim, device=device).unsqueeze(0)

    # Linearized predictive covariance (simple isotropic Q)
    S = 1e-3 * torch.eye(latent_dim, device=device).unsqueeze(0)
    # S = Fe @ e_bel["P"] @ Fe.transpose(-1, -2) + Q
    # S = 0.5 * (S + S.transpose(-1, -2))

    cholS = torch.linalg.cholesky(S)
    invS_r = torch.cholesky_solve(r.unsqueeze(-1), cholS)
    grad_ll = Fe.transpose(-1, -2) @ invS_r
    X = torch.cholesky_solve(Fe, cholS)
    curv_ll = Fe.transpose(-1, -2) @ X

    # Update belief
    Prec_old = e_bel["Prec"]
    Prec_new = Prec_old + curv_ll  # (1, De, De)
    eta_old = Prec_old @ e_bel["m"].transpose(-1, -2)  # (1, De, 1)
    eta_new = eta_old + grad_ll.squeeze(0)

    cholPrec = torch.linalg.cholesky(Prec_new)
    P_new = torch.cholesky_inverse(cholPrec)
    P_new = torch.inverse(Prec_new)

    m_new = (P_new @ eta_new).transpose(-1, -2)
    e_bel = {"m": m_new.detach(), "P": P_new.detach(), "Prec": Prec_new.detach()}

    loss_hist.append(r.norm().item())
    e_trace.append(e_bel["m"].cpu().detach().numpy())


# --- Plot ---
e_trace = torch.tensor(e_trace).squeeze((1, 2))
plt.figure(figsize=(6, 4))
plt.plot(e_trace[:, 0], label="e1 (est)")
plt.plot(e_trace[:, 1], label="e2 (est)")
plt.axhline(e_true[0, 0].item(), color="gray", ls="--", label="true e1")
plt.axhline(e_true[0, 1].item(), color="gray", ls=":", label="true e2")
plt.xlabel("Time step")
plt.ylabel("Embedding estimate")
plt.legend()
plt.title("Embedding recovery with full latent observation")
plt.show()

print(f"Final estimated e = {e_bel['m'].cpu().numpy()}, true e = {e_true.cpu().numpy()}")
