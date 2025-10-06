# %%
import os
from typing import Callable
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import actdyn.environment
import actdyn.environment.action
import actdyn.environment.observation
from actdyn.utils.torch_helper import to_np
import external.integrative_inference.src.modules as metadyn
from external.integrative_inference.experiments.model_utils import build_hypernetwork
from actdyn.utils.visualize import plot_vector_field, set_matplotlib_style

from torch.utils.data import Dataset, DataLoader

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

# %% Generate diverse environment and create trajectories
observation_dim = 30
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
    a_list = np.linspace(-2, -0.5, 10)
    b_list = np.linspace(-1.0, 1, 10)
    x_list, y_list = [], []
    # make all possible combinations of a and b
    param_list = [(a, b) for a in a_list for b in b_list]

    for a, b in param_list:
        vec_env = actdyn.VectorFieldEnv("duffing", x_range=5, a=a, b=b, dt=0.1, noise_scale=0.005)
        x0 = torch.randn(100, 1, 2) * 2
        T = 500
        x = vec_env.generate_trajectory(x0, T).to(device)
        y = obs_model(x).detach()
        x_list.append(x)
        y_list.append(y)
    data = {"x": x_list, "y": y_list, "e": param_list}
    # save data for later use
    torch.save(data, data_path)

    # Clean up memory and cuda cache
    del x, y, vec_env, x0, x_list, y_list
    torch.cuda.empty_cache()

    # Sanity check: plot some trajectories
    for j in range(100):
        for i in range(10):
            ax = plt.subplot(111)
            ax.plot(to_np(data["x"][j][i, :, 0]), to_np(data["x"][j][i, :, 1]))


# %% Generate evenly sampled latent space for training
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


z_sampler = make_uniform_sampler(-10.0, 10.0, 2)
e_sampler = make_uniform_sampler([-3.0, -2.0], [-0.1, 2.0], 2)
ds = zeDataset(100000, z_sampler, e_sampler, device)

# %% Pretrain context dependent dynamics model
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


# Train with True embedding value and  RMSE loss
optimizer = torch.optim.Adam(
    list(hypernet_dynamics.parameters()) + list(mean_dynamics.parameters()), lr=1e-3
)
n_epochs = 500
batch_size = 1000
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}")


def meta_dynamics_fn(x, e):
    out, _ = hypernet_dynamics(e)
    return mean_dynamics(x, out)


# %% Check learned meta-dynamical model (Checked)
for i in range(10):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()
    i = torch.randint(0, 100, (1,))
    plot_vector_field(
        actdyn.VectorFieldEnv(
            "duffing", x_range=5, a=data["e"][i][0], b=data["e"][i][1], dt=0.1, noise_scale=0.0
        ).dynamics,
        ax=axs[0],
        x_range=10,
        is_residual=True,
    )
    axs[0].set_title(
        "True Vector Field of Duffing System for a={:.2f}, b={:.2f}".format(
            data["e"][i][0], data["e"][i][1]
        )
    )
    plot_vector_field(
        lambda x: meta_dynamics_fn(
            x.to(device),
            torch.tensor(data["e"][i], device=device, dtype=torch.float32).unsqueeze(0),
        ),
        ax=axs[1],
        x_range=10,
        is_residual=True,
    )
    axs[1].set_title("Meta Learned Vector Field")
    plt.show()


# %% Train Amortized Embedding Gradient Network
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
        B = z.shape[0]
        x = torch.cat((z, e), dim=-1)  # [B, nz+ne]
        Fe_hat = self.net(x).view(B, self.d_latent, self.d_embed)
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


z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-2.0, -1.0], [-0.1, 1.0], 2)

ds = FeDataset(meta_dynamics_fn, 10000, z_sampler, e_sampler, device)
dl = DataLoader(ds, batch_size=500, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

Fe = Amortized_Jacobian(d_latent=2, d_embed=2, d_hidden=32, n_hidden=2, device=device)
opt = torch.optim.AdamW(Fe.parameters(), lr=1e-3, weight_decay=1e-4)
n_epochs = 500
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
pbar = tqdm(range(n_epochs))
for ep in pbar:
    Fe.train()
    total, n = 0.0, 0
    for z, e, Fe_star in dl:
        z, e, Fe_star = z.to(device), e.to(device), Fe_star.to(device)
        Fe_hat = Fe(z, e)
        loss_fit = F.mse_loss(Fe_hat, Fe_star)  # Frobenius MSE
        loss = loss_fit
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Fe.parameters(), max_norm=5.0)
        opt.step()
        total += loss.item() * z.size(0)
        n += z.size(0)
    sched.step()
    pbar.set_postfix(loss=total / n)
    # print(f"[Epoch {ep:02d}] loss={total/n:.6f}")

Fe.eval()

# %% Test Amortized Jacobian Network (Tested)
for i in range(10):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()
    i = torch.randint(0, 100, (1,))
    z = data["x"][i][:1, 0, :].to(device)
    e = torch.tensor(data["e"][i], device=device, dtype=torch.float32).unsqueeze(0)
    e = e.repeat(z.shape[0], 1)
    Fe_star = jacobian_wrt_e(meta_dynamics_fn, z, e).squeeze(0).cpu()
    Fe_hat = Fe(z, e).squeeze(0).cpu()

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

# %% Save trained models for later use
save_path = os.path.join(os.path.dirname(__file__), "models")
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save(hypernet_dynamics.state_dict(), os.path.join(save_path, "hypernet_dynamics.pth"))
torch.save(mean_dynamics.state_dict(), os.path.join(save_path, "mean_dynamics.pth"))
torch.save(Fe.state_dict(), os.path.join(save_path, "amortized_jacobian.pth"))

# %% Create Amortized State Inference model, likelihood network

# %% Create an environment for active learning
vec_env = actdyn.VectorFieldEnv("duffing", x_range=5, dt=0.1, noise_scale=0.0, device=device)
action_model = actdyn.environment.action.IdentityActionEncoder(
    action_dim=2, latent_dim=2, action_bounds=[-1.0, 1.0], device=device
)
obs_model = actdyn.environment.observation.LinearObservation(
    obs_dim=observation_dim, latent_dim=2, noise_scale=0.1, noise_type="gaussian", device=device
)
env = actdyn.environment.GymObservationWrapper(
    vec_env, obs_model, action_model, dt=0.1, device=device
)

# %% 1. Train with random actions
# %% 2. Train with active learning (myopic)
# %% 3. Train with active learning + planning
