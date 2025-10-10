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
import actdyn.models
import actdyn.models.encoder
from actdyn.utils.torch_helper import to_np
import external.integrative_inference.src.modules as metadyn
from external.integrative_inference.experiments.model_utils import build_hypernetwork
from actdyn.utils.visualize import plot_vector_field, set_matplotlib_style

from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softplus

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


z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
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

hypernet_model_path = os.path.join(os.path.dirname(__file__), "models", "hypernet_dynamics.pth")
mean_dynamics_model_path = os.path.join(os.path.dirname(__file__), "models", "mean_dynamics.pth")
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
        x_range=5,
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
        x_range=5,
        is_residual=True,
    )
    axs[1].set_title("Meta Learned Vector Field")
    plt.show()


# %% Train Amortized Embedding Jacobian (Fe) Network
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

Fe_model_path = os.path.join(os.path.dirname(__file__), "models", "amortized_Fe.pth")
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
# %% Train Amortized State Jacobian (Fz) Network
z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
e_sampler = make_uniform_sampler([-2.0, -1.0], [-0.1, 1.0], 2)

ds = FzDataset(meta_dynamics_fn, 10000, z_sampler, e_sampler, device)
dl = DataLoader(ds, batch_size=500, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

Fz_net = Amortized_Jacobian(d_latent=2, d_embed=2, d_hidden=64, n_hidden=1, device=device)

Fz_model_path = os.path.join(os.path.dirname(__file__), "models", "amortized_Fz.pth")
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
# %% Test Amortized Jacobian Network (Tested)
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

# %% Save trained models for later use
save_path = os.path.join(os.path.dirname(__file__), "models")
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save(hypernet_dynamics.state_dict(), os.path.join(save_path, "hypernet_dynamics.pth"))
torch.save(mean_dynamics.state_dict(), os.path.join(save_path, "mean_dynamics.pth"))
torch.save(Fe_net.state_dict(), os.path.join(save_path, "amortized_Fe.pth"))
torch.save(Fz_net.state_dict(), os.path.join(save_path, "amortized_Fz.pth"))

# %% Create Amortized State Inference model, likelihood network
latent_dim = 2
embedding_dim = 2
action_dim = 2

mapping = actdyn.models.decoder.LinearMapping(
    latent_dim=latent_dim, obs_dim=observation_dim, device=device
)
noise = actdyn.models.decoder.GaussianNoise(obs_dim=observation_dim, sigma=0.01, device=device)
decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
encoder = actdyn.models.encoder.HybridEncoder(
    latent_dim=latent_dim,
    obs_dim=observation_dim,
    embedding_dim=embedding_dim,
    action_dim=action_dim,
    hidden_dim=0,
    activation="relu",
    device=device,
)

params = list(decoder.parameters())

# def meta_dynamics_fn(x, e):
#     out, _ = hypernet_dynamics(e)
#     return mean_dynamics(x, out)

# %% Create an environment for active learning
data_idx = 15
a, b = data["e"][data_idx]
vec_env = actdyn.VectorFieldEnv(
    "duffing", x_range=5, a=a, b=b, dt=0.1, noise_scale=0.0, device=device
)
action_model = actdyn.environment.action.IdentityActionEncoder(
    action_dim=action_dim, latent_dim=latent_dim, action_bounds=[-1.0, 1.0], device=device
)
obs_model = actdyn.environment.observation.LinearObservation(
    obs_dim=observation_dim,
    latent_dim=latent_dim,
    noise_scale=0.01,
    noise_type="gaussian",
    device=device,
)
env = actdyn.environment.GymObservationWrapper(
    vec_env, obs_model, action_model, dt=0.1, device=device
)


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


# %% 1-1. Train with random actions (EKF/EKF + Laplace)
plt.close("all")
z_bel = {
    "m": torch.zeros(1, latent_dim, device=device),
    "P": torch.eye(latent_dim, device=device).unsqueeze(0),
}

sigma_0 = 0.01
e_bel = {
    "m": torch.zeros(1, embedding_dim, device=device),
    # "m": torch.tensor([[-0.8, -1]], device=device),
    "P": sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
    "Prec": 1 / sigma_0 * torch.eye(embedding_dim, device=device).unsqueeze(0),
}

opt = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
rows = []
env.reset()
z = []
z_hat = []

pbar = tqdm(range(100000))
for t in pbar:
    # 1) Random action sampling
    u_t = env.action_space.sample() * 10 if t % 10 == 0 else env.action_space.sample()
    u_t = torch.tensor(u_t, device=device, dtype=torch.float32)

    # 2-1) Predict latent
    dfdz = Fz_net(z_bel["m"], e_bel["m"]) * env.dt
    # Jz = jacobian_wrt_z(meta_dynamics_fn, z_bel["m"], e_bel["m"]) * env.dt
    dfdz = dfdz + torch.eye(latent_dim, device=device).unsqueeze(0)
    # f_true = env.env._get_dynamics(z_bel["m"]).to(device) # For debug

    z_pred = {
        "m": z_bel["m"] + meta_dynamics_fn(z_bel["m"], e_bel["m"]) * env.dt + u_t * env.dt,
        # "m": z_bel["m"] + f_true * env.dt + u_t * env.dt,
        "P": dfdz @ z_bel["P"] @ dfdz.transpose(-1, -2),
    }
    # 2-2) Predict observation
    R = softplus(decoder.noise.logvar).diag_embed() + eps
    y_pred = {
        "m": decoder(z_pred["m"]),
        "P": R,
    }

    # 3) Get true observation from env
    obs, reward, _, _, info = env.step(u_t)
    y_true = obs.squeeze(0)  # (1, Do)
    r = y_true - y_pred["m"]

    dhdz = decoder.jacobian.unsqueeze(0)  # (1, Do, Dz)
    R = y_pred["P"]

    # 4) Embedding update (Laplace)
    Prec = e_bel["Prec"]
    eta = Prec @ e_bel["m"].unsqueeze(-1)
    for _ in range(5):
        y_hat = decoder(z_pred["m"])
        r_t = y_true - y_hat

        # predictive covariance and Cholesky solve (as fixed earlier)
        S = dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R
        S = 0.5 * (S + S.transpose(-1, -2))

        dfde = Fe_net(z_bel["m"], e_bel["m"]) * env.dt
        HzFe = torch.einsum("byz,bzd->byd", dhdz, dfde)

        chol_S = torch.linalg.cholesky(S)
        invS_r = torch.cholesky_solve(r_t.unsqueeze(-1), chol_S)
        # inv_S = torch.inverse(S)
        # invS_r = inv_S @ r_t.unsqueeze(-1)
        grad_ll = torch.einsum("byd,byk->bd", HzFe, invS_r)
        X = torch.cholesky_solve(HzFe, chol_S)
        # X = inv_S @ HzFe
        curv_ll = torch.einsum("byd,bye->bde", HzFe, X)
        curv_ll = symmetrize(curv_ll)  # ensure symmetry

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
    # z_post = encoder(r=r, H=dhdz, R=R, z_pred=z_pred, e_mu=e_bel["m"])
    K = torch.cholesky_solve(dhdz @ z_pred["P"].transpose(-1, -2), chol_S).transpose(-1, -2)
    I = torch.eye(latent_dim, device=device).unsqueeze(0)
    KH = K @ dhdz

    P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    z_post = {
        "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
        "P": 0.5 * (P_upd + P_upd.transpose(-1, -2)),
    }

    # 6) Roll updated z posterior as new prior
    e_bel = {"m": mu_e.detach(), "P": Sigma_e.detach(), "Prec": Prec_new.detach()}
    z_bel = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}

    # 7) Optimize Likelihood
    opt.zero_grad(set_to_none=True)

    # Single-sample NLL
    ll = decoder.compute_log_prob(z_bel["m"], y_true)
    loss = -ll
    loss.backward()

    # torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 5.0)
    opt.step()

    # 7) Log (optional)
    if (t % 1000000) == 0 and t > 0:
        print(f"[t={t}] LL={ll.item():.3f}")
        print(f"Residual: {r_t.norm().item():.3f}")
        print(f"estimated e: {to_np(mu_e.squeeze())}, true e: {data['e'][data_idx]}")

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
            "t": t,
            "e_norm": float(e_bel["m"].norm()),
            "z_norm": float(z_bel["m"].norm()),
            "r_norm": float(r.norm()),
        }
    )
    z.append(info["latent_state"].squeeze(0).cpu())
    z_hat.append(z_bel["m"].squeeze(0).cpu())

    if t % 100 == 0:
        pbar.set_postfix(
            LL=f"{ll.item():.3f}",
            e_hat=f"({mu_e[..., 0].item():.2f},{mu_e[..., 1].item():.2f})",
            e_true=f"({data['e'][data_idx][0]:.2f},{data['e'][data_idx][1]:.2f})",
        )
        pbar.update(100)


# %% 1-2. Train with random action (EKF/AmortizedGain + Laplace)
# %% 1-3. Train with random action (MC Sample + Laplace)
# %% 1-4. Train with random action (Fully Amortized + Laplace)
# %% 2. Train with active learning (myopic)
# %% 3. Train with active learning + planning
