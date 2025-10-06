# %%
import os
from einops import rearrange
from tqdm import tqdm
import numpy as np
from scipy import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.functional import softplus
import actdyn.environment
import actdyn.environment.observation
import actdyn.models
from actdyn.utils.torch_helper import to_np
import external.integrative_inference.src.modules as metadyn
from external.integrative_inference.experiments.model_utils import (
    get_shared_modules,
    build_hypernetwork,
)
from actdyn.utils.visualize import plot_vector_field, set_matplotlib_style

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
data_train = {"fx": [], "x": [], "e": []}
vec_env = actdyn.VectorFieldEnv("duffing", x_range=5, dt=0.1, noise_scale=0.0)
for a, b in data["e"]:
    e = torch.tensor([[a, b]], device=device, dtype=torch.float32)
    vec_env.dynamics.a = a
    vec_env.dynamics.b = b
    xx, yy = torch.meshgrid(torch.linspace(-5, 5, 50), torch.linspace(-5, 5, 50))
    x = torch.stack((xx.flatten(), yy.flatten()), -1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        fx = vec_env._get_dynamics(x).squeeze()
    data_train["fx"].append(fx)
    data_train["x"].append(x.squeeze())
    data_train["e"].append(e.repeat(fx.shape[0], 1))
data_train["fx"] = torch.stack(data_train["fx"], 0)
data_train["x"] = torch.stack(data_train["x"], 0)
data_train["e"] = torch.stack(data_train["e"], 0)

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
n_epochs = 2000
batch_size = 10
for epoch in tqdm(range(n_epochs)):
    b_idx = torch.randint(0, data_train["fx"].shape[0], (batch_size,))
    # Flatten the batch dimension (0) with time dimesion (1)
    fx_true = data_train["fx"][b_idx].reshape(-1, cfg["d_latent"])
    x = data_train["x"][b_idx].reshape(-1, cfg["d_latent"])
    e = data_train["e"][b_idx].reshape(-1, cfg["d_embed"])

    out, _ = hypernet_dynamics(e)
    fx_pred = mean_dynamics.compute_param(x, out)

    loss = F.mse_loss(fx_pred, fx_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


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
        lambda x: mean_dynamics(
            x.to(device),
            hypernet_dynamics(
                torch.tensor(data["e"][i], device=device, dtype=torch.float32).unsqueeze(0)
            )[0],
        ),
        ax=axs[1],
        x_range=5,
        is_residual=True,
    )
    axs[1].set_title("Meta Learned Vector Field")
    plt.show()

# %% Train Amortized Embedding Gradient Network

# %% Create Amortized State Inference model, and define our VAE Model
