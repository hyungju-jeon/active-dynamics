# %%
# Standard library
import os
import random
from pathlib import Path

# Third-party
import numpy as np
import torch
import matplotlib.pyplot as plt


# Project imports
from actdyn.config import ExperimentConfig
from actdyn.utils import setup_experiment, save_load
from actdyn.utils.rollout import Rollout, RolloutBuffer
from actdyn.utils.torch_helper import to_np
from actdyn.utils.validation_helpers import rotate_latent, apply_latent_transform
from actdyn.utils.visualize import plot_vector_field


plt.rcParams["text.usetex"] = False
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.serif"] = "cmr10"
plt.rcParams["font.size"] = 12
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts
# %%

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.join(os.path.dirname(__file__), "../../results", "debug_dynamics")
    config.results_dir = results_dir

    # Set random seeds
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and config.device == "cuda":
        torch.set_default_device(config.device)

    # Set up experiment
    experiment, agent, env, model_env = setup_experiment(config)
    model = model_env.model

    # %% Basic Dynamics Test
    T = 1500
    x0 = torch.zeros(1, 1, model_env.model.latent_dim, device=config.device)
    u = torch.rand(1, T, model_env.model.action_dim, device=config.device) - 0.5

    X = env.env.generate_trajectory(x0, T, action=u)
    X = to_np(X)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    env.env.render(x_range=2, ax=axs[0])
    axs[0].plot(X[0][:, 0], X[0][:, 1])
    axs[0].set_title("Vectorfield")
    axs[0].set_aspect("equal")
    axs[1].plot(X[0])
    axs[1].legend([r"$x_1$", r"$x_2$"])
    axs[1].set_xlabel("Time")
    axs[1].set_title("State Over Time")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    plt.suptitle(r"Trajectory with random action ($u \sim U[-0.5, 0.5]$), dt = 0.1")
    plt.tight_layout()
    fig.savefig(os.path.join(config.results_dir, "random_trajectory.png"))
    plt.close()

    X = env.env.generate_trajectory(x0, T, action=None)
    X = to_np(X)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    env.env.render(x_range=2, ax=axs[0])
    axs[0].plot(X[0][:, 0], X[0][:, 1])
    axs[0].set_title("Vectorfield")
    axs[0].set_aspect("equal")
    axs[1].plot(X[0])
    axs[1].legend([r"$x_1$", r"$x_2$"])
    axs[1].set_xlabel("Time")
    axs[1].set_title("State Over Time")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    plt.suptitle(r"Trajectory with no action, dt = 0.1")
    plt.tight_layout()
    fig.savefig(os.path.join(config.results_dir, "offpolicy_trajectory.png"))
    plt.close()

    # %% Run the experiment and generate trajectory
    experiment.run()
    print(f"Experiment setup completed. Results directory: {config.results_dir}")

    offline_experiment, _, _, _ = setup_experiment(config)
    offline_experiment.offline_learning()
    print(f"Offline experiment completed. Results directory: {config.results_dir}")

    # %% Load rollout
    rollout = save_load.load_and_concatenate_rollouts(
        os.path.join(results_dir, "rollouts"), device="cpu"
    )
    rollout.finalize()
    from actdyn.models import dynamics_from_str

    # Dynamics module
    dynamics_cls = dynamics_from_str(config.model.dynamics_type)
    dyn_config = config.model.get_dynamics_cfg()
    dyn_config.update(
        {
            "state_dim": config.latent_dim,
            "device": config.device,
            "dt": config.dt,
        }
    )
    ls_dynamics = dynamics_cls(
        **dyn_config,
    )

    # %% Compute LS dynamics and plot result
    # x_t+1 = x_t + (f(x_t) + a_t + w) * dt
    fx = (rollout["next_env_state"] - rollout["env_state"]) / config.dt - rollout["action"]
    fx_pred = ls_dynamics(rollout["env_state"].to(ls_dynamics.device))
    phi_x = ls_dynamics.rbf(rollout["env_state"].to(ls_dynamics.device)).detach()
    fx_pred2 = torch.matmul(phi_x, ls_dynamics.weights)

    # Perform LS regression for ls_dynamics.weights
    ls_weights = torch.linalg.lstsq(phi_x, fx.to(ls_dynamics.device), rcond=None).solution
    ls_dynamics.weights.data = ls_weights

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plot_vector_field(env.env.dynamics, x_range=2, ax=axs[0])
    if axs[0].collections:
        mappable0 = axs[0].collections[-1]
        fig.colorbar(mappable0, ax=axs[0], label="Speed", aspect=20)
        vmin, vmax = mappable0.get_clim()
    axs[0].set_xlabel("Latent Dimension 1")
    axs[0].set_ylabel("Latent Dimension 2")
    axs[0].set_title("Vector Field of True dynamics")
    axs[0].axis("equal")
    axs[0].plot(
        rollout["env_state"][0, :, 0],
        rollout["env_state"][0, :, 1],
        label="Trajectory",
        alpha=0.75,
        lw=0.5,
    )
    plot_vector_field(ls_dynamics, x_range=2, ax=axs[1])
    if axs[1].collections:
        mappable1 = axs[1].collections[-1]
        if "vmin" in locals():
            mappable1.set_clim(vmin, vmax)  # enforce same scale as axs[0]
        fig.colorbar(mappable1, ax=axs[1], label="Speed", aspect=20)
    axs[1].set_xlabel("Latent Dimension 1")
    axs[1].set_ylabel("Latent Dimension 2")
    axs[1].set_title("Vector Field of Learned dynamics (Least Sq.)")
    axs[1].axis("equal")
    plt.suptitle("Dynamics Comparison: True vs Learned (Least Squares)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "dynamics_comparison.png"))
    plt.close()

    # %% Compute KL
    n_samples = 10
    T_batch = 100
    online_kls = []
    offline_kls = []
    ls_kls = []

    for i in range(0, T - T_batch):
        x_t = rollout["env_state"][:, i : i + T_batch].to(config.device)
        y_t = rollout["next_obs"][:, i : i + T_batch].to(config.device)
        u_t = rollout["action"][:, i : i + T_batch].to(config.device)

        mu_q = x_t + (env.env._get_dynamics(x_t) + u_t) * config.dt
        x_samples = (
            mu_q
            + torch.randn(n_samples, T_batch, 2)
            * torch.sqrt(torch.tensor(config.environment.env_noise_scale))
            * config.dt
        )
        start_states = x_samples
        # LS model
        pred = start_states
        mu_p, _ = ls_dynamics.compute_param(pred)
        var_p = torch.tensor(config.environment.env_noise_scale) * config.dt**2
        mu_p = pred + (mu_p + u_t) * config.dt

        kl_d = 0.5 * (((mu_q - mu_p) ** 2) / var_p)
        ls_kls.append(to_np(torch.sum(kl_d, (-1, -2)).mean(dim=0)) / T_batch)

        # Online model
        elbo, ll, kl = experiment.agent.model_env.model.compute_elbo(
            y=y_t, u=u_t, n_samples=n_samples, k_steps=1
        )
        online_kls.append(to_np(kl) / T_batch)

        # Offline model
        elbo, ll, kl = offline_experiment.agent.model_env.model.compute_elbo(
            y=y_t, u=u_t, n_samples=n_samples, k_steps=1
        )
        offline_kls.append(to_np(kl) / T_batch)

    # Visualize kld
    plt.plot(
        np.arange(len(online_kls)),
        online_kls,
        label="Online Model",
    )
    plt.plot(
        np.arange(len(online_kls)),
        offline_kls,
        label="Offline Model",
    )
    plt.plot(
        np.arange(len(online_kls)),
        ls_kls,
        label="Least Squares Model",
    )
    plt.xlabel("Time Step")
    plt.ylabel("KL Divergence")
    plt.legend(loc="upper right")
    plt.title("KL Divergence Comparison")
    plt.show()
