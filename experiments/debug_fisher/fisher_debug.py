# %%
import os
from pathlib import Path
import numpy as np
import torch
from actdyn.config import ExperimentConfig
from actdyn.utils.helpers import (
    setup_environment,
    setup_model,
    setup_metric,
    setup_policy,
)
from actdyn.models.model_wrapper import VAEWrapper
from actdyn.core.agent import Agent
from actdyn.core.experiment import Experiment
import matplotlib.pyplot as plt
from actdyn.utils.visualize import compute_fisher_map


def plot_fisher(agent, rollout, elite_sample):
    model_traj = torch.stack(rollout["model_state"])
    next_traj = torch.stack(rollout["next_model_state"])
    action = torch.stack(rollout["action"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Compute Fisher map
    compute_fisher_map(
        agent.policy.metric, device=agent.device, show_plot=True, ax=axs[0]
    )
    plt.plot(
        model_traj[:, 0].cpu(),
        model_traj[:, 1].cpu(),
        color="white",
        linewidth=2,
        label="Model Trajectory",
    )
    plt.scatter(
        model_traj[-1, 0].cpu(),
        model_traj[-1, 1].cpu(),
        color="white",
        s=100,
        label="end point",
    )
    plt.scatter(
        next_traj[-1, 0].cpu(),
        next_traj[-1, 1].cpu(),
        color="black",
        s=100,
        label="Expected point",
    )
    for traj in elite_sample:
        plt.plot(
            traj["model_state"][:, 0].cpu(),
            traj["model_state"][:, 1].cpu(),
            color="cyan",
            linewidth=0.5,
        )

    if agent.policy.metric.I is not None:
        X_BOUND = 2.5
        rbf_grid_x = torch.linspace(-X_BOUND, X_BOUND, 25)
        rbf_grid_y = torch.linspace(-X_BOUND, X_BOUND, 25)
        rbf_xx, rbf_yy = torch.meshgrid(rbf_grid_x, rbf_grid_y, indexing="xy")  # [H, W]
        plt.sca(axs[1])
        plt.contourf(
            rbf_xx.cpu(),
            rbf_yy.cpu(),
            agent.policy.metric.I[0, 0, :625].view(25, 25).cpu()
            + agent.policy.metric.I[0, 0, 625:].view(25, 25).cpu(),
            levels=10,
            cmap="plasma",
        )
        plt.colorbar(label="Fisher Information")
        plt.title("Fisher Information Map")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(
            model_traj[:, 0].cpu(),
            model_traj[:, 1].cpu(),
            color="white",
            linewidth=2,
            label="Model Trajectory",
        )
        plt.scatter(
            model_traj[-1, 0].cpu(),
            model_traj[-1, 1].cpu(),
            color="red",
            s=100,
            label="end point",
        )
        plt.scatter(
            model_traj[-2, 0].cpu() + action[-1, 0].cpu(),
            model_traj[-2, 1].cpu() + action[-1, 1].cpu(),
            color="white",
            s=100,
            label="Expected point",
        )
    plt.show()


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config_fisher_debug.yaml")
    exp_config = ExperimentConfig.from_yaml(config_path)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    # Create results directory
    results_dir = Path(exp_config.results_dir)
    results_dir.mkdir(exist_ok=True)
    for subdir in ["videos", "models", "buffers", "logs"]:
        (results_dir / subdir).mkdir(exist_ok=True)

    X_BOUND = 5
    rbf_grid_x = torch.linspace(-X_BOUND, X_BOUND, 25)
    rbf_grid_y = torch.linspace(-X_BOUND, X_BOUND, 25)
    rbf_xx, rbf_yy = torch.meshgrid(rbf_grid_x, rbf_grid_y, indexing="xy")  # [H, W]
    rbf_grid_pts = torch.stack([rbf_xx.flatten(), rbf_yy.flatten()], dim=1)

    # Set up experiment
    # experiment, agent, env, model_env = setup_experiment(exp_config)
    env = setup_environment(exp_config)
    model = setup_model(exp_config)
    model.decoder.mapping.network.weight.data = env.obs_model.network.weight.data
    model.decoder.mapping.network.bias.data = env.obs_model.network.bias.data
    model.decoder.mapping.network.bias.requires_grad = False
    model.decoder.mapping.network.weight.requires_grad = False

    metric = setup_metric(exp_config, model)
    policy = setup_policy(exp_config, env, model, metric)
    policy.metric.discount_factor = 1
    model_env = VAEWrapper(
        model, env.observation_space, env.action_space, device=exp_config.device
    )
    model_env.model.dynamics.set_centers(rbf_grid_pts)
    agent = Agent(env, model_env, policy, device=exp_config.device)
    experiment = Experiment(agent, exp_config)

    # Run the experiment
    agent.reset()
    env_step = 0
    rollout = experiment.rollout
    while env_step < 20:
        with torch.no_grad():
            # 1. Plan
            action, elite_sample, _, _, costs = agent.plan()
            # 2. Execute
            transition, done = agent.step(action)
        rollout.add(**transition)
        # plot fisher gain at each point and current fisher
        if env_step > 0:
            plot_fisher(agent, rollout, elite_sample)

        agent.policy.metric.update_fim(transition)
        env_step += 1
