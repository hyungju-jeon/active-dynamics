#!/usr/bin/env python3
"""
Template for Hydra-based actdyn experiments.

This template shows how to create new experiments using the general Hydra integration
utilities from actdyn.utils. Copy this file and modify for your specific experiment.
"""

import os
import torch
import numpy as np
from actdyn.config import ExperimentConfig
from actdyn.utils import hydra_experiment, save_load, setup_experiment
from actdyn.utils.validation import compute_kstep_r2
from actdyn.utils.helper import to_np
import matplotlib.pyplot as plt


def list_to_str(lst):
    return "x".join([str(x) for x in lst])


@hydra_experiment(config_path="conf", config_name="config")
def run_experiment(exp_config: ExperimentConfig) -> None:
    """
    Main experiment function.

    This function receives a fully configured ExperimentConfig dataclass
    with all Hydra overrides applied and results directory set.

    Args:
        exp_config: ExperimentConfig dataclass with all experiment parameters
    """

    print("=" * 60)
    print("EXPERIMENT: [YOUR EXPERIMENT NAME HERE]")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Seed: {exp_config.seed}")
    print(f"  Device: {exp_config.device}")
    print(f"  Environment: {exp_config.environment.environment_type}")
    print(f"  Model: {exp_config.model.model_type}")
    print(f"  Policy: {exp_config.policy.policy_type}")
    print(f"  Total Steps: {exp_config.training.total_steps}")
    print(f"  Results Directory: {exp_config.results_dir}")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    # Set up experiment
    experiment, agent, env, model_env = setup_experiment(exp_config)
    if os.path.exists(os.path.join(experiment.results_path, "../train.pkl")):
        print("Training rollout already exists. Skipping rollout generation.")
    else:
        experiment.generate_rollout(
            50, 1000, rollout_dir=os.path.join(experiment.results_path, "../")
        )
    model = model_env.model

    # -------------------------------
    # Single Trajectory - Online
    # -------------------------------
    experiment.run()

    # Post-run on validation set
    # experiment.post_run()

    # k-step Prediction R2
    single_ro = save_load.load_and_concatenate_rollouts(
        os.path.join(experiment.results_path, "rollouts")
    ).to(device=exp_config.device)
    validation_ro = save_load.load_rollout(experiment.results_path + "/../validation.pkl").to(
        device=exp_config.device
    )

    # Compute latent space k-step prediction R2
    z_pred = model.encoder(validation_ro["next_obs"], validation_ro["action"])[1]
    fig_path = os.path.join(experiment.results_path, "online_kstep_r2_latent_validate.png")
    compute_kstep_r2(
        dynamics=model.dynamics,
        action_encoder=model.action_encoder,
        z=z_pred,
        u=validation_ro["action"],
        k_max=10,
        fig_path=fig_path,
    )

    z_pred = model.encoder(single_ro["next_obs"], single_ro["action"])[1]
    fig_path = os.path.join(experiment.results_path, "online_kstep_r2_latent_train.png")
    compute_kstep_r2(
        dynamics=model.dynamics,
        z=z_pred,
        u=single_ro["action"],
        y=z_pred,
        k_max=10,
        fig_path=fig_path,
    )

    # Compute observation space k-step prediction R2
    fig_path = os.path.join(experiment.results_path, "online_kstep_r2_observation_validate.png")
    compute_kstep_r2(model=model, rollout=validation_ro, k_max=10, fig_path=fig_path)

    fig_path = os.path.join(experiment.results_path, "online_kstep_r2_observation_train.png")
    compute_kstep_r2(model=model, rollout=single_ro, k_max=10, fig_path=fig_path)

    # -------------------------------
    # Single Trajectory - Offline
    # -------------------------------
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    offline_experiment, agent, _, _ = setup_experiment(exp_config)
    offline_experiment.offline_run()
    model = offline_experiment.agent.model.model

    # Compute latent space k-step prediction R2
    z_pred = model.encoder(validation_ro["next_obs"], validation_ro["action"])[1]
    fig_path = os.path.join(experiment.results_path, "offline_single_kstep_r2_latent_validate.png")
    compute_kstep_r2(
        dynamics=model.dynamics,
        action_encoder=model.action_encoder,
        z=z_pred,
        u=validation_ro["action"],
        k_max=10,
        fig_path=fig_path,
    )

    z_pred = model.encoder(single_ro["next_obs"], single_ro["action"])[1]
    fig_path = os.path.join(experiment.results_path, "offline_single_kstep_r2_latent_train.png")
    compute_kstep_r2(
        dynamics=model.dynamics,
        action_encoder=model.action_encoder,
        z=z_pred,
        u=single_ro["action"],
        k_max=10,
        fig_path=fig_path,
    )

    # Compute observation space k-step prediction R2
    fig_path = os.path.join(
        experiment.results_path, "offline_single_kstep_r2_observation_validate.png"
    )
    compute_kstep_r2(model=model, rollout=validation_ro, k_max=10, fig_path=fig_path)

    fig_path = os.path.join(
        experiment.results_path, "offline_single_kstep_r2_observation_train.png"
    )
    compute_kstep_r2(model=model, rollout=single_ro, k_max=10, fig_path=fig_path)

    # -------------------------------
    # Multiple Trajectories - Offline
    # -------------------------------
    multi_ro = save_load.load_rollout(experiment.results_path + "/../train.pkl").to(
        device=exp_config.device
    )
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    multi_offline_experiment, agent, _, _ = setup_experiment(exp_config)

    offline_cfg = multi_offline_experiment.cfg.training.get_offline_optim_cfg()

    # Perform offline learning
    loss = multi_offline_experiment.agent.model.train_model(multi_ro, **offline_cfg)
    model = multi_offline_experiment.agent.model.model

    # Compute latent space k-step prediction R2
    z_pred = model.encoder(validation_ro["next_obs"], validation_ro["action"])[1]
    fig_path = os.path.join(experiment.results_path, "offline_multi_kstep_r2_latent_validate.png")
    compute_kstep_r2(
        dynamics=model.dynamics,
        action_encoder=model.action_encoder,
        z=z_pred,
        u=validation_ro["action"],
        k_max=10,
        fig_path=fig_path,
    )

    z_pred = model.encoder(multi_ro["next_obs"], multi_ro["action"])[1]
    fig_path = os.path.join(experiment.results_path, "offline_multi_kstep_r2_latent_train.png")
    compute_kstep_r2(
        dynamics=model.dynamics,
        action_encoder=model.action_encoder,
        z=z_pred,
        u=multi_ro["action"],
        k_max=10,
        fig_path=fig_path,
    )
    # Compute observation space k-step prediction R2
    fig_path = os.path.join(
        experiment.results_path, "offline_multi_kstep_r2_observation_validate.png"
    )
    compute_kstep_r2(model=model, rollout=validation_ro, k_max=10, fig_path=fig_path)

    fig_path = os.path.join(experiment.results_path, "offline_multi_kstep_r2_observation_train.png")
    compute_kstep_r2(model=model, rollout=multi_ro, k_max=10, fig_path=fig_path)

    ro = validation_ro
    y = ro["next_obs"]
    u = ro["action"]
    # y = ro["obs"]
    x_on = model_env.model.encoder(y, u)[1]
    y_on = model_env.model.decoder(x_on)

    offline_model = offline_experiment.agent.model.model
    x_off = offline_model.encoder(y, u)[1]
    y_off = offline_model.decoder(x_off)

    multi_model = multi_offline_experiment.agent.model.model
    x_multi = multi_model.encoder(y, u)[1]
    y_multi = multi_model.decoder(x_multi)

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.flatten()
    y_labels = [
        r"$\cos(\phi)$",
        r"$\sin(\phi)$",
        r"$\dot{\phi}$",
        r"$\cos(\theta)$",
        r"$\sin(\theta)$",
        r"$\dot{\theta}$",
    ]

    for i in range(6):
        axs[i].plot(to_np(y[0, :, i]), alpha=0.7, label="y")
        axs[i].plot(to_np(y_on[0, :, i]), alpha=0.7, label="y_on")
        axs[i].plot(to_np(y_off[0, :, i]), alpha=0.7, label="y_off")
        axs[i].plot(to_np(y_multi[0, :, i]), alpha=0.7, label="y_multi")
        axs[i].legend()
        axs[i].set_title(y_labels[i])

    plt.tight_layout()
    plt.savefig(os.path.join(exp_config.results_dir, "reconstruction.png"))
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs = axs.flatten()

    phi_y = torch.atan2(y[0, :, 1], y[0, :, 0])
    phi_y_on = torch.atan2(y_on[0, :, 1], y_on[0, :, 0])
    phi_y_off = torch.atan2(y_off[0, :, 1], y_off[0, :, 0])
    phi_y_multi = torch.atan2(y_multi[0, :, 1], y_multi[0, :, 0])

    theta_y = torch.atan2(y[0, :, 4], y[0, :, 3])
    theta_y_on = torch.atan2(y_on[0, :, 4], y_on[0, :, 3])
    theta_y_off = torch.atan2(y_off[0, :, 4], y_off[0, :, 3])
    theta_y_multi = torch.atan2(y_multi[0, :, 4], y_multi[0, :, 3])

    axs[0].plot(np.unwrap(to_np(phi_y)), to_np(y[0, :, 2]), alpha=0.7, label="y")
    axs[0].plot(np.unwrap(to_np(phi_y_on)), to_np(y_on[0, :, 2]), alpha=0.7, label="y_on")
    axs[0].plot(np.unwrap(to_np(phi_y_off)), to_np(y_off[0, :, 2]), alpha=0.7, label="y_off")
    axs[0].plot(np.unwrap(to_np(phi_y_multi)), to_np(y_multi[0, :, 2]), alpha=0.7, label="y_multi")
    axs[0].legend()
    axs[0].set_xlabel(r"$\phi$")
    axs[0].set_ylabel(r"$\dot{\phi}$")
    axs[0].set_title(r"$\phi$")

    axs[1].plot(np.unwrap(to_np(theta_y)), to_np(y[0, :, 5]), alpha=0.7, label="y")
    axs[1].plot(np.unwrap(to_np(theta_y_on)), to_np(y_on[0, :, 5]), alpha=0.7, label="y_on")
    axs[1].plot(np.unwrap(to_np(theta_y_off)), to_np(y_off[0, :, 5]), alpha=0.7, label="y_off")
    axs[1].plot(
        np.unwrap(to_np(theta_y_multi)), to_np(y_multi[0, :, 5]), alpha=0.7, label="y_multi"
    )
    axs[1].legend()
    axs[1].set_xlabel(r"$\theta$")
    axs[1].set_ylabel(r"$\dot{\theta}$")
    axs[1].set_title(r"$\theta$")

    plt.tight_layout()
    plt.savefig(os.path.join(experiment.results_path, "phase_plot.png"))

    # Memory cleanup after experiment following the project pattern
    if "cuda" in str(exp_config.device):
        torch.cuda.empty_cache()

    print("\nExperiment completed successfully!")
    print(f"Results saved to: {exp_config.results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Optional: Register configs for better type validation
    try:
        from actdyn.utils import register_actdyn_configs

        register_actdyn_configs()
    except ImportError:
        pass

    # Run the experiment
    run_experiment()
