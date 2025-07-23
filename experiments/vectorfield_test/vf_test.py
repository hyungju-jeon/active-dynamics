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


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config_vf_test.yaml")
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

    X_BOUND = 2.5
    rbf_grid_x = torch.linspace(-X_BOUND, X_BOUND, 25)
    rbf_grid_y = torch.linspace(-X_BOUND, X_BOUND, 25)
    rbf_xx, rbf_yy = torch.meshgrid(rbf_grid_x, rbf_grid_y, indexing="ij")  # [H, W]
    rbf_grid_pts = torch.stack([rbf_xx.flatten(), rbf_yy.flatten()], dim=1)

    # Set up experiment
    # experiment, agent, env, model_env = setup_experiment(exp_config)
    env = setup_environment(exp_config)
    model = setup_model(exp_config)
    model.decoder.mapping.network[0].weight.data = env.obs_model.network[0].weight.data
    model.decoder.mapping.network[0].bias.data = env.obs_model.network[0].bias.data

    metric = setup_metric(exp_config, model)
    policy = setup_policy(exp_config, env, model, metric)
    model_env = VAEWrapper(
        model, env.observation_space, env.action_space, device=exp_config.device
    )
    model_env.model.dynamics.set_centers(rbf_grid_pts)
    agent = Agent(env, model_env, policy, device=exp_config.device)
    experiment = Experiment(agent, exp_config)

    # Run the experiment using the Experiment class's run method
    experiment.run()
    print(f"Experiment completed. Results saved to {results_dir}")

    experiment.rollout.finalize()
    env.env.render()
