# %%
import os
import torch
import numpy as np
from pathlib import Path
import gymnasium
from dataclasses import asdict

from actdyn.core.agent import Agent
from actdyn.core.experiment import Experiment
from actdyn.config import ExperimentConfig
from actdyn.environment import *
from actdyn.models import *
from actdyn.policy import *
from actdyn.metrics import *


def setup_experiment(config: ExperimentConfig):
    device = torch.device(config.device)

    # Observation model
    obs_model_cls = observation_from_str(config.environment.observation_type)
    observation_model = obs_model_cls(
        latent_dim=config.latent_dim,
        obs_dim=config.observation_dim,
        noise_type=config.environment.noise_type,
        noise_scale=config.environment.noise_scale,
        device=config.device,
    )

    # Action model
    action_model_cls = action_from_str(config.environment.action_type)
    action_model = action_model_cls(
        input_dim=config.action_dim,
        latent_dim=config.latent_dim,
        device=config.device,
    )

    # Environment
    env_cls = environment_from_str(config.environment.env_type)
    base_env = env_cls(**asdict(config.environment))

    env = GymObservationWrapper(
        env=base_env,
        obs_model=observation_model,
        action_model=action_model,
        device=config.device,
    )

    # Model components
    encoder_cls = encoder_from_str(config.model.encoder_type)
    encoder = encoder_cls(
        input_dim=config.model.observation_dim,
        hidden_dims=config.model.encoder_hidden_dims,
        latent_dim=config.model.latent_dim,
        device=config.device,
    )
    mapping_cls = mapping_from_str(config.model.mapping_type)
    mapping = mapping_cls(
        latent_dim=config.model.latent_dim,
        output_dim=config.model.observation_dim,
        device=config.device,
    )
    noise_cls = noise_from_str(config.model.noise_type)
    noise = noise_cls(output_dim=config.model.latent_dim, device=config.device)
    decoder = Decoder(mapping, noise, device=config.device)
    dynamics_cls = dynamics_from_str(config.model.dynamics_type)
    dynamics = dynamics_cls(
        state_dim=config.model.latent_dim,
        device=config.device,
    )
    model_cls = model_from_str(config.model.model_type)
    model = model_cls(
        dynamics=dynamics,
        encoder=encoder,
        decoder=decoder,
        device=config.device,
    )

    # Set action_space for policy compatibility
    if not isinstance(env.action_space, gymnasium.spaces.Box):
        raise ValueError(f"env.action_space must be Box, got {type(env.action_space)}")
    model.action_space = env.action_space
    print(f"[DEBUG] model.action_space type: {type(model.action_space)}")

    # Patch C matrix if specified
    if C_matrix is not None:
        C = torch.tensor(C_matrix, dtype=torch.float32)
        mapping.network[0].weight.data = C
        mapping.network[0].bias.data.zero_()

    # Policy
    policy_type = getattr(config.policy, "policy_type", "mpc-icem")
    policy_cls = policy_factory(policy_type)
    policy = policy_cls(
        cost_fn=None, model=model, icem_params=config.policy, mpc_params=config.policy
    )

    # Model environment
    model_env = VAEWrapper(
        model, env.observation_space, env.action_space, device=config.device
    )

    # Gym environment
    gym_env = GymObservationWrapper(
        env=env,
        obs_model=observation_model,
        action_model=action_model,
        device=config.device,
    )

    # Agent
    agent = Agent(gym_env, model_env, policy, device=config.device)

    # Experiment
    experiment = Experiment(agent, vars(config.training))

    return experiment, model, decoder


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config_vf_test.yaml")
    config = ExperimentConfig.from_yaml(config_path)

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create results directory
    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)
    for subdir in ["videos", "models", "buffers", "logs"]:
        (results_dir / subdir).mkdir(exist_ok=True)

    # Set up experiment
    experiment, model, decoder = setup_experiment(config)

    # Run the experiment using the Experiment class's run method
    experiment.run()
    print(f"Experiment completed. Results saved to {results_dir}")
