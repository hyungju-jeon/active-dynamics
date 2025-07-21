# %%
import torch
import gymnasium
from dataclasses import asdict

from actdyn.config import ExperimentConfig

from actdyn.environment import *
from actdyn.models import *
from actdyn.core import *
from actdyn.policy import *
from actdyn.metrics import *


def parse_subconfig(config, prefix):
    """Parse subconfig from the main config dictionary."""
    return {
        k.replace(prefix, "", 1): v
        for k, v in asdict(config).items()
        if k.startswith(prefix)
    }


def setup_environment(config: ExperimentConfig):
    """Setup the environment based on the configuration."""
    # Environment
    env_cls = environment_from_str(config.environment.environment_type)
    env_config = parse_subconfig(config.environment, "env_")
    if isinstance(env_cls, str):
        # If the environment is a gymnasium environment, we need to create it with gymnasium.make
        base_env = gymnasium.make(env_cls, config.environment.env_render_mode)
        # Set action bounds from gymnaisum environment
        config.environment.env_action_bounds = (
            base_env.action_space.low.tolist(),
            base_env.action_space.high.tolist(),
        )
        config.environment.env_state_bounds = (
            base_env.observation_space.low.tolist(),
            base_env.observation_space.high.tolist(),
        )
        config.latent_dim = base_env.observation_space.shape[0]

    else:
        base_env = env_cls(
            **env_config,
            state_dim=config.latent_dim,
            device=config.device,
        )

    # Observation model
    obs_model_cls = observation_from_str(config.environment.observation_type)
    obs_config = parse_subconfig(config.environment, "obs_")
    observation_model = obs_model_cls(
        obs_dim=config.observation_dim,
        latent_dim=config.latent_dim,
        noise_type=config.environment.noise_type,
        noise_scale=config.environment.noise_scale,
        device=config.device,
        **obs_config,
    )

    # Action model
    action_model_cls = action_from_str(config.environment.action_type)
    if config.environment.action_type == "identity":
        # For identity action, we can use the latent dimension as the action dimension
        config.environment.act_action_dim = config.latent_dim

    action_config = parse_subconfig(config.environment, "act_")
    action_model = action_model_cls(
        **action_config,
        action_dim=config.action_dim,
        latent_dim=config.latent_dim,
        device=config.device,
    )

    # Wrap the environment with GymObservationWrapper
    env = GymObservationWrapper(
        env=base_env,
        obs_model=observation_model,
        action_model=action_model,
        device=config.device,
    )

    # Set action_space for policy compatibility
    if not isinstance(env.action_space, gymnasium.spaces.Box):
        raise ValueError(f"env.action_space must be Box, got {type(env.action_space)}")

    return env


def setup_model(config: ExperimentConfig):
    """Setup the model based on the configuration."""
    # Model components
    # Encoder module
    encoder_cls = encoder_from_str(config.model.encoder_type)
    enc_config = parse_subconfig(config.model, "enc_")
    encoder = encoder_cls(
        input_dim=config.observation_dim,
        latent_dim=config.latent_dim,
        device=config.device,
        **enc_config,
    )

    # Decoder module (mapping + noise)
    mapping_cls = mapping_from_str(config.model.mapping_type)
    map_config = parse_subconfig(config.model, "map_")
    mapping = mapping_cls(
        latent_dim=config.latent_dim,
        output_dim=config.observation_dim,
        device=config.device,
        **map_config,
    )
    noise_cls = noise_from_str(config.model.noise_type)
    noise = noise_cls(output_dim=config.observation_dim, device=config.device)

    decoder = Decoder(mapping, noise, device=config.device)

    # Dynamics module
    dynamics_cls = dynamics_from_str(config.model.dynamics_type)
    dyn_config = parse_subconfig(config.model, "dyn_")
    dyn_config.update(
        {
            "state_dim": config.latent_dim,
            "device": config.device,
        }
    )
    if config.model.is_ensemble:
        # If ensemble dynamics, we need to create multiple dynamics models
        dynamics = EnsembleDynamics(
            dynamics_cls=dynamics_cls,
            n_models=config.model.n_models,
            dynamics_kwargs=dyn_config,
        )
    else:
        dynamics = dynamics_cls(
            **dyn_config,
        )

    # Action model
    action_model_cls = action_from_str(config.model.action_type)
    if config.model.action_type == "identity":
        # For identity action, we can use the latent dimension as the action dimension
        config.model.act_action_dim = config.latent_dim

    action_config = parse_subconfig(config.model, "act_")
    action_model = action_model_cls(
        **action_config,
        action_dim=config.action_dim,
        latent_dim=config.latent_dim,
        device=config.device,
    )

    # Model
    model_cls = model_from_str(config.model.model_type)
    model = model_cls(
        dynamics=dynamics,
        encoder=encoder,
        decoder=decoder,
        action_encoder=action_model,
        device=config.device,
    )

    return model


def setup_metric(config, model):
    # Metric
    metric_cls = metric_from_str(config.metric.metric_type)
    if isinstance(config.metric.metric_type, list):
        metrics = [
            metric_cls(
                compute_type=config.metric.compute_type,
                gamma=gamma,
                device=config.device,
            )
            for gamma in config.metric.gamma
        ]
        assert len(metrics) == len(config.metric.composite_weights)

        metric = CompositeMetric(
            metrics=metrics,
            weights=config.metric.composite_weights,
            device=config.device,
        )
    else:
        metric_config = parse_subconfig(config.metric, "met_")
        if issubclass(metric_cls, FisherInformationMetric):
            metric = metric_cls(
                dynamics=model.dynamics,
                decoder=model.decoder,
                compute_type=config.metric.compute_type,
                device=config.device,
                **metric_config,
            )
        else:
            metric = metric_cls(
                compute_type=config.metric.compute_type,
                device=config.device,
                **metric_config,
            )
    return metric


def setup_policy(config, env, model, metric):
    # Policy
    policy_cls = policy_from_str(config.policy.policy_type)
    # check type of policy_cls, BasePolicy or BaseMPC
    if issubclass(policy_cls, BaseMPC):
        mpc_config = parse_subconfig(config.policy, "mpc_")
        policy = policy_cls(
            metric=metric,
            model=model,
            device=config.device,
            **mpc_config,
        )
    else:
        policy = policy_cls(
            action_space=env.action_space,
            device=config.device,
        )
    return policy


def setup_experiment(config: ExperimentConfig):
    if not torch.cuda.is_available() and config.device == "cuda":
        print("CUDA is not available. Falling back to CPU.")
        config.device = "cpu"

    env = setup_environment(config)
    model = setup_model(config)
    # ------------------------------------------------------------------------
    metric = setup_metric(config, model)
    policy = setup_policy(config, env, model, metric)

    # ------------------------------------------------------------------------
    # Wrapper for the model environment
    # Model environment
    model_env = VAEWrapper(
        model, env.observation_space, env.action_space, device=config.device
    )

    # Agent
    agent = Agent(env, model_env, policy, device=config.device)

    # Experiment
    experiment = Experiment(agent, config)

    return (
        experiment,
        agent,
        env,
        model_env,
    )


if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig()
    experiment, agent, env, model_env = setup_experiment(config)
