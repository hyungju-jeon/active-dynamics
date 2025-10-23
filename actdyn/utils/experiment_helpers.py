# %%
import torch
import gymnasium

from actdyn.metrics.uncertainty import EnsembleDisagreement
from actdyn.config import ExperimentConfig
from actdyn.environment import (
    environment_from_str,
    observation_from_str,
    action_from_str,
    GymObservationWrapper,
)
from actdyn.models import (
    encoder_from_str,
    dynamics_from_str,
    mapping_from_str,
    noise_from_str,
    model_from_str,
    Decoder,
    BaseModel,
    ModelWrapper,
)
from actdyn.models.base import BaseDynamicsEnsemble
from actdyn.models.model import SeqVae
from actdyn.policy import policy_from_str, BaseMPC
from actdyn.metrics import metric_from_str, FisherInformationMetric, CompositeMetric


def setup_environment(config: ExperimentConfig):
    """Setup the environment based on the configuration."""
    # Environment
    env_cls = environment_from_str(config.environment.environment_type)
    env_config = config.environment.get_environment_cfg()
    if isinstance(env_cls, str):
        # If the environment is a gymnasium environment, we need to create it with gymnasium.make
        base_env = gymnasium.make(env_cls)
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
        # Don't pass dt explicitly if it's already in env_config to avoid conflicts
        extra_params = {}
        if "dt" not in env_config:
            extra_params["dt"] = config.dt

        base_env = env_cls(
            **env_config,
            **extra_params,
            state_dim=config.latent_dim,
            device=config.device,
        )

    env_obs_dim = base_env.observation_space.shape[0]

    # Observation model
    obs_model_cls = observation_from_str(config.environment.observation_type)
    obs_config = config.environment.get_observation_cfg()
    observation_model = obs_model_cls(
        obs_dim=config.observation_dim,
        latent_dim=env_obs_dim,
        device=config.device,
        **obs_config,
    )

    # Action model
    action_model_cls = action_from_str(config.environment.action_type)

    action_config = config.environment.get_action_cfg()
    action_model = action_model_cls(
        **action_config,
        action_bounds=config.environment.env_action_bounds,
        action_dim=config.action_dim,
        latent_dim=config.latent_dim,
        device=config.device,
    )

    # Wrap the environment with GymObservationWrapper
    env = GymObservationWrapper(
        env=base_env,
        dt=config.environment.env_dt,
        obs_model=observation_model,
        action_model=action_model,
        device=config.device,
    )

    # Set action_space for policy compatibility
    if not isinstance(env.action_space, gymnasium.spaces.Box):
        raise ValueError(f"env.action_space must be Box, got {type(env.action_space)}")

    return env


def setup_model(config: ExperimentConfig) -> SeqVae | BaseModel:
    """Setup the model based on the configuration."""
    # Model components
    # Encoder module
    encoder_cls = encoder_from_str(config.model.encoder_type)
    enc_config = config.model.get_encoder_cfg()
    encoder = encoder_cls(
        obs_dim=config.observation_dim,
        action_dim=config.action_dim,
        latent_dim=config.latent_dim,
        device=config.device,
        **enc_config,
    )

    # Decoder module (mapping + noise)
    mapping_cls = mapping_from_str(config.model.mapping_type)
    map_config = config.model.get_decoder_cfg()
    mapping = mapping_cls(
        latent_dim=config.latent_dim,
        obs_dim=config.observation_dim,
        device=config.device,
        **map_config,
    )
    noise_cls = noise_from_str(config.model.noise_type)
    noise = noise_cls(obs_dim=config.observation_dim, device=config.device)
    decoder = Decoder(mapping, noise, device=config.device)

    # Dynamics module
    dynamics_cls = dynamics_from_str(config.model.dynamics_type)
    dyn_config = config.model.get_dynamics_cfg()
    dyn_config.update(
        {
            "state_dim": config.latent_dim,
            "device": config.device,
        }
    )
    if config.model.is_ensemble or ("ensemble_disagreement" in config.metric.metric_type):
        # If ensemble dynamics, we need to create multiple dynamics models
        ensemble_config = config.model.get_ensemble_cfg()
        dynamics = BaseDynamicsEnsemble(
            dynamics_cls=dynamics_cls, **ensemble_config, dynamics_config=dyn_config
        )
    else:
        dynamics = dynamics_cls(
            **dyn_config,
        )

    # Action model
    action_model_cls = action_from_str(config.model.action_type)

    action_config = config.model.get_action_cfg()
    action_model = action_model_cls(
        **action_config,
        action_dim=config.action_dim,
        action_bounds=config.environment.env_action_bounds,
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
    if isinstance(config.metric.metric_type, list):
        metric_config = config.metric.get_metric_cfg()
        metrics = []
        for metric_type in config.metric.metric_type:
            metric_cls = metric_from_str(metric_type)
            if issubclass(metric_cls, FisherInformationMetric):
                metrics.append(
                    metric_cls(
                        model=model,
                        device=config.device,
                        **metric_config,
                    )
                )
            elif issubclass(metric_cls, EnsembleDisagreement):
                metrics.append(
                    metric_cls(
                        ensemble_dynamics=model.dynamics,
                        device=config.device,
                        **metric_config,
                    )
                )
            else:
                metric = metric_cls(
                    device=config.device,
                    **metric_config,
                )
                metrics.append(metric)

        metric = CompositeMetric(
            metrics=metrics,
            compute_type=config.metric.compute_type,
            weights=config.metric.composite_weights,
            device=config.device,
        )
    else:
        metric_cls = metric_from_str(config.metric.metric_type)
        metric_config = config.metric.get_metric_cfg()
        if issubclass(metric_cls, FisherInformationMetric):
            metric = metric_cls(
                dynamics=model.dynamics,
                decoder=model.decoder,
                **metric_config,
                device=config.device,
            )
        elif issubclass(metric_cls, EnsembleDisagreement):
            metric = metric_cls(
                ensemble_dynamics=model.dynamics,
                device=config.device,
                **metric_config,
            )
        else:
            metric = metric_cls(
                device=config.device,
                **metric_config,
            )
    return metric


def setup_policy(config, env, model, metric):
    # Policy
    policy_cls = policy_from_str(config.policy.policy_type)
    # check type of policy_cls, BasePolicy or BaseMPC
    if issubclass(policy_cls, BaseMPC):
        mpc_config = config.policy.get_mpc_cfg()
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
    # Import here to avoid circular import
    from actdyn.core.agent import Agent
    from actdyn.core.experiment import Experiment

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
    model_env = ModelWrapper(model, env.observation_space, env.action_space, device=config.device)

    # Agent
    agent = Agent(
        env, model_env, policy, buffer_length=config.training.rollout_horizon, device=config.device
    )

    # Experiment
    experiment = Experiment(agent, config)

    return (
        experiment,
        agent,
        env,
        model_env,
    )
