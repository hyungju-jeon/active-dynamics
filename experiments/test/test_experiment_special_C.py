import os
import torch
import numpy as np
from pathlib import Path
import yaml
import gymnasium

from actdyn.core.agent import Agent
from actdyn.core.experiment import Experiment
from actdyn.config import (
    ExperimentConfig,
    EnvironmentConfig,
    ModelConfig,
    PolicyConfig,
    TrainingConfig,
    LoggingConfig,
)

from actdyn.environment import environment_factory
from actdyn.models import (
    model_from_str,
    encoder_from_str,
    mapping_from_str,
    dynamics_from_str,
    noise_from_str,
    Decoder,
)
from actdyn.policy import policy_factory
from actdyn.metrics import information_from_str
from actdyn.models.model_wrapper import VAEWrapper


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    # Extract env_type before creating the dataclass
    env_type = config_dict["environment"].pop("env_type")
    config_dict["environment"] = EnvironmentConfig(**config_dict["environment"])
    config_dict["environment_env_type"] = env_type  # Store separately

    # Extract extra model fields
    model_dict = config_dict["model"].copy()
    model_extras = {}
    for key in [
        "model_type",
        "encoder_type",
        "mapping_type",
        "noise_type",
        "dynamics_type",
        "observation_model",
        "C_matrix",
        "action_dim",
        "ensemble",
    ]:
        if key in model_dict:
            model_extras[key] = model_dict.pop(key)
    config_dict["model"] = ModelConfig(**model_dict)
    config_dict["model_extras"] = model_extras

    # Extract extra policy fields
    policy_dict = config_dict["policy"].copy()
    policy_type = policy_dict.pop("policy_type", None)
    policy_extras = {}
    if "action_dim" in policy_dict:
        policy_extras["action_dim"] = policy_dict.pop("action_dim")
    config_dict["policy"] = PolicyConfig(**policy_dict)
    config_dict["policy_type"] = policy_type
    config_dict["policy_extras"] = policy_extras

    # Extract extra training fields
    training_dict = config_dict["training"].copy()
    training_extras = {}
    for key in ["patience", "min_delta"]:
        if key in training_dict:
            training_extras[key] = training_dict.pop(key)
    config_dict["training"] = TrainingConfig(**training_dict)
    config_dict["training_extras"] = training_extras

    config_dict["logging"] = LoggingConfig(**config_dict["logging"])
    return config_dict


def setup_experiment(config_dict):
    device = torch.device(config_dict["device"])
    # Environment
    env_type = config_dict["environment_env_type"]
    env_cfg = config_dict["environment"]
    env_cls = environment_factory(env_type)
    env = env_cls(
        dynamics_type=env_cfg.dynamics_type,
        state_dim=env_cfg.dim,
        noise_scale=env_cfg.noise_scale,
        dt=env_cfg.dt,
        device=env_cfg.device,
    )

    # Model components
    model_cfg = config_dict["model"]
    model_extras = config_dict["model_extras"]
    encoder_cls = encoder_from_str(model_extras["encoder_type"])
    encoder = encoder_cls(
        input_dim=model_cfg.input_dim,
        hidden_dims=model_cfg.encoder_hidden_dims,
        latent_dim=model_cfg.latent_dim,
        device=config_dict["device"],
    )
    mapping_cls = mapping_from_str(model_extras["mapping_type"])
    mapping = mapping_cls(
        latent_dim=model_cfg.latent_dim,
        output_dim=model_cfg.latent_dim,
        device=config_dict["device"],
    )
    noise_cls = noise_from_str(model_extras["noise_type"])
    noise = noise_cls(output_dim=model_cfg.latent_dim, device=config_dict["device"])
    decoder = Decoder(mapping, noise, device=config_dict["device"])
    dynamics_cls = dynamics_from_str(model_extras["dynamics_type"])
    dynamics = dynamics_cls(
        state_dim=model_cfg.latent_dim,
        device=config_dict["device"],
    )
    model_cls = model_from_str(model_extras["model_type"])
    model = model_cls(
        dynamics=dynamics,
        encoder=encoder,
        decoder=decoder,
        device=config_dict["device"],
    )

    # Set action_space for policy compatibility
    if not isinstance(env.action_space, gymnasium.spaces.Box):
        raise ValueError(f"env.action_space must be Box, got {type(env.action_space)}")
    model.action_space = env.action_space
    print(f"[DEBUG] model.action_space type: {type(model.action_space)}")

    # Patch C matrix if specified
    if "C_matrix" in model_extras and model_extras["C_matrix"] is not None:
        C = torch.tensor(model_extras["C_matrix"], dtype=torch.float32)
        mapping.network[0].weight.data = C
        mapping.network[0].bias.data.zero_()

    # Policy
    policy_type = config_dict["policy_type"]
    policy_cfg = config_dict["policy"]
    policy_cls = policy_factory(policy_type)
    policy = policy_cls(
        cost_fn=None, model=model, icem_params=policy_cfg, mpc_params=policy_cfg
    )

    # Model environment
    model_env = VAEWrapper(
        model, env.observation_space, env.action_space, device=config_dict["device"]
    )

    # Agent
    agent = Agent(
        env, model_env, policy, action_encoder=None, device=config_dict["device"]
    )

    # Experiment
    experiment = Experiment(agent, vars(config_dict["training"]))

    return experiment, model, decoder


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config_special_C.yaml")
    config_dict = load_config(config_path)

    # Set random seeds
    torch.manual_seed(config_dict["seed"])
    np.random.seed(config_dict["seed"])

    # Create results directory
    results_dir = Path(config_dict["results_dir"])
    results_dir.mkdir(exist_ok=True)
    for subdir in ["videos", "models", "buffers", "logs"]:
        (results_dir / subdir).mkdir(exist_ok=True)

    # Set up experiment
    experiment, model, decoder = setup_experiment(config_dict)

    # Fisher Information Metric
    fim_metric = information_from_str(
        "fisher", dynamics=model.dynamics, decoder=decoder, device=config_dict["device"]
    )

    # Run experiment
    best_loss = float("inf")
    patience = getattr(config_dict["training"], "patience", 5)
    patience_counter = 0
    min_delta = getattr(config_dict["training"], "min_delta", 1e-4)
    env_step = 0
    total_steps = config_dict["training"].total_steps
    rollout_horizon = config_dict["training"].rollout_horizon
    log_every = config_dict["logging"].log_every
    train_every = config_dict["training"].train_every
    save_every = config_dict["training"].save_every

    try:
        while env_step < total_steps:
            obs = experiment.agent.reset()
            done = False
            total_reward = 0
            for _ in range(rollout_horizon):
                action = experiment.agent.plan()
                obs, reward, done, env_info, model_info = experiment.agent.step(action)
                total_reward += reward
                if done:
                    break
            # Log reward
            # (Assume experiment has a logger, or just print for now)
            print(f"Step {env_step}: reward={total_reward}")

            # Compute and log Fisher Information
            if env_step % log_every == 0:
                # You may need to adapt this to your rollout structure
                # Here, just use model parameters as a placeholder
                # (In practice, pass a real rollout to fim_metric)
                print("[INFO] Computing Fisher Information (placeholder call)")
                # fi = fim_metric.compute(rollout)
                # print(f"Fisher Information: {fi}")

            # Train model
            if env_step % train_every == 0:
                loss = experiment.agent.train_model()
                print(f"Training loss: {loss}")
                if loss < best_loss - min_delta:
                    best_loss = loss
                    patience_counter = 0
                    torch.save(
                        model.state_dict(), results_dir / "models" / "best_model.pt"
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at step {env_step}")
                        break

            # Save checkpoint
            if env_step % save_every == 0:
                torch.save(
                    model.state_dict(),
                    results_dir / "models" / f"checkpoint_{env_step}.pt",
                )

            env_step += rollout_horizon

    except KeyboardInterrupt:
        print("Experiment interrupted by user")
    print(f"Experiment completed. Results saved to {results_dir}")


if __name__ == "__main__":
    main()
