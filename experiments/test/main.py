# %%
import os
import torch
import numpy as np
from pathlib import Path

from actdyn.core.agent import Agent
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.models.model import VAE, EnsembleVAE
from actdyn.policy.simple_icem import SimpleICem
from actdyn.core.experiment import Experiment
from actdyn.metrics.fisher import FisherInformation
from actdyn.config import ExperimentConfig


def setup_experiment(config: ExperimentConfig) -> Experiment:
    """Set up experiment components."""
    # Set device
    device = torch.device(config.device)

    # Create environment
    if config.environment.observation_wrapper:
        base_env = VectorFieldEnv(**config.environment.__dict__)
        obs_model = config.environment.observation_wrapper.obs_model
        env = config.environment.observation_wrapper.wrapper(base_env, obs_model)
    else:
        env = VectorFieldEnv(**config.environment.__dict__)

    # Create model
    if config.model.ensemble:
        model = EnsembleVAE(**config.model.__dict__)
    else:
        model = VAE(**config.model.__dict__)

    # Create policy
    policy = SimpleICem(**config.policy.__dict__)

    # Create agent
    agent = Agent(env, model, policy, device=config.device)

    # Create experiment
    experiment = Experiment(agent, config)

    return experiment


# %%
if __name__ == "__main__":
    # Load configuration
    config = ExperimentConfig.from_yaml("config.yaml")

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create results directory
    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)

    # Create subdirectories
    for subdir in ["videos", "models", "buffers", "logs"]:
        (results_dir / subdir).mkdir(exist_ok=True)

    # Set up experiment
    experiment = setup_experiment(config)

    # Initialize Fisher Information tracker
    fisher_info = FisherInformation(
        latent_dim=config.model.latent_dim,
        device=config.device,
    )

    # Initialize early stopping variables
    best_loss = float("inf")
    patience = config.training.patience
    patience_counter = 0
    min_delta = config.training.min_delta

    # Run experiment
    try:
        while experiment.env_step < config.training.total_steps:
            # Run one training iteration
            state = experiment.agent.reset()
            rollout, state = experiment.agent.simulate_rollout(
                initial_state=state,
                horizon=config.training.rollout_horizon,
                animate=experiment.env_step % config.training.animate_every == 0,
                video_path=str(
                    results_dir / "videos" / f"step_{experiment.env_step}.mp4"
                ),
            )

            # Log metrics
            experiment.logger.log("reward", sum(rollout.rewards))
            experiment.logger.log("step", experiment.env_step)

            # Compute and log Fisher Information
            if experiment.env_step % config.logging.log_every == 0:
                fi = fisher_info.compute(experiment.agent.model)
                experiment.logger.log("fisher_info", fi.mean().item())
                experiment.logger.log("fisher_info_std", fi.std().item())

            # Train model
            if experiment.env_step % config.training.train_every == 0:
                loss = experiment.agent.train_model()
                experiment.logger.log("training_loss", loss)

                # Early stopping check
                if loss < best_loss - min_delta:
                    best_loss = loss
                    patience_counter = 0
                    # Save best model
                    torch.save(
                        experiment.agent.model.state_dict(),
                        results_dir / "models" / "best_model.pt",
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at step {experiment.env_step}")
                        break

            # Save checkpoint
            if experiment.env_step % config.training.save_every == 0:
                checkpoint = {
                    "step": experiment.env_step,
                    "model_state": experiment.agent.model.state_dict(),
                    "optimizer_state": experiment.agent.model.optimizer.state_dict(),
                    "loss": loss,
                    "config": config,
                }
                torch.save(
                    checkpoint,
                    results_dir / "models" / f"checkpoint_{experiment.env_step}.pt",
                )

            experiment.env_step += len(rollout)

    except KeyboardInterrupt:
        print("Experiment interrupted by user")
    finally:
        # Save final results
        experiment.save_results()
        print(f"Experiment completed. Results saved to {results_dir}")
