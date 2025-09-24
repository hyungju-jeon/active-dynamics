# %%
"""
Run Experiment (general) for Active Dynamics.

This script runs experiments using Hydra for configuration management.
"""

import torch
import numpy as np
from actdyn.config import ExperimentConfig
from actdyn.utils import hydra_experiment, setup_experiment
from omegaconf import OmegaConf


@hydra_experiment(config_path="conf")
def run_hydra_experiment(exp_config: ExperimentConfig) -> None:
    """
    Run a hydra experiment.
    """

    print("=" * 60)
    print("RUN EXPERIMENT")
    print("=" * 60)
    print("Experiment Configuration:")
    print(f"  Seed: {exp_config.seed}")
    print(f"  Device: {exp_config.device}")
    print(f"  Environment: {exp_config.environment.environment_type}")
    print(f"  Model: {exp_config.model.model_type}")
    print(f"  Policy: {exp_config.policy.policy_type}")
    print(f"  Total Steps: {exp_config.training.total_steps}")
    print(f"  Results Directory: {exp_config.results_dir}")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    # Configure device
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        print(f"Using device: {exp_config.device}")

    # Setup and run experiment using existing actdyn infrastructure
    if exp_config.run_online:
        print("\nSetting up experiment components...")
        experiment, _, _, _ = setup_experiment(exp_config)

        print("Starting experiment run...")
        experiment.run()

        print("\nExperiment completed successfully!")
        print(f"Results saved to: {exp_config.results_dir}")
        print("=" * 60)

    if exp_config.run_analysis:
        print("Performing post-experiment analysis...")
        experiment.post_run()

    if exp_config.run_offline:
        print("Performing offline learning...")
        torch.manual_seed(exp_config.seed)
        np.random.seed(exp_config.seed)
        offline_experiment, _, _, _ = setup_experiment(exp_config)
        offline_experiment.offline_learning()


if __name__ == "__main__":
    # Optional: Register configs for better IDE support and type validation
    try:
        from actdyn.utils import register_actdyn_configs

        register_actdyn_configs()
    except ImportError:
        pass  # Skip if hydra integration not available

    # Run the experiment
    run_hydra_experiment()
