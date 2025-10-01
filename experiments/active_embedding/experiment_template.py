#!/usr/bin/env python3
"""
Template for Hydra-based actdyn experiments.

This template shows how to create new experiments using the general Hydra integration
utilities from actdyn.utils. Copy this file and modify for your specific experiment.
"""

import torch
import numpy as np
from actdyn.config import ExperimentConfig
from actdyn.utils import hydra_experiment, setup_experiment


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

    # Set random seeds for reproducibility
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    # Configure device
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    # Setup experiment using actdyn infrastructure
    experiment, agent, env, model_env = setup_experiment(exp_config)

    # TODO: Add your custom experiment logic here
    # You can:
    # 1. Modify the experiment before running
    # 2. Run custom training loops
    # 3. Add custom metrics or logging
    # 4. Implement custom analysis

    # Run the experiment
    experiment.run()

    # TODO: Add post-experiment analysis here

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
