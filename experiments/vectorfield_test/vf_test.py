# %%
import os
from pathlib import Path

import numpy as np
import torch

from actdyn.config import ExperimentConfig
from actdyn.utils.helpers import setup_experiment

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config_vf_test.yaml")
    exp_config = ExperimentConfig.from_yaml(config_path)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    # Create results directory
    results_dir = Path(exp_config.results_dir)
    results_dir.mkdir(exist_ok=True)
    for subdir in ["videos", "models", "buffers", "logs"]:
        (results_dir / subdir).mkdir(exist_ok=True)

    # Set up experiment
    experiment, agent, env, model_env = setup_experiment(exp_config)

    # Run the experiment using the Experiment class's run method
    experiment.run()
    print(f"Experiment completed. Results saved to {results_dir}")
