# %%
import os
import argparse
import numpy as np
import torch
from actdyn.config import ExperimentConfig
from actdyn.utils.helpers import setup_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run Active Dynamics Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default results directory (./results)
  python run_experiment.py -c /path/to/config.yaml
  
  # Run with custom results directory
  python run_experiment.py --config /path/to/config.yaml --results-dir /path/to/results
        """,
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results (default: ./results)",
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    try:
        exp_config = ExperimentConfig.from_yaml(args.config_path)
    except Exception as e:
        raise ValueError(f"Failed to load config from {args.config_path}: {e}")

    # Set results directory
    config_name = os.path.splitext(os.path.basename(args.config_path))[0].replace(
        "config_", ""
    )
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../results", config_name)
        )
    exp_config.results_dir = results_dir

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    print(f"Loading experiment configuration from: {args.config_path}")
    print(f"Results will be saved to: {exp_config.results_dir}")
    print(f"Using device: {exp_config.device}")
    print("-" * 40)
    print("\nStarting experiment...\n")
    print("-" * 40)

    # Set up experiment components
    experiment, agent, env, model_env = setup_experiment(exp_config)

    # Run the experiment using the Experiment class's run method
    experiment.run()
    print(f"Experiment completed. Results saved to {exp_config.results_dir}")


if __name__ == "__main__":
    main()
