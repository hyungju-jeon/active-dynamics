# %%
import os
import numpy as np
import torch
from actdyn.config import ExperimentConfig
from actdyn.utils.helpers import setup_experiment


if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__), "config_fisher_comparison.yaml"
    )
    exp_config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.dirname(__file__)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    # # ---------------------------
    # # 1st order Fisher
    # # ---------------------------
    # exp_config.results_dir = os.path.join(results_dir, "results", "1st_fisher")
    # exp_config.metric.met_covariance = "1st"
    # exp_config.metric.met_sensitivity = True
    # experiment, agent, env, model_env = setup_experiment(exp_config)
    # experiment.run()
    # # ---------------------------
    # # 1st order - simple Fisher
    # # ---------------------------
    # exp_config.results_dir = os.path.join(results_dir, "results", "1st_simple_fisher")
    # exp_config.metric.met_covariance = "1st"
    # exp_config.metric.met_sensitivity = False
    # experiment, agent, env, model_env = setup_experiment(exp_config)
    # experiment.run()

    # ---------------------------
    # invariant Fisher
    # ---------------------------
    exp_config.results_dir = os.path.join(results_dir, "results", "invariant_fisher")
    exp_config.metric.met_covariance = "invariant"
    exp_config.metric.met_sensitivity = True
    experiment, agent, env, model_env = setup_experiment(exp_config)
    experiment.run()

    # ---------------------------
    # invariant - simple Fisher
    # ---------------------------
    config_path = os.path.join(
        os.path.dirname(__file__), "config_fisher_comparison.yaml"
    )
    exp_config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.dirname(__file__)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    exp_config.results_dir = os.path.join(
        results_dir, "results", "invariant_simple_fisher"
    )
    exp_config.metric.met_covariance = "invariant"
    exp_config.metric.met_sensitivity = False
    experiment, agent, env, model_env = setup_experiment(exp_config)
    experiment.run()

    # ---------------------------
    # deterministic Fisher
    # ---------------------------
    config_path = os.path.join(
        os.path.dirname(__file__), "config_fisher_comparison.yaml"
    )
    exp_config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.dirname(__file__)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    exp_config.results_dir = os.path.join(
        results_dir, "results", "deterministic_fisher"
    )
    exp_config.metric.met_covariance = "deterministic"
    exp_config.metric.met_sensitivity = True
    experiment, agent, env, model_env = setup_experiment(exp_config)
    experiment.run()

    # ---------------------------
    # deterministic - simple Fisher
    # ---------------------------
    config_path = os.path.join(
        os.path.dirname(__file__), "config_fisher_comparison.yaml"
    )
    exp_config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.dirname(__file__)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    exp_config.results_dir = os.path.join(
        results_dir, "results", "deterministic_simple_isher"
    )
    exp_config.metric.met_covariance = "deterministic"
    exp_config.metric.met_sensitivity = False
    experiment, agent, env, model_env = setup_experiment(exp_config)
    experiment.run()

    # ---------------------------
    # Flex
    # ---------------------------
    config_path = os.path.join(
        os.path.dirname(__file__), "config_fisher_comparison.yaml"
    )
    exp_config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.dirname(__file__)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    exp_config.results_dir = os.path.join(results_dir, "results", "Flex_fisher")
    exp_config.metric.met_covariance = "invariant"
    exp_config.metric.met_sensitivity = True
    exp_config.policy.mpc_horizon = 2
    exp_config.metric.met_discount_factor = 1
    experiment, agent, env, model_env = setup_experiment(exp_config)
    experiment.run()

    # ---------------------------
    # invariant Fisher + k_steps
    # ---------------------------
    config_path = os.path.join(
        os.path.dirname(__file__), "config_fisher_comparison.yaml"
    )
    exp_config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.dirname(__file__)

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    exp_config.results_dir = os.path.join(results_dir, "results", "k_steps_fisher")
    exp_config.metric.met_covariance = "invariant"
    exp_config.metric.met_sensitivity = True
    exp_config.training.k_steps = 5
    exp_config.policy.mpc_horizon = 10
    exp_config.metric.met_discount_factor = 0.99
    experiment, agent, env, model_env = setup_experiment(exp_config)
    experiment.run()
