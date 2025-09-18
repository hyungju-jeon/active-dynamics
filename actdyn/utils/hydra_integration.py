"""
General Hydra integration utilities for actdyn experiments.
This module provides reusable Hydra integration components that can be used across different experiments.
"""

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from typing import Callable, Any
import os

from actdyn.config import ExperimentConfig


def list_to_str(lst):
    """Convert a list to a string representation suitable for filenames."""
    if isinstance(lst, list):
        return "x".join([str(i) for i in lst])
    elif isinstance(lst, str):
        return "x".join([str(i) for i in eval(lst)])


OmegaConf.register_new_resolver("list_str", lambda x: list_to_str(x))


def str_to_list(s: str) -> list:
    """Convert a string representation of a list back to a list."""
    try:
        return eval(s)
    except Exception as e:
        raise ValueError(f"Could not convert string to list: {s}") from e


def register_actdyn_configs():
    """Register actdyn dataclass configs with Hydra's ConfigStore for better type safety."""
    cs = ConfigStore.instance()

    # Register the main config
    cs.store(name="base_config", node=ExperimentConfig)

    # Register sub-configs for compositional configuration
    from actdyn.config import (
        EnvironmentConfig,
        ModelConfig,
        PolicyConfig,
        MetricConfig,
        TrainingConfig,
        LoggingConfig,
    )

    cs.store(group="environment", name="base", node=EnvironmentConfig)
    cs.store(group="model", name="base", node=ModelConfig)
    cs.store(group="policy", name="base", node=PolicyConfig)
    cs.store(group="metric", name="base", node=MetricConfig)
    cs.store(group="training", name="base", node=TrainingConfig)
    cs.store(group="logging", name="base", node=LoggingConfig)


class HydraExperimentConfig:
    """Utilities for converting between Hydra and actdyn ExperimentConfig."""

    @staticmethod
    def from_hydra_dict(cfg: DictConfig) -> ExperimentConfig:
        """
        Convert Hydra DictConfig to ExperimentConfig dataclass.
        """
        # Convert to regular dict with resolved interpolations
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Ensure we have a dict
        if not isinstance(config_dict, dict):
            raise ValueError("Expected config to be a dictionary")

        # Filter out Hydra-specific sections
        hydra_keys = {"defaults", "hydra"}
        config_dict = {k: v for k, v in config_dict.items() if k not in hydra_keys}

        # Use the same logic as ExperimentConfig.from_yaml but directly from dict
        from actdyn.config import (
            EnvironmentConfig,
            ModelConfig,
            PolicyConfig,
            MetricConfig,
            TrainingConfig,
            LoggingConfig,
        )

        # Convert nested dictionaries to their respective config classes
        if "environment" in config_dict and isinstance(config_dict["environment"], dict):
            config_dict["environment"] = EnvironmentConfig(**config_dict["environment"])
        if "model" in config_dict and isinstance(config_dict["model"], dict):
            # Manually parse string-represented lists from Hydra sweeps
            if "dyn_hidden_dim" in config_dict["model"] and isinstance(
                config_dict["model"]["dyn_hidden_dim"], str
            ):
                config_dict["model"]["dyn_hidden_dim"] = str_to_list(
                    config_dict["model"]["dyn_hidden_dim"]
                )
            if "enc_hidden_dim" in config_dict["model"] and isinstance(
                config_dict["model"]["enc_hidden_dim"], str
            ):
                config_dict["model"]["enc_hidden_dim"] = str_to_list(
                    config_dict["model"]["enc_hidden_dim"]
                )
            if "act_hidden_dim" in config_dict["model"] and isinstance(
                config_dict["model"]["act_hidden_dim"], str
            ):
                config_dict["model"]["act_hidden_dim"] = str_to_list(
                    config_dict["model"]["act_hidden_dim"]
                )
            if "map_hidden_dim" in config_dict["model"] and isinstance(
                config_dict["model"]["map_hidden_dim"], str
            ):
                config_dict["model"]["map_hidden_dim"] = str_to_list(
                    config_dict["model"]["map_hidden_dim"]
                )
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "policy" in config_dict and isinstance(config_dict["policy"], dict):
            config_dict["policy"] = PolicyConfig(**config_dict["policy"])
        if "metric" in config_dict and isinstance(config_dict["metric"], dict):
            config_dict["metric"] = MetricConfig(**config_dict["metric"])
        if "training" in config_dict and isinstance(config_dict["training"], dict):
            config_dict["training"] = TrainingConfig(**config_dict["training"])
        if "logging" in config_dict and isinstance(config_dict["logging"], dict):
            config_dict["logging"] = LoggingConfig(**config_dict["logging"])

        return ExperimentConfig(**config_dict)  # type: ignore

    @staticmethod
    def to_hydra_config(exp_config: ExperimentConfig) -> DictConfig:
        """Convert ExperimentConfig back to Hydra DictConfig."""
        return OmegaConf.structured(exp_config)


def hydra_experiment(config_path: str = "conf", config_name: str | None = None):
    """
    Decorator that handles Hydra setup and config conversion for actdyn experiments.
    Args:
        config_path: Path to Hydra configuration directory
        config_name: Name of the configuration file (without .yaml). If None,
                    allows command-line --config-name to be used.
    """

    def decorator(func: Callable[[ExperimentConfig], Any]):
        # Resolve config_path relative to the current working directory
        resolved_config_path = (
            os.path.abspath(config_path) if not os.path.isabs(config_path) else config_path
        )
        if not os.path.exists(resolved_config_path):
            raise FileNotFoundError(f"Config directory not found: {resolved_config_path}")

        @hydra.main(version_base=None, config_path=resolved_config_path, config_name=config_name)
        def wrapper(cfg: DictConfig):
            exp_config = setup_hydra_experiment(cfg)
            return func(exp_config)

        return wrapper

    return decorator


def setup_hydra_experiment(cfg: DictConfig) -> ExperimentConfig:
    """
    Set up an experiment from Hydra config.
    """
    exp_config = HydraExperimentConfig.from_hydra_dict(cfg)
    exp_config.results_dir = HydraConfig.get().runtime.output_dir
    return exp_config
