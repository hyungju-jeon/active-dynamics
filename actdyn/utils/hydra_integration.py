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

from actdyn.config import (
    ExperimentConfig,
    EnvironmentConfig,
    ModelConfig,
    PolicyConfig,
    MetricConfig,
    TrainingConfig,
    LoggingConfig,
)


def list_to_str(lst):
    """Convert a list to a string representation suitable for filenames."""
    if isinstance(lst, list):
        return "x".join([str(i) for i in lst])
    elif isinstance(lst, str):
        try:
            # Handle string representation of a list
            return "x".join([str(i) for i in eval(lst)])
        except (SyntaxError, NameError):
            # Not a list-like string, return as is or handle appropriately
            return lst


def str_to_list(s: str) -> list:
    """Convert a string representation of a list back to a list."""
    try:
        return eval(s)
    except Exception as e:
        raise ValueError(f"Could not convert string to list: {s}") from e


OmegaConf.register_new_resolver("list_str", list_to_str)


def register_actdyn_configs():
    """Register actdyn dataclass configs with Hydra's ConfigStore for better type safety."""
    cs = ConfigStore.instance()

    # Register the main config
    cs.store(name="base_config", node=ExperimentConfig)

    # Register sub-configs for compositional configuration
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

        config_mapping = {
            "environment": EnvironmentConfig,
            "model": ModelConfig,
            "policy": PolicyConfig,
            "metric": MetricConfig,
            "training": TrainingConfig,
            "logging": LoggingConfig,
        }

        # Convert nested dictionaries to their respective config classes
        for key, config_class in config_mapping.items():
            if key in config_dict and isinstance(config_dict[key], dict):
                # Manually parse string-represented lists from Hydra sweeps
                for field, value in config_dict[key].items():
                    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                        try:
                            config_dict[key][field] = str_to_list(value)
                        except (ValueError, SyntaxError):
                            # Not a list-like string, so we leave it as is
                            pass
                config_dict[key] = config_class(**config_dict[key])

        exp_config = ExperimentConfig(**config_dict)  # type: ignore

        # Set env_dt from top-level dt for consistency with from_yaml
        if exp_config.environment:
            exp_config.environment.env_dt = exp_config.dt

        return exp_config

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
    exp_config.data_dir = os.path.join(HydraConfig.get().sweep.dir, "..")
    return exp_config
