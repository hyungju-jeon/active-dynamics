"""Environment factories for actdyn.

Only canonical, explicitly-registered environment keys are supported.
Unknown keys raise ImportError (no implicit fallback).
"""

from __future__ import annotations

import importlib
from typing import Type

import gymnasium as gym

from .base import BaseAction, BaseObservation
from .env_wrapper import EnvWrapper
from .vectorfield import (
    build_system_jacobians,
    jacobian_param_torch,
    jacobian_state_torch,
    residual_np,
    residual_torch,
    rollout_no_input,
    step_np,
)

__all__ = [
    "environment_from_str",
    "observation_from_str",
    "action_from_str",
    "EnvWrapper",
    "residual_torch",
    "residual_np",
    "jacobian_state_torch",
    "jacobian_param_torch",
    "build_system_jacobians",
    "rollout_no_input",
    "step_np",
]

_environment_map = {
    "vectorfield": (".vectorfield", "VectorFieldEnv"),
    "windfield": (".windfield", "WindField"),
}

_observation_map = {
    "identity": (".observation", "IdentityObservation"),
    "linear": (".observation", "LinearObservation"),
    "log-linear": (".observation", "LogLinearObservation"),
    "non-linear": (".observation", "NonlinearObservation"),
}

_action_map = {
    "identity": (".action", "IdentityActionEncoder"),
    "linear": (".action", "LinearActionEncoder"),
    "mlp": (".action", "MlpActionEncoder"),
}


def _resolve(map_table: dict[str, tuple[str, str]], key: str):
    if key not in map_table:
        raise ImportError(f"Unknown key: {key}. Available: {sorted(map_table.keys())}")
    module_name, class_name = map_table[key]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def environment_from_str(env_str: str) -> Type[gym.Env]:
    """Return an environment class from a canonical key."""
    return _resolve(_environment_map, env_str)


def observation_from_str(obs_str: str) -> type[BaseObservation]:
    """Return an observation model class from a canonical key."""
    return _resolve(_observation_map, obs_str)


def action_from_str(act_str: str) -> type[BaseAction]:
    """Return an action model class from a canonical key."""
    return _resolve(_action_map, act_str)
