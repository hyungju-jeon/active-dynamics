from typing import Type, Union
from .base import BaseDynamicsEnv, BaseObservation, BaseAction
from .env_wrapper import GymObservationWrapper
import importlib

__all__ = [
    "environment_from_str",
    "observation_from_str",
    "action_from_str",
    "GymObservationWrapper",
]

_environment_map = {
    "vectorfield": (".vectorfield", "VectorFieldEnv"),
}

_observation_map = {
    "identity": (".observation", "IdentityObservation"),
    "linear": (".observation", "LinearObservation"),
    "loglinear": (".observation", "LogLinearObservation"),
    "nonlinear": (".observation", "NonlinearObservation"),
}

_action_map = {
    "identity": (".action", "IdentityActionEncoder"),
    "linear": (".action", "LinearActionEncoder"),
    "mlp": (".action", "MlpActionEncoder"),
}


def environment_from_str(env_str: str) -> Union[Type[BaseDynamicsEnv], str]:
    """
    Dynamically import and return the environment class based on the string key.

    Args:
        env_str: String identifier for the environment

    Returns:
        Either an environment class or a string for gymnasium environments
    """
    if env_str not in _environment_map:
        print(f"Unknown environment: {env_str}. Using gymnasium environment instead.")
        module = importlib.import_module("gymnasium")
        return env_str

    module_name, class_name = _environment_map[env_str]
    if module_name == "gymnasium":
        return env_str
    else:
        module = importlib.import_module(module_name, __package__)
        return getattr(module, class_name)


def observation_from_str(obs_str: str) -> type[BaseObservation]:
    """Dynamically import and return the observation model class based on the string key."""
    if obs_str not in _observation_map:
        raise ImportError(
            f"Unknown observation model: {obs_str}. Available: {list(_observation_map.keys())}"
        )
    module_name, class_name = _observation_map[obs_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def action_from_str(act_str: str) -> type[BaseAction]:
    """Dynamically import and return the observation model class based on the string key."""
    if act_str not in _action_map:
        raise ImportError(
            f"Unknown action model: {act_str}. Available: {list(_action_map.keys())}"
        )
    module_name, class_name = _action_map[act_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)
