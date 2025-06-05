from .base import BaseDynamicsEnv
import importlib

__all__ = [
    "BaseObservation",
    "IdentityObservation",
    "LinearObservation",
    "LogLinearObservation",
    "NonlinearObservation",
]

_environment_map = {
    "vectorfield": (".vectorfield", "VectorFieldEnv"),
    # Add more mappings as needed
}


def environment_factory(env_str: str) -> type[BaseDynamicsEnv]:
    """
    Dynamically import and return the environment class based on the string key.
    Example: environment_factory('vectorfield')
    """
    if env_str not in _environment_map:
        raise ImportError(
            f"Unknown environment: {env_str}. Available: {list(_environment_map.keys())}"
        )
    module_name, class_name = _environment_map[env_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)
