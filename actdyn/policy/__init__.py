from .base import BasePolicy
import importlib

__all__ = ["policy_factory"]

_policy_map = {
    "mpc-icem": (".icem", "MpcICem"),
    "random": (".random", "RandomPolicy"),
    "lazy": (".lazy", "LazyPolicy"),
    # Add more mappings as needed
}


def policy_factory(policy_str: str) -> type[BasePolicy]:
    """
    Dynamically import and return the policy class based on the string key.
    Example: policy_factory('mpc-icem')
    """
    if policy_str not in _policy_map:
        raise ImportError(
            f"Unknown policy: {policy_str}. Available: {list(_policy_map.keys())}"
        )
    module_name, class_name = _policy_map[policy_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)
