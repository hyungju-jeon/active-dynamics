"""Policy package exports.

Expose policy base classes and a few built-in policies. Use :func:`policy_from_str`
to resolve policies by short names.
"""

from .base import BasePolicy, BaseMPC
from .policy import OffPolicy, RandomPolicy, StepPolicy

__all__ = ["policy_from_str", "RandomPolicy", "StepPolicy", "OffPolicy"]

import importlib

_policy_map = {
    "mpc-icem": (".mpc", "MpcICem"),
    "random": (".policy", "RandomPolicy"),
    "off-policy": (".policy", "OffPolicy"),
}


def policy_from_str(policy_str: str) -> type[BasePolicy]:
    if policy_str not in _policy_map:
        raise ImportError(f"Unknown policy: {policy_str}. Available: {list(_policy_map.keys())}")
    module_name, class_name = _policy_map[policy_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)
