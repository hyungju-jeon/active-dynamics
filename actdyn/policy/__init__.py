"""Policy package exports.

Expose policy base classes and a few built-in policies. Use :func:`policy_from_str`
to resolve policies by short names.
"""

from .base import BasePolicy, BaseMPC
from .baseline_ce_mpc import BaselineCEMPCPolicy
from .baseline_flex import FLEXPolicy
from .baseline_prbs import BaselinePRBSPolicy
from .baseline_random import BaselineRandomPolicy
from .baseline_thompson import BaselineThompsonPolicy
from .baseline_ucb import BaselineUCBPolicy
from .policy import OffPolicy, RandomPolicy, StepPolicy

__all__ = [
    "policy_from_str",
    "RandomPolicy",
    "StepPolicy",
    "OffPolicy",
    "BaselineRandomPolicy",
    "BaselinePRBSPolicy",
    "BaselineCEMPCPolicy",
    "FLEXPolicy",
    "RecedingHorizonCuriosityPolicy",
    "BaselineThompsonPolicy",
    "BaselineUCBPolicy",
]

import importlib

_policy_map = {
    "mpc-icem": (".mpc", "MpcICem"),
    "random": (".policy", "RandomPolicy"),
    "off-policy": (".policy", "OffPolicy"),
    "baseline-random": (".baseline_random", "BaselineRandomPolicy"),
    "baseline-prbs": (".baseline_prbs", "BaselinePRBSPolicy"),
    "baseline-ce-mpc": (".baseline_ce_mpc", "BaselineCEMPCPolicy"),
    "flex": (".baseline_flex", "FLEXPolicy"),
    "rhc": (".baseline_rhc", "RecedingHorizonCuriosityPolicy"),
    "baseline-thompson": (".baseline_thompson", "BaselineThompsonPolicy"),
    "baseline-ucb": (".baseline_ucb", "BaselineUCBPolicy"),
}


def policy_from_str(policy_str: str) -> type[BasePolicy]:
    if policy_str not in _policy_map:
        raise ImportError(f"Unknown policy: {policy_str}. Available: {list(_policy_map.keys())}")
    module_name, class_name = _policy_map[policy_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def __getattr__(name: str):
    if name == "RecedingHorizonCuriosityPolicy":
        module = importlib.import_module(".baseline_rhc", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
