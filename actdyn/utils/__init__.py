"""Utilities for actdyn experiments."""

from __future__ import annotations

from .helper import Belief, Transition, format_list, to_np, eps

__all__ = [
    "setup_experiment",
    "save_rollout",
    "load_rollout",
    "Rollout",
    "RolloutBuffer",
    "RecentRollout",
    "VideoRecorder",
    "Belief",
    "Transition",
    "format_list",
    "to_np",
    "eps",
]


def setup_experiment(*args, **kwargs):
    """Lazy import wrapper for setup_experiment."""
    from .experiment_helpers import setup_experiment as _setup

    return _setup(*args, **kwargs)


def __getattr__(name: str):
    if name in {"save_rollout", "load_rollout"}:
        from .save_load import load_rollout, save_rollout

        return {"save_rollout": save_rollout, "load_rollout": load_rollout}[name]

    if name in {"Rollout", "RolloutBuffer", "RecentRollout"}:
        from .rollout import RecentRollout, Rollout, RolloutBuffer

        return {
            "Rollout": Rollout,
            "RolloutBuffer": RolloutBuffer,
            "RecentRollout": RecentRollout,
        }[name]

    if name == "VideoRecorder":
        from .video import VideoRecorder

        return VideoRecorder

    if name in {
        "hydra_experiment",
        "HydraExperimentConfig",
        "register_actdyn_configs",
        "setup_hydra_experiment",
    }:
        from .hydra_integration import (
            HydraExperimentConfig,
            hydra_experiment,
            register_actdyn_configs,
            setup_hydra_experiment,
        )

        return {
            "hydra_experiment": hydra_experiment,
            "HydraExperimentConfig": HydraExperimentConfig,
            "register_actdyn_configs": register_actdyn_configs,
            "setup_hydra_experiment": setup_hydra_experiment,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
