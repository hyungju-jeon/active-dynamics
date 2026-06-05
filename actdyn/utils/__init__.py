"""Utilities for actdyn experiments."""

from __future__ import annotations


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
    "read_trace_csv",
    "write_trace_csv",
    "seed_range_csv",
    "to_xy_pair",
    "as_bool",
    "extract_remaining_plan_actions",
    "apply_loglinear_loading_asymmetry",
    "predict_planned_xy_trajectory",
    "extract_rollout_metrics",
    "trajectory_r2_vectorfield",
]


def setup_experiment(*args, **kwargs):
    """Lazy import wrapper for setup_experiment."""
    from .experiment_setup import setup_experiment as _setup

    return _setup(*args, **kwargs)


def __getattr__(name: str):
    if name in {"Belief", "Transition", "format_list", "to_np", "eps"}:
        from .torch_utils import Belief, Transition, eps, format_list, to_np

        return {
            "Belief": Belief,
            "Transition": Transition,
            "format_list": format_list,
            "to_np": to_np,
            "eps": eps,
        }[name]

    if name in {"save_rollout", "load_rollout"}:
        from .persistence import load_rollout, save_rollout

        return {"save_rollout": save_rollout, "load_rollout": load_rollout}[name]

    if name in {"Rollout", "RolloutBuffer", "RecentRollout"}:
        from .rollout import RecentRollout, Rollout, RolloutBuffer

        return {
            "Rollout": Rollout,
            "RolloutBuffer": RolloutBuffer,
            "RecentRollout": RecentRollout,
        }[name]

    if name in {
        "write_trace_csv",
        "read_trace_csv",
        "seed_range_csv",
        "to_xy_pair",
        "as_bool",
        "extract_remaining_plan_actions",
        "apply_loglinear_loading_asymmetry",
        "predict_planned_xy_trajectory",
        "extract_rollout_metrics",
    }:
        from .experiment_runtime import (
            apply_loglinear_loading_asymmetry,
            as_bool,
            extract_remaining_plan_actions,
            extract_rollout_metrics,
            predict_planned_xy_trajectory,
            read_trace_csv,
            to_xy_pair,
            seed_range_csv,
            write_trace_csv,
        )

        return {
            "read_trace_csv": read_trace_csv,
            "write_trace_csv": write_trace_csv,
            "seed_range_csv": seed_range_csv,
            "to_xy_pair": to_xy_pair,
            "as_bool": as_bool,
            "extract_remaining_plan_actions": extract_remaining_plan_actions,
            "apply_loglinear_loading_asymmetry": apply_loglinear_loading_asymmetry,
            "predict_planned_xy_trajectory": predict_planned_xy_trajectory,
            "extract_rollout_metrics": extract_rollout_metrics,
        }[name]

    if name == "trajectory_r2_vectorfield":
        from .validation import trajectory_r2_vectorfield

        return trajectory_r2_vectorfield

    if name == "VideoRecorder":
        from .video import VideoRecorder

        return VideoRecorder

    if name in {
        "hydra_experiment",
        "HydraExperimentConfig",
        "register_actdyn_configs",
        "setup_hydra_experiment",
    }:
        from .hydra_config import (
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
