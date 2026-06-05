#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from actdyn.environment import residual_np
from actdyn.environment.vectorfield import ResidualDynamicsCallable
from actdyn.utils.experiment_runtime import read_trace_csv
from actdyn.utils.plotting import (
    apply_manuscript_figure_style,
    compute_vector_field,
    decorate_phase_space_axis,
    style_manuscript_axis,
)

from ..experiment_definitions import get_environment_preset, get_experiment_spec
from ..experiment_io import (
    expected_loglinear_rate_hz,
    find_nested_metadata_paths,
    get_environment_preset_from_metadata,
    load_json,
    reconstruct_loglinear_rate_model,
    resolve_artifact_path,
)
from ..visualize import (
    _parse_figure_formats,
    plot_final_value_by_policy,
    plot_information_colormap,
    plot_metric_over_cpu_time,
    plot_metric_over_steps,
    plot_neuron_tuning_curve_colormap,
    plot_parameter_covariance_trace_over_steps,
)
from .run_tbme_experiments import configure_tbme_catalogs

configure_tbme_catalogs()


# GROUPS is evaluated at import time, so this constructor stays before the global table.
def _asset_latest_session(base: Path) -> Path:
    sessions = [
        path
        for path in base.glob("session_*")
        if path.is_dir() and path.name.removeprefix("session_").isdigit()
    ]
    if not sessions:
        return base / "session_1"
    return max(sessions, key=lambda path: int(path.name.removeprefix("session_")))


# Global configuration
_TBME_STROKE_COLOR = "#3A3A3A"
_TBME_GRID_COLOR = "#DDD7CE"


# Current TBME manuscript asset export
_asset_REPO_ROOT = Path(__file__).resolve().parents[2]
_asset_FIG_DIR = _asset_REPO_ROOT / "docs" / "figs" / "tbme" / "generated"
_asset_TEX_DIR = _asset_REPO_ROOT / "docs" / "tables"
_asset_RESULTS_DIR = _asset_REPO_ROOT / "results" / "tbme"
_asset_R2_THRESHOLDS = (0.90, 0.95, 0.99)
_asset_C_WRITE = "#1F4FA8"
_asset_C_WRITE_SOFT = "#6F8EC8"
_asset_C_WRITE_FILL = "#E8EEFF"
_asset_C_STROKE = "#3A3A3A"
_asset_C_NEUTRAL = "#6F6A62"
_asset_C_NEUTRAL_LIGHT = "#C8C1B8"
_asset_C_NEUTRAL_FILL = "#F4F1EC"
_asset_C_WHITE = "#FFFFFF"
_asset_R2_THRESHOLD_SEGMENT_COLORS = (_asset_C_NEUTRAL_LIGHT, _asset_C_WRITE_SOFT, _asset_C_WRITE)
_asset_R2_THRESHOLD_POINT_COLORS = {
    0.90: _asset_C_NEUTRAL,
    0.95: _asset_C_WRITE_SOFT,
    0.99: _asset_C_WRITE,
}


@dataclass(frozen=True)
class SuiteRef:
    suite_id: str
    label: str
    session_root: Path
    slug: str


GROUPS: dict[str, list[SuiteRef]] = {
    "exp01_base": [
        SuiteRef(
            "exp01_duffing",
            "Duffing",
            _asset_latest_session(_asset_RESULTS_DIR / "exp01_base"),
            "duffing",
        ),
        SuiteRef(
            "exp01_damped_pendulum",
            "Damped pendulum",
            _asset_latest_session(_asset_RESULTS_DIR / "exp01_base"),
            "damped_pendulum",
        ),
        SuiteRef(
            "exp01_asymmetric_basin",
            "Asymmetric basin",
            _asset_latest_session(_asset_RESULTS_DIR / "exp01_base"),
            "asymmetric_basin",
        ),
    ],
    "exp02_hard": [
        SuiteRef(
            "exp02_hard_duffing",
            "Duffing hard",
            _asset_latest_session(_asset_RESULTS_DIR / "exp02_hard"),
            "duffing_hard",
        ),
        SuiteRef(
            "exp02_hard_asymmetric_basin",
            "Asymmetric basin hard",
            _asset_latest_session(_asset_RESULTS_DIR / "exp02_hard"),
            "asymmetric_basin_hard",
        ),
        SuiteRef(
            "exp02_hard_damped_pendulum",
            "Damped pendulum hard",
            _asset_latest_session(_asset_RESULTS_DIR / "exp02_hard"),
            "damped_pendulum_hard",
        ),
    ],
    "exp03_schedule": [
        SuiteRef(
            "exp03_schedule_duffing",
            "Duffing",
            _asset_latest_session(_asset_RESULTS_DIR / "exp03_schedule"),
            "duffing",
        ),
        SuiteRef(
            "exp03_schedule_damped_pendulum",
            "Damped pendulum",
            _asset_latest_session(_asset_RESULTS_DIR / "exp03_schedule"),
            "damped_pendulum",
        ),
        SuiteRef(
            "exp03_schedule_asymmetric_basin",
            "Asymmetric basin",
            _asset_latest_session(_asset_RESULTS_DIR / "exp03_schedule"),
            "asymmetric_basin",
        ),
    ],
    "exp04_mismatch": [
        SuiteRef(
            "exp04_duffing_parameter_mismatch",
            "Duffing parameter mismatch",
            _asset_latest_session(_asset_RESULTS_DIR / "exp04_mismatch"),
            "duffing_parameter_mismatch",
        ),
        SuiteRef(
            "exp04_asymmetric_basin_parameter_mismatch",
            "Asymmetric basin parameter mismatch",
            _asset_latest_session(_asset_RESULTS_DIR / "exp04_mismatch"),
            "asymmetric_basin_parameter_mismatch",
        ),
    ],
    "exp05_ablation": [
        SuiteRef(
            "exp05_asymmetric_basin_objective_ablation",
            "Asymmetric basin objective ablation",
            _asset_latest_session(_asset_RESULTS_DIR / "exp05_ablation"),
            "asymmetric_basin_objective_ablation",
        ),
        SuiteRef(
            "exp05_hard_asymmetric_basin_objective_ablation",
            "Hard asymmetric basin objective ablation",
            _asset_latest_session(_asset_RESULTS_DIR / "exp05_ablation"),
            "hard_asymmetric_basin_objective_ablation",
        ),
    ],
    "exp06_bottleneck": [
        SuiteRef(
            "exp06_asymmetric_basin_bottleneck_weak_observation",
            "Asymmetric basin weak observation",
            _asset_latest_session(_asset_RESULTS_DIR / "exp06_bottleneck"),
            "asymmetric_basin_bottleneck_weak_observation",
        ),
        SuiteRef(
            "exp06_asymmetric_basin_bottleneck_tight_action",
            "Asymmetric basin tight action",
            _asset_latest_session(_asset_RESULTS_DIR / "exp06_bottleneck"),
            "asymmetric_basin_bottleneck_tight_action",
        ),
        SuiteRef(
            "exp06_asymmetric_basin_bottleneck_combined",
            "Asymmetric basin bottleneck",
            _asset_latest_session(_asset_RESULTS_DIR / "exp06_bottleneck"),
            "asymmetric_basin_bottleneck_combined",
        ),
    ],
    "exp07_mismatch_stress": [
        SuiteRef(
            "exp07_duffing_parameter_mismatch_mild",
            "Duffing parameter mismatch mild",
            _asset_latest_session(_asset_RESULTS_DIR / "exp07_mismatch_stress"),
            "duffing_parameter_mismatch_mild",
        ),
        SuiteRef(
            "exp07_duffing_parameter_mismatch_strong",
            "Duffing parameter mismatch strong",
            _asset_latest_session(_asset_RESULTS_DIR / "exp07_mismatch_stress"),
            "duffing_parameter_mismatch_strong",
        ),
        SuiteRef(
            "exp07_asymmetric_basin_parameter_mismatch_mild",
            "Asymmetric basin parameter mismatch mild",
            _asset_latest_session(_asset_RESULTS_DIR / "exp07_mismatch_stress"),
            "asymmetric_basin_parameter_mismatch_mild",
        ),
        SuiteRef(
            "exp07_asymmetric_basin_parameter_mismatch_strong",
            "Asymmetric basin parameter mismatch strong",
            _asset_latest_session(_asset_RESULTS_DIR / "exp07_mismatch_stress"),
            "asymmetric_basin_parameter_mismatch_strong",
        ),
    ],
}


POLICY_LABELS = {
    "active_planning": "Planning",
    "active_planning_u1_r1_h40": "Planning u1/r1/h40",
    "active_planning_u5_r5_h40": "Planning u5/r5/h40",
    "active_planning_u1_r5_h40": "Planning u1/r5/h40",
    "active_planning_u10_r10_h40": "Planning u10/r10/h40",
    "active_planning_u5_r10_h40": "Planning u5/r10/h40",
    "active_planning_u20_r20_h40": "Planning u20/r20/h40",
    "active_planning_u5_r20_h40": "Planning u5/r20/h40",
    "active_planning_u10_r20_h40": "Planning u10/r20/h40",
    "active_fully_observable_u20_r20_h40": "Full-observable EIG",
    "active_e_optimality_u20_r20_h40": "E-optimality",
    "active_state_information_u20_r20_h40": "State information",
    "active_dynamics_u20_r20_h40": "Dynamics objective",
    "active_sampling_variance_u20_r20_h40": "Sampling variance",
    "active_myopic": "Myopic",
    "prbs": "PRBS",
    "random": "Random",
    "flex": "FLEX",
    "flex_true_state": "FLEX true state",
    "ensemble": "Ensemble",
    "rhc": "RHC-US",
}

POLICY_ORDER = [
    "active_planning",
    "active_planning_u1_r1_h40",
    "active_planning_u5_r5_h40",
    "active_planning_u1_r5_h40",
    "active_planning_u10_r10_h40",
    "active_planning_u5_r10_h40",
    "active_planning_u20_r20_h40",
    "active_planning_u5_r20_h40",
    "active_planning_u10_r20_h40",
    "active_fully_observable_u20_r20_h40",
    "active_e_optimality_u20_r20_h40",
    "active_state_information_u20_r20_h40",
    "active_dynamics_u20_r20_h40",
    "active_sampling_variance_u20_r20_h40",
    "active_myopic",
    "prbs",
    "random",
    "flex",
    "flex_true_state",
    "ensemble",
    "rhc",
]

POLICY_COLORS = {
    "active_planning": "#1F4FA8",
    "active_planning_u1_r1_h40": "#4B74B9",
    "active_planning_u5_r5_h40": "#2F6F9F",
    "active_planning_u1_r5_h40": "#5B8D5A",
    "active_planning_u10_r10_h40": "#7A6AAE",
    "active_planning_u5_r10_h40": "#4B8F8C",
    "active_planning_u20_r20_h40": "#1F4FA8",
    "active_planning_u5_r20_h40": "#6F8EC8",
    "active_planning_u10_r20_h40": "#3C6D99",
    "active_fully_observable_u20_r20_h40": "#5B8D5A",
    "active_e_optimality_u20_r20_h40": "#7E5AA6",
    "active_state_information_u20_r20_h40": "#C27A2C",
    "active_dynamics_u20_r20_h40": "#2F7C7A",
    "active_sampling_variance_u20_r20_h40": "#9C5C38",
    "active_myopic": "#B5361C",
    "prbs": "#7C6A45",
    "random": "#6F6A62",
    "flex": "#7E5AA6",
    "flex_true_state": "#4F8A62",
    "ensemble": "#C27A2C",
    "rhc": "#2F7C7A",
}
FALLBACK_COLORS = (
    "#1F4FA8",
    "#B5361C",
    "#4F8A62",
    "#7E5AA6",
    "#C27A2C",
    "#6F6A62",
    "#2F7C7A",
    "#6F8EC8",
)


# Current TBME summary figure export
_summary_TBME_DIR = Path(__file__).resolve().parent
_summary_EXPERIMENTS_DIR = _summary_TBME_DIR.parent


# Trajectory figure family
_trajectory_C_WRITE = "#1F4FA8"
_trajectory_C_READ = "#B5361C"
_trajectory_C_STROKE = "#3A3A3A"
_trajectory_C_NEUTRAL = "#6F6A62"
_trajectory_C_NEUTRAL_LIGHT = "#C8C1B8"
_trajectory_C_NEUTRAL_FILL = "#F4F1EC"
_trajectory_C_WHITE = "#FFFFFF"
_trajectory_TBME_DIR = Path(__file__).resolve().parent
_trajectory_EXPERIMENTS_DIR = _trajectory_TBME_DIR.parent


# Requested figure family
_requested_REPO_ROOT = Path(__file__).resolve().parents[2]
_requested_TBME_DIR = Path(__file__).resolve().parent
_requested_EXPERIMENTS_DIR = _requested_TBME_DIR.parent
_requested_FIGURE_DIR = _requested_REPO_ROOT / "docs" / "figs"
_requested_GENERATED_DIR = _requested_REPO_ROOT / "docs" / "tables"

_requested_C_WRITE = "#1F4FA8"
_requested_C_READ = "#B5361C"
_requested_C_ENSEMBLE = "#C27A2C"
_requested_C_PRBS = "#7C6A45"
_requested_C_RANDOM = "#6F6A62"
_requested_C_STROKE = "#3A3A3A"
_requested_C_NEUTRAL = "#6F6A62"
_requested_C_NEUTRAL_LIGHT = "#C8C1B8"
_requested_C_NEUTRAL_FILL = "#F4F1EC"
_requested_C_GRID = "#DDD7CE"
_requested_POLICY_COLORS = {
    "active_planning_u20_r20_h40": _requested_C_WRITE,
    "active_fully_observable_u20_r20_h40": "#5B8D5A",
    "active_e_optimality_u20_r20_h40": "#7E5AA6",
    "active_state_information_u20_r20_h40": _requested_C_ENSEMBLE,
    "active_dynamics_u20_r20_h40": "#2F7C7A",
    "active_sampling_variance_u20_r20_h40": "#9C5C38",
    "active_myopic": _requested_C_READ,
    "ensemble": _requested_C_ENSEMBLE,
    "prbs": _requested_C_PRBS,
    "random": _requested_C_RANDOM,
}

_requested_BOTTLENECK_POLICIES = [
    "active_planning_u20_r20_h40",
    "active_myopic",
    "ensemble",
    "prbs",
    "random",
]
_requested_OBJECTIVE_POLICIES = [
    "active_planning_u20_r20_h40",
    "active_fully_observable_u20_r20_h40",
    "active_state_information_u20_r20_h40",
    "active_dynamics_u20_r20_h40",
    "active_sampling_variance_u20_r20_h40",
    "active_e_optimality_u20_r20_h40",
    "ensemble",
    "prbs",
]
_requested_OBJECTIVE_DEFINITIONS = [
    {
        "policy_id": "active_planning_u20_r20_h40",
        "objective_name": "Parameter EIG",
        "objective_formula": (
            r"$J(u_{0:H-1})=\frac{1}{2}\log\det(I+P_\theta "
            r"\sum_{i=0}^{H-1}\gamma^i \Delta\Lambda_i)$, "
            r"$\Delta\Lambda_i=S_i^\top(I+P_i^- I_{z,i})^{-1}I_{z,i}S_i$"
        ),
        "objective_notes": "Main objective; partial-observation attenuation uses the predicted latent covariance.",
    },
    {
        "policy_id": "active_fully_observable_u20_r20_h40",
        "objective_name": "Full-observable EIG",
        "objective_formula": (
            r"$J(u_{0:H-1})=\frac{1}{2}\log\det(I+P_\theta "
            r"\sum_{i=0}^{H-1}\gamma^i S_i^\top I_{z,i}S_i)$"
        ),
        "objective_notes": "Ablation that removes partial-observation attenuation.",
    },
    {
        "policy_id": "active_state_information_u20_r20_h40",
        "objective_name": "State information",
        "objective_formula": (
            r"$J(u_{0:H-1})=\sum_{i=0}^{H-1}\gamma^i "
            r"\log\det\operatorname{chol}(I+P_i^- I_{z,i})$"
        ),
        "objective_notes": "Scores latent-state observability, not parameter sensitivity.",
    },
    {
        "policy_id": "active_dynamics_u20_r20_h40",
        "objective_name": "Dynamics sensitivity",
        "objective_formula": (
            r"$J(u_{0:H-1})=\sum_{i=0}^{H-1}\gamma^i " r"\operatorname{tr}(S_i^\top P_i^- S_i)$"
        ),
        "objective_notes": "Scores predicted state sensitivity to parameters without the decoder Fisher term.",
    },
    {
        "policy_id": "active_sampling_variance_u20_r20_h40",
        "objective_name": "Sampling variance",
        "objective_formula": (
            r"$J(u_{0:H-1})=\sum_{i=0}^{H-1}\gamma^i "
            r"\sum_j\log(1+\operatorname{Var}_{\theta\sim q(\theta)}"
            r"[\lambda_j(z_i(\theta))])$"
        ),
        "objective_notes": "Monte Carlo objective using posterior samples of the dynamics parameters.",
    },
    {
        "policy_id": "active_e_optimality_u20_r20_h40",
        "objective_name": "E-optimality",
        "objective_formula": (
            r"$J(u_{0:H-1})=\lambda_{\min}(P_\theta " r"\sum_{i=0}^{H-1}\gamma^i \Delta\Lambda_i)$"
        ),
        "objective_notes": "Maximizes the least-informed parameter direction.",
    },
    {
        "policy_id": "ensemble",
        "objective_name": "Ensemble state variance",
        "objective_formula": (
            r"$J(u_{0:H-1})=\sum_{i=0}^{H-1}\gamma^i "
            r"\sum_d \operatorname{Var}_{\theta\sim q(\theta)}[z_{i,d}(\theta)]$"
        ),
        "objective_notes": "Practical adaptive baseline; it is not a Fisher-information objective.",
    },
    {
        "policy_id": "prbs",
        "objective_name": "PRBS",
        "objective_formula": r"Preset pseudo-random binary excitation sequence.",
        "objective_notes": "Passive baseline with no model-based acquisition optimization.",
    },
]
_requested_DOSE_POLICIES = [
    "active_planning_u20_r20_h40",
    "active_myopic",
    "ensemble",
    "prbs",
    "random",
]


@dataclass(frozen=True)
class _requested_SuiteSource:
    exp_id: str
    label: str
    suite_dir: Path
    dose: str | None = None
    family: str | None = None


@dataclass(frozen=True)
class _requested_RunRecord:
    policy_id: str
    seed: int
    run_dir: Path
    metadata: dict[str, Any]


_requested_PLOTS = (
    "bottleneck_sweep",
    "objective_ablation",
    "mismatch_dose_response",
    "downstream_control",
)
_requested_OBJECTIVE_DEFINITION_PLOTS = {"objective_ablation", "downstream_control"}
_requested_REQUIRED_SUITES_BY_PLOT = {
    "bottleneck_sweep": (
        ("exp01_base", "exp01_asymmetric_basin"),
        ("exp06_bottleneck", "exp06_asymmetric_basin_bottleneck_weak_observation"),
        ("exp06_bottleneck", "exp06_asymmetric_basin_bottleneck_tight_action"),
        ("exp06_bottleneck", "exp06_asymmetric_basin_bottleneck_combined"),
    ),
    "objective_ablation": (
        ("exp05_ablation", "exp05_asymmetric_basin_objective_ablation"),
        ("exp05_ablation", "exp05_hard_asymmetric_basin_objective_ablation"),
    ),
    "mismatch_dose_response": (
        ("exp01_base", "exp01_duffing"),
        ("exp01_base", "exp01_asymmetric_basin"),
        ("exp04_mismatch", "exp04_duffing_parameter_mismatch"),
        ("exp04_mismatch", "exp04_asymmetric_basin_parameter_mismatch"),
        ("exp07_mismatch_stress", "exp07_duffing_parameter_mismatch_mild"),
        ("exp07_mismatch_stress", "exp07_duffing_parameter_mismatch_strong"),
        ("exp07_mismatch_stress", "exp07_asymmetric_basin_parameter_mismatch_mild"),
        ("exp07_mismatch_stress", "exp07_asymmetric_basin_parameter_mismatch_strong"),
    ),
    "downstream_control": (("exp05_ablation", "exp05_asymmetric_basin_objective_ablation"),),
}


# Additional figure family
_additional_REPO_ROOT = Path(__file__).resolve().parents[2]
_additional_TBME_DIR = Path(__file__).resolve().parent
_additional_EXPERIMENTS_DIR = _additional_TBME_DIR.parent
_additional_FIGURE_DIR = _additional_REPO_ROOT / "docs" / "figs"
_additional_GENERATED_DIR = _additional_REPO_ROOT / "docs" / "tables"

_additional_C_WRITE = "#1F4FA8"
_additional_C_READ = "#B5361C"
_additional_C_ENSEMBLE = "#C27A2C"
_additional_C_PRBS = "#7C6A45"
_additional_C_RANDOM = "#6F6A62"
_additional_C_RHC = "#2F7C7A"
_additional_C_STROKE = "#3A3A3A"
_additional_C_NEUTRAL = "#6F6A62"
_additional_C_NEUTRAL_LIGHT = "#C8C1B8"
_additional_C_NEUTRAL_FILL = "#F4F1EC"
_additional_C_GRID = "#DDD7CE"
_additional_POLICY_COLORS = {
    "active_planning_u20_r20_h40": _additional_C_WRITE,
    "active_planning_u5_r5_h40": "#2F6F9F",
    "active_planning_u10_r10_h40": "#7A6AAE",
    "active_planning_u5_r10_h40": "#4B8F8C",
    "active_planning_u5_r20_h40": "#6F8EC8",
    "active_planning_u10_r20_h40": "#3C6D99",
    "active_myopic": _additional_C_READ,
    "prbs": _additional_C_PRBS,
    "random": _additional_C_RANDOM,
    "ensemble": _additional_C_ENSEMBLE,
    "rhc": _additional_C_RHC,
    "flex": "#7E5AA6",
    "flex_true_state": "#4F8A62",
}


@dataclass(frozen=True)
class _additional_RunRecord:
    policy_id: str
    seed: int
    run_dir: Path
    metadata: dict[str, Any]


_additional_PLOTS = (
    "true_dynamics_all",
    "asymmetric_basin_mechanism",
    "learned_vectorfield_snapshots",
    "sample_efficiency_thresholds",
    "compute_accuracy_pareto",
    "per_parameter_recovery",
    "information_learning_coupling",
)

# Helper functions


def _apply_style(plt_module: Any | None = None) -> None:
    if plt_module is None:
        plt_module = plt
    apply_manuscript_figure_style(plt_module, stroke_color=_TBME_STROKE_COLOR)


def _style_manuscript_axis(
    ax: Any,
    *,
    grid_axis: str | None = None,
    grid_color: str = _TBME_GRID_COLOR,
    grid_alpha: float = 0.42,
) -> None:
    style_manuscript_axis(
        ax,
        grid_axis=grid_axis,
        grid_color=grid_color,
        grid_alpha=float(grid_alpha),
        stroke_color=_TBME_STROKE_COLOR,
    )


def _style_requested_axis(ax: Any) -> None:
    _style_manuscript_axis(ax, grid_alpha=0.55)


def _suite_dir(group_name: str, suite_id: str) -> Path:
    for ref in GROUPS[group_name]:
        if ref.suite_id == suite_id:
            return ref.session_root / ref.suite_id
    raise KeyError(f"Unknown suite {group_name}/{suite_id}")


def _policy_sort_key(policy_id: str) -> tuple[int, str]:
    try:
        return POLICY_ORDER.index(policy_id), policy_id
    except ValueError:
        return len(POLICY_ORDER), policy_id


def _policy_label(policy_id: str) -> str:
    return POLICY_LABELS.get(policy_id, policy_id.replace("_", " "))


def _policy_color(policy_id: str, fallback_idx: int = 0) -> str:
    return POLICY_COLORS.get(policy_id, FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])


def _asset_safe_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _asset_summary_dir(ref: SuiteRef) -> Path:
    return ref.session_root / ref.suite_id / "summary"


def _asset_policy_sort_key(policy_id: str) -> tuple[int, str]:
    try:
        idx = POLICY_ORDER.index(policy_id)
    except ValueError:
        idx = len(POLICY_ORDER)
    return idx, policy_id


def _asset_fmt(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return "--"
    if not math.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _asset_aggregate_suite(ref: SuiteRef) -> list[dict[str, object]]:
    metrics_path = _asset_summary_dir(ref) / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    rows = read_trace_csv(metrics_path)
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        pid = str(row["policy_id"])
        bucket = grouped.setdefault(pid, {"value": [], "r2": [], "runtime": []})
        if row.get("value_final_mean"):
            bucket["value"].append(float(row["value_final_mean"]))
        if row.get("trajectory_r2_final_mean"):
            bucket["r2"].append(float(row["trajectory_r2_final_mean"]))
        if row.get("runtime_sec_mean"):
            bucket["runtime"].append(float(row["runtime_sec_mean"]))
    out: list[dict[str, object]] = []
    for pid, bucket in grouped.items():
        vals = np.asarray(bucket["value"], dtype=np.float64)
        r2s = np.asarray(bucket["r2"], dtype=np.float64)
        runtimes = np.asarray(bucket["runtime"], dtype=np.float64)
        out.append(
            {
                "suite_id": ref.suite_id,
                "suite_label": ref.label,
                "policy_id": pid,
                "policy_label": POLICY_LABELS.get(pid, pid),
                "n": int(vals.size),
                "parameter_error_mean": float(vals.mean()) if vals.size else math.nan,
                "parameter_error_std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "trajectory_r2_mean": float(r2s.mean()) if r2s.size else math.nan,
                "trajectory_r2_std": float(r2s.std(ddof=1)) if r2s.size > 1 else 0.0,
                "runtime_sec_mean": float(runtimes.mean()) if runtimes.size else math.nan,
                "runtime_sec_std": float(runtimes.std(ddof=1)) if runtimes.size > 1 else 0.0,
            }
        )
    out.sort(key=lambda row: _asset_policy_sort_key(str(row["policy_id"])))
    return out


def _asset_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite_id",
        "suite_label",
        "policy_id",
        "policy_label",
        "n",
        "parameter_error_mean",
        "parameter_error_std",
        "trajectory_r2_mean",
        "trajectory_r2_std",
        "runtime_sec_mean",
        "runtime_sec_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _asset_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _asset_write_tex(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    lines = [
        "% Auto-generated by experiments/tbme/generate_figures.py assets",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Suite & Policy & Param. error & Trajectory $R^2$ & Runtime (s) \\",
        r"\midrule",
    ]
    current_suite = None
    for row in rows:
        suite = str(row["suite_label"])
        suite_cell = _asset_escape(suite) if suite != current_suite else ""
        current_suite = suite
        line = (
            " & ".join(
                [
                    suite_cell,
                    _asset_escape(str(row["policy_label"])),
                    _asset_fmt(
                        float(row["parameter_error_mean"]), float(row["parameter_error_std"])
                    ),
                    _asset_fmt(float(row["trajectory_r2_mean"]), float(row["trajectory_r2_std"])),
                    _asset_fmt(
                        float(row["runtime_sec_mean"]), float(row["runtime_sec_std"]), digits=1
                    ),
                ]
            )
            + r" \\"
        )
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _asset_threshold_suffix(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _asset_threshold_rows_for_suite(ref: SuiteRef) -> list[dict[str, object]]:
    threshold_path = _asset_summary_dir(ref) / "trajectory_r2_thresholds.csv"
    if threshold_path.exists():
        rows = read_trace_csv(threshold_path)
        out: list[dict[str, object]] = []
        for row in rows:
            payload: dict[str, object] = {
                "suite_id": ref.suite_id,
                "suite_label": ref.label,
                "policy_id": row["policy_id"],
                "policy_label": POLICY_LABELS.get(row["policy_id"], row["policy_id"]),
            }
            for threshold in _asset_R2_THRESHOLDS:
                suffix = _asset_threshold_suffix(threshold)
                payload[f"step_to_r2_{suffix}"] = _asset_safe_float(row.get(f"step_to_r2_{suffix}"))
                payload[f"cpu_time_sec_to_r2_{suffix}"] = _asset_safe_float(
                    row.get(f"cpu_time_sec_to_r2_{suffix}")
                )
            out.append(payload)
        out.sort(key=lambda row: _asset_policy_sort_key(str(row["policy_id"])))
        return out

    trace_path = _asset_summary_dir(ref) / "trajectory_r2_over_steps.csv"
    if not trace_path.exists():
        return []
    by_policy: dict[str, list[dict[str, str]]] = {}
    for row in read_trace_csv(trace_path):
        by_policy.setdefault(row["policy_id"], []).append(row)

    out = []
    for policy_id, series in by_policy.items():
        series.sort(key=lambda row: int(float(row["step"])))
        payload = {
            "suite_id": ref.suite_id,
            "suite_label": ref.label,
            "policy_id": policy_id,
            "policy_label": POLICY_LABELS.get(policy_id, policy_id),
        }
        for threshold in _asset_R2_THRESHOLDS:
            suffix = _asset_threshold_suffix(threshold)
            crossing = None
            for row in series:
                value = _asset_safe_float(row.get("trajectory_r2_mean"))
                if value is not None and value >= threshold:
                    crossing = row
                    break
            payload[f"step_to_r2_{suffix}"] = (
                int(float(crossing["step"])) if crossing is not None else None
            )
            payload[f"cpu_time_sec_to_r2_{suffix}"] = (
                _asset_safe_float(crossing.get("cpu_time_sec_mean"))
                if crossing is not None
                else None
            )
        out.append(payload)
    out.sort(key=lambda row: _asset_policy_sort_key(str(row["policy_id"])))
    return out


def _asset_write_threshold_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite_id",
        "suite_label",
        "policy_id",
        "policy_label",
    ]
    for threshold in _asset_R2_THRESHOLDS:
        suffix = _asset_threshold_suffix(float(threshold))
        fields.extend([f"step_to_r2_{suffix}", f"cpu_time_sec_to_r2_{suffix}"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _asset_fmt_optional_int(value: object) -> str:
    parsed = _asset_safe_float(value)
    if parsed is None:
        return "--"
    return str(int(parsed))


def _asset_fmt_optional_sec(value: object) -> str:
    parsed = _asset_safe_float(value)
    if parsed is None:
        return "--"
    return f"{parsed:.1f}"


def _asset_write_threshold_tex(path: Path, rows: list[dict[str, object]]) -> None:
    header_cells = ["Suite", "Policy"]
    for threshold in _asset_R2_THRESHOLDS:
        label = f"{threshold:.2f}"
        header_cells.extend([rf"Step to $R^2\ge {label}$", rf"CPU to $R^2\ge {label}$ (s)"])
    header = " & ".join(header_cells) + r" \\"
    lines = [
        "% Auto-generated by experiments/tbme/generate_figures.py assets",
        r"\begin{tabular}{ll" + ("rr" * len(_asset_R2_THRESHOLDS)) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    current_suite = None
    for row in rows:
        suite = str(row["suite_label"])
        suite_cell = _asset_escape(suite) if suite != current_suite else ""
        current_suite = suite
        row_cells = [suite_cell, _asset_escape(str(row["policy_label"]))]
        for threshold in _asset_R2_THRESHOLDS:
            suffix = _asset_threshold_suffix(float(threshold))
            row_cells.extend(
                [
                    _asset_fmt_optional_int(row.get(f"step_to_r2_{suffix}")),
                    _asset_fmt_optional_sec(row.get(f"cpu_time_sec_to_r2_{suffix}")),
                ]
            )
        lines.append(" & ".join(row_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _asset_clear_group_figures(group_name: str) -> None:
    group_dir = _asset_FIG_DIR / group_name
    if group_dir.exists():
        shutil.rmtree(group_dir)
    for legacy in _asset_FIG_DIR.glob(f"{group_name}_*.pdf"):
        legacy.unlink()


def _asset_copy_figures(group_name: str, refs: list[SuiteRef]) -> list[Path]:
    copied: list[Path] = []
    _asset_clear_group_figures(group_name)
    for ref in refs:
        fig_dir = _asset_summary_dir(ref) / "figures"
        if not fig_dir.exists():
            continue
        dst_dir = _asset_FIG_DIR / group_name / ref.slug
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(fig_dir.glob("*.pdf")):
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _asset_short_policy_label(policy_id: str) -> str:
    labels = {
        "active_planning": "Planning",
        "active_planning_u1_r1_h40": "Plan\nU1/R1",
        "active_planning_u5_r5_h40": "Plan\nU5/R5",
        "active_planning_u1_r5_h40": "Plan\nU1/R5",
        "active_planning_u10_r10_h40": "Plan\nU10/R10",
        "active_planning_u5_r10_h40": "Plan\nU5/R10",
        "active_planning_u20_r20_h40": "Plan\nU20/R20",
        "active_planning_u5_r20_h40": "Plan\nU5/R20",
        "active_planning_u10_r20_h40": "Plan\nU10/R20",
        "active_fully_observable_u20_r20_h40": "Full\nobs",
        "active_e_optimality_u20_r20_h40": "E-opt",
        "active_state_information_u20_r20_h40": "State\ninfo",
        "active_dynamics_u20_r20_h40": "Dyn",
        "active_sampling_variance_u20_r20_h40": "Sample\nvar",
        "active_myopic": "Myopic",
        "prbs": "PRBS",
        "random": "Random",
        "flex": "FLEX",
        "flex_true_state": "FLEX\ntrue",
        "ensemble": "Ensemble",
        "rhc": "RHC-US",
    }
    return labels.get(policy_id, policy_id.replace("_", "\n"))


def _asset_inline_policy_label(policy_id: str) -> str:
    labels = {
        "active_planning_u1_r1_h40": "U1/R1",
        "active_planning_u5_r5_h40": "U5/R5",
        "active_planning_u10_r10_h40": "U10/R10",
        "active_planning_u5_r10_h40": "U5/R10",
        "active_planning_u20_r20_h40": "U20/R20",
        "active_planning_u5_r20_h40": "U5/R20",
        "active_planning_u10_r20_h40": "U10/R20",
        "active_myopic": "Myopic",
    }
    return labels.get(policy_id, POLICY_LABELS.get(policy_id, policy_id))


def _asset_threshold_segments(
    row: dict[str, object], *, field_prefix: str
) -> tuple[list[float], bool]:
    segments: list[float] = []
    previous = 0.0
    reached_all = True
    for threshold in _asset_R2_THRESHOLDS:
        suffix = _asset_threshold_suffix(float(threshold))
        value = _asset_safe_float(row.get(f"{field_prefix}_{suffix}"))
        if value is None:
            segments.append(0.0)
            reached_all = False
            continue
        value = max(value, previous)
        segments.append(value - previous)
        previous = value
    return segments, reached_all


def _asset_threshold_value_penalty(
    threshold_rows: list[dict[str, object]],
    *,
    field_prefix: str,
) -> float:
    values: list[float] = []
    for row in threshold_rows:
        for threshold in _asset_R2_THRESHOLDS:
            suffix = _asset_threshold_suffix(float(threshold))
            value = _asset_safe_float(row.get(f"{field_prefix}_{suffix}"))
            if value is not None:
                values.append(value)
    return max(values) * 1.25 if values else 1.0


def _asset_policy_threshold_sort_key(
    policy_id: str,
    refs: list[SuiteRef],
    row_by_key: dict[tuple[str, str], dict[str, object]],
    *,
    field_prefix: str,
    missing_penalty: float,
) -> tuple[float, int, tuple[int, str]]:
    values: list[float] = []
    reached_count = 0
    for ref in refs:
        row = row_by_key.get((ref.suite_id, policy_id), {})
        for threshold in _asset_R2_THRESHOLDS:
            suffix = _asset_threshold_suffix(float(threshold))
            value = _asset_safe_float(row.get(f"{field_prefix}_{suffix}"))
            if value is None:
                values.append(missing_penalty)
            else:
                values.append(value)
                reached_count += 1
    return (
        float(np.mean(values)) if values else missing_penalty,
        -reached_count,
        _asset_policy_sort_key(policy_id),
    )


def _asset_apply_plot_style(plt: object) -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 7.8,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": _asset_C_STROKE,
            "axes.linewidth": 0.55,
            "axes.labelcolor": _asset_C_STROKE,
            "xtick.color": _asset_C_STROKE,
            "ytick.color": _asset_C_STROKE,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _asset_plot_r2_threshold_stacked_bars(
    group_name: str,
    refs: list[SuiteRef],
    threshold_rows: list[dict[str, object]],
    *,
    field_prefix: str,
    ylabel: str,
    title_metric: str,
    output_name: str,
    log_y: bool = False,
) -> Path | None:
    if not threshold_rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except Exception:
        return None

    _asset_apply_plot_style(plt)
    row_by_key = {(str(row["suite_id"]), str(row["policy_id"])): row for row in threshold_rows}
    missing_penalty = _asset_threshold_value_penalty(threshold_rows, field_prefix=field_prefix)
    policy_ids = sorted(
        {str(row["policy_id"]) for row in threshold_rows},
        key=lambda policy_id: _asset_policy_threshold_sort_key(
            policy_id,
            refs,
            row_by_key,
            field_prefix=field_prefix,
            missing_penalty=missing_penalty,
        ),
    )
    if not policy_ids:
        return None
    positive_values = [
        value
        for row in threshold_rows
        for threshold in _asset_R2_THRESHOLDS
        if (
            value := _asset_safe_float(
                row.get(f"{field_prefix}_{_asset_threshold_suffix(float(threshold))}")
            )
        )
        is not None
        and value > 0.0
    ]
    log_floor = min(positive_values) * 0.72 if log_y and positive_values else 0.0
    n_methods = len(policy_ids)
    group_gap = 3.0
    bar_width = 0.48
    x_positions: list[float] = []
    x_labels: list[str] = []
    group_centers: list[float] = []
    max_height = 1.0
    fig_width = max(6.8, 0.42 * n_methods * len(refs) + 0.35 * len(refs) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 3.45))

    for env_idx, ref in enumerate(refs):
        base = env_idx * (n_methods + group_gap)
        group_centers.append(base + (n_methods - 1) / 2.0)
        if env_idx > 0:
            ax.axvline(
                base - group_gap / 2.0,
                color=_asset_C_NEUTRAL_LIGHT,
                linewidth=0.7,
                alpha=0.85,
                zorder=1,
            )
        if env_idx % 2 == 1:
            ax.axvspan(
                base - 0.62,
                base + n_methods - 0.38,
                color=_asset_C_NEUTRAL_FILL,
                alpha=0.52,
                zorder=0,
            )
        for method_idx, policy_id in enumerate(policy_ids):
            x = base + method_idx
            x_positions.append(x)
            x_labels.append(_asset_short_policy_label(policy_id))
            row = row_by_key.get((ref.suite_id, policy_id), {})
            segments, reached_all = _asset_threshold_segments(row, field_prefix=field_prefix)
            bottom = 0.0
            for seg_idx, segment in enumerate(segments):
                if segment <= 0.0:
                    continue
                ax.bar(
                    x,
                    segment,
                    width=bar_width,
                    bottom=bottom,
                    color=_asset_R2_THRESHOLD_SEGMENT_COLORS[seg_idx],
                    edgecolor=_asset_C_STROKE,
                    linewidth=0.35,
                    zorder=3,
                )
                bottom += segment
            max_height = max(max_height, bottom)
            if bottom == 0.0:
                ax.plot(
                    [x - bar_width / 2.0, x + bar_width / 2.0],
                    [log_floor, log_floor],
                    color=_asset_C_NEUTRAL,
                    linewidth=0.7,
                    zorder=4,
                )
            if not reached_all:
                ax.scatter(
                    [x],
                    [bottom if bottom > 0.0 else log_floor],
                    marker="x",
                    s=12,
                    color=_asset_C_STROKE,
                    linewidths=0.6,
                    zorder=5,
                )

    for center, ref in zip(group_centers, refs):
        ax.text(
            center,
            -0.31,
            ref.label,
            ha="center",
            va="top",
            color=_asset_C_STROKE,
            transform=ax.get_xaxis_transform(),
            fontsize=7.5,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=5.8)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{group_name}: {title_metric} to trajectory R2 thresholds", pad=6.0)
    ax.set_xlim(min(x_positions) - 0.8, max(x_positions) + 0.8)
    if log_y and positive_values:
        ax.set_yscale("log")
        ax.set_ylim(log_floor, max_height * 1.45)
    else:
        ax.set_ylim(0.0, max_height * 1.16)
    ax.grid(axis="y", color=_asset_C_NEUTRAL_LIGHT, linewidth=0.35, alpha=0.38, zorder=1)
    for spine in ax.spines.values():
        spine.set_color(_asset_C_STROKE)
        spine.set_linewidth(0.55)
    legend_handles = [
        Patch(
            facecolor=_asset_R2_THRESHOLD_SEGMENT_COLORS[0],
            edgecolor=_asset_C_STROKE,
            linewidth=0.35,
            label="0 -> 0.90",
        ),
        Patch(
            facecolor=_asset_R2_THRESHOLD_SEGMENT_COLORS[1],
            edgecolor=_asset_C_STROKE,
            linewidth=0.35,
            label="0.90 -> 0.95",
        ),
        Patch(
            facecolor=_asset_R2_THRESHOLD_SEGMENT_COLORS[2],
            edgecolor=_asset_C_STROKE,
            linewidth=0.35,
            label="0.95 -> 0.99",
        ),
        Line2D(
            [0],
            [0],
            color=_asset_C_STROKE,
            marker="x",
            linestyle="None",
            markersize=4.5,
            markeredgewidth=0.7,
            label="threshold not reached",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        ncol=1,
        fontsize=6.5,
        handlelength=1.4,
    )
    fig.subplots_adjust(left=0.075, right=0.83, top=0.88, bottom=0.37)
    out_path = _asset_FIG_DIR / group_name / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _asset_plot_schedule_threshold_pareto() -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return None

    schedule_rows: list[dict[str, object]] = []
    for ref in GROUPS["exp03_schedule"]:
        schedule_rows.extend(
            row
            for row in _asset_threshold_rows_for_suite(ref)
            if str(row["policy_id"]).startswith("active_planning")
        )
    myopic_rows: list[dict[str, object]] = []
    for ref in GROUPS["exp01_base"]:
        myopic_rows.extend(
            row
            for row in _asset_threshold_rows_for_suite(ref)
            if str(row["policy_id"]) == "active_myopic"
        )
    rows = [*schedule_rows, *myopic_rows]
    if not rows:
        return None

    _asset_apply_plot_style(plt)
    env_labels = [ref.label for ref in GROUPS["exp03_schedule"]]
    policy_ids = sorted({str(row["policy_id"]) for row in rows}, key=_asset_policy_sort_key)
    policy_offsets = {
        policy_id: ((idx % 4) - 1.5, (idx // 4) - 0.5) for idx, policy_id in enumerate(policy_ids)
    }
    fig, axes = plt.subplots(1, len(env_labels), figsize=(8.2, 2.95), sharex=False, sharey=True)
    if len(env_labels) == 1:
        axes = [axes]

    max_step_seen = 1.0
    for ax, env_label in zip(axes, env_labels):
        env_rows = [row for row in rows if str(row["suite_label"]) == env_label]
        plotted_for_policy: dict[str, tuple[float, float, float]] = {}
        max_cpu_seen = 1.0
        for row in env_rows:
            policy_id = str(row["policy_id"])
            marker = "D" if policy_id == "active_myopic" else "o"
            for threshold in _asset_R2_THRESHOLDS:
                suffix = _asset_threshold_suffix(float(threshold))
                step = _asset_safe_float(row.get(f"step_to_r2_{suffix}"))
                cpu_time = _asset_safe_float(row.get(f"cpu_time_sec_to_r2_{suffix}"))
                if step is None or cpu_time is None:
                    continue
                ax.scatter(
                    cpu_time,
                    step,
                    s=24 if policy_id != "active_myopic" else 30,
                    marker=marker,
                    facecolor=_asset_R2_THRESHOLD_POINT_COLORS[float(threshold)],
                    edgecolor=_asset_C_STROKE,
                    linewidth=0.45,
                    alpha=0.92,
                    zorder=4,
                )
                max_step_seen = max(max_step_seen, step)
                max_cpu_seen = max(max_cpu_seen, cpu_time)
                if (
                    policy_id not in plotted_for_policy
                    or threshold > plotted_for_policy[policy_id][2]
                ):
                    plotted_for_policy[policy_id] = (cpu_time, step, threshold)
        for policy_id, (cpu_time, step, _threshold) in plotted_for_policy.items():
            dx, dy = policy_offsets.get(policy_id, (0.0, 0.0))
            ax.annotate(
                _asset_inline_policy_label(policy_id),
                (cpu_time, step),
                xytext=(4.0 + 3.0 * dx, 3.0 + 3.0 * dy),
                textcoords="offset points",
                fontsize=5.8,
                color=_asset_C_STROKE,
                ha="left",
                va="bottom",
                bbox={"facecolor": _asset_C_WHITE, "edgecolor": "none", "alpha": 0.72, "pad": 0.25},
            )
        ax.set_title(env_label, fontsize=8.0, pad=3.0)
        ax.set_xlabel("CPU time (sec)")
        ax.set_xlim(left=0.0, right=max_cpu_seen * 1.13)
        ax.grid(color=_asset_C_NEUTRAL_LIGHT, linewidth=0.32, alpha=0.36)
        for spine in ax.spines.values():
            spine.set_color(_asset_C_STROKE)
            spine.set_linewidth(0.55)
        ax.tick_params(width=0.45, length=2.0, colors=_asset_C_STROKE)
    axes[0].set_ylabel("Environment steps")
    axes[0].set_ylim(bottom=0.0, top=max_step_seen * 1.16)
    fig.suptitle("Exp03 schedule Pareto: time and steps to trajectory R2 thresholds", y=0.99)
    threshold_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=_asset_R2_THRESHOLD_POINT_COLORS[float(threshold)],
            markeredgecolor=_asset_C_STROKE,
            markeredgewidth=0.45,
            markersize=5.0,
            label=f"R2 {threshold:.2f}",
        )
        for threshold in _asset_R2_THRESHOLDS
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=_asset_C_WHITE,
            markeredgecolor=_asset_C_STROKE,
            markersize=5.0,
            label="Active schedule",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor=_asset_C_WHITE,
            markeredgecolor=_asset_C_STROKE,
            markersize=5.0,
            label="Myopic",
        ),
    ]
    fig.legend(
        handles=[*threshold_handles, *method_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=5,
        fontsize=6.4,
        columnspacing=0.9,
        handlelength=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.73, bottom=0.18, wspace=0.22)
    out_path = _asset_FIG_DIR / "exp03_schedule" / "r2_threshold_pareto_step_cpu_by_environment.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _asset_export_group(
    group_name: str, refs: list[SuiteRef]
) -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for ref in refs:
        rows.extend(_asset_aggregate_suite(ref))
        threshold_rows.extend(_asset_threshold_rows_for_suite(ref))
    rows.sort(
        key=lambda row: (str(row["suite_label"]), _asset_policy_sort_key(str(row["policy_id"])))
    )
    threshold_rows.sort(
        key=lambda row: (str(row["suite_label"]), _asset_policy_sort_key(str(row["policy_id"])))
    )
    csv_path = _asset_TEX_DIR / f"tbme_{group_name}_table.csv"
    tex_path = _asset_TEX_DIR / f"tbme_{group_name}_table.tex"
    _asset_write_csv(csv_path, rows)
    _asset_write_tex(tex_path, group_name, rows)
    _asset_write_threshold_csv(
        _asset_TEX_DIR / f"tbme_{group_name}_r2_threshold_table.csv", threshold_rows
    )
    _asset_write_threshold_tex(
        _asset_TEX_DIR / f"tbme_{group_name}_r2_threshold_table.tex", threshold_rows
    )
    copied = _asset_copy_figures(group_name, refs)
    threshold_plots = [
        _asset_plot_r2_threshold_stacked_bars(
            group_name,
            refs,
            threshold_rows,
            field_prefix="step_to_r2",
            ylabel="Environment steps",
            title_metric="steps",
            output_name="r2_threshold_stacked_steps_by_environment.pdf",
        ),
        _asset_plot_r2_threshold_stacked_bars(
            group_name,
            refs,
            threshold_rows,
            field_prefix="cpu_time_sec_to_r2",
            ylabel="CPU time (sec)",
            title_metric="CPU time",
            output_name="r2_threshold_stacked_cpu_time_by_environment.pdf",
            log_y=True,
        ),
    ]
    copied.extend(path for path in threshold_plots if path is not None)
    return rows, copied


def _summary_curve_rows(path: Path, value_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_trace_csv(path):
        if value_column not in row:
            continue
        payload: dict[str, Any] = dict(row)
        payload["value_mean"] = row[value_column]
        rows.append(payload)
    return rows


def _summary_style_colorbar(cbar: Any) -> None:
    cbar.outline.set_edgecolor(_TBME_STROKE_COLOR)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.tick_params(width=0.45, length=2.0, colors=_TBME_STROKE_COLOR)


def _summary_existing_records(suite_dir: Path, policy_ids: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    track_root = suite_dir / "track"
    if not track_root.exists():
        return records
    for policy_id in policy_ids:
        policy_dir = track_root / policy_id
        if not policy_dir.exists():
            continue
        seed_dirs: list[tuple[int, Path]] = []
        for seed_dir in policy_dir.glob("seed_*"):
            suffix = seed_dir.name.removeprefix("seed_")
            if suffix.isdigit():
                seed_dirs.append((int(suffix), seed_dir))
        for seed, seed_dir in sorted(seed_dirs):
            for metadata_path in find_nested_metadata_paths(seed_dir):
                records.append(
                    {
                        "policy_id": policy_id,
                        "seed": seed,
                        "run_dir": metadata_path.parent,
                        "metadata": load_json(metadata_path),
                    }
                )
    records.sort(key=lambda rec: (str(rec["policy_id"]), int(rec["seed"]), str(rec["run_dir"])))
    return records


def _summary_write_figures(suite_dir: Path, figure_formats: Sequence[str]) -> list[Path]:
    exp_spec = get_experiment_spec(suite_dir.name)
    summary_dir = suite_dir / "summary"
    figures_dir = summary_dir / "figures"
    value_prefix = "parameter_error"
    value_label = "Parameter Error"
    rows = read_trace_csv(summary_dir / "metrics.csv")
    trace_rows = _summary_curve_rows(
        summary_dir / f"{value_prefix}_over_steps.csv",
        f"{value_prefix}_mean",
    )
    traj_rows = _summary_curve_rows(
        summary_dir / "trajectory_r2_over_steps.csv", "trajectory_r2_mean"
    )
    cov_rows = _summary_curve_rows(
        summary_dir / "parameter_covariance_trace_over_steps.csv",
        "parameter_covariance_trace_mean",
    )
    plot_final_value_by_policy(
        figures_dir,
        rows=rows,
        ylabel=f"Final {value_label} (mean +/- SEM over seeds)",
        title=f"{value_label} by Policy",
        output_stem=f"final_{value_prefix}_by_policy",
        figure_formats=figure_formats,
        policy_sort_key=_policy_sort_key,
        policy_label=_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        stroke_color=_TBME_STROKE_COLOR,
    )
    plot_metric_over_steps(
        figures_dir,
        rows=trace_rows,
        ylabel=f"{value_label} (mean ± SEM)",
        title=f"{value_label} Over Steps",
        output_stem=f"{value_prefix}_over_steps",
        figure_formats=figure_formats,
        policy_sort_key=_policy_sort_key,
        policy_label=_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
    )
    plot_metric_over_cpu_time(
        figures_dir,
        rows=trace_rows,
        ylabel=f"{value_label} (mean ± SEM)",
        title=f"{value_label} Over CPU Time",
        output_stem=f"{value_prefix}_over_cpu_time",
        figure_formats=figure_formats,
        policy_sort_key=_policy_sort_key,
        policy_label=_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
    )
    plot_metric_over_steps(
        figures_dir,
        rows=traj_rows,
        ylabel="Trajectory R2 (mean ± SEM)",
        title="Trajectory R2 Over Steps",
        output_stem="trajectory_r2_over_steps",
        figure_formats=figure_formats,
        ylim=(0.0, 1.0),
        policy_sort_key=_policy_sort_key,
        policy_label=_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
    )
    plot_metric_over_cpu_time(
        figures_dir,
        rows=traj_rows,
        ylabel="Trajectory R2 (mean ± SEM)",
        title="Trajectory R2 Over CPU Time",
        output_stem="trajectory_r2_over_cpu_time",
        figure_formats=figure_formats,
        policy_sort_key=_policy_sort_key,
        policy_label=_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
    )
    plot_parameter_covariance_trace_over_steps(
        figures_dir,
        cov_rows=cov_rows,
        figure_formats=figure_formats,
        ylabel="Trace of Parameter Covariance (mean +/- SEM)",
        title="Trace of Parameter Covariance Over Steps",
        output_stem="parameter_covariance_trace_over_steps",
        policy_sort_key=_policy_sort_key,
        policy_label=_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
    )
    records = _summary_existing_records(suite_dir, exp_spec.policy_ids)
    plot_neuron_tuning_curve_colormap(
        figures_dir,
        records=records,
        figure_formats=figure_formats,
        get_environment_preset_from_metadata=get_environment_preset_from_metadata,
        reconstruct_loglinear_rate_model=reconstruct_loglinear_rate_model,
        expected_loglinear_rate_hz=expected_loglinear_rate_hz,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        style_colorbar=_summary_style_colorbar,
        output_stem="neuron_tuning_curve_colormap",
        axis_labels=("x", "v"),
        colorbar_label="Total firing rate (Hz)",
        title_template="Total firing rate (mean over {n_seeds} seeds)",
    )
    plot_information_colormap(
        figures_dir,
        records=records,
        figure_formats=figure_formats,
        get_environment_preset_from_metadata=get_environment_preset_from_metadata,
        reconstruct_loglinear_rate_model=reconstruct_loglinear_rate_model,
        expected_loglinear_rate_hz=expected_loglinear_rate_hz,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        style_colorbar=_summary_style_colorbar,
        output_stem="I_z_t_colormap",
        axis_labels=("x", "v"),
        colorbar_label="log det(I_z)",
        title_template="log det(I_z) (mean over {n_seeds} seeds)",
    )
    stems = [
        f"final_{value_prefix}_by_policy",
        f"{value_prefix}_over_steps",
        f"{value_prefix}_over_cpu_time",
        "trajectory_r2_over_steps",
        "trajectory_r2_over_cpu_time",
        "parameter_covariance_trace_over_steps",
        "neuron_tuning_curve_colormap",
        "I_z_t_colormap",
    ]
    return [
        figures_dir / f"{stem}{fmt}"
        for stem in stems
        for fmt in figure_formats
        if (figures_dir / f"{stem}{fmt}").exists()
    ]


def _summary_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write current TBME summary figures from existing summary CSV files."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(GROUPS),
        help="Comma-separated TBME group names.",
    )
    parser.add_argument("--figure-formats", type=str, default=".pdf")
    return parser


def _trajectory_policy_sort_key(policy_id: str) -> tuple[int, str]:
    try:
        return POLICY_ORDER.index(policy_id), policy_id
    except ValueError:
        return len(POLICY_ORDER), policy_id


def _trajectory_policy_label(policy_id: str) -> str:
    labels = {
        "active_planning": "Planning",
        "active_planning_u1_r1_h40": "Planning U1/R1",
        "active_planning_u5_r5_h40": "Planning U5/R5",
        "active_planning_u1_r5_h40": "Planning U1/R5",
        "active_planning_u10_r10_h40": "Planning U10/R10",
        "active_planning_u5_r10_h40": "Planning U5/R10",
        "active_planning_u20_r20_h40": "Planning U20/R20",
        "active_planning_u5_r20_h40": "Planning U5/R20",
        "active_planning_u10_r20_h40": "Planning U10/R20",
        "active_myopic": "Myopic",
        "prbs": "PRBS",
        "random": "Random",
        "flex": "FLEX",
        "flex_true_state": "FLEX true state",
        "ensemble": "Ensemble",
        "rhc": "RHC-US",
    }
    return labels.get(policy_id, policy_id.replace("_", " "))


def _trajectory_read_trace_xy(path: Path) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for row in read_trace_csv(path):
        try:
            x_val = float(row["true_x"])
            v_val = float(row["true_v"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x_val) and math.isfinite(v_val):
            pts.append((x_val, v_val))
    return np.asarray(pts, dtype=np.float32)


def _trajectory_collect_policy_traces(
    suite_dir: Path,
    *,
    max_seeds: int,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    track_dir = suite_dir / "track"
    grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
    if not track_dir.exists():
        return grouped
    for policy_dir in sorted(
        (p for p in track_dir.iterdir() if p.is_dir()),
        key=lambda p: _trajectory_policy_sort_key(p.name),
    ):
        traces: list[tuple[int, np.ndarray]] = []
        seed_dirs: list[tuple[int, Path]] = []
        for seed_dir in policy_dir.glob("seed_*"):
            suffix = seed_dir.name.removeprefix("seed_")
            if suffix.isdigit():
                seed_dirs.append((int(suffix), seed_dir))
        for seed, seed_dir in sorted(seed_dirs)[:max_seeds]:
            trace_path = None
            for repeat_dir in sorted(seed_dir.glob("repeat_*")):
                candidate = repeat_dir / "state_action_trace.csv"
                if candidate.exists():
                    trace_path = candidate
                    break
            if trace_path is None:
                continue
            traj = _trajectory_read_trace_xy(trace_path)
            if traj.size:
                traces.append((seed, traj))
        if traces:
            grouped[policy_dir.name] = traces
    return grouped


def _trajectory_reference_metadata(suite_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(suite_dir.glob("track/*/seed_*/repeat_*/run_metadata.json"))
    if not candidates:
        return None
    return load_json(candidates[0])


def _trajectory_build_true_dynamics(metadata: dict[str, Any]) -> tuple[Any, float, str] | None:
    env_preset = get_environment_preset_from_metadata(metadata)
    if bool(getattr(env_preset, "real_data", False)):
        return None
    theta_true = np.asarray(metadata.get("embedding_true", [0.0, 0.0]), dtype=np.float32)
    dynamics_alpha = float(metadata.get("dynamics_alpha", 0.7))
    grid_lim = float(env_preset.resolved_plot_limit())
    dyn_true = ResidualDynamicsCallable(
        dynamics_type=env_preset.resolved_dynamics_type(),
        dyn_params=env_preset.params_from_embedding(theta_true),
        dynamics_alpha=dynamics_alpha,
        device="cpu",
    )
    label = str(getattr(env_preset, "system_label", None) or env_preset.system_id)
    return dyn_true, grid_lim, label


def _trajectory_seed_color_map(seeds: list[int]) -> dict[int, tuple[float, float, float, float]]:
    if not seeds:
        return {}
    cmap = plt.get_cmap("turbo")
    denom = max(len(seeds) - 1, 1)
    return {seed: cmap(idx / denom) for idx, seed in enumerate(sorted(seeds))}


def _trajectory_layout(n_panels: int) -> tuple[int, int, tuple[float, float]]:
    if n_panels <= 1:
        return 1, 1, (3.0, 2.8)
    if n_panels <= 4:
        n_cols = 2
    elif n_panels <= 8:
        n_cols = 4
    else:
        n_cols = 3
    n_rows = int(math.ceil(n_panels / n_cols))
    return n_rows, n_cols, (2.35 * n_cols, 2.25 * n_rows)


def _trajectory_format_axis(ax: Any, grid_lim: float, *, title: str) -> None:
    ax.set_xlim(-grid_lim, grid_lim)
    ax.set_ylim(-grid_lim, grid_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8.0, color=_trajectory_C_STROKE, pad=2.0)
    ax.set_xlabel("x", labelpad=1.5)
    ax.set_ylabel("v", labelpad=1.5)
    ax.grid(color=_trajectory_C_NEUTRAL_LIGHT, linewidth=0.28, alpha=0.28)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(_trajectory_C_STROKE)


def _trajectory_trajectory_plot_limit(
    grouped: dict[str, list[tuple[int, np.ndarray]]],
    grid_lim: float,
) -> float:
    max_abs = float(grid_lim)
    for traces in grouped.values():
        for _seed, traj in traces:
            if traj.size == 0:
                continue
            finite = traj[np.isfinite(traj).all(axis=1)]
            if finite.size:
                max_abs = max(max_abs, float(np.max(np.abs(finite[:, :2]))))
    return max(max_abs * 1.08, float(grid_lim))


def _trajectory_plot_vectorfield_background(ax: Any, dyn_true: Any, grid_lim: float) -> None:
    x_grid, y_grid, u_grid, v_grid = compute_vector_field(
        dyn_true,
        x_range=grid_lim,
        n_grid=36,
        is_residual=True,
        device="cpu",
    )
    ax.streamplot(
        x_grid.cpu().numpy(),
        y_grid.cpu().numpy(),
        u_grid.cpu().numpy(),
        v_grid.cpu().numpy(),
        color=_trajectory_C_NEUTRAL_LIGHT,
        linewidth=0.34,
        density=1.35,
        arrowsize=0.55,
        zorder=1,
    )


def _trajectory_trajectory_density_cmap() -> Any:
    try:
        import seaborn as sns

        return sns.color_palette("crest", as_cmap=True)
    except Exception:
        return plt.get_cmap("viridis")


def _trajectory_plot_overlay_figure(
    suite_dir: Path,
    *,
    grouped: dict[str, list[tuple[int, np.ndarray]]],
    dyn_true: Any,
    grid_lim: float,
    system_label: str,
    max_seeds: int,
) -> Path:
    policies = sorted(grouped, key=_trajectory_policy_sort_key)
    plot_lim = _trajectory_trajectory_plot_limit(grouped, grid_lim)
    seeds = sorted({seed for traces in grouped.values() for seed, _traj in traces})
    seed_colors = _trajectory_seed_color_map(seeds)
    n_rows, n_cols, figsize = _trajectory_layout(len(policies))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True, sharey=True
    )
    for idx, policy_id in enumerate(policies):
        ax = axes[idx // n_cols, idx % n_cols]
        _trajectory_plot_vectorfield_background(ax, dyn_true, plot_lim)
        traces = grouped[policy_id]
        for seed, traj in traces:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=seed_colors.get(seed, _trajectory_C_WRITE),
                linewidth=0.55,
                alpha=0.72,
                zorder=3,
            )
            ax.scatter(
                traj[0, 0],
                traj[0, 1],
                s=5.0,
                color=seed_colors.get(seed, _trajectory_C_WRITE),
                edgecolors="none",
                alpha=0.95,
                zorder=4,
            )
        _trajectory_format_axis(
            ax, plot_lim, title=f"{_trajectory_policy_label(policy_id)}  n={len(traces)}"
        )
    for idx in range(len(policies), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle(
        f"{suite_dir.name}: trajectory overlays on true {system_label} vector field "
        f"(first {max_seeds} seeds)",
        fontsize=9.0,
        color=_trajectory_C_STROKE,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path = suite_dir / "summary" / "figures" / "trajectory_overlay_vectorfield_by_policy.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _trajectory_histogram(
    traces: list[tuple[int, np.ndarray]], grid_lim: float, bins: int
) -> np.ndarray:
    if not traces:
        return np.zeros((bins, bins), dtype=np.float64)
    pts = np.concatenate([traj[:, :2] for _seed, traj in traces if traj.size], axis=0)
    if pts.size == 0:
        return np.zeros((bins, bins), dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    hist, _x_edges, _y_edges = np.histogram2d(
        pts[:, 0],
        pts[:, 1],
        bins=bins,
        range=[[-grid_lim, grid_lim], [-grid_lim, grid_lim]],
    )
    return hist.T


def _trajectory_plot_density_figure(
    suite_dir: Path,
    *,
    grouped: dict[str, list[tuple[int, np.ndarray]]],
    dyn_true: Any,
    grid_lim: float,
    system_label: str,
    max_seeds: int,
    bins: int,
) -> Path:
    policies = sorted(grouped, key=_trajectory_policy_sort_key)
    plot_lim = _trajectory_trajectory_plot_limit(grouped, grid_lim)
    hists = {
        policy_id: _trajectory_histogram(grouped[policy_id], plot_lim, bins)
        for policy_id in policies
    }
    max_count = max((float(np.nanmax(hist)) for hist in hists.values() if hist.size), default=1.0)
    max_count = max(max_count, 1.0)
    max_log_count = float(np.log10(max_count + 1.0))
    norm = Normalize(vmin=0.0, vmax=max_log_count)
    cmap = _trajectory_trajectory_density_cmap()

    n_rows, n_cols, figsize = _trajectory_layout(len(policies))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True, sharey=True
    )
    im = None
    for idx, policy_id in enumerate(policies):
        ax = axes[idx // n_cols, idx % n_cols]
        _trajectory_plot_vectorfield_background(ax, dyn_true, plot_lim)
        hist = np.log10(hists[policy_id] + 1.0)
        im = ax.imshow(
            hist,
            origin="lower",
            extent=(-plot_lim, plot_lim, -plot_lim, plot_lim),
            cmap=cmap,
            norm=norm,
            alpha=0.7,
            interpolation="nearest",
            zorder=2,
        )
        _trajectory_format_axis(
            ax,
            plot_lim,
            title=f"{_trajectory_policy_label(policy_id)}  n={len(grouped[policy_id])}",
        )
    for idx in range(len(policies), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle(
        f"{suite_dir.name}: trajectory density on true {system_label} state space "
        f"(first {max_seeds} seeds)",
        fontsize=9.0,
        color=_trajectory_C_STROKE,
        y=0.995,
    )
    fig.subplots_adjust(left=0.065, right=0.895, bottom=0.075, top=0.91, wspace=0.22, hspace=0.32)
    if im is None:
        im = ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.915, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log10(1 + trajectory samples per bin)", color=_trajectory_C_STROKE)
    cbar.outline.set_linewidth(0.45)
    out_path = suite_dir / "summary" / "figures" / "trajectory_density_by_policy.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _trajectory_suite_dirs(groups: list[str] | None) -> list[Path]:
    selected = GROUPS if groups is None else {name: GROUPS[name] for name in groups}
    suite_dirs: list[Path] = []
    for refs in selected.values():
        for ref in refs:
            suite_dir = ref.session_root / ref.suite_id
            if suite_dir.exists():
                suite_dirs.append(suite_dir)
    return suite_dirs


def _trajectory_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TBME trajectory overlay and density summary figures."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(GROUPS),
        help="Comma-separated TBME group names.",
    )
    parser.add_argument("--max-seeds", type=int, default=50)
    parser.add_argument("--density-bins", type=int, default=96)
    return parser


def _requested_parse_plots(raw: str) -> list[str]:
    plot_ids = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(plot_ids) - set(_requested_PLOTS))
    if unknown:
        raise ValueError(f"Unknown requested plot(s): {', '.join(unknown)}")
    return plot_ids


def _requested_required_suite_dirs(plot_ids: Sequence[str]) -> list[Path]:
    suite_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for plot_id in plot_ids:
        for key in _requested_REQUIRED_SUITES_BY_PLOT[plot_id]:
            if key not in seen:
                suite_keys.append(key)
                seen.add(key)
    return [_suite_dir(group_name, suite_id) for group_name, suite_id in suite_keys]


def _requested_write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _requested_safe_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _requested_sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def _requested_policy_sort_key(policy_id: str) -> tuple[int, str]:
    try:
        return POLICY_ORDER.index(policy_id), policy_id
    except ValueError:
        return len(POLICY_ORDER), policy_id


def _requested_policy_label(policy_id: str) -> str:
    short = {
        "active_planning_u20_r20_h40": "Planning",
        "active_fully_observable_u20_r20_h40": "Full obs.",
        "active_e_optimality_u20_r20_h40": "E-opt.",
        "active_state_information_u20_r20_h40": "State info",
        "active_dynamics_u20_r20_h40": "Dynamics",
        "active_sampling_variance_u20_r20_h40": "Sampling var.",
        "active_myopic": "Myopic",
        "ensemble": "Ensemble",
        "prbs": "PRBS",
        "random": "Random",
    }
    return short.get(policy_id, POLICY_LABELS.get(policy_id, policy_id.replace("_", " ")))


def _requested_policy_color(policy_id: str) -> str:
    return _requested_POLICY_COLORS.get(policy_id, _requested_C_NEUTRAL)


def _requested_escape_tex(text: object) -> str:
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _requested_suite_dir(group_name: str, exp_id: str) -> Path:
    return _suite_dir(group_name, exp_id)


def _requested_metrics_by_policy(suite_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows = read_trace_csv(suite_dir / "summary" / "metrics.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") not in {None, "", "completed"}:
            continue
        grouped.setdefault(str(row.get("policy_id", "")), []).append(row)
    return grouped


def _requested_metric_values(suite_dir: Path, policy_id: str, field: str) -> list[float]:
    values: list[float] = []
    for row in _requested_metrics_by_policy(suite_dir).get(policy_id, []):
        value = _requested_safe_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _requested_metric_mean_sem(
    suite_dir: Path, policy_id: str, field: str
) -> tuple[float | None, float, int]:
    values = _requested_metric_values(suite_dir, policy_id, field)
    if not values:
        return None, 0.0, 0
    return float(np.mean(values)), _requested_sem(values), len(values)


def _requested_curve_rows(
    suite_dir: Path, name: str, value_col: str
) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in read_trace_csv(suite_dir / "summary" / name):
        policy_id = str(row.get("policy_id", ""))
        step = _requested_safe_float(row.get("step"))
        value = _requested_safe_float(row.get(value_col))
        sem = _requested_safe_float(row.get("value_sem"))
        if not policy_id or step is None or value is None:
            continue
        grouped.setdefault(policy_id, []).append(
            {"step": step, "value": value, "sem": 0.0 if sem is None else sem}
        )
    for policy_rows in grouped.values():
        policy_rows.sort(key=lambda row: row["step"])
    return grouped


def _requested_r2_threshold_step(
    suite_dir: Path, policy_id: str, threshold: float = 0.90
) -> float | None:
    suffix = f"{float(threshold):.2f}".replace(".", "p")
    for row in read_trace_csv(suite_dir / "summary" / "trajectory_r2_thresholds.csv"):
        if str(row.get("policy_id", "")) != policy_id:
            continue
        return _requested_safe_float(row.get(f"step_to_r2_{suffix}"))
    return None


def _requested_r2_threshold_times(
    suite_dir: Path,
    policy_id: str,
    threshold: float,
) -> tuple[float | None, float | None, float | None]:
    suffix = f"{float(threshold):.2f}".replace(".", "p")
    for row in read_trace_csv(suite_dir / "summary" / "trajectory_r2_thresholds.csv"):
        if str(row.get("policy_id", "")) != policy_id:
            continue
        return (
            _requested_safe_float(row.get(f"step_to_r2_{suffix}")),
            _requested_safe_float(row.get(f"cpu_time_sec_to_r2_{suffix}")),
            _requested_safe_float(row.get(f"r2_at_{suffix}")),
        )
    return None, None, None


def _requested_save_pdf(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    return path


def _requested_plot_bottleneck_sweep(plt: Any) -> tuple[Path, Path]:
    sources = [
        _requested_SuiteSource(
            "exp01_asymmetric_basin",
            "Nominal",
            _requested_suite_dir("exp01_base", "exp01_asymmetric_basin"),
        ),
        _requested_SuiteSource(
            "exp06_asymmetric_basin_bottleneck_weak_observation",
            "Weak obs.",
            _requested_suite_dir(
                "exp06_bottleneck",
                "exp06_asymmetric_basin_bottleneck_weak_observation",
            ),
        ),
        _requested_SuiteSource(
            "exp06_asymmetric_basin_bottleneck_tight_action",
            "Tight action",
            _requested_suite_dir(
                "exp06_bottleneck",
                "exp06_asymmetric_basin_bottleneck_tight_action",
            ),
        ),
        _requested_SuiteSource(
            "exp06_asymmetric_basin_bottleneck_combined",
            "Combined",
            _requested_suite_dir(
                "exp06_bottleneck",
                "exp06_asymmetric_basin_bottleneck_combined",
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for source in sources:
        for policy_id in _requested_BOTTLENECK_POLICIES:
            r2, r2_sem, n_r2 = _requested_metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            step_to_r2 = _requested_r2_threshold_step(source.suite_dir, policy_id, threshold=0.90)
            rows.append(
                {
                    "experiment": source.exp_id,
                    "condition": source.label,
                    "policy_id": policy_id,
                    "policy_label": _requested_policy_label(policy_id),
                    "trajectory_r2_mean": r2,
                    "trajectory_r2_sem": r2_sem,
                    "step_to_r2_0p90": step_to_r2,
                    "n_r2": n_r2,
                }
            )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95), sharex=True)
    x = np.arange(len(sources), dtype=np.float64)
    offsets = np.linspace(-0.30, 0.30, len(_requested_BOTTLENECK_POLICIES))
    max_step = 1.0
    finite_r2: list[float] = []
    for idx, policy_id in enumerate(_requested_BOTTLENECK_POLICIES):
        color = _requested_policy_color(policy_id)
        r2_y = []
        r2_sem = []
        step_y = []
        missing_x = []
        for source in sources:
            match = [
                row
                for row in rows
                if row["condition"] == source.label and row["policy_id"] == policy_id
            ][0]
            r2_y.append(
                np.nan if match["trajectory_r2_mean"] is None else match["trajectory_r2_mean"]
            )
            r2_sem.append(match["trajectory_r2_sem"])
            step = match["step_to_r2_0p90"]
            if step is None:
                step_y.append(np.nan)
                missing_x.append(x[len(step_y) - 1] + offsets[idx])
            else:
                step_y.append(float(step))
                max_step = max(max_step, float(step))
        xpos = x + offsets[idx]
        axes[0].errorbar(
            xpos,
            r2_y,
            yerr=r2_sem,
            fmt="o-",
            color=color,
            linewidth=1.0,
            markersize=3.4,
            capsize=2.0,
            label=_requested_policy_label(policy_id),
        )
        finite_r2.extend(float(v) for v in r2_y if np.isfinite(v))
        axes[1].plot(
            xpos,
            step_y,
            marker="o",
            color=color,
            linewidth=1.0,
            markersize=3.4,
            label=_requested_policy_label(policy_id),
        )
        if missing_x:
            axes[1].scatter(
                missing_x,
                [max_step * 1.04 for _ in missing_x],
                marker="x",
                s=14,
                color=color,
                linewidths=0.75,
            )
    for ax in axes:
        _style_requested_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([source.label for source in sources], rotation=18, ha="right")
    axes[0].set_ylabel("Final prediction R2")
    axes[0].set_ylim(min(-0.1, min(finite_r2) - 0.05) if finite_r2 else -0.1, 1.05)
    axes[0].set_title("A. Prediction under bottlenecks")
    axes[1].set_ylabel("Steps to prediction R2 >= 0.90")
    axes[1].set_ylim(0.0, max_step * 1.15)
    axes[1].set_title("B. Predictive sample efficiency")
    axes[1].legend(loc="upper left", fontsize=6.6, ncol=1)
    fig.tight_layout(w_pad=1.1)

    csv_path = _requested_GENERATED_DIR / "tbme_requested_bottleneck_sweep.csv"
    _requested_write_csv(
        csv_path,
        rows,
        [
            "experiment",
            "condition",
            "policy_id",
            "policy_label",
            "trajectory_r2_mean",
            "trajectory_r2_sem",
            "step_to_r2_0p90",
            "n_r2",
        ],
    )
    return (
        _requested_save_pdf(fig, _requested_FIGURE_DIR / "tbme_requested_bottleneck_sweep.pdf"),
        csv_path,
    )


def _requested_objective_sources() -> list[_requested_SuiteSource]:
    return [
        _requested_SuiteSource(
            "exp05_asymmetric_basin_objective_ablation",
            "Nominal asymmetric basin",
            _requested_suite_dir(
                "exp05_ablation",
                "exp05_asymmetric_basin_objective_ablation",
            ),
        ),
        _requested_SuiteSource(
            "exp05_hard_asymmetric_basin_objective_ablation",
            "Hard asymmetric basin",
            _requested_suite_dir(
                "exp05_ablation",
                "exp05_hard_asymmetric_basin_objective_ablation",
            ),
        ),
    ]


def _requested_write_objective_definition_tables() -> tuple[Path, Path]:
    rows = [
        {
            "policy_id": str(row["policy_id"]),
            "policy_label": _requested_policy_label(str(row["policy_id"])),
            "objective_name": str(row["objective_name"]),
            "objective_formula": str(row["objective_formula"]),
            "objective_notes": str(row["objective_notes"]),
        }
        for row in _requested_OBJECTIVE_DEFINITIONS
    ]
    csv_path = _requested_GENERATED_DIR / "tbme_requested_objective_ablation_objectives.csv"
    _requested_write_csv(
        csv_path,
        rows,
        [
            "policy_id",
            "policy_label",
            "objective_name",
            "objective_formula",
            "objective_notes",
        ],
    )

    tex_path = _requested_GENERATED_DIR / "tbme_requested_objective_ablation_objectives.tex"
    lines = [
        "% Auto-generated by experiments/tbme/generate_figures.py requested",
        (
            r"\noindent Active policies maximize \(J(u_{0:H-1})\) over candidate "
            r"action sequences; the runtime minimizes \(-J\). Here "
            r"\(S_i=\partial z_i/\partial\theta\), \(P_\theta\) is the current "
            r"parameter covariance, \(P_i^-\) is the predicted latent covariance, "
            r"\(I_{z,i}=H_i^\top R_i^{-1}H_i\), \(H_i=\partial h/\partial z_i\), "
            r"and \(\gamma\) is the horizon discount."
        ),
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Policy & Objective & Maximized acquisition \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _requested_escape_tex(row["policy_label"]),
                    _requested_escape_tex(row["objective_name"]),
                    str(row["objective_formula"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def _requested_plot_objective_ablation(plt: Any) -> tuple[Path, Path]:
    sources = _requested_objective_sources()
    threshold = 0.95
    metric_rows: list[dict[str, Any]] = []
    curves_by_source: dict[str, dict[str, list[dict[str, float]]]] = {}
    for source in sources:
        curves_by_source[source.exp_id] = _requested_curve_rows(
            source.suite_dir,
            "trajectory_r2_over_steps.csv",
            "trajectory_r2_mean",
        )
        for policy_id in _requested_OBJECTIVE_POLICIES:
            err, err_sem, n_err = _requested_metric_mean_sem(
                source.suite_dir, policy_id, "value_final_mean"
            )
            r2, r2_sem, n_r2 = _requested_metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            step_to_r2, cpu_time_to_r2, r2_at_threshold = _requested_r2_threshold_times(
                source.suite_dir,
                policy_id,
                threshold,
            )
            metric_rows.append(
                {
                    "experiment": source.exp_id,
                    "condition": source.label,
                    "policy_id": policy_id,
                    "policy_label": _requested_policy_label(policy_id),
                    "parameter_error_mean": err,
                    "parameter_error_sem": err_sem,
                    "trajectory_r2_mean": r2,
                    "trajectory_r2_sem": r2_sem,
                    "step_to_r2_0p95": step_to_r2,
                    "cpu_time_sec_to_r2_0p95": cpu_time_to_r2,
                    "r2_at_0p95": r2_at_threshold,
                    "n_error": n_err,
                    "n_r2": n_r2,
                }
            )

    fig, axes = plt.subplots(len(sources), 2, figsize=(7.15, 5.05), squeeze=False)
    x = np.arange(len(_requested_OBJECTIVE_POLICIES), dtype=np.float64)
    x_labels = [_requested_policy_label(policy_id) for policy_id in _requested_OBJECTIVE_POLICIES]
    letters = ["A", "B", "C", "D"]
    for source_idx, source in enumerate(sources):
        row_metrics = [row for row in metric_rows if row["experiment"] == source.exp_id]
        bars = [
            np.nan if row["step_to_r2_0p95"] is None else row["step_to_r2_0p95"]
            for row in row_metrics
        ]
        colors = [_requested_policy_color(str(row["policy_id"])) for row in row_metrics]
        ax_bar = axes[source_idx, 0]
        ax_curve = axes[source_idx, 1]
        ax_bar.bar(x, bars, color=colors, edgecolor=_requested_C_STROKE, linewidth=0.45)
        finite_steps = [float(v) for v in bars if np.isfinite(v)]
        max_step = max(finite_steps) if finite_steps else 1.0
        missing_x = [float(x[idx]) for idx, value in enumerate(bars) if not np.isfinite(value)]
        if missing_x:
            ax_bar.scatter(
                missing_x,
                [max_step * 1.05 for _ in missing_x],
                marker="x",
                s=15,
                color=_requested_C_STROKE,
                linewidths=0.8,
            )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(x_labels, rotation=35, ha="right")
        ax_bar.set_ylabel("Steps to prediction R2 >= 0.95")
        ax_bar.set_ylim(0.0, max_step * 1.18)
        ax_bar.set_title(f"{letters[2 * source_idx]}. {source.label}: threshold")
        _style_requested_axis(ax_bar)

        curves = curves_by_source[source.exp_id]
        for policy_id in _requested_OBJECTIVE_POLICIES:
            rows = curves.get(policy_id, [])
            if not rows:
                continue
            steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
            values = np.asarray([row["value"] for row in rows], dtype=np.float64)
            sem = np.asarray([row["sem"] for row in rows], dtype=np.float64)
            color = _requested_policy_color(policy_id)
            ax_curve.plot(
                steps,
                values,
                color=color,
                linewidth=1.0,
                label=_requested_policy_label(policy_id),
            )
            if np.any(sem > 0):
                ax_curve.fill_between(
                    steps,
                    values - sem,
                    values + sem,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
        ax_curve.axhline(0.95, color=_requested_C_NEUTRAL_LIGHT, linestyle="--", linewidth=0.7)
        ax_curve.set_xlabel("Environment step")
        ax_curve.set_ylabel("Prediction R2")
        ax_curve.set_ylim(-0.1, 1.05)
        ax_curve.set_title(f"{letters[2 * source_idx + 1]}. {source.label}: recovery")
        _style_requested_axis(ax_curve)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=6.2,
        columnspacing=0.9,
        handlelength=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=1.0, h_pad=1.15)

    csv_path = _requested_GENERATED_DIR / "tbme_requested_objective_ablation.csv"
    _requested_write_csv(
        csv_path,
        metric_rows,
        [
            "experiment",
            "condition",
            "policy_id",
            "policy_label",
            "trajectory_r2_mean",
            "trajectory_r2_sem",
            "step_to_r2_0p95",
            "cpu_time_sec_to_r2_0p95",
            "r2_at_0p95",
            "parameter_error_mean",
            "parameter_error_sem",
            "n_error",
            "n_r2",
        ],
    )
    return (
        _requested_save_pdf(
            fig, _requested_FIGURE_DIR / "tbme_requested_objective_ablation_asymmetric_basin.pdf"
        ),
        csv_path,
    )


def _requested_dose_sources() -> list[_requested_SuiteSource]:
    return [
        _requested_SuiteSource(
            "exp01_duffing",
            "None",
            _requested_suite_dir("exp01_base", "exp01_duffing"),
            dose="none",
            family="Duffing",
        ),
        _requested_SuiteSource(
            "exp07_duffing_parameter_mismatch_mild",
            "Mild",
            _requested_suite_dir(
                "exp07_mismatch_stress",
                "exp07_duffing_parameter_mismatch_mild",
            ),
            dose="mild",
            family="Duffing",
        ),
        _requested_SuiteSource(
            "exp04_duffing_parameter_mismatch",
            "Medium",
            _requested_suite_dir("exp04_mismatch", "exp04_duffing_parameter_mismatch"),
            dose="medium",
            family="Duffing",
        ),
        _requested_SuiteSource(
            "exp07_duffing_parameter_mismatch_strong",
            "Strong",
            _requested_suite_dir(
                "exp07_mismatch_stress",
                "exp07_duffing_parameter_mismatch_strong",
            ),
            dose="strong",
            family="Duffing",
        ),
        _requested_SuiteSource(
            "exp01_asymmetric_basin",
            "None",
            _requested_suite_dir("exp01_base", "exp01_asymmetric_basin"),
            dose="none",
            family="Asymmetric basin",
        ),
        _requested_SuiteSource(
            "exp07_asymmetric_basin_parameter_mismatch_mild",
            "Mild",
            _requested_suite_dir(
                "exp07_mismatch_stress",
                "exp07_asymmetric_basin_parameter_mismatch_mild",
            ),
            dose="mild",
            family="Asymmetric basin",
        ),
        _requested_SuiteSource(
            "exp04_asymmetric_basin_parameter_mismatch",
            "Medium",
            _requested_suite_dir("exp04_mismatch", "exp04_asymmetric_basin_parameter_mismatch"),
            dose="medium",
            family="Asymmetric basin",
        ),
        _requested_SuiteSource(
            "exp07_asymmetric_basin_parameter_mismatch_strong",
            "Strong",
            _requested_suite_dir(
                "exp07_mismatch_stress",
                "exp07_asymmetric_basin_parameter_mismatch_strong",
            ),
            dose="strong",
            family="Asymmetric basin",
        ),
    ]


def _requested_plot_mismatch_dose_response(plt: Any) -> tuple[Path, Path]:
    sources = _requested_dose_sources()
    rows: list[dict[str, Any]] = []
    for source in sources:
        for policy_id in _requested_DOSE_POLICIES:
            err, err_sem, n_err = _requested_metric_mean_sem(
                source.suite_dir, policy_id, "value_final_mean"
            )
            r2, r2_sem, n_r2 = _requested_metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            rows.append(
                {
                    "family": source.family,
                    "dose": source.dose,
                    "dose_label": source.label,
                    "experiment": source.exp_id,
                    "policy_id": policy_id,
                    "policy_label": _requested_policy_label(policy_id),
                    "parameter_error_mean": err,
                    "parameter_error_sem": err_sem,
                    "trajectory_r2_mean": r2,
                    "trajectory_r2_sem": r2_sem,
                    "n_error": n_err,
                    "n_r2": n_r2,
                }
            )

    dose_order = ["none", "mild", "medium", "strong"]
    dose_labels = ["None", "Mild", "Medium", "Strong"]
    x = np.arange(len(dose_order), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=False)
    for ax, family in zip(axes, ["Duffing", "Asymmetric basin"], strict=True):
        family_rows = [row for row in rows if row["family"] == family]
        for policy_id in _requested_DOSE_POLICIES:
            y = []
            yerr = []
            for dose in dose_order:
                match = [
                    row
                    for row in family_rows
                    if row["dose"] == dose and row["policy_id"] == policy_id
                ]
                if not match or match[0]["trajectory_r2_mean"] is None:
                    y.append(np.nan)
                    yerr.append(0.0)
                else:
                    y.append(float(match[0]["trajectory_r2_mean"]))
                    yerr.append(float(match[0]["trajectory_r2_sem"]))
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.0,
                markersize=3.4,
                capsize=2.0,
                color=_requested_policy_color(policy_id),
                label=_requested_policy_label(policy_id),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(dose_labels)
        ax.set_ylabel("Final prediction R2")
        finite_family_r2 = [
            float(row["trajectory_r2_mean"])
            for row in family_rows
            if row.get("trajectory_r2_mean") is not None
            and np.isfinite(float(row["trajectory_r2_mean"]))
        ]
        ax.set_ylim(
            min(-0.1, min(finite_family_r2) - 0.05) if finite_family_r2 else -0.1,
            1.05,
        )
        ax.set_title(f"{family} mismatch dose-response")
        _style_requested_axis(ax)
    axes[1].legend(loc="upper left", fontsize=6.4)
    fig.tight_layout(w_pad=1.0)

    csv_path = _requested_GENERATED_DIR / "tbme_requested_mismatch_dose_response.csv"
    _requested_write_csv(
        csv_path,
        rows,
        [
            "family",
            "dose",
            "dose_label",
            "experiment",
            "policy_id",
            "policy_label",
            "trajectory_r2_mean",
            "trajectory_r2_sem",
            "parameter_error_mean",
            "parameter_error_sem",
            "n_error",
            "n_r2",
        ],
    )
    return (
        _requested_save_pdf(
            fig, _requested_FIGURE_DIR / "tbme_requested_mismatch_dose_response.pdf"
        ),
        csv_path,
    )


def _requested_collect_records(
    suite_dir: Path, policy_ids: Sequence[str]
) -> list[_requested_RunRecord]:
    records: list[_requested_RunRecord] = []
    for policy_id in sorted(policy_ids, key=_requested_policy_sort_key):
        policy_dir = suite_dir / "track" / policy_id
        if not policy_dir.exists():
            continue
        for seed_dir in sorted(policy_dir.glob("seed_*")):
            seed_suffix = seed_dir.name.removeprefix("seed_")
            if not seed_suffix.isdigit():
                continue
            for metadata_path in find_nested_metadata_paths(seed_dir):
                metadata = load_json(metadata_path)
                if metadata.get("status") != "completed":
                    continue
                records.append(
                    _requested_RunRecord(
                        policy_id=policy_id,
                        seed=int(seed_suffix),
                        run_dir=metadata_path.parent,
                        metadata=metadata,
                    )
                )
    return records


def _requested_pad_params(
    estimate: Sequence[float], full_params: Sequence[float], min_dim: int
) -> np.ndarray:
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    full = np.asarray(full_params, dtype=np.float64).reshape(-1).copy()
    n = min(max(int(min_dim), est.size), full.size)
    full[:n] = est[:n]
    return full


def _requested_step_batch(
    *,
    dynamics_type: str,
    states: np.ndarray,
    actions: np.ndarray,
    params: np.ndarray,
    dt: float,
    dynamics_alpha: float,
    clip_limit: float,
) -> np.ndarray:
    states_np = np.asarray(states, dtype=np.float64)
    actions_np = np.asarray(actions, dtype=np.float64)
    params_np = np.asarray(params, dtype=np.float64)
    if params_np.ndim == 1:
        params_np = np.broadcast_to(
            params_np[None, :], (states_np.shape[0], params_np.shape[0])
        ).copy()
    drift = residual_np(
        dynamics_type,
        states_np,
        params_np,
        dynamics_alpha=float(dynamics_alpha),
    )
    next_states = states_np + float(dt) * (drift + actions_np)
    return np.clip(next_states, -float(clip_limit), float(clip_limit))


def _requested_evaluate_regulation_cost(
    *,
    metadata: dict[str, Any],
    model_params: np.ndarray,
    true_params: np.ndarray,
    starts: np.ndarray,
    targets: np.ndarray,
    seed: int,
    n_control_steps: int = 240,
) -> float:
    del seed
    dt = float(metadata.get("dt", 0.01))
    action_max = float(metadata.get("action_max", 1.0))
    dynamics_alpha = float(metadata.get("dynamics_alpha", 1.0))
    clip_limit = float(
        np.max(
            np.abs(
                np.asarray(
                    metadata.get("state_high", [5.0, 5.0])
                    + metadata.get("state_low", [-5.0, -5.0]),
                    dtype=np.float64,
                )
            )
        )
    )
    clip_limit = max(clip_limit, 5.0)
    true_type = str(metadata.get("dynamics_type", "asymmetric_basin"))
    estimator_type = str(metadata.get("estimator_dynamics_type", true_type))
    total_cost = 0.0
    feedback_gain = 0.18
    for start, target in zip(starts, targets, strict=True):
        state = np.asarray(start, dtype=np.float64).reshape(1, -1)
        target_np = np.asarray(target, dtype=np.float64).reshape(1, -1)
        task_cost = 0.0
        for control_step in range(n_control_steps):
            model_drift = residual_np(
                estimator_type,
                state,
                model_params,
                dynamics_alpha=dynamics_alpha,
            )
            action = np.clip(
                -model_drift + feedback_gain * (target_np - state),
                -action_max,
                action_max,
            )
            state = _requested_step_batch(
                dynamics_type=true_type,
                states=state,
                actions=action,
                params=true_params,
                dt=dt,
                dynamics_alpha=dynamics_alpha,
                clip_limit=clip_limit,
            )
            err = state - target_np
            task_cost += float(np.sum(err * err) + 0.02 * np.sum(action * action))
            if control_step == n_control_steps - 1:
                task_cost += 25.0 * float(np.sum(err * err))
        total_cost += task_cost / float(n_control_steps)
    return total_cost / float(max(1, len(starts)))


def _requested_control_tasks() -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [
            [-1.70, 0.00],
            [-1.05, 0.55],
            [1.05, -0.55],
            [1.70, 0.00],
        ],
        dtype=np.float64,
    )
    starts = targets + np.asarray(
        [
            [0.35, -0.25],
            [-0.25, 0.20],
            [0.25, -0.20],
            [-0.35, 0.25],
        ],
        dtype=np.float64,
    )
    return starts, targets


def _requested_compute_downstream_rows() -> list[dict[str, Any]]:
    suite_dir = _requested_suite_dir(
        "exp05_ablation",
        "exp05_asymmetric_basin_objective_ablation",
    )
    records = _requested_collect_records(suite_dir, _requested_OBJECTIVE_POLICIES)
    starts, targets = _requested_control_tasks()
    rows: list[dict[str, Any]] = []
    oracle_costs: list[float] = []
    for record in records:
        metadata = record.metadata
        true_params = np.asarray(metadata.get("true_params_full", []), dtype=np.float64)
        if true_params.size == 0:
            continue
        learned_params = _requested_pad_params(
            metadata.get("embedding_estimate", true_params),
            metadata.get("estimator_true_params_full", true_params),
            int(metadata.get("min_embedding_dim", true_params.size)),
        )
        base_seed = 17_000 + int(record.seed) * 101 + len(rows)
        learned_cost = _requested_evaluate_regulation_cost(
            metadata=metadata,
            model_params=learned_params,
            true_params=true_params,
            starts=starts,
            targets=targets,
            seed=base_seed,
        )
        oracle_cost = _requested_evaluate_regulation_cost(
            metadata=metadata,
            model_params=true_params,
            true_params=true_params,
            starts=starts,
            targets=targets,
            seed=base_seed,
        )
        oracle_costs.append(oracle_cost)
        rows.append(
            {
                "policy_id": record.policy_id,
                "policy_label": _requested_policy_label(record.policy_id),
                "seed": record.seed,
                "parameter_error_final": _requested_safe_float(
                    metadata.get("embedding_error_final")
                ),
                "trajectory_r2_final": _requested_safe_float(metadata.get("trajectory_r2_final")),
                "downstream_control_cost": learned_cost,
                "oracle_control_cost": oracle_cost,
                "relative_control_cost": learned_cost / max(oracle_cost, 1e-8),
            }
        )
    if oracle_costs:
        oracle_mean = float(np.mean(oracle_costs))
        rows.append(
            {
                "policy_id": "oracle_true_model",
                "policy_label": "Oracle true model",
                "seed": -1,
                "parameter_error_final": 0.0,
                "trajectory_r2_final": 1.0,
                "downstream_control_cost": oracle_mean,
                "oracle_control_cost": oracle_mean,
                "relative_control_cost": 1.0,
            }
        )
    return rows


def _requested_plot_downstream_control(plt: Any) -> tuple[Path, Path]:
    rows = _requested_compute_downstream_rows()
    policy_ids = [*_requested_OBJECTIVE_POLICIES, "oracle_true_model"]
    grouped: dict[str, list[dict[str, Any]]] = {
        policy_id: [row for row in rows if row["policy_id"] == policy_id]
        for policy_id in policy_ids
    }
    summary_rows: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        items = grouped.get(policy_id, [])
        if not items:
            continue
        costs = [float(row["relative_control_cost"]) for row in items]
        param = [
            float(row["parameter_error_final"])
            for row in items
            if row.get("parameter_error_final") is not None
        ]
        r2 = [
            float(row["trajectory_r2_final"])
            for row in items
            if row.get("trajectory_r2_final") is not None
        ]
        summary_rows.append(
            {
                "policy_id": policy_id,
                "policy_label": (
                    _requested_policy_label(policy_id)
                    if policy_id != "oracle_true_model"
                    else "Oracle"
                ),
                "relative_control_cost_mean": float(np.mean(costs)),
                "relative_control_cost_sem": _requested_sem(costs),
                "parameter_error_mean": float(np.mean(param)) if param else None,
                "trajectory_r2_mean": float(np.mean(r2)) if r2 else None,
                "n": len(costs),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    x = np.arange(len(summary_rows), dtype=np.float64)
    y = [float(row["relative_control_cost_mean"]) for row in summary_rows]
    yerr = [float(row["relative_control_cost_sem"]) for row in summary_rows]
    colors = [
        (
            _requested_C_NEUTRAL_FILL
            if row["policy_id"] == "oracle_true_model"
            else _requested_policy_color(str(row["policy_id"]))
        )
        for row in summary_rows
    ]
    edge_colors = [
        _requested_C_STROKE if row["policy_id"] == "oracle_true_model" else _requested_C_STROKE
        for row in summary_rows
    ]
    axes[0].bar(x, y, yerr=yerr, color=colors, edgecolor=edge_colors, linewidth=0.45, capsize=2.0)
    axes[0].axhline(1.0, color=_requested_C_NEUTRAL_LIGHT, linestyle="--", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [str(row["policy_label"]) for row in summary_rows], rotation=35, ha="right"
    )
    axes[0].set_ylabel("Control cost / oracle")
    axes[0].set_title("A. Downstream control utility")
    _style_requested_axis(axes[0])

    for row in rows:
        if row["policy_id"] == "oracle_true_model":
            continue
        r2 = row.get("trajectory_r2_final")
        cost = row.get("relative_control_cost")
        if r2 is None or cost is None:
            continue
        axes[1].scatter(
            [float(r2)],
            [float(cost)],
            s=22,
            color=_requested_policy_color(str(row["policy_id"])),
            edgecolor=_requested_C_STROKE,
            linewidth=0.35,
            label=_requested_policy_label(str(row["policy_id"])),
        )
    handles, labels = axes[1].get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        dedup.setdefault(label, handle)
    axes[1].set_xlabel("Final prediction R2")
    axes[1].set_ylabel("Control cost / oracle")
    axes[1].set_title("B. Prediction quality vs control cost")
    axes[1].legend(dedup.values(), dedup.keys(), fontsize=6.0, loc="upper left")
    _style_requested_axis(axes[1])
    fig.tight_layout(w_pad=1.0)

    csv_path = _requested_GENERATED_DIR / "tbme_requested_downstream_control_utility.csv"
    _requested_write_csv(
        csv_path,
        rows,
        [
            "policy_id",
            "policy_label",
            "seed",
            "parameter_error_final",
            "trajectory_r2_final",
            "downstream_control_cost",
            "oracle_control_cost",
            "relative_control_cost",
        ],
    )
    return (
        _requested_save_pdf(
            fig, _requested_FIGURE_DIR / "tbme_requested_downstream_control_utility.pdf"
        ),
        csv_path,
    )


def _requested_write_manifest(paths: list[Path]) -> Path:
    manifest = _requested_GENERATED_DIR / "tbme_requested_experiment_figures_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "Requested TBME experiment figures:\n"
        + "\n".join(str(path.relative_to(_requested_REPO_ROOT)) for path in paths)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _requested_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate requested TBME follow-up experiment figures.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--plots",
        type=str,
        default=",".join(_requested_PLOTS),
        help="Comma-separated requested TBME plot ids.",
    )
    return parser


def _additional_safe_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _additional_sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def _additional_policy_sort_key(policy_id: str) -> tuple[int, str]:
    try:
        return POLICY_ORDER.index(policy_id), policy_id
    except ValueError:
        return len(POLICY_ORDER), policy_id


def _additional_policy_label(policy_id: str) -> str:
    return POLICY_LABELS.get(policy_id, policy_id.replace("_", " "))


def _additional_short_policy_label(policy_id: str) -> str:
    labels = {
        "active_planning_u20_r20_h40": "Plan U20/R20",
        "active_planning_u10_r20_h40": "Plan U10/R20",
        "active_planning_u5_r20_h40": "Plan U5/R20",
        "active_planning_u10_r10_h40": "Plan U10/R10",
        "active_planning_u5_r10_h40": "Plan U5/R10",
        "active_planning_u5_r5_h40": "Plan U5/R5",
        "active_planning_u1_r1_h40": "Plan U1/R1",
        "active_fully_observable_u20_r20_h40": "Full-observable EIG",
        "active_e_optimality_u20_r20_h40": "E-optimality",
        "active_state_information_u20_r20_h40": "State information",
        "active_dynamics_u20_r20_h40": "Dynamics",
        "active_sampling_variance_u20_r20_h40": "Sampling variance",
        "active_myopic": "Myopic",
        "prbs": "PRBS",
        "random": "Random",
        "ensemble": "Ensemble",
        "rhc": "RHC-US",
        "flex": "FLEX",
        "flex_true_state": "FLEX true",
    }
    return labels.get(policy_id, _additional_policy_label(policy_id))


def _additional_policy_color(policy_id: str) -> str:
    return _additional_POLICY_COLORS.get(policy_id, _additional_C_NEUTRAL)


def _additional_suite_dir(group_name: str, suite_id: str) -> Path:
    return _suite_dir(group_name, suite_id)


def _additional_state_bounds_from_metadata(metadata: dict[str, Any]) -> tuple[float, float]:
    low = np.asarray(metadata.get("state_low", [-5.0, -5.0]), dtype=np.float64)
    high = np.asarray(metadata.get("state_high", [5.0, 5.0]), dtype=np.float64)
    return float(np.min(low)), float(np.max(high))


def _additional_collect_records(
    suite_dir: Path,
    policy_ids: Sequence[str],
    *,
    max_seeds: int,
) -> list[_additional_RunRecord]:
    records: list[_additional_RunRecord] = []
    for policy_id in sorted(policy_ids, key=_additional_policy_sort_key):
        policy_dir = suite_dir / "track" / policy_id
        if not policy_dir.exists():
            continue
        seed_dirs: list[tuple[int, Path]] = []
        for seed_dir in policy_dir.glob("seed_*"):
            suffix = seed_dir.name.removeprefix("seed_")
            if suffix.isdigit():
                seed_dirs.append((int(suffix), seed_dir))
        for seed, seed_dir in sorted(seed_dirs)[:max_seeds]:
            for metadata_path in find_nested_metadata_paths(seed_dir):
                records.append(
                    _additional_RunRecord(
                        policy_id=policy_id,
                        seed=seed,
                        run_dir=metadata_path.parent,
                        metadata=load_json(metadata_path),
                    )
                )
    return records


def _additional_trace_path(
    record: _additional_RunRecord, metadata_key: str, fallback_name: str
) -> Path:
    return resolve_artifact_path(
        record.run_dir,
        record.metadata,
        key=metadata_key,
        fallback_name=fallback_name,
    )


def _additional_load_xy_trace(record: _additional_RunRecord) -> np.ndarray:
    path = _additional_trace_path(record, "state_action_trace_path", "state_action_trace.csv")
    points: list[tuple[float, float]] = []
    for row in read_trace_csv(path):
        x_val = _additional_safe_float(row.get("true_x"))
        v_val = _additional_safe_float(row.get("true_v"))
        if x_val is None or v_val is None:
            continue
        points.append((x_val, v_val))
    return np.asarray(points, dtype=np.float32)


def _additional_logdet_information(
    latent: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Compute log det of the state Fisher information for Poisson log-linear observations.

    For mean counts mu(z)=dt*exp(W z + b), H=dmu/dz=diag(mu)W and
    R=diag(mu), so I_z = H^T R^{-1} H = W^T diag(mu) W.
    """
    latent = np.asarray(latent, dtype=np.float64)
    if latent.ndim == 1:
        latent = latent.reshape(1, -1)
    weights, bias, dt = reconstruct_loglinear_rate_model(
        metadata,
        obs_dim=int(metadata.get("observation_dim", 20)),
        latent_dim=int(metadata.get("latent_dim", 2)),
    )
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    log_rate_hz = latent @ weights.T + bias.reshape(1, -1)
    rate_hz = np.exp(np.clip(log_rate_hz, -20.0, 20.0))
    mean_counts = np.clip(rate_hz * float(dt), 1e-12, 1e12)
    info_mats = np.einsum("nd,di,dj->nij", mean_counts, weights, weights, optimize=True)
    info_mats = 0.5 * (info_mats + np.swapaxes(info_mats, -1, -2))
    info_mats = info_mats + 1e-9 * np.eye(weights.shape[1], dtype=np.float64)[None, :, :]
    sign, logabsdet = np.linalg.slogdet(info_mats)
    return np.where(sign > 0.0, logabsdet, np.nan)


def _additional_observation_model_key(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(metadata.get("seed", 0)),
        str(metadata.get("env_preset_id", "")),
        int(metadata.get("observation_dim", 20)),
        int(metadata.get("latent_dim", 2)),
        float(metadata.get("dt", 0.01)),
        float(metadata.get("mean_firing_rate_target", 10.0)),
        float(metadata.get("max_firing_rate_target", 100.0)),
    )


def _additional_information_reference_records(
    records: Sequence[_additional_RunRecord],
) -> list[_additional_RunRecord]:
    out: list[_additional_RunRecord] = []
    seen: set[tuple[Any, ...]] = set()
    for record in sorted(
        records, key=lambda item: (item.seed, _additional_policy_sort_key(item.policy_id))
    ):
        key = _additional_observation_model_key(record.metadata)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _additional_make_information_grid(
    metadata: dict[str, Any],
    *,
    n_grid: int = 121,
    axis_min: float | None = None,
    axis_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis_min is None or axis_max is None:
        state_min, state_max = _additional_state_bounds_from_metadata(metadata)
    else:
        state_min, state_max = float(axis_min), float(axis_max)
    axis = np.linspace(state_min, state_max, n_grid, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    latent = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    logdet = _additional_logdet_information(latent, metadata=metadata).reshape(n_grid, n_grid)
    return axis, axis, logdet


def _additional_make_mean_information_grid(
    records: Sequence[_additional_RunRecord],
    *,
    n_grid: int = 121,
    axis_min: float | None = None,
    axis_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("At least one record is required to compute an information grid")
    x_axis, y_axis, first_grid = _additional_make_information_grid(
        records[0].metadata,
        n_grid=n_grid,
        axis_min=axis_min,
        axis_max=axis_max,
    )
    maps = [first_grid.astype(np.float64)]
    for record in records[1:]:
        _x, _y, grid = _additional_make_information_grid(
            record.metadata,
            n_grid=n_grid,
            axis_min=axis_min,
            axis_max=axis_max,
        )
        maps.append(grid.astype(np.float64))
    return x_axis, y_axis, np.nanmean(np.stack(maps, axis=0), axis=0)


def _additional_true_vectorfield_dynamics(metadata: dict[str, Any]) -> ResidualDynamicsCallable:
    env_preset = get_environment_preset_from_metadata(metadata)
    theta_true = np.asarray(metadata.get("embedding_true", []), dtype=np.float32)
    if theta_true.size == 0:
        theta_true = np.asarray(env_preset.true_embedding_vector(), dtype=np.float32)
    return ResidualDynamicsCallable(
        dynamics_type=env_preset.resolved_dynamics_type(),
        dyn_params=env_preset.params_from_embedding(theta_true),
        dynamics_alpha=float(metadata.get("dynamics_alpha", 1.0)),
        device="cpu",
    )


def _additional_learned_vectorfield_dynamics(
    metadata: dict[str, Any],
    theta: np.ndarray,
) -> ResidualDynamicsCallable:
    env_preset = get_environment_preset_from_metadata(metadata)
    return ResidualDynamicsCallable(
        dynamics_type=env_preset.resolved_dynamics_type(estimator=True),
        dyn_params=env_preset.params_from_embedding(theta, estimator=True),
        dynamics_alpha=float(metadata.get("dynamics_alpha", 1.0)),
        device="cpu",
    )


def _additional_plot_neutral_vector_field(
    ax: Any,
    dynamics: ResidualDynamicsCallable,
    *,
    grid_lim: float,
    n_grid: int,
    arrowsize: float = 0.70,
) -> None:
    x_grid, y_grid, u_grid, v_grid = compute_vector_field(
        dynamics,
        x_range=float(grid_lim),
        n_grid=int(n_grid),
        is_residual=True,
        device="cpu",
    )
    from matplotlib.colors import to_rgba

    ax.streamplot(
        x_grid.cpu().numpy(),
        y_grid.cpu().numpy(),
        u_grid.cpu().numpy(),
        v_grid.cpu().numpy(),
        color=to_rgba(_additional_C_STROKE, 0.42),
        linewidth=0.34,
        density=1.55,
        arrowsize=float(arrowsize),
        zorder=2,
    )


def _additional_save_pdf(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    return path


def _additional_plot_true_dynamics_all() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    _apply_style(plt)
    panel_specs = [
        ("tbme_duffing", "Duffing"),
        ("tbme_damped_pendulum", "Damped pendulum"),
        ("tbme_asymmetric_basin", "Asymmetric basin"),
        ("tbme_asymmetric_basin_hard", "Asymmetric basin (hard)"),
        ("tbme_multi_stable", "Multi-stable"),
    ]
    grid_lim = 6.0
    fields = []
    log_speeds = []
    for preset_id, title in panel_specs:
        env_preset = get_environment_preset(preset_id)
        theta_true = env_preset.true_embedding_vector()
        dynamics = ResidualDynamicsCallable(
            dynamics_type=env_preset.resolved_dynamics_type(),
            dyn_params=env_preset.params_from_embedding(theta_true),
            dynamics_alpha=float(env_preset.dynamics_alpha),
            device="cpu",
        )
        x_grid, y_grid, u_grid, v_grid = compute_vector_field(
            dynamics,
            x_range=grid_lim,
            n_grid=53,
            is_residual=True,
            device="cpu",
        )
        x_np = x_grid.cpu().numpy()
        y_np = y_grid.cpu().numpy()
        u_np = np.nan_to_num(u_grid.cpu().numpy(), nan=0.0, posinf=1e6, neginf=-1e6)
        v_np = np.nan_to_num(v_grid.cpu().numpy(), nan=0.0, posinf=1e6, neginf=-1e6)
        speed = np.hypot(u_np, v_np)
        log_speed = np.log1p(np.nan_to_num(speed, nan=0.0, posinf=1e6, neginf=0.0))
        fields.append((title, x_np, y_np, u_np, v_np, log_speed))
        log_speeds.append(log_speed)

    finite_speed = np.concatenate([arr[np.isfinite(arr)].reshape(-1) for arr in log_speeds])
    vmax = float(np.percentile(finite_speed, 98.0)) if finite_speed.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    fig = plt.figure(figsize=(7.25, 4.05))
    gs = fig.add_gridspec(
        2,
        7,
        wspace=0.36,
        hspace=0.42,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.08],
    )
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    cax = fig.add_subplot(gs[:, 6])
    for panel_idx, (ax, (title, x_np, y_np, u_np, v_np, log_speed)) in enumerate(zip(axes, fields)):
        ax.pcolormesh(
            x_np,
            y_np,
            log_speed,
            cmap="viridis",
            norm=norm,
            shading="auto",
            alpha=0.82,
            rasterized=True,
            zorder=0,
        )
        ax.streamplot(
            x_np,
            y_np,
            u_np,
            v_np,
            color=_additional_C_STROKE,
            linewidth=0.38,
            density=1.25,
            arrowsize=0.62,
            zorder=2,
        )
        decorate_phase_space_axis(
            ax,
            xlim=(-grid_lim, grid_lim),
            ylim=(-grid_lim, grid_lim),
            title=f"{chr(ord('A') + panel_idx)}. {title}",
            xlabel="x",
            ylabel="v",
            grid_alpha=0.20,
        )
        ax.set_xticks([-6, 0, 6])
        ax.set_yticks([-6, 0, 6])

    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\log(1 + \|f(z)\|)$")
    cbar.outline.set_linewidth(0.45)
    return _additional_save_pdf(
        fig, _additional_FIGURE_DIR / "tbme_additional_true_dynamics_all.pdf"
    )


def _additional_plot_asymmetric_basin_mechanism(max_seeds: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _apply_style(plt)
    suite_dir = _additional_suite_dir("exp02_hard", "exp02_hard_asymmetric_basin")
    policy_ids = ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "flex", "prbs"]
    records = _additional_collect_records(suite_dir, policy_ids, max_seeds=max_seeds)
    if not records:
        raise RuntimeError(f"No records found under {suite_dir}")
    ref_metadata = records[0].metadata
    by_policy: dict[str, list[np.ndarray]] = {policy_id: [] for policy_id in policy_ids}
    record_traces: list[tuple[_additional_RunRecord, np.ndarray]] = []
    env_preset = get_environment_preset_from_metadata(ref_metadata)
    panel_abs = max(float(env_preset.resolved_plot_limit()), 6.0)
    boundary_radius = _additional_safe_float(ref_metadata.get("boundary_radius"))
    if boundary_radius is None:
        boundary_radius = _additional_safe_float(getattr(env_preset, "boundary_radius", None))
    if boundary_radius is not None:
        panel_abs = max(panel_abs, boundary_radius)
    for record in records:
        traj = _additional_load_xy_trace(record)
        if traj.size == 0:
            continue
        by_policy.setdefault(record.policy_id, []).append(traj)
        record_traces.append((record, traj))
        finite = traj[np.isfinite(traj).all(axis=1)]
        if finite.size:
            panel_abs = max(panel_abs, 1.04 * float(np.max(np.abs(finite[:, :2]))))
    panel_min, panel_max = -panel_abs, panel_abs
    information_refs = _additional_information_reference_records(records)
    x_axis, y_axis, logdet_grid = _additional_make_mean_information_grid(
        information_refs,
        axis_min=panel_min,
        axis_max=panel_max,
    )
    finite_grid = logdet_grid[np.isfinite(logdet_grid)]
    info_threshold = float(np.percentile(finite_grid, 75.0))
    info_vmin = float(np.percentile(finite_grid, 1.0))
    info_vmax = float(np.percentile(finite_grid, 99.0))
    if info_vmax <= info_vmin:
        info_vmax = info_vmin + 1e-6

    informative_fraction: dict[str, list[float]] = {policy_id: [] for policy_id in policy_ids}
    coverage_fraction: dict[str, list[float]] = {policy_id: [] for policy_id in policy_ids}
    threshold_by_model: dict[tuple[Any, ...], float] = {}
    bins = 48
    for record, traj in record_traces:
        values = _additional_logdet_information(traj[:, :2], metadata=record.metadata)
        finite = values[np.isfinite(values)]
        if finite.size:
            model_key = _additional_observation_model_key(record.metadata)
            if model_key not in threshold_by_model:
                _x, _y, record_grid = _additional_make_information_grid(
                    record.metadata,
                    axis_min=panel_min,
                    axis_max=panel_max,
                )
                record_finite = record_grid[np.isfinite(record_grid)]
                threshold_by_model[model_key] = (
                    float(np.percentile(record_finite, 75.0))
                    if record_finite.size
                    else info_threshold
                )
            informative_fraction[record.policy_id].append(
                float(np.mean(finite >= threshold_by_model[model_key]))
            )
        hist, _x_edges, _y_edges = np.histogram2d(
            traj[:, 0],
            traj[:, 1],
            bins=bins,
            range=[[panel_min, panel_max], [panel_min, panel_max]],
        )
        coverage_fraction[record.policy_id].append(float(np.count_nonzero(hist) / hist.size))

    metrics = read_trace_csv(suite_dir / "summary" / "metrics.csv")
    final_r2: dict[str, list[float]] = {policy_id: [] for policy_id in policy_ids}
    for row in metrics:
        policy_id = str(row.get("policy_id", ""))
        if policy_id not in final_r2 or row.get("status") != "completed":
            continue
        value = _additional_safe_float(row.get("trajectory_r2_final_mean"))
        if value is not None:
            final_r2[policy_id].append(value)

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.75))
    ax = axes[0, 0]
    im = ax.imshow(
        logdet_grid,
        origin="lower",
        extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
        cmap="magma",
        vmin=info_vmin,
        vmax=info_vmax,
        interpolation="nearest",
        aspect="equal",
        alpha=0.72,
    )
    contour_levels = [info_threshold]
    ax.contour(
        x_axis,
        y_axis,
        logdet_grid,
        levels=contour_levels,
        colors=[_additional_C_STROKE],
        linewidths=0.7,
        linestyles="--",
    )
    _additional_plot_neutral_vector_field(
        ax,
        _additional_true_vectorfield_dynamics(ref_metadata),
        n_grid=28,
        grid_lim=panel_max,
        arrowsize=0.70,
    )
    for policy_id in ["active_planning_u20_r20_h40", "active_myopic", "flex", "prbs"]:
        for traj in by_policy.get(policy_id, [])[:8]:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=_additional_policy_color(policy_id),
                linewidth=0.55,
                alpha=0.68,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
    cbar.set_label("mean log det(I_z)")
    cbar.outline.set_linewidth(0.45)
    decorate_phase_space_axis(
        ax,
        xlim=(panel_min, panel_max),
        ylim=(panel_min, panel_max),
        title="A. Hard asymmetric-basin information and vector field",
        grid_alpha=0.20,
    )
    _style_manuscript_axis(ax)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_additional_policy_color(policy_id),
                linewidth=0.9,
                label=_additional_short_policy_label(policy_id),
            )
            for policy_id in ["active_planning_u20_r20_h40", "active_myopic", "flex", "prbs"]
        ],
        loc="lower right",
        fontsize=5.8,
        frameon=True,
        framealpha=0.78,
        borderpad=0.25,
    )

    panels = [
        (
            axes[0, 1],
            informative_fraction,
            "B. Occupancy of high-information states",
            "Fraction of samples",
        ),
        (axes[1, 0], coverage_fraction, "C. State-space coverage", "Visited-bin fraction"),
        (axes[1, 1], final_r2, "D. Endpoint prediction", "Final prediction R2"),
    ]
    labels = [_additional_short_policy_label(policy_id) for policy_id in policy_ids]
    x = np.arange(len(policy_ids), dtype=np.float64)
    for ax_i, data, title, ylabel in panels:
        means = []
        errors = []
        for policy_id in policy_ids:
            vals = [v for v in data.get(policy_id, []) if math.isfinite(v)]
            means.append(float(np.mean(vals)) if vals else np.nan)
            errors.append(_additional_sem(vals))
        ax_i.bar(
            x,
            means,
            yerr=errors,
            color=[_additional_policy_color(policy_id) for policy_id in policy_ids],
            edgecolor=_additional_C_STROKE,
            linewidth=0.45,
            capsize=2.3,
            error_kw={"elinewidth": 0.55, "ecolor": _additional_C_STROKE, "capthick": 0.55},
        )
        ax_i.set_xticks(x)
        ax_i.set_xticklabels(labels, rotation=25, ha="right")
        ax_i.set_ylabel(ylabel)
        ax_i.set_title(title)
        _style_manuscript_axis(ax_i, grid_axis="y")
    axes[1, 1].set_ylim(-0.05, 1.05)
    fig.suptitle(
        "Hard asymmetric-basin mechanism: information geometry, coverage, and prediction",
        y=0.995,
    )
    fig.tight_layout()
    return _additional_save_pdf(
        fig, _additional_FIGURE_DIR / "tbme_additional_asymmetric_basin_mechanism.pdf"
    )


def _additional_common_seed(
    records: Sequence[_additional_RunRecord],
    policy_ids: Sequence[str],
) -> int:
    seeds_by_policy: dict[str, set[int]] = {policy_id: set() for policy_id in policy_ids}
    for record in records:
        if record.metadata.get("status") != "completed":
            continue
        if record.policy_id in seeds_by_policy:
            seeds_by_policy[record.policy_id].add(int(record.seed))
    common = set.intersection(*(seeds for seeds in seeds_by_policy.values() if seeds))
    if not common:
        raise RuntimeError("No completed seed is shared by all requested policies")
    return min(common)


def _additional_embedding_at_step(record: _additional_RunRecord, step: int) -> np.ndarray:
    path = _additional_trace_path(
        record, "embedding_estimate_trace_path", "embedding_estimate_trace.csv"
    )
    selected: dict[str, str] | None = None
    selected_step = -math.inf
    fallback: dict[str, str] | None = None
    fallback_step = math.inf
    for row in read_trace_csv(path):
        row_step = _additional_safe_float(row.get("step"))
        if row_step is None:
            continue
        if row_step <= step and row_step >= selected_step:
            selected = row
            selected_step = row_step
        if row_step >= step and row_step <= fallback_step:
            fallback = row
            fallback_step = row_step
    row = selected if selected is not None else fallback
    if row is None:
        raise RuntimeError(f"No embedding estimates found for {record.run_dir}")
    embedding_dim = int(
        _additional_safe_float(row.get("embedding_dim")) or record.metadata.get("embedding_dim", 0)
    )
    values = []
    for idx in range(embedding_dim):
        value = _additional_safe_float(row.get(f"e{idx}"))
        if value is None:
            raise RuntimeError(f"Missing e{idx} in {path}")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _additional_xy_trace_until(record: _additional_RunRecord, step: int) -> np.ndarray:
    path = _additional_trace_path(record, "state_action_trace_path", "state_action_trace.csv")
    points: list[tuple[float, float]] = []
    for row in read_trace_csv(path):
        row_step = _additional_safe_float(row.get("step"))
        x_val = _additional_safe_float(row.get("true_x"))
        v_val = _additional_safe_float(row.get("true_v"))
        if row_step is None or x_val is None or v_val is None:
            continue
        if row_step <= step:
            points.append((x_val, v_val))
    return np.asarray(points, dtype=np.float32)


def _additional_plot_learned_vectorfield_snapshots(max_seeds: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    suite_dir = _additional_suite_dir("exp02_hard", "exp02_hard_asymmetric_basin")
    policy_ids = ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "flex", "prbs"]
    checkpoints = [250, 500, 1000]
    row_ids = ["true", *policy_ids]
    records = _additional_collect_records(suite_dir, policy_ids, max_seeds=max_seeds)
    if not records:
        raise RuntimeError(f"No records found under {suite_dir}")
    seed = _additional_common_seed(records, policy_ids)
    record_by_policy: dict[str, _additional_RunRecord] = {}
    for policy_id in policy_ids:
        matches = [
            record
            for record in records
            if record.policy_id == policy_id
            and int(record.seed) == seed
            and record.metadata.get("status") == "completed"
        ]
        if not matches:
            raise RuntimeError(f"Missing completed seed {seed} for {policy_id}")
        record_by_policy[policy_id] = matches[0]

    ref_metadata = record_by_policy[policy_ids[0]].metadata
    env_preset = get_environment_preset_from_metadata(ref_metadata)
    plot_abs = max(float(env_preset.resolved_plot_limit()), 6.0)
    boundary_radius = _additional_safe_float(ref_metadata.get("boundary_radius"))
    if boundary_radius is None:
        boundary_radius = _additional_safe_float(getattr(env_preset, "boundary_radius", None))
    if boundary_radius is not None:
        plot_abs = max(plot_abs, boundary_radius)
    for record in record_by_policy.values():
        traj = _additional_xy_trace_until(record, max(checkpoints))
        finite = traj[np.isfinite(traj).all(axis=1)]
        if finite.size:
            plot_abs = max(plot_abs, 1.04 * float(np.max(np.abs(finite[:, :2]))))

    fig, axes = plt.subplots(
        len(row_ids),
        len(checkpoints),
        figsize=(7.25, 8.85),
        sharex=True,
        sharey=True,
    )
    true_dynamics = _additional_true_vectorfield_dynamics(ref_metadata)
    for row_idx, row_id in enumerate(row_ids):
        record = None if row_id == "true" else record_by_policy[row_id]
        color = _additional_C_STROKE if row_id == "true" else _additional_policy_color(row_id)
        for col_idx, checkpoint in enumerate(checkpoints):
            ax = axes[row_idx, col_idx]
            if row_id == "true":
                dynamics = true_dynamics
            else:
                assert record is not None
                theta = _additional_embedding_at_step(record, checkpoint)
                dynamics = _additional_learned_vectorfield_dynamics(record.metadata, theta)
            _additional_plot_neutral_vector_field(
                ax,
                dynamics,
                grid_lim=plot_abs,
                n_grid=22,
                arrowsize=0.58,
            )
            traj = (
                np.empty((0, 2), dtype=np.float32)
                if record is None
                else _additional_xy_trace_until(record, checkpoint)
            )
            if traj.size:
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.8, alpha=0.92, zorder=4)
                ax.scatter(
                    [traj[-1, 0]],
                    [traj[-1, 1]],
                    s=13,
                    color=color,
                    edgecolor=_additional_C_STROKE,
                    linewidth=0.35,
                    zorder=5,
                )
                ax.scatter(
                    [traj[0, 0]],
                    [traj[0, 1]],
                    s=9,
                    color=_additional_C_NEUTRAL_FILL,
                    edgecolor=_additional_C_STROKE,
                    linewidth=0.3,
                    zorder=5,
                )
            ax.set_xlim(-plot_abs, plot_abs)
            ax.set_ylim(-plot_abs, plot_abs)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color=_additional_C_GRID, linewidth=0.28, alpha=0.25)
            for spine in ax.spines.values():
                spine.set_color(_additional_C_STROKE)
                spine.set_linewidth(0.48)
            ax.tick_params(width=0.4, length=1.6, labelsize=5.8)
            if row_idx == 0:
                ax.set_title(f"step {checkpoint}", fontsize=7.4, pad=2.0)
            if col_idx == 0:
                ylabel = "True" if row_id == "true" else _additional_short_policy_label(row_id)
                ax.set_ylabel(ylabel, fontsize=7.2)
            else:
                ax.set_ylabel("")
            if row_idx == len(row_ids) - 1:
                ax.set_xlabel("x")
            else:
                ax.set_xlabel("")
    fig.suptitle(
        f"Hard asymmetric-basin true and learned vector fields, seed {seed}",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), w_pad=0.25, h_pad=0.45)
    return _additional_save_pdf(
        fig, _additional_FIGURE_DIR / "tbme_additional_asymmetric_basin_learned_vectorfields.pdf"
    )


def _additional_threshold_value(row: dict[str, str], threshold: float) -> float | None:
    suffix = f"{threshold:.2f}".replace(".", "p")
    return _additional_safe_float(row.get(f"step_to_r2_{suffix}"))


def _additional_plot_sample_efficiency_thresholds() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    selected = [
        ("exp01_base", "exp01_duffing", "Duffing"),
        ("exp01_base", "exp01_asymmetric_basin", "Asym. basin"),
        ("exp02_hard", "exp02_hard_duffing", "Hard Duffing"),
        ("exp02_hard", "exp02_hard_asymmetric_basin", "Hard asym."),
        ("exp04_mismatch", "exp04_duffing_parameter_mismatch", "Duffing mismatch"),
        ("exp04_mismatch", "exp04_asymmetric_basin_parameter_mismatch", "Asym. mismatch"),
    ]
    policy_ids = [
        "active_planning_u20_r20_h40",
        "active_myopic",
        "ensemble",
        "prbs",
        "random",
        "rhc",
    ]
    thresholds = {
        "exp04_duffing_parameter_mismatch": 0.90,
        "exp04_asymmetric_basin_parameter_mismatch": 0.90,
    }
    default_threshold = 0.95

    values: list[tuple[str, str, float | None, float]] = []
    max_step = 1.0
    for group_name, suite_id, suite_label in selected:
        suite_dir = _additional_suite_dir(group_name, suite_id)
        threshold = thresholds.get(suite_id, default_threshold)
        rows = read_trace_csv(suite_dir / "summary" / "trajectory_r2_thresholds.csv")
        row_by_policy = {str(row.get("policy_id", "")): row for row in rows}
        for policy_id in policy_ids:
            step = _additional_threshold_value(row_by_policy.get(policy_id, {}), threshold)
            if step is not None:
                max_step = max(max_step, step)
            values.append((suite_label, policy_id, step, threshold))

    fig, ax = plt.subplots(figsize=(7.1, 3.55))
    width = 0.12
    suite_labels = [item[2] for item in selected]
    group_x = np.arange(len(suite_labels), dtype=np.float64)
    for idx, policy_id in enumerate(policy_ids):
        xs = group_x + (idx - (len(policy_ids) - 1) / 2.0) * width
        heights = []
        missing_x = []
        missing_y = []
        for suite_label in suite_labels:
            match = [v for v in values if v[0] == suite_label and v[1] == policy_id]
            step = match[0][2] if match else None
            if step is None:
                heights.append(0.0)
                missing_x.append(xs[len(heights) - 1])
                missing_y.append(max_step * 1.04)
            else:
                heights.append(float(step))
        ax.bar(
            xs,
            heights,
            width=width * 0.92,
            color=_additional_policy_color(policy_id),
            edgecolor=_additional_C_STROKE,
            linewidth=0.35,
            label=_additional_short_policy_label(policy_id),
        )
        if missing_x:
            ax.scatter(
                missing_x,
                missing_y,
                marker="x",
                s=13,
                color=_additional_policy_color(policy_id),
                linewidths=0.75,
            )
    ax.set_xticks(group_x)
    ax.set_xticklabels(suite_labels, rotation=18, ha="right")
    ax.set_ylabel("Environment steps")
    ax.set_title("Steps to predictive-accuracy thresholds")
    ax.text(
        0.02,
        0.97,
        "Base and hard bars use R2 >= 0.95; mismatch bars use R2 >= 0.90. x = threshold not reached.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=_additional_C_STROKE,
    )
    ax.set_ylim(0.0, max_step * 1.18)
    _style_manuscript_axis(ax, grid_axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=6.3)
    fig.tight_layout()
    return _additional_save_pdf(
        fig, _additional_FIGURE_DIR / "tbme_additional_sample_efficiency_thresholds.pdf"
    )


def _additional_aggregate_metric_rows(group_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in GROUPS[group_name]:
        summary_path = ref.session_root / ref.suite_id / "summary" / "metrics.csv"
        for row in read_trace_csv(summary_path):
            if row.get("status") != "completed":
                continue
            value = _additional_safe_float(row.get("value_final_mean"))
            r2 = _additional_safe_float(row.get("trajectory_r2_final_mean"))
            runtime = _additional_safe_float(row.get("runtime_sec_mean"))
            if r2 is None or runtime is None:
                continue
            payload = dict(row)
            payload["suite_id"] = ref.suite_id
            payload["suite_label"] = ref.label
            payload["parameter_error"] = value
            payload["trajectory_r2"] = r2
            payload["runtime_sec"] = runtime
            rows.append(payload)
    return rows


def _additional_mean_rows_by_policy(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float] | str]] = {}
    for row in rows:
        key = (str(row["suite_label"]), str(row["policy_id"]))
        bucket = grouped.setdefault(
            key,
            {
                "suite_label": str(row["suite_label"]),
                "policy_id": str(row["policy_id"]),
                "error": [],
                "r2": [],
                "runtime": [],
            },
        )
        assert isinstance(bucket["error"], list)
        assert isinstance(bucket["r2"], list)
        assert isinstance(bucket["runtime"], list)
        if row.get("parameter_error") is not None:
            bucket["error"].append(float(row["parameter_error"]))
        bucket["r2"].append(float(row["trajectory_r2"]))
        bucket["runtime"].append(float(row["runtime_sec"]))
    out: list[dict[str, Any]] = []
    for bucket in grouped.values():
        errors = np.asarray(bucket["error"], dtype=np.float64)
        r2s = np.asarray(bucket["r2"], dtype=np.float64)
        runtimes = np.asarray(bucket["runtime"], dtype=np.float64)
        out.append(
            {
                "suite_label": str(bucket["suite_label"]),
                "policy_id": str(bucket["policy_id"]),
                "parameter_error": float(np.mean(errors)) if errors.size else math.nan,
                "trajectory_r2": float(np.mean(r2s)),
                "runtime_sec": float(np.mean(runtimes)),
            }
        )
    return out


def _additional_plot_compute_accuracy_pareto() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))

    schedule_rows = _additional_mean_rows_by_policy(
        _additional_aggregate_metric_rows("exp03_schedule")
    )
    markers = {"Duffing": "o", "Damped pendulum": "s", "Asymmetric basin": "^"}
    ax = axes[0]
    for row in schedule_rows:
        policy_id = str(row["policy_id"])
        suite_label = str(row["suite_label"])
        ax.scatter(
            float(row["runtime_sec"]),
            float(row["trajectory_r2"]),
            s=30,
            marker=markers.get(suite_label, "o"),
            color=_additional_policy_color(policy_id),
            edgecolor=_additional_C_STROKE,
            linewidth=0.35,
            alpha=0.9,
        )
        if policy_id in {
            "active_planning_u20_r20_h40",
            "active_planning_u10_r20_h40",
            "active_planning_u1_r1_h40",
        }:
            ax.annotate(
                policy_id.replace("active_planning_", "").replace("_h40", ""),
                (float(row["runtime_sec"]), float(row["trajectory_r2"])),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.8,
                color=_additional_C_STROKE,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Runtime per run (sec, log scale)")
    ax.set_ylabel("Final prediction R2")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("A. Planning schedule prediction-cost tradeoff")
    _style_manuscript_axis(ax)

    group_rows = []
    for group_name in ("exp01_base", "exp02_hard", "exp04_mismatch"):
        group_rows.extend(
            _additional_mean_rows_by_policy(_additional_aggregate_metric_rows(group_name))
        )
    focus_policies = [
        "active_planning_u20_r20_h40",
        "active_myopic",
        "ensemble",
        "prbs",
        "random",
        "rhc",
    ]
    ax = axes[1]
    group_markers = {
        "Duffing": "o",
        "Damped pendulum": "s",
        "Asymmetric basin": "^",
        "Duffing hard": "D",
        "Asymmetric basin hard": "P",
        "Damped pendulum hard": "X",
        "Duffing parameter mismatch": "v",
        "Asymmetric basin parameter mismatch": ">",
    }
    for row in group_rows:
        policy_id = str(row["policy_id"])
        if policy_id not in focus_policies:
            continue
        suite_label = str(row["suite_label"])
        ax.scatter(
            float(row["runtime_sec"]),
            float(row["trajectory_r2"]),
            s=31,
            marker=group_markers.get(suite_label, "o"),
            color=_additional_policy_color(policy_id),
            edgecolor=_additional_C_STROKE,
            linewidth=0.35,
            alpha=0.86,
            label=_additional_short_policy_label(policy_id),
        )
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.set_xscale("log")
    ax.set_xlabel("Runtime per run (sec, log scale)")
    ax.set_ylabel("Final prediction R2")
    ax.set_ylim(-0.25, 1.05)
    ax.set_title("B. Policy-level prediction-cost tradeoff")
    _style_manuscript_axis(ax)
    ax.legend([h for h, _l in unique], [l for _h, l in unique], fontsize=6.0, loc="upper right")
    fig.tight_layout()
    return _additional_save_pdf(
        fig, _additional_FIGURE_DIR / "tbme_additional_compute_accuracy_pareto.pdf"
    )


def _additional_aggregate_parameter_traces(
    suite_dir: Path,
    policy_ids: Sequence[str],
    *,
    max_seeds: int,
    stride: int,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], np.ndarray]:
    records = _additional_collect_records(suite_dir, policy_ids, max_seeds=max_seeds)
    if not records:
        raise RuntimeError(f"No records found under {suite_dir}")
    true_params = np.asarray(records[0].metadata.get("embedding_true", []), dtype=np.float64)
    traces: dict[str, dict[int, list[np.ndarray]]] = {policy_id: {} for policy_id in policy_ids}
    for record in records:
        path = _additional_trace_path(
            record, "embedding_estimate_trace_path", "embedding_estimate_trace.csv"
        )
        for row in read_trace_csv(path):
            step_raw = _additional_safe_float(row.get("step"))
            if step_raw is None:
                continue
            step = int(step_raw)
            if step % stride != 0 and step != int(record.metadata.get("total_steps", 0)):
                continue
            params: list[float] = []
            for idx in range(true_params.size):
                value = _additional_safe_float(row.get(f"e{idx}"))
                if value is None:
                    break
                params.append(value)
            if len(params) != true_params.size:
                continue
            traces.setdefault(record.policy_id, {}).setdefault(step, []).append(
                np.asarray(params, dtype=np.float64)
            )
    return traces, true_params


def _additional_plot_per_parameter_recovery(max_seeds: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    suite_dir = _additional_suite_dir("exp01_base", "exp01_asymmetric_basin")
    policy_ids = ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "flex", "prbs"]
    traces, true_params = _additional_aggregate_parameter_traces(
        suite_dir,
        policy_ids,
        max_seeds=max_seeds,
        stride=20,
    )
    names = ["a_L", "b_L", "a_R", "b_R"]
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.75), sharex=True)
    for param_idx, ax in enumerate(axes.ravel()):
        for policy_id in policy_ids:
            by_step = traces.get(policy_id, {})
            steps = sorted(by_step)
            if not steps:
                continue
            means = []
            sems = []
            for step in steps:
                vals = np.asarray([arr[param_idx] for arr in by_step[step]], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                means.append(float(np.mean(vals)) if vals.size else np.nan)
                sems.append(_additional_sem(vals.tolist()))
            means_arr = np.asarray(means, dtype=np.float64)
            sems_arr = np.asarray(sems, dtype=np.float64)
            color = _additional_policy_color(policy_id)
            ax.plot(
                steps,
                means_arr,
                color=color,
                linewidth=1.0,
                label=_additional_short_policy_label(policy_id),
            )
            ax.fill_between(
                steps,
                means_arr - sems_arr,
                means_arr + sems_arr,
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )
        ax.axhline(
            float(true_params[param_idx]), color=_additional_C_STROKE, linewidth=0.8, linestyle="--"
        )
        ax.set_title(f"{chr(65 + param_idx)}. {names[param_idx]}")
        ax.set_ylabel("Estimate")
        _style_manuscript_axis(ax)
    axes[1, 0].set_xlabel("Environment step")
    axes[1, 1].set_xlabel("Environment step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=5, fontsize=6.4, bbox_to_anchor=(0.5, 1.015)
    )
    fig.suptitle("Asymmetric-basin per-parameter recovery", y=1.06)
    fig.tight_layout()
    return _additional_save_pdf(
        fig, _additional_FIGURE_DIR / "tbme_additional_asymmetric_basin_parameter_recovery.pdf"
    )


def _additional_plot_information_learning_coupling(max_seeds: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    suite_dir = _additional_suite_dir("exp01_base", "exp01_asymmetric_basin")
    policy_ids = [
        "active_planning_u20_r20_h40",
        "active_myopic",
        "ensemble",
        "prbs",
        "random",
        "rhc",
    ]
    records = _additional_collect_records(suite_dir, policy_ids, max_seeds=max_seeds)
    points: dict[str, list[tuple[float, float, float]]] = {
        policy_id: [] for policy_id in policy_ids
    }
    for record in records:
        info_rows = read_trace_csv(
            _additional_trace_path(record, "information_trace_path", "information_trace.csv")
        )
        r2_rows = read_trace_csv(
            _additional_trace_path(record, "trajectory_r2_trace_path", "trajectory_r2_trace.csv")
        )
        info_vals = [
            value
            for value in (_additional_safe_float(row.get("I_theta_t")) for row in info_rows)
            if value is not None and value >= 0.0
        ]
        r2_vals = [
            value
            for value in (_additional_safe_float(row.get("trajectory_r2")) for row in r2_rows)
            if value is not None
        ]
        if not info_vals or len(r2_vals) < 2:
            continue
        cumulative_info = float(np.sum(info_vals))
        initial_r2 = float(r2_vals[0])
        final_r2_value = float(r2_vals[-1])
        points.setdefault(record.policy_id, []).append(
            (cumulative_info, final_r2_value, final_r2_value - initial_r2)
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    for policy_id in policy_ids:
        vals = points.get(policy_id, [])
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        x = np.log10(1.0 + arr[:, 0])
        axes[0].scatter(
            x,
            arr[:, 1],
            s=13,
            color=_additional_policy_color(policy_id),
            alpha=0.38,
            edgecolors="none",
        )
        axes[0].scatter(
            [float(np.median(x))],
            [float(np.median(arr[:, 1]))],
            s=42,
            color=_additional_policy_color(policy_id),
            edgecolor=_additional_C_STROKE,
            linewidth=0.45,
            label=_additional_short_policy_label(policy_id),
        )
        axes[1].scatter(
            x,
            arr[:, 2],
            s=13,
            color=_additional_policy_color(policy_id),
            alpha=0.38,
            edgecolors="none",
        )
        axes[1].scatter(
            [float(np.median(x))],
            [float(np.median(arr[:, 2]))],
            s=42,
            color=_additional_policy_color(policy_id),
            edgecolor=_additional_C_STROKE,
            linewidth=0.45,
        )
    axes[0].set_title("A. Information versus endpoint prediction")
    axes[0].set_xlabel("log10(1 + cumulative I_theta)")
    axes[0].set_ylabel("Final prediction R2")
    axes[0].set_ylim(-0.1, 1.05)
    axes[1].set_title("B. Information versus R2 improvement")
    axes[1].set_xlabel("log10(1 + cumulative I_theta)")
    axes[1].set_ylabel("Final minus initial prediction R2")
    for ax in axes:
        _style_manuscript_axis(ax)
    axes[0].legend(fontsize=6.0, loc="best")
    fig.tight_layout()
    return _additional_save_pdf(
        fig, _additional_FIGURE_DIR / "tbme_additional_information_learning_coupling.pdf"
    )


def _additional_write_manifest(paths: Sequence[Path]) -> Path:
    _additional_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _additional_GENERATED_DIR / "tbme_additional_figures_manifest.txt"
    lines = ["Additional TBME manuscript figures:"]
    for path in paths:
        lines.append(str(path.relative_to(_additional_REPO_ROOT)))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _additional_write_latex_snippet(paths: Sequence[Path]) -> Path:
    _additional_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    snippet = _additional_GENERATED_DIR / "tbme_additional_figures.tex"
    captions = {
        "tbme_additional_true_dynamics_all.pdf": (
            "True phase portraits for the distinct TBME synthetic dynamics. Background color shows "
            "log speed, and streamlines show the direction of the latent drift over $[-6, 6]^2$; "
            "observation-only and mismatch-dose variants reuse one of these true vector fields."
        ),
        "tbme_additional_asymmetric_basin_mechanism.pdf": (
            "Hard asymmetric-basin mechanism diagnostics under asymmetric observation loading. "
            "Panel A overlays trajectories and the true dynamics vector field on the seed-averaged "
            "spatial observation information geometry; the remaining panels connect this geometry "
            "to state-space coverage and endpoint prediction."
        ),
        "tbme_additional_asymmetric_basin_learned_vectorfields.pdf": (
            "Hard asymmetric-basin vector fields for one shared seed. The first row shows the true "
            "vector field; remaining rows correspond to methods, and columns show checkpoints at "
            "250, 500, and 1000 interaction steps. Learned-field panels overlay the trajectory "
            "prefix on the vector field induced by the current parameter estimate."
        ),
        "tbme_additional_sample_efficiency_thresholds.pdf": (
            "Sample efficiency measured by the first environment step at which each method reaches "
            "the indicated trajectory-$R^2$ threshold."
        ),
        "tbme_additional_compute_accuracy_pareto.pdf": (
            "Prediction-cost tradeoffs across planning schedules and policy families."
        ),
        "tbme_additional_asymmetric_basin_parameter_recovery.pdf": (
            "Per-parameter recovery in the asymmetric-basin benchmark, including the FLEX baseline."
        ),
        "tbme_additional_information_learning_coupling.pdf": (
            "Relationship between accumulated parameter information and predictive-$R^2$ improvement."
        ),
    }
    labels = {
        "tbme_additional_true_dynamics_all.pdf": "fig:tbme_additional_true_dynamics_all",
        "tbme_additional_asymmetric_basin_mechanism.pdf": "fig:tbme_additional_asymmetric_basin_mechanism",
        "tbme_additional_asymmetric_basin_learned_vectorfields.pdf": "fig:tbme_additional_learned_vectorfields",
        "tbme_additional_sample_efficiency_thresholds.pdf": "fig:tbme_additional_sample_efficiency",
        "tbme_additional_compute_accuracy_pareto.pdf": "fig:tbme_additional_compute_pareto",
        "tbme_additional_asymmetric_basin_parameter_recovery.pdf": "fig:tbme_additional_parameter_recovery",
        "tbme_additional_information_learning_coupling.pdf": "fig:tbme_additional_information_learning",
    }
    lines: list[str] = ["% Auto-generated by experiments/tbme/generate_figures.py additional"]
    for path in paths:
        name = path.name
        lines.extend(
            [
                r"\begin{figure*}[t]",
                r"	\centering",
                rf"	\includegraphics[width=\textwidth]{{../figs/{name}}}",
                rf"	\caption{{{captions[name]}}}",
                rf"	\label{{{labels[name]}}}",
                r"\end{figure*}",
                "",
            ]
        )
    snippet.write_text("\n".join(lines), encoding="utf-8")
    return snippet


def _additional_parse_plots(raw: str) -> list[str]:
    plot_ids = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(plot_ids) - set(_additional_PLOTS))
    if unknown:
        raise ValueError(f"Unknown additional plot(s): {', '.join(unknown)}")
    return plot_ids


def _additional_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate additive TBME manuscript figures from existing experiment logs."
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=100,
        help="Maximum seeds per policy to read for trace-derived figures.",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default=",".join(_additional_PLOTS),
        help="Comma-separated additional TBME plot ids.",
    )
    return parser

# Main functions
def asset_main() -> int:
    summary_lines = []
    copied_by_group: dict[str, list[Path]] = {}
    for group_name, refs in GROUPS.items():
        rows, copied = _asset_export_group(group_name, refs)
        copied_by_group[group_name] = copied
        summary_lines.append(f"{group_name}: {len(rows)} table rows, {len(copied)} copied figures")
    pareto_path = _asset_plot_schedule_threshold_pareto()
    if pareto_path is not None:
        copied_by_group.setdefault("exp03_schedule", []).append(pareto_path)
        summary_lines.append(f"exp03_schedule_pareto: 1 copied figure")
    manifest = _asset_TEX_DIR / "tbme_current_export_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(manifest)
    return 0


def summary_main(argv: list[str] | None = None) -> int:
    args = _summary_build_parser().parse_args(argv)
    groups = [item.strip() for item in str(args.groups).split(",") if item.strip()]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")
    figure_formats = _parse_figure_formats(str(args.figure_formats))

    written: list[Path] = []
    for group_name in groups:
        for ref in GROUPS[group_name]:
            suite_dir = ref.session_root / ref.suite_id
            if suite_dir.exists():
                written.extend(_summary_write_figures(suite_dir, figure_formats))
    for path in written:
        print(path)
    print(f"wrote {len(written)} summary figure files")
    return 0


def trajectory_main(argv: list[str] | None = None) -> int:
    _apply_style()
    args = _trajectory_build_parser().parse_args(argv)
    groups = [item.strip() for item in str(args.groups).split(",") if item.strip()]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")

    written: list[Path] = []
    for suite_dir in _trajectory_suite_dirs(groups):
        metadata = _trajectory_reference_metadata(suite_dir)
        if metadata is None:
            continue
        dynamics_payload = _trajectory_build_true_dynamics(metadata)
        if dynamics_payload is None:
            continue
        dyn_true, grid_lim, system_label = dynamics_payload
        grouped = _trajectory_collect_policy_traces(suite_dir, max_seeds=int(args.max_seeds))
        if not grouped:
            continue
        written.append(
            _trajectory_plot_overlay_figure(
                suite_dir,
                grouped=grouped,
                dyn_true=dyn_true,
                grid_lim=grid_lim,
                system_label=system_label,
                max_seeds=int(args.max_seeds),
            )
        )
        written.append(
            _trajectory_plot_density_figure(
                suite_dir,
                grouped=grouped,
                dyn_true=dyn_true,
                grid_lim=grid_lim,
                system_label=system_label,
                max_seeds=int(args.max_seeds),
                bins=int(args.density_bins),
            )
        )
    for path in written:
        print(path)
    print(f"wrote {len(written)} trajectory summary figures")
    return 0


def requested_main(argv: list[str] | None = None) -> int:
    args = _requested_build_parser().parse_args(argv)
    plot_ids = _requested_parse_plots(str(args.plots))
    missing_suite_dirs = sorted(
        (
            suite_dir
            for suite_dir in _requested_required_suite_dirs(plot_ids)
            if not suite_dir.exists()
        ),
        key=str,
    )
    if missing_suite_dirs:
        missing_text = ", ".join(str(path) for path in missing_suite_dirs)
        raise FileNotFoundError(f"Missing requested experiment suite(s): {missing_text}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    figure_paths: list[Path] = []
    csv_paths: list[Path] = []
    if any(plot_id in _requested_OBJECTIVE_DEFINITION_PLOTS for plot_id in plot_ids):
        objective_definition_csv, objective_definition_tex = (
            _requested_write_objective_definition_tables()
        )
        csv_paths.extend([objective_definition_csv, objective_definition_tex])

    plotters = {
        "bottleneck_sweep": lambda: _requested_plot_bottleneck_sweep(plt),
        "objective_ablation": lambda: _requested_plot_objective_ablation(plt),
        "mismatch_dose_response": lambda: _requested_plot_mismatch_dose_response(plt),
        "downstream_control": lambda: _requested_plot_downstream_control(plt),
    }
    for plot_id in plot_ids:
        figure_path, csv_path = plotters[plot_id]()
        figure_paths.append(figure_path)
        csv_paths.append(csv_path)
        plt.close("all")
    manifest_path = _requested_write_manifest([*figure_paths, *csv_paths])
    for path in [*figure_paths, *csv_paths, manifest_path]:
        print(path)
    return 0


def additional_main(argv: list[str] | None = None) -> int:
    args = _additional_build_parser().parse_args(argv)
    max_seeds = int(args.max_seeds)
    plot_ids = _additional_parse_plots(str(args.plots))
    plotters = {
        "true_dynamics_all": lambda: _additional_plot_true_dynamics_all(),
        "asymmetric_basin_mechanism": lambda: _additional_plot_asymmetric_basin_mechanism(
            max_seeds=max_seeds
        ),
        "learned_vectorfield_snapshots": lambda: _additional_plot_learned_vectorfield_snapshots(
            max_seeds=max_seeds
        ),
        "sample_efficiency_thresholds": lambda: _additional_plot_sample_efficiency_thresholds(),
        "compute_accuracy_pareto": lambda: _additional_plot_compute_accuracy_pareto(),
        "per_parameter_recovery": lambda: _additional_plot_per_parameter_recovery(
            max_seeds=max_seeds
        ),
        "information_learning_coupling": lambda: _additional_plot_information_learning_coupling(
            max_seeds=max_seeds
        ),
    }
    paths = [plotters[plot_id]() for plot_id in plot_ids]
    manifest = _additional_write_manifest(paths)
    snippet = _additional_write_latex_snippet(paths)
    for path in paths:
        print(path)
    print(manifest)
    print(snippet)
    return 0
