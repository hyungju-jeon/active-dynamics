#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from actdyn.environment import residual_np
from actdyn.environment.vectorfield import ResidualDynamicsCallable
from actdyn.utils.experiment_runtime import read_trace_csv
from actdyn.utils.plotting import (
    apply_manuscript_figure_style,
    compute_vector_field,
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
    apply_tbme_asset_plot_style,
    plot_final_value_by_policy,
    plot_information_colormap,
    plot_metric_over_cpu_time,
    plot_metric_over_steps,
    plot_neuron_tuning_curve_colormap,
    plot_tbme_asymmetric_basin_mechanism,
    plot_tbme_bottleneck_sweep,
    plot_tbme_compute_accuracy_pareto,
    plot_tbme_downstream_control,
    plot_tbme_information_learning_coupling,
    plot_tbme_learned_vectorfield_snapshots,
    plot_tbme_mismatch_dose_response,
    plot_tbme_objective_ablation,
    plot_tbme_per_parameter_recovery,
    plot_tbme_r2_threshold_stacked_bars,
    plot_tbme_sample_efficiency_thresholds,
    plot_tbme_schedule_threshold_pareto,
    plot_tbme_trajectory_density,
    plot_tbme_trajectory_overlay,
    plot_tbme_true_dynamics_all,
)
from .run_tbme_experiments import configure_tbme_catalogs

configure_tbme_catalogs()


# Common variables
_TBME_STROKE_COLOR = "#3A3A3A"
_TBME_GRID_COLOR = "#DDD7CE"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_TBME_RESULTS_DIR = _RESULTS_ROOT / "tbme"
_DOCS_DIR = _REPO_ROOT / "docs"
_DOCS_FIGURE_DIR = _DOCS_DIR / "figs"
_DOCS_TABLE_DIR = _DOCS_DIR / "tables"


# GROUPS is evaluated at import time, so this constructor stays before GROUPS.
def _latest_session(base: Path) -> Path:
    sessions = [
        path
        for path in base.glob("session_*")
        if path.is_dir() and path.name.removeprefix("session_").isdigit()
    ]
    if not sessions:
        return base / "session_1"
    return max(sessions, key=lambda path: int(path.name.removeprefix("session_")))


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
            _latest_session(_TBME_RESULTS_DIR / "exp01_base"),
            "duffing",
        ),
        SuiteRef(
            "exp01_damped_pendulum",
            "Damped pendulum",
            _latest_session(_TBME_RESULTS_DIR / "exp01_base"),
            "damped_pendulum",
        ),
        SuiteRef(
            "exp01_asymmetric_basin",
            "Asymmetric basin",
            _latest_session(_TBME_RESULTS_DIR / "exp01_base"),
            "asymmetric_basin",
        ),
    ],
    "exp02_hard": [
        SuiteRef(
            "exp02_hard_duffing",
            "Duffing hard",
            _latest_session(_TBME_RESULTS_DIR / "exp02_hard"),
            "duffing_hard",
        ),
        SuiteRef(
            "exp02_hard_asymmetric_basin",
            "Asymmetric basin hard",
            _latest_session(_TBME_RESULTS_DIR / "exp02_hard"),
            "asymmetric_basin_hard",
        ),
        SuiteRef(
            "exp02_hard_damped_pendulum",
            "Damped pendulum hard",
            _latest_session(_TBME_RESULTS_DIR / "exp02_hard"),
            "damped_pendulum_hard",
        ),
    ],
    "exp03_schedule": [
        SuiteRef(
            "exp03_schedule_duffing",
            "Duffing",
            _latest_session(_TBME_RESULTS_DIR / "exp03_schedule"),
            "duffing",
        ),
        SuiteRef(
            "exp03_schedule_damped_pendulum",
            "Damped pendulum",
            _latest_session(_TBME_RESULTS_DIR / "exp03_schedule"),
            "damped_pendulum",
        ),
        SuiteRef(
            "exp03_schedule_asymmetric_basin",
            "Asymmetric basin",
            _latest_session(_TBME_RESULTS_DIR / "exp03_schedule"),
            "asymmetric_basin",
        ),
    ],
    "exp04_mismatch": [
        SuiteRef(
            "exp04_duffing_parameter_mismatch",
            "Duffing parameter mismatch",
            _latest_session(_TBME_RESULTS_DIR / "exp04_mismatch"),
            "duffing_parameter_mismatch",
        ),
        SuiteRef(
            "exp04_asymmetric_basin_parameter_mismatch",
            "Asymmetric basin parameter mismatch",
            _latest_session(_TBME_RESULTS_DIR / "exp04_mismatch"),
            "asymmetric_basin_parameter_mismatch",
        ),
    ],
    "exp05_ablation": [
        SuiteRef(
            "exp05_asymmetric_basin_objective_ablation",
            "Asymmetric basin objective ablation",
            _latest_session(_TBME_RESULTS_DIR / "exp05_ablation"),
            "asymmetric_basin_objective_ablation",
        ),
        SuiteRef(
            "exp05_hard_asymmetric_basin_objective_ablation",
            "Hard asymmetric basin objective ablation",
            _latest_session(_TBME_RESULTS_DIR / "exp05_ablation"),
            "hard_asymmetric_basin_objective_ablation",
        ),
    ],
    "exp06_bottleneck": [
        SuiteRef(
            "exp06_asymmetric_basin_bottleneck_weak_observation",
            "Asymmetric basin weak observation",
            _latest_session(_TBME_RESULTS_DIR / "exp06_bottleneck"),
            "asymmetric_basin_bottleneck_weak_observation",
        ),
        SuiteRef(
            "exp06_asymmetric_basin_bottleneck_tight_action",
            "Asymmetric basin tight action",
            _latest_session(_TBME_RESULTS_DIR / "exp06_bottleneck"),
            "asymmetric_basin_bottleneck_tight_action",
        ),
        SuiteRef(
            "exp06_asymmetric_basin_bottleneck_combined",
            "Asymmetric basin bottleneck",
            _latest_session(_TBME_RESULTS_DIR / "exp06_bottleneck"),
            "asymmetric_basin_bottleneck_combined",
        ),
    ],
    "exp07_mismatch_stress": [
        SuiteRef(
            "exp07_duffing_parameter_mismatch_mild",
            "Duffing parameter mismatch mild",
            _latest_session(_TBME_RESULTS_DIR / "exp07_mismatch_stress"),
            "duffing_parameter_mismatch_mild",
        ),
        SuiteRef(
            "exp07_duffing_parameter_mismatch_strong",
            "Duffing parameter mismatch strong",
            _latest_session(_TBME_RESULTS_DIR / "exp07_mismatch_stress"),
            "duffing_parameter_mismatch_strong",
        ),
        SuiteRef(
            "exp07_asymmetric_basin_parameter_mismatch_mild",
            "Asymmetric basin parameter mismatch mild",
            _latest_session(_TBME_RESULTS_DIR / "exp07_mismatch_stress"),
            "asymmetric_basin_parameter_mismatch_mild",
        ),
        SuiteRef(
            "exp07_asymmetric_basin_parameter_mismatch_strong",
            "Asymmetric basin parameter mismatch strong",
            _latest_session(_TBME_RESULTS_DIR / "exp07_mismatch_stress"),
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


# Shared helpers


def _apply_style(plt_module: Any | None = None) -> None:
    if plt_module is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_module
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


def _safe_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def _r2_threshold_suffix(threshold: float) -> str:
    return f"{float(threshold):.2f}".replace(".", "p")


def _read_xy_trace(path: Path) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for row in read_trace_csv(path):
        x_val = _safe_float(row.get("true_x"))
        v_val = _safe_float(row.get("true_v"))
        if x_val is None or v_val is None:
            continue
        points.append((x_val, v_val))
    return np.asarray(points, dtype=np.float32)


# summary_main variables
_summary_trace_C_WRITE = "#1F4FA8"
_summary_trace_C_STROKE = "#3A3A3A"
_summary_trace_C_NEUTRAL_LIGHT = "#C8C1B8"

# summary_main helpers
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
    plot_metric_over_steps(
        figures_dir,
        rows=cov_rows,
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


def _summary_write_trace_figures(
    suite_dir: Path,
    *,
    max_seeds: int,
    density_bins: int,
) -> list[Path]:
    metadata = _summary_trace_reference_metadata(suite_dir)
    if metadata is None:
        return []
    dynamics_payload = _summary_trace_build_true_dynamics(metadata)
    if dynamics_payload is None:
        return []
    dyn_true, grid_lim, system_label = dynamics_payload
    grouped = _summary_trace_collect_policy_traces(suite_dir, max_seeds=max_seeds)
    if not grouped:
        return []
    figures_dir = suite_dir / "summary" / "figures"
    written = [
        plot_tbme_trajectory_overlay(
            figures_dir / "trajectory_overlay_vectorfield_by_policy.pdf",
            suite_name=suite_dir.name,
            grouped=grouped,
            dyn_true=dyn_true,
            grid_lim=grid_lim,
            system_label=system_label,
            max_seeds=max_seeds,
            policy_sort_key=_policy_sort_key,
            policy_label=_summary_trace_policy_label,
            apply_style=_apply_style,
            stroke_color=_summary_trace_C_STROKE,
            write_color=_summary_trace_C_WRITE,
            neutral_light=_summary_trace_C_NEUTRAL_LIGHT,
        ),
        plot_tbme_trajectory_density(
            figures_dir / "trajectory_density_by_policy.pdf",
            suite_name=suite_dir.name,
            grouped=grouped,
            dyn_true=dyn_true,
            grid_lim=grid_lim,
            system_label=system_label,
            max_seeds=max_seeds,
            bins=density_bins,
            policy_sort_key=_policy_sort_key,
            policy_label=_summary_trace_policy_label,
            apply_style=_apply_style,
            stroke_color=_summary_trace_C_STROKE,
            neutral_light=_summary_trace_C_NEUTRAL_LIGHT,
        ),
    ]
    return [path for path in written if path is not None]


def _summary_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write current TBME summary and trajectory figures."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(GROUPS),
        help="Comma-separated TBME group names.",
    )
    parser.add_argument("--figure-formats", type=str, default=".pdf")
    parser.add_argument("--trajectory-max-seeds", type=int, default=50)
    parser.add_argument("--density-bins", type=int, default=96)
    return parser


def _summary_trace_policy_label(policy_id: str) -> str:
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


def _summary_trace_collect_policy_traces(
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
        key=lambda p: _policy_sort_key(p.name),
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
            traj = _read_xy_trace(trace_path)
            if traj.size:
                traces.append((seed, traj))
        if traces:
            grouped[policy_dir.name] = traces
    return grouped


def _summary_trace_reference_metadata(suite_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(suite_dir.glob("track/*/seed_*/repeat_*/run_metadata.json"))
    if not candidates:
        return None
    return load_json(candidates[0])


def _summary_trace_build_true_dynamics(metadata: dict[str, Any]) -> tuple[Any, float, str] | None:
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


def summary_main(argv: list[str] | None = None) -> int:
    _apply_style()
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
                written.extend(
                    _summary_write_trace_figures(
                        suite_dir,
                        max_seeds=int(args.trajectory_max_seeds),
                        density_bins=int(args.density_bins),
                    )
                )
    for path in written:
        print(path)
    print(f"wrote {len(written)} summary figure files")
    return 0


# group_overview_main variables
_overview_R2_THRESHOLDS = (0.90, 0.95, 0.99)
_overview_C_WRITE = "#1F4FA8"
_overview_C_WRITE_SOFT = "#6F8EC8"
_overview_C_STROKE = "#3A3A3A"
_overview_C_NEUTRAL = "#6F6A62"
_overview_C_NEUTRAL_LIGHT = "#C8C1B8"
_overview_C_NEUTRAL_FILL = "#F4F1EC"
_overview_C_WHITE = "#FFFFFF"
_overview_R2_THRESHOLD_SEGMENT_COLORS = (
    _overview_C_NEUTRAL_LIGHT,
    _overview_C_WRITE_SOFT,
    _overview_C_WRITE,
)
_overview_R2_THRESHOLD_POINT_COLORS = {
    0.90: _overview_C_NEUTRAL,
    0.95: _overview_C_WRITE_SOFT,
    0.99: _overview_C_WRITE,
}


# group_overview_main helpers
def _overview_dir(group_name: str) -> Path:
    return _TBME_RESULTS_DIR / group_name / "overview"


def _overview_figures_dir(group_name: str) -> Path:
    return _overview_dir(group_name) / "figures"


def _overview_summary_dir(ref: SuiteRef) -> Path:
    return ref.session_root / ref.suite_id / "summary"


def _overview_fmt(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return "--"
    if not math.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _overview_aggregate_suite(ref: SuiteRef) -> list[dict[str, object]]:
    metrics_path = _overview_summary_dir(ref) / "metrics.csv"
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
    out.sort(key=lambda row: _policy_sort_key(str(row["policy_id"])))
    return out


def _overview_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _overview_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _overview_write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "% Auto-generated by experiments/tbme/generate_figures.py overview",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Suite & Policy & Param. error & Trajectory $R^2$ & Runtime (s) \\",
        r"\midrule",
    ]
    current_suite = None
    for row in rows:
        suite = str(row["suite_label"])
        suite_cell = _overview_escape(suite) if suite != current_suite else ""
        current_suite = suite
        line = (
            " & ".join(
                [
                    suite_cell,
                    _overview_escape(str(row["policy_label"])),
                    _overview_fmt(
                        float(row["parameter_error_mean"]), float(row["parameter_error_std"])
                    ),
                    _overview_fmt(float(row["trajectory_r2_mean"]), float(row["trajectory_r2_std"])),
                    _overview_fmt(
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


def _overview_threshold_rows_for_suite(ref: SuiteRef) -> list[dict[str, object]]:
    threshold_path = _overview_summary_dir(ref) / "trajectory_r2_thresholds.csv"
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
            for threshold in _overview_R2_THRESHOLDS:
                suffix = _r2_threshold_suffix(threshold)
                payload[f"step_to_r2_{suffix}"] = _safe_float(row.get(f"step_to_r2_{suffix}"))
                payload[f"cpu_time_sec_to_r2_{suffix}"] = _safe_float(
                    row.get(f"cpu_time_sec_to_r2_{suffix}")
                )
            out.append(payload)
        out.sort(key=lambda row: _policy_sort_key(str(row["policy_id"])))
        return out

    trace_path = _overview_summary_dir(ref) / "trajectory_r2_over_steps.csv"
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
        for threshold in _overview_R2_THRESHOLDS:
            suffix = _r2_threshold_suffix(threshold)
            crossing = None
            for row in series:
                value = _safe_float(row.get("trajectory_r2_mean"))
                if value is not None and value >= threshold:
                    crossing = row
                    break
            payload[f"step_to_r2_{suffix}"] = (
                int(float(crossing["step"])) if crossing is not None else None
            )
            payload[f"cpu_time_sec_to_r2_{suffix}"] = (
                _safe_float(crossing.get("cpu_time_sec_mean"))
                if crossing is not None
                else None
            )
        out.append(payload)
    out.sort(key=lambda row: _policy_sort_key(str(row["policy_id"])))
    return out


def _overview_short_policy_label(policy_id: str) -> str:
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


def _overview_inline_policy_label(policy_id: str) -> str:
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


def _overview_threshold_segments(
    row: dict[str, object], *, field_prefix: str
) -> tuple[list[float], bool]:
    segments: list[float] = []
    previous = 0.0
    reached_all = True
    for threshold in _overview_R2_THRESHOLDS:
        suffix = _r2_threshold_suffix(float(threshold))
        value = _safe_float(row.get(f"{field_prefix}_{suffix}"))
        if value is None:
            segments.append(0.0)
            reached_all = False
            continue
        value = max(value, previous)
        segments.append(value - previous)
        previous = value
    return segments, reached_all


def _overview_threshold_value_penalty(
    threshold_rows: list[dict[str, object]],
    *,
    field_prefix: str,
) -> float:
    values: list[float] = []
    for row in threshold_rows:
        for threshold in _overview_R2_THRESHOLDS:
            suffix = _r2_threshold_suffix(float(threshold))
            value = _safe_float(row.get(f"{field_prefix}_{suffix}"))
            if value is not None:
                values.append(value)
    return max(values) * 1.25 if values else 1.0


def _overview_policy_threshold_sort_key(
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
        for threshold in _overview_R2_THRESHOLDS:
            suffix = _r2_threshold_suffix(float(threshold))
            value = _safe_float(row.get(f"{field_prefix}_{suffix}"))
            if value is None:
                values.append(missing_penalty)
            else:
                values.append(value)
                reached_count += 1
    return (
        float(np.mean(values)) if values else missing_penalty,
        -reached_count,
        _policy_sort_key(policy_id),
    )


def _overview_apply_plot_style(plt: object) -> None:
    apply_tbme_asset_plot_style(plt, stroke_color=_overview_C_STROKE)


def _overview_plot_r2_threshold_stacked_bars(
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
    out_path = _overview_figures_dir(group_name) / output_name
    return plot_tbme_r2_threshold_stacked_bars(
        out_path,
        group_name=group_name,
        refs=refs,
        threshold_rows=threshold_rows,
        thresholds=_overview_R2_THRESHOLDS,
        field_prefix=field_prefix,
        ylabel=ylabel,
        title_metric=title_metric,
        log_y=log_y,
        threshold_suffix=_r2_threshold_suffix,
        safe_float=_safe_float,
        threshold_segments=lambda row, prefix: _overview_threshold_segments(
            row, field_prefix=prefix
        ),
        threshold_value_penalty=lambda rows, prefix: _overview_threshold_value_penalty(
            rows, field_prefix=prefix
        ),
        policy_threshold_sort_key=lambda policy_id, refs_arg, row_by_key, prefix, penalty: (
            _overview_policy_threshold_sort_key(
                policy_id,
                list(refs_arg),
                row_by_key,
                field_prefix=prefix,
                missing_penalty=penalty,
            )
        ),
        short_policy_label=_overview_short_policy_label,
        apply_style=_overview_apply_plot_style,
        stroke_color=_overview_C_STROKE,
        neutral_color=_overview_C_NEUTRAL,
        neutral_light=_overview_C_NEUTRAL_LIGHT,
        neutral_fill=_overview_C_NEUTRAL_FILL,
        segment_colors=_overview_R2_THRESHOLD_SEGMENT_COLORS,
    )


def _overview_plot_schedule_threshold_pareto() -> Path | None:
    schedule_rows: list[dict[str, object]] = []
    for ref in GROUPS["exp03_schedule"]:
        schedule_rows.extend(
            row
            for row in _overview_threshold_rows_for_suite(ref)
            if str(row["policy_id"]).startswith("active_planning")
        )
    myopic_rows: list[dict[str, object]] = []
    for ref in GROUPS["exp01_base"]:
        myopic_rows.extend(
            row
            for row in _overview_threshold_rows_for_suite(ref)
            if str(row["policy_id"]) == "active_myopic"
        )
    rows = [*schedule_rows, *myopic_rows]
    if not rows:
        return None

    env_labels = [ref.label for ref in GROUPS["exp03_schedule"]]
    policy_ids = sorted({str(row["policy_id"]) for row in rows}, key=_policy_sort_key)
    out_path = (
        _overview_figures_dir("exp03_schedule")
        / "r2_threshold_pareto_step_cpu_by_environment.pdf"
    )
    return plot_tbme_schedule_threshold_pareto(
        out_path,
        rows=rows,
        env_labels=env_labels,
        policy_ids=policy_ids,
        thresholds=_overview_R2_THRESHOLDS,
        threshold_suffix=_r2_threshold_suffix,
        safe_float=_safe_float,
        short_policy_label=_overview_inline_policy_label,
        threshold_point_colors=_overview_R2_THRESHOLD_POINT_COLORS,
        apply_style=_overview_apply_plot_style,
        stroke_color=_overview_C_STROKE,
        neutral_light=_overview_C_NEUTRAL_LIGHT,
        white_color=_overview_C_WHITE,
    )


def _overview_export_group(
    group_name: str, refs: list[SuiteRef]
) -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for ref in refs:
        rows.extend(_overview_aggregate_suite(ref))
        threshold_rows.extend(_overview_threshold_rows_for_suite(ref))
    rows.sort(
        key=lambda row: (str(row["suite_label"]), _policy_sort_key(str(row["policy_id"])))
    )
    threshold_rows.sort(
        key=lambda row: (str(row["suite_label"]), _policy_sort_key(str(row["policy_id"])))
    )
    overview_dir = _overview_dir(group_name)
    csv_path = overview_dir / "overview_table.csv"
    tex_path = overview_dir / "overview_table.tex"
    _overview_write_csv(csv_path, rows)
    _overview_write_tex(tex_path, rows)
    written = [csv_path, tex_path]
    threshold_plots = [
        _overview_plot_r2_threshold_stacked_bars(
            group_name,
            refs,
            threshold_rows,
            field_prefix="step_to_r2",
            ylabel="Environment steps",
            title_metric="steps",
            output_name="r2_threshold_stacked_steps_by_environment.pdf",
        ),
        _overview_plot_r2_threshold_stacked_bars(
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
    written.extend(path for path in threshold_plots if path is not None)
    return rows, written


def _group_overview_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write group-level TBME overview tables and figures into results/tbme."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(GROUPS),
        help="Comma-separated TBME group names.",
    )
    return parser


def group_overview_main(argv: list[str] | None = None) -> int:
    args = _group_overview_build_parser().parse_args(argv)
    groups = [item.strip() for item in str(args.groups).split(",") if item.strip()]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")

    row_counts: dict[str, int] = {}
    written_by_group: dict[str, list[Path]] = {}
    for group_name in groups:
        rows, written = _overview_export_group(group_name, GROUPS[group_name])
        row_counts[group_name] = len(rows)
        written_by_group[group_name] = written

    if "exp03_schedule" in groups:
        pareto_path = _overview_plot_schedule_threshold_pareto()
        if pareto_path is not None:
            written_by_group.setdefault("exp03_schedule", []).append(pareto_path)

    for group_name in groups:
        written = written_by_group.get(group_name, [])
        print(f"{group_name}: {row_counts[group_name]} table rows, {len(written)} overview files")
        for path in written:
            print(path)
    return 0


# experiment_main variables
_experiment_C_STROKE = "#3A3A3A"
_experiment_C_NEUTRAL_LIGHT = "#C8C1B8"
_experiment_C_NEUTRAL_FILL = "#F4F1EC"
_experiment_C_GRID = "#DDD7CE"

_experiment_BOTTLENECK_POLICIES = [
    "active_planning_u20_r20_h40",
    "active_myopic",
    "ensemble",
    "prbs",
    "random",
]
_experiment_OBJECTIVE_POLICIES = [
    "active_planning_u20_r20_h40",
    "active_fully_observable_u20_r20_h40",
    "active_state_information_u20_r20_h40",
    "active_dynamics_u20_r20_h40",
    "active_sampling_variance_u20_r20_h40",
    "active_e_optimality_u20_r20_h40",
    "ensemble",
    "prbs",
]
_experiment_OBJECTIVE_DEFINITIONS = [
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
_experiment_DOSE_POLICIES = [
    "active_planning_u20_r20_h40",
    "active_myopic",
    "ensemble",
    "prbs",
    "random",
]


@dataclass(frozen=True)
class _ExperimentSuiteSource:
    exp_id: str
    label: str
    suite_dir: Path
    dose: str | None = None
    family: str | None = None


@dataclass(frozen=True)
class _ExperimentRunRecord:
    policy_id: str
    seed: int
    run_dir: Path
    metadata: dict[str, Any]


_experiment_PLOTS = (
    "bottleneck_sweep",
    "objective_ablation",
    "mismatch_dose_response",
    "downstream_control",
    "true_dynamics_all",
    "asymmetric_basin_mechanism",
    "learned_vectorfield_snapshots",
    "sample_efficiency_thresholds",
    "compute_accuracy_pareto",
    "per_parameter_recovery",
    "information_learning_coupling",
)
EXPERIMENT_PLOTS = _experiment_PLOTS

_experiment_OBJECTIVE_DEFINITION_PLOTS = {"objective_ablation", "downstream_control"}
_experiment_REQUIRED_SUITES_BY_PLOT = {
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


# experiment_main helpers
def _style_experiment_axis(ax: Any) -> None:
    _style_manuscript_axis(ax, grid_alpha=0.55)

def _experiment_parse_plots(raw: str) -> list[str]:
    plot_ids = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(plot_ids) - set(EXPERIMENT_PLOTS))
    if unknown:
        raise ValueError(f"Unknown experiment plot(s): {', '.join(unknown)}")

    ordered: list[str] = []
    seen: set[str] = set()
    for plot_id in plot_ids:
        if plot_id not in seen:
            ordered.append(plot_id)
            seen.add(plot_id)
    return ordered


def _experiment_required_suite_dirs(plot_ids: Sequence[str]) -> list[Path]:
    suite_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for plot_id in plot_ids:
        for key in _experiment_REQUIRED_SUITES_BY_PLOT.get(plot_id, ()):
            if key not in seen:
                suite_keys.append(key)
                seen.add(key)
    return [_suite_dir(group_name, suite_id) for group_name, suite_id in suite_keys]


def _experiment_write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _experiment_policy_label(policy_id: str) -> str:
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


def _experiment_escape_tex(text: object) -> str:
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _experiment_metrics_by_policy(suite_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows = read_trace_csv(suite_dir / "summary" / "metrics.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") not in {None, "", "completed"}:
            continue
        grouped.setdefault(str(row.get("policy_id", "")), []).append(row)
    return grouped


def _experiment_metric_values(suite_dir: Path, policy_id: str, field: str) -> list[float]:
    values: list[float] = []
    for row in _experiment_metrics_by_policy(suite_dir).get(policy_id, []):
        value = _safe_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _experiment_metric_mean_sem(
    suite_dir: Path, policy_id: str, field: str
) -> tuple[float | None, float, int]:
    values = _experiment_metric_values(suite_dir, policy_id, field)
    if not values:
        return None, 0.0, 0
    return float(np.mean(values)), _sem(values), len(values)


def _experiment_curve_rows(
    suite_dir: Path, name: str, value_col: str
) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in read_trace_csv(suite_dir / "summary" / name):
        policy_id = str(row.get("policy_id", ""))
        step = _safe_float(row.get("step"))
        value = _safe_float(row.get(value_col))
        sem = _safe_float(row.get("value_sem"))
        if not policy_id or step is None or value is None:
            continue
        grouped.setdefault(policy_id, []).append(
            {"step": step, "value": value, "sem": 0.0 if sem is None else sem}
        )
    for policy_rows in grouped.values():
        policy_rows.sort(key=lambda row: row["step"])
    return grouped


def _experiment_r2_threshold_step(
    suite_dir: Path, policy_id: str, threshold: float = 0.90
) -> float | None:
    suffix = _r2_threshold_suffix(threshold)
    for row in read_trace_csv(suite_dir / "summary" / "trajectory_r2_thresholds.csv"):
        if str(row.get("policy_id", "")) != policy_id:
            continue
        return _safe_float(row.get(f"step_to_r2_{suffix}"))
    return None


def _experiment_r2_threshold_times(
    suite_dir: Path,
    policy_id: str,
    threshold: float,
) -> tuple[float | None, float | None, float | None]:
    suffix = _r2_threshold_suffix(threshold)
    for row in read_trace_csv(suite_dir / "summary" / "trajectory_r2_thresholds.csv"):
        if str(row.get("policy_id", "")) != policy_id:
            continue
        return (
            _safe_float(row.get(f"step_to_r2_{suffix}")),
            _safe_float(row.get(f"cpu_time_sec_to_r2_{suffix}")),
            _safe_float(row.get(f"r2_at_{suffix}")),
        )
    return None, None, None


def _experiment_plot_bottleneck_sweep() -> tuple[Path, Path]:
    sources = [
        _ExperimentSuiteSource(
            "exp01_asymmetric_basin",
            "Nominal",
            _suite_dir("exp01_base", "exp01_asymmetric_basin"),
        ),
        _ExperimentSuiteSource(
            "exp06_asymmetric_basin_bottleneck_weak_observation",
            "Weak obs.",
            _suite_dir(
                "exp06_bottleneck",
                "exp06_asymmetric_basin_bottleneck_weak_observation",
            ),
        ),
        _ExperimentSuiteSource(
            "exp06_asymmetric_basin_bottleneck_tight_action",
            "Tight action",
            _suite_dir(
                "exp06_bottleneck",
                "exp06_asymmetric_basin_bottleneck_tight_action",
            ),
        ),
        _ExperimentSuiteSource(
            "exp06_asymmetric_basin_bottleneck_combined",
            "Combined",
            _suite_dir(
                "exp06_bottleneck",
                "exp06_asymmetric_basin_bottleneck_combined",
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for source in sources:
        for policy_id in _experiment_BOTTLENECK_POLICIES:
            r2, r2_sem, n_r2 = _experiment_metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            step_to_r2 = _experiment_r2_threshold_step(source.suite_dir, policy_id, threshold=0.90)
            rows.append(
                {
                    "experiment": source.exp_id,
                    "condition": source.label,
                    "policy_id": policy_id,
                    "policy_label": _experiment_policy_label(policy_id),
                    "trajectory_r2_mean": r2,
                    "trajectory_r2_sem": r2_sem,
                    "step_to_r2_0p90": step_to_r2,
                    "n_r2": n_r2,
                }
            )

    csv_path = _DOCS_TABLE_DIR / "tbme_experiment_bottleneck_sweep.csv"
    _experiment_write_csv(
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
    figure_path = plot_tbme_bottleneck_sweep(
        _DOCS_FIGURE_DIR / "tbme_experiment_bottleneck_sweep.pdf",
        sources=sources,
        rows=rows,
        policy_ids=_experiment_BOTTLENECK_POLICIES,
        policy_label=_experiment_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_experiment_axis,
    )
    return (
        figure_path,
        csv_path,
    )


def _experiment_objective_sources() -> list[_ExperimentSuiteSource]:
    return [
        _ExperimentSuiteSource(
            "exp05_asymmetric_basin_objective_ablation",
            "Nominal asymmetric basin",
            _suite_dir(
                "exp05_ablation",
                "exp05_asymmetric_basin_objective_ablation",
            ),
        ),
        _ExperimentSuiteSource(
            "exp05_hard_asymmetric_basin_objective_ablation",
            "Hard asymmetric basin",
            _suite_dir(
                "exp05_ablation",
                "exp05_hard_asymmetric_basin_objective_ablation",
            ),
        ),
    ]


def _experiment_write_objective_definition_tables() -> tuple[Path, Path]:
    rows = [
        {
            "policy_id": str(row["policy_id"]),
            "policy_label": _experiment_policy_label(str(row["policy_id"])),
            "objective_name": str(row["objective_name"]),
            "objective_formula": str(row["objective_formula"]),
            "objective_notes": str(row["objective_notes"]),
        }
        for row in _experiment_OBJECTIVE_DEFINITIONS
    ]
    csv_path = _DOCS_TABLE_DIR / "tbme_experiment_objective_ablation_objectives.csv"
    _experiment_write_csv(
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

    tex_path = _DOCS_TABLE_DIR / "tbme_experiment_objective_ablation_objectives.tex"
    lines = [
        "% Auto-generated by experiments/tbme/generate_figures.py experiment",
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
                    _experiment_escape_tex(row["policy_label"]),
                    _experiment_escape_tex(row["objective_name"]),
                    str(row["objective_formula"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def _experiment_plot_objective_ablation() -> tuple[Path, Path]:
    sources = _experiment_objective_sources()
    threshold = 0.95
    metric_rows: list[dict[str, Any]] = []
    curves_by_source: dict[str, dict[str, list[dict[str, float]]]] = {}
    for source in sources:
        curves_by_source[source.exp_id] = _experiment_curve_rows(
            source.suite_dir,
            "trajectory_r2_over_steps.csv",
            "trajectory_r2_mean",
        )
        for policy_id in _experiment_OBJECTIVE_POLICIES:
            err, err_sem, n_err = _experiment_metric_mean_sem(
                source.suite_dir, policy_id, "value_final_mean"
            )
            r2, r2_sem, n_r2 = _experiment_metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            step_to_r2, cpu_time_to_r2, r2_at_threshold = _experiment_r2_threshold_times(
                source.suite_dir,
                policy_id,
                threshold,
            )
            metric_rows.append(
                {
                    "experiment": source.exp_id,
                    "condition": source.label,
                    "policy_id": policy_id,
                    "policy_label": _experiment_policy_label(policy_id),
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

    csv_path = _DOCS_TABLE_DIR / "tbme_experiment_objective_ablation.csv"
    _experiment_write_csv(
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
    figure_path = plot_tbme_objective_ablation(
        _DOCS_FIGURE_DIR / "tbme_experiment_objective_ablation_asymmetric_basin.pdf",
        sources=sources,
        metric_rows=metric_rows,
        curves_by_source=curves_by_source,
        policy_ids=_experiment_OBJECTIVE_POLICIES,
        policy_label=_experiment_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_experiment_axis,
        stroke_color=_experiment_C_STROKE,
        neutral_light=_experiment_C_NEUTRAL_LIGHT,
    )
    return (
        figure_path,
        csv_path,
    )


def _experiment_dose_sources() -> list[_ExperimentSuiteSource]:
    return [
        _ExperimentSuiteSource(
            "exp01_duffing",
            "None",
            _suite_dir("exp01_base", "exp01_duffing"),
            dose="none",
            family="Duffing",
        ),
        _ExperimentSuiteSource(
            "exp07_duffing_parameter_mismatch_mild",
            "Mild",
            _suite_dir(
                "exp07_mismatch_stress",
                "exp07_duffing_parameter_mismatch_mild",
            ),
            dose="mild",
            family="Duffing",
        ),
        _ExperimentSuiteSource(
            "exp04_duffing_parameter_mismatch",
            "Medium",
            _suite_dir("exp04_mismatch", "exp04_duffing_parameter_mismatch"),
            dose="medium",
            family="Duffing",
        ),
        _ExperimentSuiteSource(
            "exp07_duffing_parameter_mismatch_strong",
            "Strong",
            _suite_dir(
                "exp07_mismatch_stress",
                "exp07_duffing_parameter_mismatch_strong",
            ),
            dose="strong",
            family="Duffing",
        ),
        _ExperimentSuiteSource(
            "exp01_asymmetric_basin",
            "None",
            _suite_dir("exp01_base", "exp01_asymmetric_basin"),
            dose="none",
            family="Asymmetric basin",
        ),
        _ExperimentSuiteSource(
            "exp07_asymmetric_basin_parameter_mismatch_mild",
            "Mild",
            _suite_dir(
                "exp07_mismatch_stress",
                "exp07_asymmetric_basin_parameter_mismatch_mild",
            ),
            dose="mild",
            family="Asymmetric basin",
        ),
        _ExperimentSuiteSource(
            "exp04_asymmetric_basin_parameter_mismatch",
            "Medium",
            _suite_dir("exp04_mismatch", "exp04_asymmetric_basin_parameter_mismatch"),
            dose="medium",
            family="Asymmetric basin",
        ),
        _ExperimentSuiteSource(
            "exp07_asymmetric_basin_parameter_mismatch_strong",
            "Strong",
            _suite_dir(
                "exp07_mismatch_stress",
                "exp07_asymmetric_basin_parameter_mismatch_strong",
            ),
            dose="strong",
            family="Asymmetric basin",
        ),
    ]


def _experiment_plot_mismatch_dose_response() -> tuple[Path, Path]:
    sources = _experiment_dose_sources()
    rows: list[dict[str, Any]] = []
    for source in sources:
        for policy_id in _experiment_DOSE_POLICIES:
            err, err_sem, n_err = _experiment_metric_mean_sem(
                source.suite_dir, policy_id, "value_final_mean"
            )
            r2, r2_sem, n_r2 = _experiment_metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            rows.append(
                {
                    "family": source.family,
                    "dose": source.dose,
                    "dose_label": source.label,
                    "experiment": source.exp_id,
                    "policy_id": policy_id,
                    "policy_label": _experiment_policy_label(policy_id),
                    "parameter_error_mean": err,
                    "parameter_error_sem": err_sem,
                    "trajectory_r2_mean": r2,
                    "trajectory_r2_sem": r2_sem,
                    "n_error": n_err,
                    "n_r2": n_r2,
                }
            )

    csv_path = _DOCS_TABLE_DIR / "tbme_experiment_mismatch_dose_response.csv"
    _experiment_write_csv(
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
    figure_path = plot_tbme_mismatch_dose_response(
        _DOCS_FIGURE_DIR / "tbme_experiment_mismatch_dose_response.pdf",
        rows=rows,
        policy_ids=_experiment_DOSE_POLICIES,
        policy_label=_experiment_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_experiment_axis,
    )
    return (
        figure_path,
        csv_path,
    )


def _experiment_collect_records(
    suite_dir: Path,
    policy_ids: Sequence[str],
    *,
    max_seeds: int | None = None,
    completed_only: bool = False,
) -> list[_ExperimentRunRecord]:
    records: list[_ExperimentRunRecord] = []
    for policy_id in sorted(policy_ids, key=_policy_sort_key):
        policy_dir = suite_dir / "track" / policy_id
        if not policy_dir.exists():
            continue
        seed_dirs: list[tuple[int, Path]] = []
        for seed_dir in policy_dir.glob("seed_*"):
            suffix = seed_dir.name.removeprefix("seed_")
            if suffix.isdigit():
                seed_dirs.append((int(suffix), seed_dir))
        selected_seed_dirs = sorted(seed_dirs)
        if max_seeds is not None:
            selected_seed_dirs = selected_seed_dirs[: int(max_seeds)]
        for seed, seed_dir in selected_seed_dirs:
            for metadata_path in find_nested_metadata_paths(seed_dir):
                metadata = load_json(metadata_path)
                if completed_only and metadata.get("status") != "completed":
                    continue
                records.append(
                    _ExperimentRunRecord(
                        policy_id=policy_id,
                        seed=seed,
                        run_dir=metadata_path.parent,
                        metadata=metadata,
                    )
                )
    return records


def _experiment_pad_params(
    estimate: Sequence[float], full_params: Sequence[float], min_dim: int
) -> np.ndarray:
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    full = np.asarray(full_params, dtype=np.float64).reshape(-1).copy()
    n = min(max(int(min_dim), est.size), full.size)
    full[:n] = est[:n]
    return full


def _experiment_step_batch(
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


def _experiment_evaluate_regulation_cost(
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
            state = _experiment_step_batch(
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


def _experiment_control_tasks() -> tuple[np.ndarray, np.ndarray]:
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


def _experiment_compute_downstream_rows() -> list[dict[str, Any]]:
    suite_dir = _suite_dir(
        "exp05_ablation",
        "exp05_asymmetric_basin_objective_ablation",
    )
    records = _experiment_collect_records(
        suite_dir,
        _experiment_OBJECTIVE_POLICIES,
        completed_only=True,
    )
    starts, targets = _experiment_control_tasks()
    rows: list[dict[str, Any]] = []
    oracle_costs: list[float] = []
    for record in records:
        metadata = record.metadata
        true_params = np.asarray(metadata.get("true_params_full", []), dtype=np.float64)
        if true_params.size == 0:
            continue
        learned_params = _experiment_pad_params(
            metadata.get("embedding_estimate", true_params),
            metadata.get("estimator_true_params_full", true_params),
            int(metadata.get("min_embedding_dim", true_params.size)),
        )
        base_seed = 17_000 + int(record.seed) * 101 + len(rows)
        learned_cost = _experiment_evaluate_regulation_cost(
            metadata=metadata,
            model_params=learned_params,
            true_params=true_params,
            starts=starts,
            targets=targets,
            seed=base_seed,
        )
        oracle_cost = _experiment_evaluate_regulation_cost(
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
                "policy_label": _experiment_policy_label(record.policy_id),
                "seed": record.seed,
                "parameter_error_final": _safe_float(
                    metadata.get("embedding_error_final")
                ),
                "trajectory_r2_final": _safe_float(metadata.get("trajectory_r2_final")),
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


def _experiment_plot_downstream_control() -> tuple[Path, Path]:
    rows = _experiment_compute_downstream_rows()
    policy_ids = [*_experiment_OBJECTIVE_POLICIES, "oracle_true_model"]

    csv_path = _DOCS_TABLE_DIR / "tbme_experiment_downstream_control_utility.csv"
    _experiment_write_csv(
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
    figure_path = plot_tbme_downstream_control(
        _DOCS_FIGURE_DIR / "tbme_experiment_downstream_control_utility.pdf",
        rows=rows,
        policy_ids=policy_ids,
        sem=_sem,
        policy_label=_experiment_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_experiment_axis,
        stroke_color=_experiment_C_STROKE,
        neutral_light=_experiment_C_NEUTRAL_LIGHT,
        neutral_fill=_experiment_C_NEUTRAL_FILL,
    )
    return (
        figure_path,
        csv_path,
    )


def _experiment_write_manifest(paths: Sequence[Path]) -> Path:
    manifest = _DOCS_TABLE_DIR / "tbme_experiment_figures_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "TBME experiment figures:\n"
        + "\n".join(str(path.relative_to(_REPO_ROOT)) for path in paths)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _experiment_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TBME experiment-level manuscript figures.",
        allow_abbrev=False,
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
        default=",".join(EXPERIMENT_PLOTS),
        help="Comma-separated TBME experiment plot ids.",
    )
    return parser


def _experiment_short_policy_label(policy_id: str) -> str:
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
    return labels.get(policy_id, _policy_label(policy_id))


def _experiment_state_bounds_from_metadata(metadata: dict[str, Any]) -> tuple[float, float]:
    low = np.asarray(metadata.get("state_low", [-5.0, -5.0]), dtype=np.float64)
    high = np.asarray(metadata.get("state_high", [5.0, 5.0]), dtype=np.float64)
    return float(np.min(low)), float(np.max(high))


def _experiment_trace_path(
    record: _ExperimentRunRecord, metadata_key: str, fallback_name: str
) -> Path:
    return resolve_artifact_path(
        record.run_dir,
        record.metadata,
        key=metadata_key,
        fallback_name=fallback_name,
    )


def _experiment_load_xy_trace(record: _ExperimentRunRecord) -> np.ndarray:
    path = _experiment_trace_path(record, "state_action_trace_path", "state_action_trace.csv")
    return _read_xy_trace(path)


def _experiment_logdet_information(
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


def _experiment_observation_model_key(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(metadata.get("seed", 0)),
        str(metadata.get("env_preset_id", "")),
        int(metadata.get("observation_dim", 20)),
        int(metadata.get("latent_dim", 2)),
        float(metadata.get("dt", 0.01)),
        float(metadata.get("mean_firing_rate_target", 10.0)),
        float(metadata.get("max_firing_rate_target", 100.0)),
    )


def _experiment_information_reference_records(
    records: Sequence[_ExperimentRunRecord],
) -> list[_ExperimentRunRecord]:
    out: list[_ExperimentRunRecord] = []
    seen: set[tuple[Any, ...]] = set()
    for record in sorted(
        records, key=lambda item: (item.seed, _policy_sort_key(item.policy_id))
    ):
        key = _experiment_observation_model_key(record.metadata)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _experiment_make_information_grid(
    metadata: dict[str, Any],
    *,
    n_grid: int = 121,
    axis_min: float | None = None,
    axis_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis_min is None or axis_max is None:
        state_min, state_max = _experiment_state_bounds_from_metadata(metadata)
    else:
        state_min, state_max = float(axis_min), float(axis_max)
    axis = np.linspace(state_min, state_max, n_grid, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    latent = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    logdet = _experiment_logdet_information(latent, metadata=metadata).reshape(n_grid, n_grid)
    return axis, axis, logdet


def _experiment_make_mean_information_grid(
    records: Sequence[_ExperimentRunRecord],
    *,
    n_grid: int = 121,
    axis_min: float | None = None,
    axis_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("At least one record is required to compute an information grid")
    x_axis, y_axis, first_grid = _experiment_make_information_grid(
        records[0].metadata,
        n_grid=n_grid,
        axis_min=axis_min,
        axis_max=axis_max,
    )
    maps = [first_grid.astype(np.float64)]
    for record in records[1:]:
        _x, _y, grid = _experiment_make_information_grid(
            record.metadata,
            n_grid=n_grid,
            axis_min=axis_min,
            axis_max=axis_max,
        )
        maps.append(grid.astype(np.float64))
    return x_axis, y_axis, np.nanmean(np.stack(maps, axis=0), axis=0)


def _experiment_true_vectorfield_dynamics(metadata: dict[str, Any]) -> ResidualDynamicsCallable:
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


def _experiment_learned_vectorfield_dynamics(
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


def _experiment_plot_true_dynamics_all() -> Path:
    panel_specs = [
        ("tbme_duffing", "Duffing"),
        ("tbme_damped_pendulum", "Damped pendulum"),
        ("tbme_asymmetric_basin", "Asymmetric basin"),
        ("tbme_asymmetric_basin_hard", "Asymmetric basin (hard)"),
        ("tbme_multi_stable", "Multi-stable"),
    ]
    grid_lim = 6.0
    fields = []
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
    return plot_tbme_true_dynamics_all(
        _DOCS_FIGURE_DIR / "tbme_experiment_true_dynamics_all.pdf",
        fields=fields,
        grid_lim=grid_lim,
        apply_style=_apply_style,
        stroke_color=_experiment_C_STROKE,
    )


def _experiment_plot_asymmetric_basin_mechanism(max_seeds: int) -> Path:
    suite_dir = _suite_dir("exp02_hard", "exp02_hard_asymmetric_basin")
    policy_ids = ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "flex", "prbs"]
    records = _experiment_collect_records(suite_dir, policy_ids, max_seeds=max_seeds)
    if not records:
        raise RuntimeError(f"No records found under {suite_dir}")
    ref_metadata = records[0].metadata
    by_policy: dict[str, list[np.ndarray]] = {policy_id: [] for policy_id in policy_ids}
    record_traces: list[tuple[_ExperimentRunRecord, np.ndarray]] = []
    env_preset = get_environment_preset_from_metadata(ref_metadata)
    panel_abs = max(float(env_preset.resolved_plot_limit()), 6.0)
    boundary_radius = _safe_float(ref_metadata.get("boundary_radius"))
    if boundary_radius is None:
        boundary_radius = _safe_float(getattr(env_preset, "boundary_radius", None))
    if boundary_radius is not None:
        panel_abs = max(panel_abs, boundary_radius)
    for record in records:
        traj = _experiment_load_xy_trace(record)
        if traj.size == 0:
            continue
        by_policy.setdefault(record.policy_id, []).append(traj)
        record_traces.append((record, traj))
        finite = traj[np.isfinite(traj).all(axis=1)]
        if finite.size:
            panel_abs = max(panel_abs, 1.04 * float(np.max(np.abs(finite[:, :2]))))
    panel_min, panel_max = -panel_abs, panel_abs
    information_refs = _experiment_information_reference_records(records)
    x_axis, y_axis, logdet_grid = _experiment_make_mean_information_grid(
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
        values = _experiment_logdet_information(traj[:, :2], metadata=record.metadata)
        finite = values[np.isfinite(values)]
        if finite.size:
            model_key = _experiment_observation_model_key(record.metadata)
            if model_key not in threshold_by_model:
                _x, _y, record_grid = _experiment_make_information_grid(
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
        value = _safe_float(row.get("trajectory_r2_final_mean"))
        if value is not None:
            final_r2[policy_id].append(value)

    return plot_tbme_asymmetric_basin_mechanism(
        _DOCS_FIGURE_DIR / "tbme_experiment_asymmetric_basin_mechanism.pdf",
        x_axis=x_axis,
        y_axis=y_axis,
        logdet_grid=logdet_grid,
        info_threshold=info_threshold,
        info_vmin=info_vmin,
        info_vmax=info_vmax,
        panel_min=panel_min,
        panel_max=panel_max,
        true_dynamics=_experiment_true_vectorfield_dynamics(ref_metadata),
        traces_by_policy=by_policy,
        policy_ids=policy_ids,
        informative_fraction=informative_fraction,
        coverage_fraction=coverage_fraction,
        final_r2=final_r2,
        sem=_sem,
        short_policy_label=_experiment_short_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        stroke_color=_experiment_C_STROKE,
    )


def _experiment_common_seed(
    records: Sequence[_ExperimentRunRecord],
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


def _experiment_embedding_at_step(record: _ExperimentRunRecord, step: int) -> np.ndarray:
    path = _experiment_trace_path(
        record, "embedding_estimate_trace_path", "embedding_estimate_trace.csv"
    )
    selected: dict[str, str] | None = None
    selected_step = -math.inf
    fallback: dict[str, str] | None = None
    fallback_step = math.inf
    for row in read_trace_csv(path):
        row_step = _safe_float(row.get("step"))
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
        _safe_float(row.get("embedding_dim")) or record.metadata.get("embedding_dim", 0)
    )
    values = []
    for idx in range(embedding_dim):
        value = _safe_float(row.get(f"e{idx}"))
        if value is None:
            raise RuntimeError(f"Missing e{idx} in {path}")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _experiment_xy_trace_until(record: _ExperimentRunRecord, step: int) -> np.ndarray:
    path = _experiment_trace_path(record, "state_action_trace_path", "state_action_trace.csv")
    points: list[tuple[float, float]] = []
    for row in read_trace_csv(path):
        row_step = _safe_float(row.get("step"))
        x_val = _safe_float(row.get("true_x"))
        v_val = _safe_float(row.get("true_v"))
        if row_step is None or x_val is None or v_val is None:
            continue
        if row_step <= step:
            points.append((x_val, v_val))
    return np.asarray(points, dtype=np.float32)


def _experiment_plot_learned_vectorfield_snapshots(max_seeds: int) -> Path:
    suite_dir = _suite_dir("exp02_hard", "exp02_hard_asymmetric_basin")
    policy_ids = ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "flex", "prbs"]
    checkpoints = [250, 500, 1000]
    row_ids = ["true", *policy_ids]
    records = _experiment_collect_records(suite_dir, policy_ids, max_seeds=max_seeds)
    if not records:
        raise RuntimeError(f"No records found under {suite_dir}")
    seed = _experiment_common_seed(records, policy_ids)
    record_by_policy: dict[str, _ExperimentRunRecord] = {}
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
    boundary_radius = _safe_float(ref_metadata.get("boundary_radius"))
    if boundary_radius is None:
        boundary_radius = _safe_float(getattr(env_preset, "boundary_radius", None))
    if boundary_radius is not None:
        plot_abs = max(plot_abs, boundary_radius)
    for record in record_by_policy.values():
        traj = _experiment_xy_trace_until(record, max(checkpoints))
        finite = traj[np.isfinite(traj).all(axis=1)]
        if finite.size:
            plot_abs = max(plot_abs, 1.04 * float(np.max(np.abs(finite[:, :2]))))

    true_dynamics = _experiment_true_vectorfield_dynamics(ref_metadata)
    dynamics_by_cell: dict[tuple[str, int], Any] = {}
    traces_by_cell: dict[tuple[str, int], np.ndarray] = {}
    for row_id in row_ids:
        record = None if row_id == "true" else record_by_policy[row_id]
        for checkpoint in checkpoints:
            if row_id == "true":
                dynamics = true_dynamics
            else:
                assert record is not None
                theta = _experiment_embedding_at_step(record, checkpoint)
                dynamics = _experiment_learned_vectorfield_dynamics(record.metadata, theta)
            dynamics_by_cell[(row_id, checkpoint)] = dynamics
            traces_by_cell[(row_id, checkpoint)] = (
                np.empty((0, 2), dtype=np.float32)
                if record is None
                else _experiment_xy_trace_until(record, checkpoint)
            )
    return plot_tbme_learned_vectorfield_snapshots(
        _DOCS_FIGURE_DIR / "tbme_experiment_asymmetric_basin_learned_vectorfields.pdf",
        seed=seed,
        row_ids=row_ids,
        checkpoints=checkpoints,
        dynamics_by_cell=dynamics_by_cell,
        traces_by_cell=traces_by_cell,
        plot_abs=plot_abs,
        short_policy_label=_experiment_short_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        stroke_color=_experiment_C_STROKE,
        neutral_fill=_experiment_C_NEUTRAL_FILL,
        grid_color=_experiment_C_GRID,
    )


def _experiment_threshold_value(row: dict[str, str], threshold: float) -> float | None:
    suffix = _r2_threshold_suffix(threshold)
    return _safe_float(row.get(f"step_to_r2_{suffix}"))


def _experiment_plot_sample_efficiency_thresholds() -> Path:
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
    for group_name, suite_id, suite_label in selected:
        suite_dir = _suite_dir(group_name, suite_id)
        threshold = thresholds.get(suite_id, default_threshold)
        rows = read_trace_csv(suite_dir / "summary" / "trajectory_r2_thresholds.csv")
        row_by_policy = {str(row.get("policy_id", "")): row for row in rows}
        for policy_id in policy_ids:
            step = _experiment_threshold_value(row_by_policy.get(policy_id, {}), threshold)
            values.append((suite_label, policy_id, step, threshold))

    suite_labels = [item[2] for item in selected]
    return plot_tbme_sample_efficiency_thresholds(
        _DOCS_FIGURE_DIR / "tbme_experiment_sample_efficiency_thresholds.pdf",
        values=values,
        suite_labels=suite_labels,
        policy_ids=policy_ids,
        short_policy_label=_experiment_short_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        stroke_color=_experiment_C_STROKE,
    )


def _experiment_aggregate_metric_rows(group_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in GROUPS[group_name]:
        summary_path = ref.session_root / ref.suite_id / "summary" / "metrics.csv"
        for row in read_trace_csv(summary_path):
            if row.get("status") != "completed":
                continue
            value = _safe_float(row.get("value_final_mean"))
            r2 = _safe_float(row.get("trajectory_r2_final_mean"))
            runtime = _safe_float(row.get("runtime_sec_mean"))
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


def _experiment_mean_rows_by_policy(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _experiment_plot_compute_accuracy_pareto() -> Path:
    schedule_rows = _experiment_mean_rows_by_policy(
        _experiment_aggregate_metric_rows("exp03_schedule")
    )
    group_rows = []
    for group_name in ("exp01_base", "exp02_hard", "exp04_mismatch"):
        group_rows.extend(
            _experiment_mean_rows_by_policy(_experiment_aggregate_metric_rows(group_name))
        )
    focus_policies = [
        "active_planning_u20_r20_h40",
        "active_myopic",
        "ensemble",
        "prbs",
        "random",
        "rhc",
    ]
    return plot_tbme_compute_accuracy_pareto(
        _DOCS_FIGURE_DIR / "tbme_experiment_compute_accuracy_pareto.pdf",
        schedule_rows=schedule_rows,
        group_rows=group_rows,
        focus_policies=focus_policies,
        short_policy_label=_experiment_short_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        stroke_color=_experiment_C_STROKE,
    )


def _experiment_aggregate_parameter_traces(
    suite_dir: Path,
    policy_ids: Sequence[str],
    *,
    max_seeds: int,
    stride: int,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], np.ndarray]:
    records = _experiment_collect_records(suite_dir, policy_ids, max_seeds=max_seeds)
    if not records:
        raise RuntimeError(f"No records found under {suite_dir}")
    true_params = np.asarray(records[0].metadata.get("embedding_true", []), dtype=np.float64)
    traces: dict[str, dict[int, list[np.ndarray]]] = {policy_id: {} for policy_id in policy_ids}
    for record in records:
        path = _experiment_trace_path(
            record, "embedding_estimate_trace_path", "embedding_estimate_trace.csv"
        )
        for row in read_trace_csv(path):
            step_raw = _safe_float(row.get("step"))
            if step_raw is None:
                continue
            step = int(step_raw)
            if step % stride != 0 and step != int(record.metadata.get("total_steps", 0)):
                continue
            params: list[float] = []
            for idx in range(true_params.size):
                value = _safe_float(row.get(f"e{idx}"))
                if value is None:
                    break
                params.append(value)
            if len(params) != true_params.size:
                continue
            traces.setdefault(record.policy_id, {}).setdefault(step, []).append(
                np.asarray(params, dtype=np.float64)
            )
    return traces, true_params


def _experiment_plot_per_parameter_recovery(max_seeds: int) -> Path:
    suite_dir = _suite_dir("exp01_base", "exp01_asymmetric_basin")
    policy_ids = ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "flex", "prbs"]
    traces, true_params = _experiment_aggregate_parameter_traces(
        suite_dir,
        policy_ids,
        max_seeds=max_seeds,
        stride=20,
    )
    return plot_tbme_per_parameter_recovery(
        _DOCS_FIGURE_DIR / "tbme_experiment_asymmetric_basin_parameter_recovery.pdf",
        traces=traces,
        true_params=true_params,
        policy_ids=policy_ids,
        sem=_sem,
        short_policy_label=_experiment_short_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        stroke_color=_experiment_C_STROKE,
    )


def _experiment_plot_information_learning_coupling(max_seeds: int) -> Path:
    suite_dir = _suite_dir("exp01_base", "exp01_asymmetric_basin")
    policy_ids = [
        "active_planning_u20_r20_h40",
        "active_myopic",
        "ensemble",
        "prbs",
        "random",
        "rhc",
    ]
    records = _experiment_collect_records(suite_dir, policy_ids, max_seeds=max_seeds)
    points: dict[str, list[tuple[float, float, float]]] = {
        policy_id: [] for policy_id in policy_ids
    }
    for record in records:
        info_rows = read_trace_csv(
            _experiment_trace_path(record, "information_trace_path", "information_trace.csv")
        )
        r2_rows = read_trace_csv(
            _experiment_trace_path(record, "trajectory_r2_trace_path", "trajectory_r2_trace.csv")
        )
        info_vals = [
            value
            for value in (_safe_float(row.get("I_theta_t")) for row in info_rows)
            if value is not None and value >= 0.0
        ]
        r2_vals = [
            value
            for value in (_safe_float(row.get("trajectory_r2")) for row in r2_rows)
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

    return plot_tbme_information_learning_coupling(
        _DOCS_FIGURE_DIR / "tbme_experiment_information_learning_coupling.pdf",
        points=points,
        policy_ids=policy_ids,
        short_policy_label=_experiment_short_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        stroke_color=_experiment_C_STROKE,
    )


def _experiment_write_latex_snippet(paths: Sequence[Path]) -> Path:
    _DOCS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    snippet = _DOCS_TABLE_DIR / "tbme_experiment_figures.tex"
    captions = {
        "tbme_experiment_true_dynamics_all.pdf": (
            "True phase portraits for the distinct TBME synthetic dynamics. Background color shows "
            "log speed, and streamlines show the direction of the latent drift over $[-6, 6]^2$; "
            "observation-only and mismatch-dose variants reuse one of these true vector fields."
        ),
        "tbme_experiment_asymmetric_basin_mechanism.pdf": (
            "Hard asymmetric-basin mechanism diagnostics under asymmetric observation loading. "
            "Panel A overlays trajectories and the true dynamics vector field on the seed-averaged "
            "spatial observation information geometry; the remaining panels connect this geometry "
            "to state-space coverage and endpoint prediction."
        ),
        "tbme_experiment_asymmetric_basin_learned_vectorfields.pdf": (
            "Hard asymmetric-basin vector fields for one shared seed. The first row shows the true "
            "vector field; remaining rows correspond to methods, and columns show checkpoints at "
            "250, 500, and 1000 interaction steps. Learned-field panels overlay the trajectory "
            "prefix on the vector field induced by the current parameter estimate."
        ),
        "tbme_experiment_sample_efficiency_thresholds.pdf": (
            "Sample efficiency measured by the first environment step at which each method reaches "
            "the indicated trajectory-$R^2$ threshold."
        ),
        "tbme_experiment_compute_accuracy_pareto.pdf": (
            "Prediction-cost tradeoffs across planning schedules and policy families."
        ),
        "tbme_experiment_asymmetric_basin_parameter_recovery.pdf": (
            "Per-parameter recovery in the asymmetric-basin benchmark, including the FLEX baseline."
        ),
        "tbme_experiment_information_learning_coupling.pdf": (
            "Relationship between accumulated parameter information and predictive-$R^2$ improvement."
        ),
    }
    labels = {
        "tbme_experiment_true_dynamics_all.pdf": "fig:tbme_experiment_true_dynamics_all",
        "tbme_experiment_asymmetric_basin_mechanism.pdf": "fig:tbme_experiment_asymmetric_basin_mechanism",
        "tbme_experiment_asymmetric_basin_learned_vectorfields.pdf": "fig:tbme_experiment_learned_vectorfields",
        "tbme_experiment_sample_efficiency_thresholds.pdf": "fig:tbme_experiment_sample_efficiency",
        "tbme_experiment_compute_accuracy_pareto.pdf": "fig:tbme_experiment_compute_pareto",
        "tbme_experiment_asymmetric_basin_parameter_recovery.pdf": "fig:tbme_experiment_parameter_recovery",
        "tbme_experiment_information_learning_coupling.pdf": "fig:tbme_experiment_information_learning",
    }
    lines: list[str] = ["% Auto-generated by experiments/tbme/generate_figures.py experiment"]
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


def experiment_main(argv: list[str] | None = None) -> int:
    args = _experiment_build_parser().parse_args(argv)
    max_seeds = int(args.max_seeds)
    plot_ids = _experiment_parse_plots(str(args.plots))
    missing_suite_dirs = sorted(
        (
            suite_dir
            for suite_dir in _experiment_required_suite_dirs(plot_ids)
            if not suite_dir.exists()
        ),
        key=str,
    )
    if missing_suite_dirs:
        missing_text = ", ".join(str(path) for path in missing_suite_dirs)
        raise FileNotFoundError(f"Missing TBME experiment suite(s): {missing_text}")

    figure_paths: list[Path] = []
    csv_paths: list[Path] = []
    figure_only_paths: list[Path] = []
    if any(plot_id in _experiment_OBJECTIVE_DEFINITION_PLOTS for plot_id in plot_ids):
        objective_definition_csv, objective_definition_tex = (
            _experiment_write_objective_definition_tables()
        )
        csv_paths.extend([objective_definition_csv, objective_definition_tex])

    experiment_plotters = {
        "bottleneck_sweep": _experiment_plot_bottleneck_sweep,
        "objective_ablation": _experiment_plot_objective_ablation,
        "mismatch_dose_response": _experiment_plot_mismatch_dose_response,
        "downstream_control": _experiment_plot_downstream_control,
    }
    figure_only_plotters = {
        "true_dynamics_all": lambda: _experiment_plot_true_dynamics_all(),
        "asymmetric_basin_mechanism": lambda: _experiment_plot_asymmetric_basin_mechanism(
            max_seeds=max_seeds
        ),
        "learned_vectorfield_snapshots": lambda: _experiment_plot_learned_vectorfield_snapshots(
            max_seeds=max_seeds
        ),
        "sample_efficiency_thresholds": lambda: _experiment_plot_sample_efficiency_thresholds(),
        "compute_accuracy_pareto": lambda: _experiment_plot_compute_accuracy_pareto(),
        "per_parameter_recovery": lambda: _experiment_plot_per_parameter_recovery(
            max_seeds=max_seeds
        ),
        "information_learning_coupling": lambda: _experiment_plot_information_learning_coupling(
            max_seeds=max_seeds
        ),
    }
    for plot_id in plot_ids:
        if plot_id in experiment_plotters:
            figure_path, csv_path = experiment_plotters[plot_id]()
            figure_paths.append(figure_path)
            csv_paths.append(csv_path)
        else:
            figure_only_paths.append(figure_only_plotters[plot_id]())

    written = [*figure_paths, *csv_paths, *figure_only_paths]
    if figure_only_paths:
        written.append(_experiment_write_latex_snippet(figure_only_paths))
    if written:
        written.append(_experiment_write_manifest(written))
    for path in written:
        print(path)
    return 0
