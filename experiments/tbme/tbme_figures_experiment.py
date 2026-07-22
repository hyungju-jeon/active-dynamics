#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from actdyn.environment.vectorfield import ResidualDynamicsCallable
from actdyn.utils.experiment_runtime import read_trace_csv, safe_float as _safe_float
from actdyn.utils.figure_io import load_plotting, sample_sem, save_figure
from actdyn.utils.plotting import compute_vector_field

from ..experiment_definitions import get_environment_preset
from ..experiment_io import (
    find_nested_metadata_paths,
    load_json,
    reconstruct_loglinear_rate_model,
)
from .figures import ablation as _ablation
from .figures import bottleneck as _bottleneck
from .figures import mismatch as _mismatch
from .figures import information as _information
from .figures import records as _records
from .figures import artifacts as _fig_artifacts
from .figures import data as _fig_data
from .figures import theme as _fig_theme
from .figures.groups import SuiteSource
from .figures.records import RunRecord
from .tbme_figures import (
    GROUPS,
    _apply_style,
    _policy_color,
    _policy_label,
    _policy_sort_key,
    _style_manuscript_axis,
    _suite_dir,
    _unique_paths,
    _write_csv,
)
from .tbme_io import (
    read_embedding_trace,
    read_xy_trace as _read_xy_trace,
    trace_path as _tbme_trace_path,
)

# Experiment manuscript output
_experiment_C_STROKE = _fig_theme.STROKE_COLOR
_experiment_C_NEUTRAL_LIGHT = _fig_theme.NEUTRAL_LIGHT
_experiment_C_NEUTRAL_FILL = _fig_theme.NEUTRAL_FILL
_experiment_C_GRID = _fig_theme.GRID_COLOR

_experiment_BOTTLENECK_POLICIES = _bottleneck.BOTTLENECK_POLICIES
_experiment_OBJECTIVE_POLICIES = _ablation.OBJECTIVE_POLICIES
_experiment_OBJECTIVE_DEFINITIONS = _ablation.OBJECTIVE_DEFINITIONS
_experiment_DOSE_POLICIES = _mismatch.DOSE_POLICIES


_ExperimentSuiteSource = SuiteSource


_ExperimentRunRecord = RunRecord


_experiment_PLOTS = (
    "bottleneck_sweep",
    "objective_ablation",
    "mismatch_dose_response",
    "true_dynamics_all",
    "per_parameter_recovery",
)
EXPERIMENT_PLOTS = _experiment_PLOTS

_experiment_OBJECTIVE_DEFINITION_PLOTS = {"objective_ablation"}
_experiment_REQUIRED_SUITES_BY_PLOT = {
    "bottleneck_sweep": _bottleneck.REQUIRED_SUITES,
    "objective_ablation": _ablation.REQUIRED_SUITES,
    "mismatch_dose_response": _mismatch.REQUIRED_SUITES,
}


_experiment_artifact_paths = _fig_artifacts.artifact_paths
_experiment_write_csv_artifacts = _fig_artifacts.write_csv_artifacts
_experiment_write_text_artifacts = _fig_artifacts.write_text_artifacts
_experiment_copy_artifact = _fig_artifacts.copy_artifact


_style_experiment_axis = _fig_theme.style_experiment_axis


plot_bottleneck_sweep = _bottleneck.plot_bottleneck_sweep


plot_objective_ablation = _ablation.plot_objective_ablation


plot_mismatch_dose_response = _mismatch.plot_mismatch_dose_response


def plot_neutral_vector_field(
    ax: Any,
    dynamics: Any,
    *,
    grid_lim: float,
    n_grid: int,
    arrowsize: float,
    stroke_color: str,
) -> None:
    """Draw a neutral TBME vector field background on an existing axis."""
    from matplotlib.colors import to_rgba

    x_grid, y_grid, u_grid, v_grid = compute_vector_field(
        dynamics,
        x_range=grid_lim,
        n_grid=n_grid,
        is_residual=True,
        device="cpu",
    )
    ax.streamplot(
        x_grid.cpu().numpy(),
        y_grid.cpu().numpy(),
        u_grid.cpu().numpy(),
        v_grid.cpu().numpy(),
        color=to_rgba(stroke_color, 0.42),
        linewidth=0.34,
        density=1.55,
        arrowsize=arrowsize,
        zorder=1,
    )


def plot_per_parameter_recovery(
    output_path: Path,
    *,
    traces: Mapping[str, Mapping[int, Sequence[np.ndarray]]],
    true_params: np.ndarray,
    policy_ids: Sequence[str],
    sem: Callable[[Sequence[float]], float],
    short_policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
    stroke_color: str,
) -> Path:
    """Plot per-parameter recovery traces for gated-Duffing dynamics."""
    plt_module = load_plotting(output_path, apply_style=apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    names = ["a_L", "b_L", "a_R", "b_R"]
    fig, axes = plt_module.subplots(2, 2, figsize=(7.35, 4.75), sharex=True)
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
                sems.append(sem(vals.tolist()))
            means_arr = np.asarray(means, dtype=np.float64)
            sems_arr = np.asarray(sems, dtype=np.float64)
            color = policy_color(policy_id)
            ax.plot(
                steps,
                means_arr,
                color=color,
                linewidth=1.0,
                label=short_policy_label(policy_id),
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
            float(true_params[param_idx]),
            color=stroke_color,
            linewidth=0.8,
            linestyle="--",
        )
        ax.set_title(f"{chr(65 + param_idx)}. {names[param_idx]}")
        ax.set_ylabel("Estimate")
        style_axis(ax)
    axes[1, 0].set_xlabel("Environment step")
    axes[1, 1].set_xlabel("Environment step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=6.4,
        bbox_to_anchor=(0.5, 1.015),
    )
    fig.suptitle("Gated-Duffing per-parameter recovery", y=1.06)
    fig.tight_layout()
    return save_figure(fig, output_path, plt_module=plt_module)


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


_experiment_policy_label = _fig_theme.short_policy_label


_experiment_metrics_by_policy = _fig_data.metrics_by_policy
_experiment_metric_values = _fig_data.metric_values
_experiment_metric_mean_sem = _fig_data.metric_mean_sem
_experiment_curve_rows = _fig_data.curve_rows
_experiment_r2_threshold_step = _fig_data.r2_threshold_step
_experiment_r2_threshold_times = _fig_data.r2_threshold_times


_experiment_plot_bottleneck_sweep = _bottleneck.generate


_experiment_objective_sources = _ablation.objective_sources
_experiment_write_objective_definition_tables = _ablation.write_definition_tables
_experiment_plot_objective_ablation = _ablation.generate


_experiment_dose_sources = _mismatch.dose_sources
_experiment_plot_mismatch_dose_response = _mismatch.generate


_experiment_collect_records = _records.collect_records


def _experiment_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TBME experiment-level figures into suite result folders.",
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
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(GROUPS),
        help="Comma-separated TBME groups whose suite folders receive global experiment figures.",
    )
    return parser


def _experiment_suite_dirs_from_groups(raw: str) -> list[Path]:
    group_ids = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(group_ids) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")
    if not group_ids:
        raise ValueError("At least one TBME group is required")
    return _unique_paths(
        ref.session_root / "tracks" / ref.suite_id
        for group_id in group_ids
        for ref in GROUPS[group_id]
    )


_experiment_short_policy_label = _fig_theme.extended_policy_label


_experiment_state_bounds_from_metadata = _records.state_bounds_from_metadata
_experiment_trace_path = _records.record_trace_path
_experiment_load_xy_trace = _records.load_xy_trace
_experiment_logdet_information = _information.logdet_information
_experiment_observation_model_key = _information.observation_model_key
_experiment_information_reference_records = _information.information_reference_records
_experiment_make_information_grid = _information.make_information_grid
_experiment_make_mean_information_grid = _information.make_mean_information_grid


def _experiment_plot_true_dynamics_all(suite_dirs: Sequence[Path]) -> list[Path]:
    """Plot the true vector fields for the TBME synthetic systems."""
    figure_paths = _experiment_artifact_paths(
        suite_dirs,
        subdir="figures",
        filename="tbme_experiment_true_dynamics_all.pdf",
    )
    output_path = figure_paths[0]
    plt_module = load_plotting(output_path, apply_style=_apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    from actdyn.utils.plotting import decorate_phase_space_axis
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    panel_specs = [
        ("tbme_duffing", "Duffing"),
        ("tbme_damped_pendulum", "Damped pendulum"),
        ("tbme_gated_duffing", "Gated Duffing"),
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

    finite_speed = np.concatenate([arr[np.isfinite(arr)].reshape(-1) for *_rest, arr in fields])
    vmax = float(np.percentile(finite_speed, 98.0)) if finite_speed.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    panel_title_size = 12.0
    label_size = 10.0

    fig = plt_module.figure(figsize=(7.25, 2.35))
    gs = fig.add_gridspec(
        1,
        4,
        wspace=0.05,
        width_ratios=[1, 1, 1, 0.08],
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]
    cax = fig.add_subplot(gs[0, 3])
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
            color=_experiment_C_STROKE,
            linewidth=0.38,
            density=1.25,
            arrowsize=0.62,
            zorder=2,
        )
        decorate_phase_space_axis(
            ax,
            xlim=(-grid_lim, grid_lim),
            ylim=(-grid_lim, grid_lim),
            title=title,
            xlabel=r"$z_1$" if panel_idx == 0 else "",
            ylabel=r"$z_2$" if panel_idx == 0 else "",
            grid_alpha=0.20,
        )
        ax.title.set_fontsize(panel_title_size)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.set_xticks([-6, 0, 6])
        ax.set_yticks([-6, 0, 6])
        ax.tick_params(labelbottom=False, labelleft=False)

    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\log(1 + \|f(z)\|)$")
    cbar.ax.yaxis.label.set_fontsize(label_size)
    cbar.ax.tick_params(labelright=False, labelleft=False, length=0)
    cbar.outline.set_linewidth(0.45)
    fig.canvas.draw()
    panel_pos = axes[-1].get_position()
    cbar_pos = cax.get_position()
    cax.set_position([cbar_pos.x0, panel_pos.y0, cbar_pos.width, panel_pos.height])
    figure_path = save_figure(
        fig,
        output_path,
        plt_module=plt_module,
    )
    return _experiment_copy_artifact(figure_path, figure_paths)


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
    metadata = records[0].metadata
    true_params = np.asarray(
        metadata.get("embedding_true")
        or metadata.get("true_embedding")
        or metadata.get("true_params_full")
        or [],
        dtype=np.float64,
    )
    traces: dict[str, dict[int, list[np.ndarray]]] = {policy_id: {} for policy_id in policy_ids}
    if true_params.size == 0:
        return traces, true_params
    for record in records:
        path = _experiment_trace_path(
            record, "embedding_estimate_trace_path", "embedding_estimate_trace.csv"
        )
        steps, theta_trace = read_embedding_trace(path)
        for step, theta in zip(steps, theta_trace, strict=False):
            step = int(step)
            if step % stride != 0 and step != int(record.metadata.get("total_steps", 0)):
                continue
            if theta.shape[0] < true_params.size:
                continue
            traces.setdefault(record.policy_id, {}).setdefault(step, []).append(
                np.asarray(theta[: true_params.size], dtype=np.float64)
            )
    return traces, true_params


def _experiment_plot_per_parameter_recovery(max_seeds: int) -> list[Path]:
    suite_dir = _suite_dir("simple_system_identification", "gated_duffing")
    policy_ids = [
        "active_planning",
        "active_myopic",
        "active_state_variance",
        "flex",
        "prbs",
    ]
    traces, true_params = _experiment_aggregate_parameter_traces(
        suite_dir,
        policy_ids,
        max_seeds=max_seeds,
        stride=20,
    )
    if true_params.size < 4:
        return []
    figure_paths = _experiment_artifact_paths(
        [suite_dir],
        subdir="figures",
        filename="tbme_experiment_gated_duffing_parameter_recovery.pdf",
    )
    figure_path = plot_per_parameter_recovery(
        figure_paths[0],
        traces=traces,
        true_params=true_params,
        policy_ids=policy_ids,
        sem=sample_sem,
        short_policy_label=_experiment_short_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        stroke_color=_experiment_C_STROKE,
    )
    return _experiment_copy_artifact(figure_path, figure_paths)


_COMPOUND_POLICY_ORDER = (
    "compound_active_planning",
    "compound_active_fully_observable",
    "compound_active_e_optimality",
    "compound_active_state_information",
    "compound_active_dynamics",
    "compound_active_observation_variance",
    "compound_active_state_variance",
    "prbs",
    "random",
)
_COMPOUND_POLICY_LABELS = {
    "compound_active_planning": "PALDI",
    "compound_active_fully_observable": "Full-observed EIG",
    "compound_active_e_optimality": "E-optimality",
    "compound_active_state_information": "State information",
    "compound_active_dynamics": "Dynamics sensitivity",
    "compound_active_observation_variance": "Observation variance",
    "compound_active_state_variance": "State variance",
    "prbs": "PRBS",
    "random": "Random",
}
_COMPOUND_POLICY_COLORS = {
    "compound_active_planning": "#0072B2",
    "compound_active_fully_observable": "#D55E00",
    "compound_active_e_optimality": "#CC79A7",
    "compound_active_state_information": "#009E73",
    "compound_active_dynamics": "#E69F00",
    "compound_active_observation_variance": "#56B4E9",
    "compound_active_state_variance": "#F0E442",
    "prbs": "#666666",
    "random": "#AAAAAA",
}


def _compound_trace_records(
    result_roots: Sequence[Path],
    *,
    exp_id: str,
) -> list[_ExperimentRunRecord]:
    records: list[_ExperimentRunRecord] = []
    for root in result_roots:
        for metadata_path in sorted(Path(root).glob("**/run_metadata.json")):
            metadata = load_json(metadata_path)
            if str(metadata.get("exp_id")) != str(exp_id):
                continue
            policy_id = str(metadata.get("policy_id"))
            if policy_id not in _COMPOUND_POLICY_ORDER:
                continue
            records.append(
                _ExperimentRunRecord(
                    policy_id=policy_id,
                    seed=int(metadata.get("seed", 0)),
                    run_dir=metadata_path.parent,
                    metadata=metadata,
                )
            )
    return records


def _compound_curve(
    records: Sequence[_ExperimentRunRecord],
    *,
    filename: str,
    value_key: str,
    stride: int,
) -> dict[str, list[dict[str, float]]]:
    values: dict[str, dict[int, list[float]]] = {}
    for record in records:
        rows = read_trace_csv(record.run_dir / filename)
        for row in rows:
            step = int(float(row["step"]))
            if step % max(1, int(stride)) != 0 and step != int(
                record.metadata.get("total_steps", 0)
            ):
                continue
            value = _safe_float(row.get(value_key))
            if value is None:
                continue
            values.setdefault(record.policy_id, {}).setdefault(step, []).append(value)
    curves: dict[str, list[dict[str, float]]] = {}
    for policy_id, by_step in values.items():
        curves[policy_id] = [
            {
                "step": float(step),
                "mean": float(np.mean(step_values)),
                "sem": float(sample_sem(step_values)),
                "median": float(np.median(step_values)),
                "q25": float(np.quantile(step_values, 0.25)),
                "q75": float(np.quantile(step_values, 0.75)),
            }
            for step, step_values in sorted(by_step.items())
        ]
    return curves


def _compound_summary_rows(
    records: Sequence[_ExperimentRunRecord],
    *,
    gate_centers: tuple[float, float, float] = (-0.5, -0.32, 0.0),
    rest_cutoff: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gate_a, gate_b, gate_m = (float(value) for value in gate_centers)
    if not gate_a < gate_b < gate_m:
        raise ValueError("gate_centers must be ordered as A < B < M")
    for policy_id in _COMPOUND_POLICY_ORDER:
        policy_records = [record for record in records if record.policy_id == policy_id]
        if not policy_records:
            continue
        errors: list[float] = []
        final_r2: list[float] = []
        occupancies: list[tuple[float, float, float, float]] = []
        for record in policy_records:
            error = _safe_float(record.metadata.get("embedding_error_final"))
            if error is not None:
                errors.append(error)
            r2_rows = read_trace_csv(record.run_dir / "trajectory_r2_trace.csv")
            if r2_rows:
                value = _safe_float(r2_rows[-1].get("trajectory_r2"))
                if value is not None:
                    final_r2.append(value)
            state_rows = read_trace_csv(record.run_dir / "state_action_trace.csv")
            selector = np.asarray([float(row["true_x"]) for row in state_rows], dtype=np.float64)
            if selector.size:
                occupancies.append(
                    _selector_gate_occupancy(
                        selector,
                        gate_centers=gate_centers,
                        rest_cutoff=rest_cutoff,
                    )
                )
        occupancy = np.mean(np.asarray(occupancies), axis=0)
        rows.append(
            {
                "policy_id": policy_id,
                "label": _COMPOUND_POLICY_LABELS[policy_id],
                "n_seeds": len(policy_records),
                "parameter_error_mean": float(np.mean(errors)),
                "parameter_error_sem": float(sample_sem(errors)),
                "parameter_error_median": float(np.median(errors)),
                "parameter_error_q25": float(np.quantile(errors, 0.25)),
                "parameter_error_q75": float(np.quantile(errors, 0.75)),
                "trajectory_r2_mean": float(np.mean(final_r2)),
                "trajectory_r2_sem": float(sample_sem(final_r2)),
                "trajectory_r2_median": float(np.median(final_r2)),
                "trajectory_r2_q25": float(np.quantile(final_r2, 0.25)),
                "trajectory_r2_q75": float(np.quantile(final_r2, 0.75)),
                "rest_fraction": float(occupancy[0]),
                "gate_A_fraction": float(occupancy[1]),
                "gate_B_fraction": float(occupancy[2]),
                "gate_M_fraction": float(occupancy[3]),
            }
        )
    return rows


def _selector_gate_occupancy(
    selector: np.ndarray,
    *,
    gate_centers: tuple[float, float, float],
    rest_cutoff: float | None,
) -> tuple[float, float, float, float]:
    """Return rest/A/B/M occupancy under midpoint gate assignments."""
    values = np.asarray(selector, dtype=np.float64).reshape(-1)
    gate_a, gate_b, gate_m = (float(value) for value in gate_centers)
    gate_mid_ab = 0.5 * (gate_a + gate_b)
    gate_mid_bm = 0.5 * (gate_b + gate_m)
    rest = 0.0 if rest_cutoff is None else float(np.mean(values < rest_cutoff))
    gate_a_mask = values < gate_mid_ab
    if rest_cutoff is not None:
        gate_a_mask &= values >= float(rest_cutoff)
    return (
        rest,
        float(np.mean(gate_a_mask)),
        float(np.mean((values >= gate_mid_ab) & (values < gate_mid_bm))),
        float(np.mean(values >= gate_mid_bm)),
    )


def _compound_paired_summary_rows(
    records: Sequence[_ExperimentRunRecord],
) -> list[dict[str, Any]]:
    """Compare every policy with PALDI on matched seeds at the final step."""
    final_by_policy: dict[str, dict[int, tuple[float, float]]] = {}
    for record in records:
        r2_rows = read_trace_csv(record.run_dir / "trajectory_r2_trace.csv")
        if not r2_rows:
            continue
        final_r2 = _safe_float(r2_rows[-1].get("trajectory_r2"))
        final_error = _safe_float(record.metadata.get("embedding_error_final"))
        if final_r2 is None or final_error is None:
            continue
        final_by_policy.setdefault(record.policy_id, {})[int(record.seed)] = (
            final_r2,
            final_error,
        )
    paldi = final_by_policy.get("compound_active_planning", {})
    rows: list[dict[str, Any]] = []
    for policy_id in _COMPOUND_POLICY_ORDER:
        if policy_id == "compound_active_planning":
            continue
        comparison = final_by_policy.get(policy_id, {})
        seeds = sorted(set(paldi) & set(comparison))
        if not seeds:
            continue
        r2_difference = np.asarray(
            [paldi[seed][0] - comparison[seed][0] for seed in seeds],
            dtype=np.float64,
        )
        error_difference = np.asarray(
            [paldi[seed][1] - comparison[seed][1] for seed in seeds],
            dtype=np.float64,
        )
        rows.append(
            {
                "policy_id": policy_id,
                "label": _COMPOUND_POLICY_LABELS[policy_id],
                "n_paired": len(seeds),
                "paldi_r2_wins": int(np.sum(r2_difference > 0.0)),
                "r2_difference_mean": float(np.mean(r2_difference)),
                "r2_difference_median": float(np.median(r2_difference)),
                "r2_difference_q25": float(np.quantile(r2_difference, 0.25)),
                "r2_difference_q75": float(np.quantile(r2_difference, 0.75)),
                "paldi_error_wins": int(np.sum(error_difference < 0.0)),
                "error_difference_mean": float(np.mean(error_difference)),
                "error_difference_median": float(np.median(error_difference)),
                "error_difference_q25": float(np.quantile(error_difference, 0.25)),
                "error_difference_q75": float(np.quantile(error_difference, 0.75)),
            }
        )
    return rows


def _reach_hold_selector_occupancy(
    *,
    rest_center: float,
    target_center: float,
    gate_centers: tuple[float, float, float],
    rest_cutoff: float,
    selector_contraction: float,
    dt: float,
    total_steps: int,
) -> tuple[float, float, float, float]:
    """Deterministic transit baseline for moving from rest to a held selector gate."""
    selector = float(rest_center)
    holding_action = float(selector_contraction) * (float(target_center) - float(rest_center))
    trace = np.empty(int(total_steps), dtype=np.float64)
    for step in range(int(total_steps)):
        trace[step] = selector
        selector += float(dt) * (
            -float(selector_contraction) * (selector - float(rest_center)) + holding_action
        )
    return _selector_gate_occupancy(
        trace,
        gate_centers=gate_centers,
        rest_cutoff=float(rest_cutoff),
    )


def generate_compound_tri_gate_figures(
    result_roots: Sequence[Path],
    *,
    output_dir: Path,
    exemplar_seed: int = 0,
    exp_id: str = "compound_tri_gate",
    file_stem: str = "compound_tri_gate",
    observation_label: str = "linear Gaussian",
    system_label: str = "CompoundTriGate",
    r2_label: str = "Trajectory $R^2$",
    r2_ylim: tuple[float, float] = (-3.1, 1.02),
    dynamics_type: str = "compound_tri_gate",
    gate_centers: tuple[float, float, float] = (-0.5, -0.32, 0.0),
    gate_width: float = 0.04,
    gate_span_sigma: float = 1.0,
    rest_center: float | None = None,
    rest_cutoff: float | None = None,
    state_dim: int = 5,
    parameter_dim: int = 3,
    selector_contraction: float = 1.0,
    summary_kind: str = "mean_sem",
    figure_suffix: str = ".png",
    response_ranges: tuple[tuple[float, float], ...] = (
        (-1.0, 6.0),
        (-0.8, 1.5),
        (-0.03, 0.03),
    ),
) -> list[Path]:
    """Write the CompoundTriGate comparison, trajectory, and vector-field panels.

    The R2 curves use each run's targeted deterministic trajectory evaluation:
    selector starts cover all three gates, response starts are bounded, and the
    nuisance coordinate is fixed at zero. This isolates learned dynamics from
    independent process-noise realizations.
    """
    if summary_kind not in {"mean_sem", "median_iqr"}:
        raise ValueError("summary_kind must be 'mean_sem' or 'median_iqr'")
    if not str(figure_suffix).startswith("."):
        raise ValueError("figure_suffix must include the leading period")
    records = _compound_trace_records(result_roots, exp_id=exp_id)
    if not records:
        raise FileNotFoundError(f"No CompoundTriGate metadata found for exp_id={exp_id!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _compound_summary_rows(
        records, gate_centers=gate_centers, rest_cutoff=rest_cutoff
    )
    summary_csv_path = output_dir / f"{file_stem}_summary.csv"
    _write_csv(
        summary_csv_path,
        summary_rows,
        (
            "policy_id",
            "label",
            "n_seeds",
            "parameter_error_mean",
            "parameter_error_sem",
            "parameter_error_median",
            "parameter_error_q25",
            "parameter_error_q75",
            "trajectory_r2_mean",
            "trajectory_r2_sem",
            "trajectory_r2_median",
            "trajectory_r2_q25",
            "trajectory_r2_q75",
            "rest_fraction",
            "gate_A_fraction",
            "gate_B_fraction",
            "gate_M_fraction",
        ),
    )
    paired_summary_rows = _compound_paired_summary_rows(records)
    paired_summary_path = output_dir / f"{file_stem}_paired_comparisons.csv"
    _write_csv(
        paired_summary_path,
        paired_summary_rows,
        (
            "policy_id",
            "label",
            "n_paired",
            "paldi_r2_wins",
            "r2_difference_mean",
            "r2_difference_median",
            "r2_difference_q25",
            "r2_difference_q75",
            "paldi_error_wins",
            "error_difference_mean",
            "error_difference_median",
            "error_difference_q25",
            "error_difference_q75",
        ),
    )
    transit_baseline_path: Path | None = None
    if rest_center is not None and rest_cutoff is not None:
        transit_occupancy = _reach_hold_selector_occupancy(
            rest_center=float(rest_center),
            target_center=float(gate_centers[-1]),
            gate_centers=gate_centers,
            rest_cutoff=float(rest_cutoff),
            selector_contraction=float(selector_contraction),
            dt=float(records[0].metadata.get("dt", 0.01)),
            total_steps=int(records[0].metadata.get("total_steps", 0)),
        )
        transit_baseline_path = output_dir / f"{file_stem}_reach_hold_M_baseline.csv"
        _write_csv(
            transit_baseline_path,
            [
                {
                    "holding_action": float(selector_contraction)
                    * (float(gate_centers[-1]) - float(rest_center)),
                    "rest_fraction": transit_occupancy[0],
                    "gate_A_fraction": transit_occupancy[1],
                    "gate_B_fraction": transit_occupancy[2],
                    "gate_M_fraction": transit_occupancy[3],
                }
            ],
            (
                "holding_action",
                "rest_fraction",
                "gate_A_fraction",
                "gate_B_fraction",
                "gate_M_fraction",
            ),
        )
    error_curves = _compound_curve(
        records,
        filename="parameter_error_trace.csv",
        value_key="parameter_error",
        stride=20,
    )
    r2_curves = _compound_curve(
        records,
        filename="trajectory_r2_trace.csv",
        value_key="trajectory_r2",
        stride=20,
    )

    summary_path = output_dir / f"{file_stem}_step_r2{figure_suffix}"
    plt_module = load_plotting(summary_path, apply_style=_apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(2, 2, figsize=(10.0, 6.3))
    for policy_id in _COMPOUND_POLICY_ORDER:
        color = _COMPOUND_POLICY_COLORS[policy_id]
        linewidth = 2.2 if policy_id == "compound_active_planning" else 0.9
        alpha = 1.0 if policy_id == "compound_active_planning" else 0.78
        for ax, curve_map in ((axes[0, 0], error_curves), (axes[0, 1], r2_curves)):
            curve = curve_map.get(policy_id, [])
            if not curve:
                continue
            step = np.asarray([row["step"] for row in curve])
            if summary_kind == "mean_sem":
                center = np.asarray([row["mean"] for row in curve])
                lower = center - np.asarray([row["sem"] for row in curve])
                upper = center + np.asarray([row["sem"] for row in curve])
            else:
                center = np.asarray([row["median"] for row in curve])
                lower = np.asarray([row["q25"] for row in curve])
                upper = np.asarray([row["q75"] for row in curve])
            ax.plot(
                step,
                center,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=_COMPOUND_POLICY_LABELS[policy_id],
            )
            if policy_id == "compound_active_planning":
                ax.fill_between(step, lower, upper, color=color, alpha=0.18)
    axes[0, 0].set_title("A. Parameter recovery")
    axes[0, 0].set_xlabel("Environment step")
    axes[0, 0].set_ylabel(r"$\|\hat\theta-\theta^*\|_2$")
    axes[0, 0].set_ylim(bottom=0.0)
    axes[0, 1].set_title("B. Targeted trajectory recovery")
    axes[0, 1].set_xlabel("Environment step")
    axes[0, 1].set_ylabel(r2_label)
    axes[0, 1].axhline(0.8, color=_experiment_C_NEUTRAL_LIGHT, linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylim(*r2_ylim)

    x = np.arange(len(summary_rows))
    colors = [_COMPOUND_POLICY_COLORS[str(row["policy_id"])] for row in summary_rows]
    labels = [str(row["label"]) for row in summary_rows]
    if summary_kind == "mean_sem":
        error_center = np.asarray([row["parameter_error_mean"] for row in summary_rows])
        error_yerr = np.asarray([row["parameter_error_sem"] for row in summary_rows])
        final_statistic_label = "Mean at step 2000"
    else:
        error_center = np.asarray([row["parameter_error_median"] for row in summary_rows])
        error_q25 = np.asarray([row["parameter_error_q25"] for row in summary_rows])
        error_q75 = np.asarray([row["parameter_error_q75"] for row in summary_rows])
        error_yerr = np.vstack((error_center - error_q25, error_q75 - error_center))
        final_statistic_label = "Median at step 2000"
    axes[1, 0].bar(
        x,
        error_center,
        yerr=error_yerr,
        color=colors,
        edgecolor=_experiment_C_STROKE,
        linewidth=0.4,
        capsize=2.0,
    )
    axes[1, 0].set_title("C. Final parameter error")
    axes[1, 0].set_ylabel(final_statistic_label)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=35, ha="right")

    bottom = np.zeros(len(summary_rows), dtype=np.float64)
    gate_colors = ("#B8B8B8", "#D95F5F", "#7B6FD0", "#3D9B6D")
    for key, label, color in zip(
        ("rest_fraction", "gate_A_fraction", "gate_B_fraction", "gate_M_fraction"),
        ("Rest", "A: confounded", "B: weak balanced", "M: main"),
        gate_colors,
        strict=True,
    ):
        value = np.asarray([row[key] for row in summary_rows], dtype=np.float64)
        axes[1, 1].bar(x, value, bottom=bottom, color=color, label=label)
        bottom += value
    axes[1, 1].set_title("D. Gate occupancy")
    axes[1, 1].set_ylabel("Fraction of 2000 steps")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1, 1].legend(fontsize=7.0, loc="upper right")
    for ax in axes.ravel():
        _style_manuscript_axis(ax)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{system_label} with {observation_label} observations", y=1.04)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        fontsize=7.0,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=1.5, w_pad=1.3)
    save_figure(fig, summary_path, plt_module=plt_module)

    trajectory_path = output_dir / f"{file_stem}_exemplary_trajectories{figure_suffix}"
    plt_module = load_plotting(trajectory_path, apply_style=_apply_style, path_is_file=True)
    fig, axes = plt_module.subplots(3, 3, figsize=(10.0, 7.0), sharex=True, sharey=True)
    exemplar_records = {
        record.policy_id: record for record in records if int(record.seed) == int(exemplar_seed)
    }
    gate_specs = tuple(
        (center, label, color)
        for center, label, color in zip(
            gate_centers,
            ("A", "B", "M"),
            ("#D95F5F", "#7B6FD0", "#3D9B6D"),
            strict=True,
        )
    )
    for ax, policy_id in zip(axes.ravel(), _COMPOUND_POLICY_ORDER, strict=True):
        if rest_center is not None:
            ax.axhline(float(rest_center), color="#777777", linestyle=":", linewidth=0.7)
        for center, label, color in gate_specs:
            span = float(gate_span_sigma) * float(gate_width)
            ax.axhspan(center - span, center + span, color=color, alpha=0.12)
            ax.axhline(center, color=color, linewidth=0.55, alpha=0.65)
        record = exemplar_records[policy_id]
        trace = read_trace_csv(record.run_dir / "state_action_trace.csv")
        step = np.asarray([float(row["step"]) for row in trace])
        selector = np.asarray([float(row["true_x"]) for row in trace])
        ax.plot(step, selector, color=_COMPOUND_POLICY_COLORS[policy_id], linewidth=0.7)
        ax.set_title(_COMPOUND_POLICY_LABELS[policy_id], fontsize=9.0)
        ax.set_ylim(
            min(
                min(gate_centers),
                min(gate_centers) if rest_center is None else float(rest_center),
            )
            - 1.2 * float(gate_width),
            max(gate_centers) + 2.8 * float(gate_width),
        )
        _style_manuscript_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Environment step")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Selector $r$")
    for center, label, color in gate_specs:
        axes[0, 0].text(30, center, label, color=color, va="center")
    fig.suptitle(
        f"{system_label} / {observation_label}: representative selector trajectories "
        f"(seed {exemplar_seed})"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_figure(fig, trajectory_path, plt_module=plt_module)

    vector_path = output_dir / f"{file_stem}_vectorfields{figure_suffix}"
    plt_module = load_plotting(vector_path, apply_style=_apply_style, path_is_file=True)
    import torch
    from actdyn.environment.vectorfield import build_vectorfield

    if len(response_ranges) != int(parameter_dim):
        raise ValueError("response_ranges must contain one range per parameter")
    fig, axes = plt_module.subplots(
        1, int(parameter_dim), figsize=(3.35 * int(parameter_dim), 3.0), squeeze=False
    )
    axes = axes.ravel()
    for param_idx, (ax, response_range) in enumerate(zip(axes, response_ranges, strict=True)):
        r_grid = np.linspace(
            min(gate_centers) - 1.2 * float(gate_width),
            max(gate_centers) + 1.2 * float(gate_width),
            90,
        )
        s_grid = np.linspace(response_range[0], response_range[1], 90)
        rr, ss = np.meshgrid(r_grid, s_grid)
        state = np.zeros((rr.size, int(state_dim)), dtype=np.float32)
        state[:, 0] = rr.reshape(-1)
        state[:, 1 + param_idx] = ss.reshape(-1)
        theta = np.zeros(int(parameter_dim), dtype=np.float32)
        theta[param_idx] = 1.0
        vf = build_vectorfield(dynamics_type, theta.tolist())
        drift = vf.compute(torch.as_tensor(state)).detach().cpu().numpy()
        uu = drift[:, 0].reshape(rr.shape)
        vv = drift[:, 1 + param_idx].reshape(rr.shape)
        speed = np.sqrt(uu**2 + vv**2)
        ax.pcolormesh(rr, ss, np.log1p(speed), cmap="viridis", shading="auto", alpha=0.72)
        ax.streamplot(
            r_grid,
            s_grid,
            uu,
            vv,
            color=_experiment_C_STROKE,
            linewidth=0.42,
            density=1.25,
            arrowsize=0.65,
        )
        for center, label, color in gate_specs:
            ax.axvline(center, color=color, linestyle="--", linewidth=0.8)
            ax.text(center, response_range[1], label, color=color, ha="center", va="bottom")
        ax.set_title(rf"Unit $\theta_{param_idx + 1}$ basis field")
        ax.set_xlabel(r"Selector $r$")
        ax.set_ylabel(rf"Response $s_{param_idx + 1}$")
        _style_manuscript_axis(ax)
    fig.suptitle(
        f"{system_label} parameter-basis vector fields; " f"{observation_label} observations"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), w_pad=1.3)
    save_figure(fig, vector_path, plt_module=plt_module)

    parameter_path = output_dir / f"{file_stem}_parameter_estimates{figure_suffix}"
    plt_module = load_plotting(parameter_path, apply_style=_apply_style, path_is_file=True)
    fig, axes = plt_module.subplots(
        1,
        int(parameter_dim),
        figsize=(3.35 * int(parameter_dim), 3.0),
        squeeze=False,
        sharex=True,
    )
    axes = axes.ravel()
    true_params = np.asarray(records[0].metadata["true_params_full"], dtype=np.float64)
    for param_idx, ax in enumerate(axes):
        curve_map = _compound_curve(
            records,
            filename="embedding_estimate_trace.csv",
            value_key=f"e{param_idx}",
            stride=20,
        )
        for policy_id in _COMPOUND_POLICY_ORDER:
            curve = curve_map.get(policy_id, [])
            if not curve:
                continue
            ax.plot(
                [row["step"] for row in curve],
                [row["mean"] for row in curve],
                color=_COMPOUND_POLICY_COLORS[policy_id],
                linewidth=2.0 if policy_id == "compound_active_planning" else 0.8,
                alpha=1.0 if policy_id == "compound_active_planning" else 0.75,
                label=_COMPOUND_POLICY_LABELS[policy_id],
            )
        ax.axhline(true_params[param_idx], color="#222222", linestyle="--", linewidth=0.8)
        ax.set_title(rf"$\theta_{param_idx + 1}$ estimate")
        ax.set_xlabel("Environment step")
        ax.set_ylabel("Posterior mean")
        _style_manuscript_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{system_label}: per-parameter recovery", y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82), w_pad=1.2)
    save_figure(fig, parameter_path, plt_module=plt_module)

    paths = [
        summary_path,
        trajectory_path,
        vector_path,
        parameter_path,
        summary_csv_path,
        paired_summary_path,
    ]
    if transit_baseline_path is not None:
        paths.append(transit_baseline_path)
    if str(records[0].metadata.get("observation_model")) == "log_linear":
        poisson_path = output_dir / f"{file_stem}_poisson_observation{figure_suffix}"
        _compound_poisson_observation_figure(records[0].metadata, poisson_path)
        paths.append(poisson_path)
    return paths


def _compound_poisson_observation_figure(metadata: dict[str, Any], output_path: Path) -> None:
    """Plot realized rate and state-Fisher geometry for the paired Poisson decoder."""
    weights, bias, dt = reconstruct_loglinear_rate_model(metadata)
    radius = float(metadata.get("boundary_radius", 6.0))
    coordinate = np.linspace(-radius, radius, 241)
    latent_dim = int(weights.shape[1])
    fisher_curves = np.zeros((latent_dim, coordinate.size), dtype=np.float64)
    primary_rate_min = np.zeros(coordinate.size, dtype=np.float64)
    primary_rate_mean = np.zeros(coordinate.size, dtype=np.float64)
    primary_rate_max = np.zeros(coordinate.size, dtype=np.float64)
    for coordinate_idx in range(latent_dim):
        state = np.zeros((coordinate.size, latent_dim), dtype=np.float64)
        state[:, coordinate_idx] = coordinate
        rate_hz = np.exp(state @ weights.T + bias)
        count = float(dt) * rate_hz
        fisher = np.einsum("no,oi,oj->nij", count, weights, weights)
        fisher_curves[coordinate_idx] = fisher[:, coordinate_idx, coordinate_idx]
        if coordinate_idx == 1:
            primary_rate_min = rate_hz.min(axis=1)
            primary_rate_mean = rate_hz.mean(axis=1)
            primary_rate_max = rate_hz.max(axis=1)

    plt_module = load_plotting(output_path, apply_style=_apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(1, 3, figsize=(10.0, 2.9))
    for index in range(latent_dim - 1):
        axes[0].plot(
            coordinate,
            fisher_curves[index],
            linewidth=0.9,
            alpha=0.75,
            label=rf"$I_{{z,{index + 1}{index + 1}}}$",
        )
    axes[0].plot(
        coordinate,
        fisher_curves[-1],
        color="#222222",
        linewidth=1.4,
        label=r"$I_{z,hh}$",
    )
    axes[0].axhline(1.0, color="#777777", linestyle="--", linewidth=0.6)
    axes[0].axhline(0.01, color="#777777", linestyle=":", linewidth=0.6)
    axes[0].set_yscale("log")
    axes[0].set_title("A. Coordinate Fisher")
    axes[0].set_xlabel("Coordinate value")
    axes[0].set_ylabel(r"Diagonal $I_z$")
    axes[0].legend(fontsize=6.0, ncol=2)

    axes[1].plot(coordinate, primary_rate_min, label="minimum", linewidth=0.9)
    axes[1].plot(coordinate, primary_rate_mean, label="mean", linewidth=0.9)
    axes[1].plot(coordinate, primary_rate_max, label="maximum", linewidth=1.2)
    axes[1].axhline(
        float(metadata.get("max_firing_rate_target", 210.0)),
        color="#777777",
        linestyle="--",
        linewidth=0.7,
        label="configured cap",
    )
    axes[1].set_title(r"B. Rates while varying $s_1$")
    axes[1].set_xlabel(r"Response $s_1$")
    axes[1].set_ylabel("Rate per neuron (Hz)")
    axes[1].legend(fontsize=6.0)

    image = axes[2].imshow(
        np.log10(np.clip(fisher_curves, 1e-8, None)),
        aspect="auto",
        origin="lower",
        extent=(coordinate[0], coordinate[-1], 0.5, latent_dim + 0.5),
        cmap="viridis",
    )
    axes[2].set_title("C. Rate-dependent Fisher")
    axes[2].set_xlabel("Own coordinate value")
    axes[2].set_ylabel("Latent coordinate")
    axes[2].set_yticks(np.arange(1, latent_dim + 1))
    coordinate_labels = [r"$r$"]
    coordinate_labels.extend(rf"$s_{index}$" for index in range(1, max(1, latent_dim - 1)))
    coordinate_labels.append(r"$h$")
    axes[2].set_yticklabels(coordinate_labels)
    fig.colorbar(image, ax=axes[2], label=r"$\log_{10} I_{z,ii}$", fraction=0.05)
    for ax in axes:
        _style_manuscript_axis(ax)
    fig.suptitle("Paired log-linear Poisson observation geometry")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), w_pad=1.2)
    save_figure(fig, output_path, plt_module=plt_module)


def generate_three_gate_diagnostic_figures(
    result_roots: Sequence[Path],
    *,
    output_dir: Path,
    exemplar_seed: int = 0,
    summary_kind: str = "mean_sem",
    figure_suffix: str = ".png",
) -> list[Path]:
    """Write the complete comparison for the designed three-gate diagnostic."""
    return generate_compound_tri_gate_figures(
        result_roots,
        output_dir=output_dir,
        exemplar_seed=exemplar_seed,
        summary_kind=summary_kind,
        figure_suffix=figure_suffix,
        exp_id="three_gate_diagnostic",
        file_stem="three_gate_diagnostic",
        observation_label="compact log-linear Poisson",
        system_label="ThreeGateDiagnostic",
        r2_label="Response-balanced rollout $R^2$",
        r2_ylim=(-0.2, 1.02),
        dynamics_type="three_gate_diagnostic",
        gate_centers=(-0.5, -0.1, 0.3),
        gate_width=0.1,
        gate_span_sigma=2.0,
        rest_center=-1.0,
        rest_cutoff=-0.75,
        state_dim=5,
        parameter_dim=3,
        response_ranges=((-1.0, 6.0), (-1.0, 2.0), (-0.5, 0.5)),
    )


def experiment_main(argv: list[str] | None = None) -> int:
    args = _experiment_build_parser().parse_args(argv)
    max_seeds = int(args.max_seeds)
    plot_ids = _experiment_parse_plots(str(args.plots))
    output_suite_dirs = _experiment_suite_dirs_from_groups(str(args.groups))
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

    written: list[Path] = []
    if any(plot_id in _experiment_OBJECTIVE_DEFINITION_PLOTS for plot_id in plot_ids):
        written.extend(
            _experiment_write_objective_definition_tables(
                _experiment_required_suite_dirs(["objective_ablation"])
            )
        )

    experiment_plotters = {
        "bottleneck_sweep": _experiment_plot_bottleneck_sweep,
        "objective_ablation": _experiment_plot_objective_ablation,
        "mismatch_dose_response": _experiment_plot_mismatch_dose_response,
    }
    figure_only_plotters = {
        "true_dynamics_all": lambda: _experiment_plot_true_dynamics_all(output_suite_dirs),
        "per_parameter_recovery": lambda: _experiment_plot_per_parameter_recovery(
            max_seeds=max_seeds
        ),
    }
    for plot_id in plot_ids:
        if plot_id in experiment_plotters:
            written.extend(experiment_plotters[plot_id]())
        else:
            written.extend(figure_only_plotters[plot_id]())

    for path in written:
        print(path)
    return 0
