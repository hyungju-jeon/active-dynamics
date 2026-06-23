#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from actdyn.utils.experiment_runtime import read_trace_csv
from actdyn.utils.figure_io import load_plotting, save_figure
from experiments.experiment_io import get_environment_preset_from_metadata

from . import tbme_figures as _figures
from .tbme_figures import (
    _ExperimentRunRecord,
    _ExperimentSuiteSource,
    _REPO_ROOT,
    _RESULTS_ROOT,
    _apply_style,
    _experiment_C_NEUTRAL_FILL,
    _experiment_C_NEUTRAL_LIGHT,
    _experiment_C_STROKE,
    _experiment_OBJECTIVE_POLICIES,
    _experiment_collect_records,
    _experiment_curve_rows,
    _experiment_load_xy_trace,
    _experiment_make_information_grid,
    _experiment_metric_mean_sem,
    _experiment_objective_sources,
    _experiment_r2_threshold_step,
    _experiment_r2_threshold_times,
    _experiment_short_policy_label,
    _experiment_trace_path,
    _latest_session,
    _overview_figures_dir,
    _policy_color,
    _style_experiment_axis,
    _style_manuscript_axis,
    _suite_dir,
    _write_csv,
    _write_text,
    plot_neutral_vector_field,
)
from .tbme_io import (
    load_planned_trace,
    planned_xy_cycle_for_step,
    safe_float as _safe_float,
    true_dynamics_from_metadata,
)

# Manuscript asset assembly
_ASSET_POLICY_LABELS = {
    "active_planning_adaptive_u20_r20_h40": "PALDI",
    "active_planning_u20_r20_h40": "Fixed PALDI",
    "active_myopic": "Myopic",
    "ensemble": "Ensemble",
    "prbs": "PRBS",
    "random": "Random",
    "active_fully_observable_u20_r20_h40": "Full obs.",
    "active_state_information_u20_r20_h40": "State info",
    "active_dynamics_u20_r20_h40": "Dyn. sens.",
    "active_sampling_variance_u20_r20_h40": "Sample var.",
    "active_e_optimality_u20_r20_h40": "E-opt.",
    "flex": "FLEX",
    "rhc": "RHC-US",
}
_ASSET_BENCHMARK_POLICIES = [
    "active_planning_adaptive_u20_r20_h40",
    "active_planning_u20_r20_h40",
    "active_myopic",
    "ensemble",
    "prbs",
    "random",
]
_ASSET_MATCHED_POLICIES = [
    "active_planning_adaptive_u20_r20_h40",
    "active_myopic",
    "flex",
    "rhc",
    "prbs",
    "random",
]
_ASSET_R2_THRESHOLDS = (0.90, 0.95, 0.99)


def _asset_policy_label(policy_id: str) -> str:
    return _ASSET_POLICY_LABELS.get(policy_id, _experiment_short_policy_label(policy_id))


def _asset_require_suite_dirs(paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing TBME result suite(s): " + ", ".join(str(path) for path in missing)
        )


def _asset_first_record(group_name: str, suite_id: str, policy_id: str) -> _ExperimentRunRecord:
    suite_dir = _suite_dir(group_name, suite_id)
    _asset_require_suite_dirs([suite_dir])
    records = _experiment_collect_records(
        suite_dir,
        [policy_id],
        completed_only=True,
    )
    if not records:
        raise RuntimeError(f"No completed records for {suite_id}/{policy_id}")
    for record in records:
        if int(record.seed) == 0:
            return record
    return records[0]


def _asset_read_information(record: _ExperimentRunRecord) -> list[dict[str, str]]:
    path = _experiment_trace_path(record, "information_trace_path", "information_trace.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return read_trace_csv(path)


def _asset_row_series(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []
    for row in rows:
        step = _safe_float(row.get("step"))
        value = _safe_float(row.get(field))
        if step is None:
            continue
        steps.append(step)
        values.append(np.nan if value is None else value)
    return np.asarray(steps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _asset_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    width = max(1, int(window))
    kernel = np.ones(width, dtype=np.float64)
    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(np.float64), kernel, mode="same")
    out = np.full_like(values, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _asset_plot_r2_curves(
    ax: Any,
    suite_dir: Path,
    policy_ids: Sequence[str],
    *,
    title: str,
    ylabel: bool,
) -> None:
    curves = _experiment_curve_rows(
        suite_dir,
        "trajectory_r2_over_steps.csv",
        "trajectory_r2_mean",
    )
    for policy_id in policy_ids:
        rows = curves.get(policy_id, [])
        if not rows:
            continue
        steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
        values = np.asarray([row["value"] for row in rows], dtype=np.float64)
        sem = np.asarray([row["sem"] for row in rows], dtype=np.float64)
        color = _policy_color(policy_id)
        ax.plot(
            steps,
            values,
            color=color,
            linewidth=0.95,
            label=_asset_policy_label(policy_id),
        )
        ax.fill_between(
            steps,
            values - sem,
            values + sem,
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )
    for threshold in _ASSET_R2_THRESHOLDS:
        ax.axhline(threshold, color=_experiment_C_NEUTRAL_LIGHT, linestyle="--", linewidth=0.55)
    ax.set_title(title, pad=3.0, fontsize=9.5)
    ax.set_xlabel("Environment step")
    if ylabel:
        ax.set_ylabel("Predictive R2")
    ax.set_ylim(-0.05, 1.04)
    _style_experiment_axis(ax)


def _asset_sort_r2_strip_rows(
    suite_policy_rows: Sequence[tuple[Path, str, str]],
    curve_cache: Mapping[Path, dict[str, list[dict[str, float]]]],
) -> list[tuple[Path, str, str]]:
    threshold_steps = []
    for suite_dir, policy_id, _label in suite_policy_rows:
        step = _experiment_r2_threshold_step(suite_dir, policy_id, 0.90)
        if step is not None:
            threshold_steps.append(step)
    comparison_step = max(threshold_steps) if threshold_steps else math.inf

    keyed_rows = []
    for order, row in enumerate(suite_policy_rows):
        suite_dir, policy_id, _label = row
        step = _experiment_r2_threshold_step(suite_dir, policy_id, 0.90)
        if step is not None:
            keyed_rows.append((0, step, order, row))
            continue
        values = [
            float(curve_row["value"])
            for curve_row in curve_cache[suite_dir].get(policy_id, [])
            if float(curve_row["step"]) <= comparison_step
            and math.isfinite(float(curve_row["value"]))
        ]
        keyed_rows.append((1, -max(values) if values else math.inf, order, row))
    keyed_rows.sort(key=lambda item: item[:3])
    return [row for _rank, _score, _order, row in keyed_rows]


def _asset_plot_r2_strips(
    ax: Any,
    suite_policy_rows: Sequence[tuple[Path, str, str]],
    *,
    title: str,
    show_ylabels: bool = True,
    policy_order: Sequence[str] | None = None,
) -> Any:
    curve_cache = {
        suite_dir: _experiment_curve_rows(
            suite_dir,
            "trajectory_r2_over_steps.csv",
            "trajectory_r2_mean",
        )
        for suite_dir, _policy_id, _label in suite_policy_rows
    }
    if policy_order is None:
        suite_policy_rows = _asset_sort_r2_strip_rows(suite_policy_rows, curve_cache)
    else:
        order_by_policy = {policy_id: order for order, policy_id in enumerate(policy_order)}
        suite_policy_rows = sorted(
            suite_policy_rows,
            key=lambda row: order_by_policy.get(row[1], len(order_by_policy)),
        )
    steps = sorted(
        {
            float(row["step"])
            for suite_dir, policy_id, _label in suite_policy_rows
            for row in curve_cache[suite_dir].get(policy_id, [])
        }
    )
    if not steps:
        raise RuntimeError(f"No trajectory R2 curves available for {title}")
    step_to_idx = {step: idx for idx, step in enumerate(steps)}
    matrix = np.full((len(suite_policy_rows), len(steps)), np.nan, dtype=np.float64)
    for row_idx, (suite_dir, policy_id, _label) in enumerate(suite_policy_rows):
        for row in curve_cache[suite_dir].get(policy_id, []):
            matrix[row_idx, step_to_idx[float(row["step"])]] = float(row["value"])

    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    strip_cmap = plt.get_cmap("afmhot_r", 20).copy()
    strip_cmap.set_bad("#F1F1EE")
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap=strip_cmap,
        norm=PowerNorm(gamma=2.0, vmin=0.5, vmax=1.0),
        extent=[steps[0], steps[-1], len(suite_policy_rows) - 0.5, -0.5],
    )
    threshold_styles = {0.90: ":", 0.95: "--", 0.99: "-."}
    for row_idx, (suite_dir, policy_id, _label) in enumerate(suite_policy_rows):
        for threshold in _ASSET_R2_THRESHOLDS:
            step = _experiment_r2_threshold_step(suite_dir, policy_id, threshold)
            if step is None:
                continue
            ax.vlines(
                step,
                row_idx - 0.42,
                row_idx + 0.42,
                color="white",
                linestyle=threshold_styles[threshold],
                linewidth=0.75,
                alpha=0.95,
            )
    ax.set_title(title, pad=3.0, fontsize=9.5)
    ax.set_xlabel("Environment step")
    ax.set_yticks(np.arange(len(suite_policy_rows), dtype=np.float64))
    ax.set_yticklabels(
        [label for _suite_dir, _policy_id, label in suite_policy_rows] if show_ylabels else [],
        fontsize=5.7,
    )
    _style_manuscript_axis(ax, grid_alpha=0.0)
    return im


def _asset_plot_active_vs_baselines(output_path: Path) -> Path:
    sources = [
        _ExperimentSuiteSource(ref.suite_id, ref.label, ref.session_root / ref.suite_id)
        for ref in _figures.GROUPS["exp01_base"]
    ]
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    plt_module = load_plotting(output_path, apply_style=_apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(2, len(sources), figsize=(7.25, 4.65), squeeze=False)
    im = None
    short_labels = {"Damped pendulum": "Pendulum", "Asymmetric basin": "Basin"}
    leftmost_strip_rows = [
        (sources[0].suite_dir, policy_id, _asset_policy_label(policy_id))
        for policy_id in _ASSET_MATCHED_POLICIES
    ]
    leftmost_curve_cache = {
        sources[0].suite_dir: _experiment_curve_rows(
            sources[0].suite_dir,
            "trajectory_r2_over_steps.csv",
            "trajectory_r2_mean",
        )
    }
    strip_policy_order = [
        policy_id
        for _suite_dir, policy_id, _label in _asset_sort_r2_strip_rows(
            leftmost_strip_rows, leftmost_curve_cache
        )
    ]
    for idx, source in enumerate(sources):
        label = short_labels.get(source.label, source.label)
        _asset_plot_r2_curves(
            axes[0, idx],
            source.suite_dir,
            _ASSET_MATCHED_POLICIES,
            title=f"{chr(65 + idx)}. {label}: recovery",
            ylabel=idx == 0,
        )
        im = _asset_plot_r2_strips(
            axes[1, idx],
            [
                (source.suite_dir, policy_id, _asset_policy_label(policy_id))
                for policy_id in _ASSET_MATCHED_POLICIES
            ],
            title=f"{chr(68 + idx)}. R2 over time",
            show_ylabels=idx == 0,
            policy_order=strip_policy_order,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(_ASSET_MATCHED_POLICIES),
        fontsize=6.4,
        columnspacing=0.9,
        handlelength=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.94), w_pad=0.75, h_pad=0.95)
    if im is not None:
        cax = fig.add_axes([0.94, 0.12, 0.018, 0.36])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Predictive R2")
        cbar.outline.set_linewidth(0.45)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_bottleneck_sources() -> list[_ExperimentSuiteSource]:
    return [
        _ExperimentSuiteSource(
            "exp01_asymmetric_basin",
            "Nominal",
            _suite_dir("exp01_base", "exp01_asymmetric_basin"),
        ),
        _ExperimentSuiteSource(
            "exp06_asymmetric_basin_observation_bottleneck_mild",
            "SNR -10",
            _suite_dir("exp06_bottleneck", "exp06_asymmetric_basin_observation_bottleneck_mild"),
        ),
        _ExperimentSuiteSource(
            "exp06_asymmetric_basin_observation_bottleneck_strong",
            "SNR -15",
            _suite_dir(
                "exp06_bottleneck",
                "exp06_asymmetric_basin_observation_bottleneck_strong",
            ),
        ),
        _ExperimentSuiteSource(
            "exp06_asymmetric_basin_action_bottleneck_mild",
            "Act. 0.55",
            _suite_dir("exp06_bottleneck", "exp06_asymmetric_basin_action_bottleneck_mild"),
        ),
        _ExperimentSuiteSource(
            "exp06_asymmetric_basin_action_bottleneck_strong",
            "Act. 0.35",
            _suite_dir("exp06_bottleneck", "exp06_asymmetric_basin_action_bottleneck_strong"),
        ),
    ]


def _asset_bottleneck_rows(
    sources: Sequence[_ExperimentSuiteSource],
    policy_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for policy_id in policy_ids:
            r2, r2_sem, n_r2 = _experiment_metric_mean_sem(
                source.suite_dir,
                policy_id,
                "trajectory_r2_final_mean",
            )
            rows.append(
                {
                    "experiment": source.exp_id,
                    "condition": source.label,
                    "policy_id": policy_id,
                    "policy_label": _asset_policy_label(policy_id),
                    "trajectory_r2_mean": r2,
                    "trajectory_r2_sem": r2_sem,
                    "step_to_r2_0p90": _experiment_r2_threshold_step(
                        source.suite_dir,
                        policy_id,
                        threshold=0.90,
                    ),
                    "n_r2": n_r2,
                }
            )
    return rows


def _asset_plot_constraints(output_path: Path) -> Path:
    sources = _asset_bottleneck_sources()
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    rows = _asset_bottleneck_rows(sources, _ASSET_BENCHMARK_POLICIES)
    row_by_key = {(row["condition"], row["policy_id"]): row for row in rows}
    for row in rows:
        baseline = row_by_key.get(("Nominal", row["policy_id"]), {})
        baseline_r2 = baseline.get("trajectory_r2_mean")
        row["trajectory_r2_delta_from_nominal"] = (
            None
            if row["condition"] == "Nominal"
            or row["trajectory_r2_mean"] is None
            or baseline_r2 is None
            else float(row["trajectory_r2_mean"]) - float(baseline_r2)
        )
    _write_csv(
        output_path.with_suffix(".csv"),
        rows,
        [
            "experiment",
            "condition",
            "policy_id",
            "policy_label",
            "trajectory_r2_mean",
            "trajectory_r2_sem",
            "trajectory_r2_delta_from_nominal",
            "step_to_r2_0p90",
            "n_r2",
        ],
    )
    plt_module = load_plotting(output_path, apply_style=_apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(
        2,
        1,
        figsize=(7.25, 5.15),
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )
    strip_rows = [
        (source.suite_dir, policy_id, f"{source.label}: {_asset_policy_label(policy_id)}")
        for source in sources
        for policy_id in _ASSET_BENCHMARK_POLICIES
    ]
    im = _asset_plot_r2_strips(
        axes[0],
        strip_rows,
        title="A. Predictive R2 over bottlenecked rollouts",
    )
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.030, pad=0.01)
    cbar.set_label("Predictive R2")
    cbar.outline.set_linewidth(0.45)

    x = np.arange(len(sources), dtype=np.float64)
    for idx, policy_id in enumerate(_ASSET_BENCHMARK_POLICIES):
        deltas = []
        delta_sem = []
        baseline = row_by_key[("Nominal", policy_id)]
        baseline_r2 = baseline["trajectory_r2_mean"]
        baseline_sem = baseline["trajectory_r2_sem"]
        for source in sources:
            row = row_by_key[(source.label, policy_id)]
            value = row["trajectory_r2_mean"]
            if value is None or baseline_r2 is None:
                deltas.append(np.nan)
                delta_sem.append(0.0)
            elif source.label == "Nominal":
                deltas.append(0.0)
                delta_sem.append(0.0)
            else:
                deltas.append(float(value) - float(baseline_r2))
                delta_sem.append(
                    math.sqrt(float(row["trajectory_r2_sem"]) ** 2 + float(baseline_sem) ** 2)
                )
        axes[1].errorbar(
            x,
            deltas,
            yerr=delta_sem,
            marker="o",
            color=_policy_color(policy_id),
            markersize=3.6,
            capsize=2.0,
            linewidth=0.9,
            label=_asset_policy_label(policy_id),
        )
    axes[1].axhline(0.0, color=_experiment_C_STROKE, linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([source.label for source in sources], rotation=18, ha="right")
    axes[1].set_ylabel("Final R2 change vs nominal")
    axes[1].set_title("B. Degradation from easy observation/action setting")
    axes[1].legend(loc="lower left", fontsize=6.0, ncol=3)
    _style_experiment_axis(axes[1])
    fig.tight_layout(h_pad=1.0)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_trace_abs(record: _ExperimentRunRecord, traj: np.ndarray) -> float:
    env_preset = get_environment_preset_from_metadata(record.metadata)
    panel_abs = max(float(env_preset.resolved_plot_limit()), 6.0)
    boundary_radius = _safe_float(record.metadata.get("boundary_radius"))
    if boundary_radius is None:
        boundary_radius = _safe_float(getattr(env_preset, "boundary_radius", None))
    if boundary_radius is not None:
        panel_abs = max(panel_abs, boundary_radius)
    finite = traj[np.isfinite(traj).all(axis=1)]
    if finite.size:
        panel_abs = max(panel_abs, 1.04 * float(np.max(np.abs(finite[:, :2]))))
    return panel_abs


def _asset_plot_mechanism(output_path: Path) -> Path:
    policy_id = "active_planning_adaptive_u20_r20_h40"
    nominal = _asset_first_record("exp02_hard", "exp02_hard_asymmetric_basin", policy_id)
    obs_mismatch = _asset_first_record(
        "exp07_mismatch_stress",
        "exp07_asymmetric_basin_observation_mismatch_strong",
        policy_id,
    )
    param_mismatch = _asset_first_record(
        "exp08_parameter_mismatch_stress",
        "exp08_asymmetric_basin_parameter_mismatch_strong",
        policy_id,
    )

    nominal_traj = _experiment_load_xy_trace(nominal)
    panel_abs = _asset_trace_abs(nominal, nominal_traj)
    x_axis, y_axis, logdet_grid = _experiment_make_information_grid(
        nominal.metadata,
        n_grid=101,
        axis_min=-panel_abs,
        axis_max=panel_abs,
    )
    finite_grid = logdet_grid[np.isfinite(logdet_grid)]
    info_vmin = float(np.percentile(finite_grid, 2.0))
    info_vmax = float(np.percentile(finite_grid, 98.0))
    if info_vmax <= info_vmin:
        info_vmax = info_vmin + 1e-6

    info_rows = {
        "Nominal": _asset_read_information(nominal),
        "Obs. mismatch": _asset_read_information(obs_mismatch),
        "Param. mismatch": _asset_read_information(param_mismatch),
    }

    plt_module = load_plotting(output_path, apply_style=_apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(2, 2, figsize=(7.25, 5.15))
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
        alpha=0.70,
    )
    plot_neutral_vector_field(
        ax,
        true_dynamics_from_metadata(nominal.metadata),
        grid_lim=panel_abs,
        n_grid=24,
        arrowsize=0.58,
        stroke_color=_experiment_C_STROKE,
    )
    ax.plot(
        nominal_traj[:, 0],
        nominal_traj[:, 1],
        color=_experiment_C_STROKE,
        linewidth=0.75,
        alpha=0.72,
        label="executed",
        zorder=4,
    )
    planned_trace = load_planned_trace(nominal.run_dir, nominal.metadata)
    planned_paths: list[np.ndarray] = []
    for step, color, label in (
        (40, _policy_color("active_planning_adaptive_u20_r20_h40"), "early plan"),
        (1000, _policy_color("active_myopic"), "late plan"),
    ):
        planned = planned_xy_cycle_for_step(planned_trace, step)
        if planned is not None:
            planned_paths.append(planned)
            ax.plot(
                planned[:, 0],
                planned[:, 1],
                color=color,
                linewidth=1.05,
                linestyle="--",
                alpha=0.92,
                label=label,
                zorder=5,
            )
    zoom_points = [nominal_traj[:, :2], *planned_paths]
    finite_points = np.concatenate(
        [arr[np.isfinite(arr).all(axis=1), :2] for arr in zoom_points if arr.size]
    )
    if finite_points.size:
        x_min, y_min = np.min(finite_points, axis=0)
        x_max, y_max = np.max(finite_points, axis=0)
        x_pad = max(0.6, 0.12 * float(x_max - x_min))
        y_pad = max(0.6, 0.12 * float(y_max - y_min))
        ax.set_xlim(max(-panel_abs, x_min - x_pad), min(panel_abs, x_max + x_pad))
        ax.set_ylim(max(-panel_abs, y_min - y_pad), min(panel_abs, y_max + y_pad))
    else:
        ax.set_xlim(-panel_abs, panel_abs)
        ax.set_ylim(-panel_abs, panel_abs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A. EIG plan in information geometry")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    _style_manuscript_axis(ax, grid_alpha=0.20)
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
    cbar.set_label(r"$\log\det I_z$")
    cbar.outline.set_linewidth(0.45)
    ax.legend(loc="lower right", fontsize=5.8, framealpha=0.78, borderpad=0.25)

    ax = axes[0, 1]
    event_specs = [
        (
            "adaptive_replan_reason",
            "parameter_update",
            "parameter replan",
            _policy_color(policy_id),
        ),
        (
            "adaptive_replan_reason",
            "state_tracking_error",
            "state-error replan",
            _policy_color("active_myopic"),
        ),
        ("parameter_update_reason", "max_interval", "interval update", _policy_color("ensemble")),
        (
            "parameter_update_reason",
            "block_eig",
            "block-EIG update",
            _policy_color("active_planning_u20_r20_h40"),
        ),
    ]
    event_steps: list[list[float]] = []
    for field, value, _label, _color in event_specs:
        event_steps.append(
            [
                float(step)
                for row in info_rows["Nominal"]
                if (step := _safe_float(row.get("step"))) is not None
                and str(row.get(field, "")) == value
            ]
        )
    ax.axvspan(0.0, 200.0, color=_experiment_C_NEUTRAL_FILL, alpha=0.72, linewidth=0.0)
    ax.axvline(200.0, color=_experiment_C_NEUTRAL_LIGHT, linewidth=0.7)
    ax.axvline(1000.0, color=_experiment_C_NEUTRAL_LIGHT, linewidth=0.7)
    ax.eventplot(
        event_steps,
        lineoffsets=np.arange(len(event_specs), dtype=np.float64),
        linelengths=0.62,
        colors=[color for _field, _value, _label, color in event_specs],
        linewidths=0.85,
    )
    ax.set_yticks(np.arange(len(event_specs), dtype=np.float64))
    ax.set_yticklabels([label for _field, _value, label, _color in event_specs], fontsize=6.3)
    ax.set_xlim(0.0, 2000.0)
    ax.set_ylim(-0.65, len(event_specs) - 0.35)
    ax.set_title("B. Adaptive cadence event timeline")
    ax.set_xlabel("Environment step")
    _style_manuscript_axis(ax, grid_axis="x")

    mismatch_specs = [
        ("Nominal", _policy_color("active_planning_adaptive_u20_r20_h40")),
        ("Obs. mismatch", _policy_color("active_myopic")),
        ("Param. mismatch", _policy_color("ensemble")),
    ]
    ax = axes[1, 0]
    rolling_values = []
    for label, color in mismatch_specs:
        steps, err = _asset_row_series(info_rows[label], "adaptive_state_tracking_error")
        rolled = _asset_rolling_mean(err, 75)
        rolling_values.extend(float(value) for value in rolled if np.isfinite(value))
        ax.plot(
            steps,
            rolled,
            color=color,
            linewidth=1.0,
            label=label,
        )
    if rolling_values:
        ax.set_ylim(-0.05, max(0.5, float(np.percentile(rolling_values, 98.0)) * 1.18))
    ax.set_title("C. Mismatch raises tracking error")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("State-tracking error")
    ax.legend(loc="upper right", fontsize=6.0)
    _style_experiment_axis(ax)

    ax = axes[1, 1]
    x = np.arange(len(mismatch_specs), dtype=np.float64)
    state_replans = []
    block_updates = []
    for label, _color in mismatch_specs:
        rows = info_rows[label]
        state_replans.append(
            sum(
                str(row.get("adaptive_replan_reason", "")) == "state_tracking_error" for row in rows
            )
        )
        block_updates.append(
            sum(str(row.get("parameter_update_reason", "")) == "block_eig" for row in rows)
        )
    ax.bar(
        x - 0.16,
        state_replans,
        width=0.30,
        color=_policy_color("active_myopic"),
        edgecolor=_experiment_C_STROKE,
        linewidth=0.35,
        label="state-error replans",
    )
    ax.bar(
        x + 0.16,
        block_updates,
        width=0.30,
        color=_policy_color("active_planning_u20_r20_h40"),
        edgecolor=_experiment_C_STROKE,
        linewidth=0.35,
        label="block-EIG updates",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(" mismatch", "\nmis.") for label, _color in mismatch_specs])
    ax.set_title("D. Mismatch-triggered adaptation")
    ax.set_ylabel("Trigger count")
    ax.legend(loc="upper left", fontsize=5.9)
    _style_manuscript_axis(ax, grid_axis="y")

    fig.tight_layout(w_pad=0.95, h_pad=0.95)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_plot_objective_ablation(output_path: Path) -> Path:
    sources = _experiment_objective_sources()
    _asset_require_suite_dirs([source.suite_dir for source in sources])
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
                source.suite_dir,
                policy_id,
                "value_final_mean",
            )
            r2, r2_sem, n_r2 = _experiment_metric_mean_sem(
                source.suite_dir,
                policy_id,
                "trajectory_r2_final_mean",
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
                    "policy_label": _asset_policy_label(policy_id),
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
    _write_csv(
        output_path.with_suffix(".csv"),
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
    plt_module = load_plotting(output_path, apply_style=_apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(2, 2, figsize=(7.25, 5.15), squeeze=False)
    for source_idx, source in enumerate(sources):
        ax_curve = axes[source_idx, 0]
        curves = curves_by_source[source.exp_id]
        for policy_id in _experiment_OBJECTIVE_POLICIES:
            curve_rows = curves.get(policy_id, [])
            if not curve_rows:
                continue
            steps = np.asarray([row["step"] for row in curve_rows], dtype=np.float64)
            values = np.asarray([row["value"] for row in curve_rows], dtype=np.float64)
            sem = np.asarray([row["sem"] for row in curve_rows], dtype=np.float64)
            color = _policy_color(policy_id)
            ax_curve.plot(
                steps,
                values,
                color=color,
                linewidth=0.95,
                label=_asset_policy_label(policy_id),
            )
            ax_curve.fill_between(
                steps,
                values - sem,
                values + sem,
                color=color,
                alpha=0.10,
                linewidth=0.0,
            )
        ax_curve.axhline(0.95, color=_experiment_C_NEUTRAL_LIGHT, linestyle="--", linewidth=0.75)
        ax_curve.set_xlabel("Environment step")
        ax_curve.set_ylabel("Predictive R2")
        ax_curve.set_ylim(-0.1, 1.05)
        ax_curve.set_title(f"{chr(65 + source_idx)}. {source.label}: recovery")
        _style_experiment_axis(ax_curve)

    x = np.arange(len(_experiment_OBJECTIVE_POLICIES), dtype=np.float64)
    width = 0.34
    offsets = np.linspace(-0.18, 0.18, len(sources))
    x_labels = [_asset_policy_label(policy_id) for policy_id in _experiment_OBJECTIVE_POLICIES]
    ax_final = axes[0, 1]
    ax_delta = axes[1, 1]
    for source_idx, source in enumerate(sources):
        source_rows = [row for row in metric_rows if row["experiment"] == source.exp_id]
        row_by_policy = {str(row["policy_id"]): row for row in source_rows}
        baseline = row_by_policy["active_planning_u20_r20_h40"]["trajectory_r2_mean"]
        final_r2 = []
        final_sem = []
        delta_r2 = []
        for policy_id in _experiment_OBJECTIVE_POLICIES:
            row = row_by_policy[policy_id]
            value = row["trajectory_r2_mean"]
            final_r2.append(np.nan if value is None else float(value))
            final_sem.append(
                0.0 if row["trajectory_r2_sem"] is None else float(row["trajectory_r2_sem"])
            )
            delta_r2.append(
                np.nan if value is None or baseline is None else float(value) - float(baseline)
            )
        color = _experiment_C_STROKE if source_idx == 0 else _experiment_C_NEUTRAL_LIGHT
        ax_final.bar(
            x + offsets[source_idx],
            final_r2,
            width=width,
            yerr=final_sem,
            color=color,
            edgecolor=_experiment_C_STROKE,
            linewidth=0.4,
            alpha=0.78,
            capsize=1.6,
            label=source.label,
        )
        ax_delta.bar(
            x + offsets[source_idx],
            delta_r2,
            width=width,
            color=color,
            edgecolor=_experiment_C_STROKE,
            linewidth=0.4,
            alpha=0.78,
            label=source.label,
        )
    ax_final.set_ylabel("Final predictive R2")
    ax_final.set_ylim(-0.05, 1.05)
    ax_final.set_title("C. Final ablation performance")
    ax_final.text(
        0.02,
        1.01,
        "dark: nominal, light: hard",
        transform=ax_final.transAxes,
        fontsize=6.0,
        va="bottom",
    )
    ax_delta.axhline(0.0, color=_experiment_C_STROKE, linewidth=0.7)
    ax_delta.set_ylabel("R2 change vs full EIG")
    ax_delta.set_title("D. Contribution relative to full EIG")
    for ax in (ax_final, ax_delta):
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=34, ha="right", fontsize=6.0)
        _style_manuscript_axis(ax, grid_axis="y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=6.2,
        columnspacing=0.9,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=0.85, h_pad=1.0)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_existing_dir(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing TBME result directory: " + " or ".join(str(path) for path in candidates)
    )


def _asset_plot_model_mismatch(output_path: Path) -> Path:
    from experiments.tbme import plot_exp08_exp09_mismatch_final as mismatch_final

    default_tbme_dir = _RESULTS_ROOT / "tbme"
    exp09_dir = _latest_session(
        _asset_existing_dir(
            [
                _figures._TBME_RESULTS_DIR / "exp09_observation_tuning_mismatch",
                default_tbme_dir / "exp09_observation_tuning_mismatch",
            ]
        )
    )
    exp08_dir = _latest_session(
        _asset_existing_dir(
            [
                _figures._TBME_RESULTS_DIR / "exp08_parameter_mismatch_stress",
                default_tbme_dir / "exp08_parameter_mismatch_stress",
            ]
        )
    )
    runs = mismatch_final._collect(exp09_dir) + mismatch_final._collect(exp08_dir)
    if not runs:
        raise RuntimeError("No exp08/exp09 mismatch runs found")
    events = mismatch_final._matched_events(runs, "observation") + mismatch_final._matched_events(
        runs,
        "parameter",
    )
    output_base = output_path.with_suffix("")
    mismatch_final._write_events(output_path.with_suffix(".events.csv"), events)
    mismatch_final._plot(runs, events, output_base)
    return output_path


def _assets_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare TBME manuscript asset assembly outputs.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(_figures.GROUPS),
        help="Comma-separated TBME groups to scan for component figures.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="TBME results root. Defaults to results/tbme.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_figures._TBME_RESULTS_DIR / "assets",
        help="Directory for assembled manuscript assets.",
    )
    return parser


def assets_main(argv: list[str] | None = None) -> int:
    """Generate TBME manuscript asset figures from existing result summaries."""
    args = _assets_build_parser().parse_args(argv)
    if args.results_dir is not None:
        _figures._set_tbme_results_dir(args.results_dir)
    group_ids = [item.strip() for item in str(args.groups).split(",") if item.strip()]
    unknown = sorted(set(group_ids) - set(_figures.GROUPS))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")
    if not group_ids:
        raise ValueError("At least one TBME group is required")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_groups = set(group_ids)
    asset_specs = [
        (
            "tbme_fig03_active_vs_baselines.pdf",
            {"exp01_base"},
            _asset_plot_active_vs_baselines,
        ),
        (
            "tbme_fig04_bottlenecks.pdf",
            {"exp01_base", "exp06_bottleneck"},
            _asset_plot_constraints,
        ),
        (
            "tbme_fig05_mechanism_cadence_mismatch.pdf",
            {"exp02_hard", "exp07_mismatch_stress", "exp08_parameter_mismatch_stress"},
            _asset_plot_mechanism,
        ),
        (
            "tbme_fig06_objective_ablation.pdf",
            {"exp05_ablation"},
            _asset_plot_objective_ablation,
        ),
        (
            "tbme_fig07_mismatch_adaptive_planning.pdf",
            {"exp08_parameter_mismatch_stress"},
            _asset_plot_model_mismatch,
        ),
    ]
    written: list[Path] = []
    skipped: list[tuple[str, set[str]]] = []
    for filename, required_groups, plotter in asset_specs:
        if not required_groups.issubset(selected_groups):
            skipped.append((filename, required_groups - selected_groups))
            continue
        written.append(plotter(output_dir / filename))

    lines = [
        "TBME manuscript asset assembly",
        "",
        "Generated assets:",
        *[str(path.relative_to(_REPO_ROOT)) for path in written],
        "",
    ]
    if skipped:
        lines += [
            "Skipped assets:",
            *[
                f"{filename}: missing groups {', '.join(sorted(missing))}"
                for filename, missing in skipped
            ],
            "",
        ]
    lines.append("Component roots:")
    for group_id in group_ids:
        lines.append(str(_overview_figures_dir(group_id).relative_to(_REPO_ROOT)))
        for ref in _figures.GROUPS[group_id]:
            suite_dir = ref.session_root / ref.suite_id
            lines.append(str((suite_dir / "summary" / "figures").relative_to(_REPO_ROOT)))
            lines.append(str((suite_dir / "experiment" / "figures").relative_to(_REPO_ROOT)))
    manifest = output_dir / "tbme_assets_manifest.txt"
    _write_text(manifest, "\n".join(lines) + "\n")
    for path in written:
        print(path)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(assets_main())
