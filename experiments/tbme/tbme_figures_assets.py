#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from actdyn.utils.experiment_runtime import read_trace_csv, safe_float as _safe_float
from actdyn.utils.figure_io import load_plotting, save_figure

from . import tbme_figures as _figures
from .tbme_figures import (
    _REPO_ROOT,
    _RESULTS_ROOT,
    _apply_style,
    _latest_session,
    _policy_color,
    _style_manuscript_axis,
    _suite_dir,
    _write_csv,
    _write_text,
)
from .tbme_figures_experiment import (
    _ExperimentRunRecord,
    _ExperimentSuiteSource,
    _experiment_C_NEUTRAL_FILL,
    _experiment_C_NEUTRAL_LIGHT,
    _experiment_C_STROKE,
    _experiment_OBJECTIVE_POLICIES,
    _experiment_curve_rows,
    _experiment_load_xy_trace,
    _experiment_make_information_grid,
    _experiment_metric_mean_sem,
    _experiment_objective_sources,
    _experiment_r2_threshold_step,
    _experiment_r2_threshold_times,
    _experiment_short_policy_label,
    _style_experiment_axis,
    plot_neutral_vector_field,
)
from .tbme_io import (
    load_planned_trace,
    planned_xy_cycle_for_step,
    true_dynamics_from_metadata,
)

# Manuscript asset assembly
_POLICY_LABELS = {
    "adaptive": "PALDI",
    "adaptive_async_anytime": "Async PALDI(anytime)",
    "adaptive_async_realtime": "Async PALDI",
    "active_planning": "Fixed PALDI",
    "active_myopic": "Myopic",
    "prbs": "PRBS",
    "random": "Random",
    "active_fully_observable": "Full obs.",
    "active_state_information": "State info",
    "active_dynamics": "Dyn. sens.",
    "active_e_optimality": "E-opt.",
    "active_observation_variance": "Obs. var.",
    "active_state_variance": "State var.",
    "flex": "FLEX",
    "rhc": "RHC-US",
    "off_policy": "Off-policy",
}
_ASSET_MATCHED_POLICIES = [
    "adaptive",
    "active_myopic",
    "flex",
    "rhc",
    "prbs",
    "random",
]
_ASSET_R2_THRESHOLDS = (0.90, 0.95, 0.99)


def _asset_policy_label(policy_id: str) -> str:
    return _POLICY_LABELS.get(policy_id, _experiment_short_policy_label(policy_id))


def _asset_require_suite_dirs(paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing TBME result suite(s): " + ", ".join(str(path) for path in missing)
        )


def _asset_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


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
    from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter

    curves = _experiment_curve_rows(
        suite_dir,
        "trajectory_r2_over_steps.csv",
        "trajectory_r2_mean",
    )
    curve_series = []
    for policy_id in policy_ids:
        rows = curves.get(policy_id, [])
        if not rows:
            continue
        steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
        values = np.asarray([row["value"] for row in rows], dtype=np.float64)
        sem = np.asarray([row["sem"] for row in rows], dtype=np.float64)
        color = _policy_color(policy_id)
        curve_series.append((steps, values, sem, color, _asset_policy_label(policy_id)))

    inset = ax.inset_axes([0.55, 0.13, 0.40, 0.40])
    for curve_ax, linewidth, alpha, labels in (
        (ax, 0.95, 0.10, True),
        (inset, 0.65, 0.08, False),
    ):
        for steps, values, sem, color, label in curve_series:
            curve_ax.plot(
                steps,
                values,
                color=color,
                linewidth=linewidth,
                label=label if labels else None,
            )
            curve_ax.fill_between(
                steps,
                values - sem,
                values + sem,
                color=color,
                alpha=alpha,
                linewidth=0.0,
            )
        for threshold in _ASSET_R2_THRESHOLDS:
            curve_ax.axhline(
                threshold,
                color=_experiment_C_NEUTRAL_LIGHT,
                linestyle="--",
                linewidth=0.55,
        )
        curve_ax.set_xlim(left=0.0)
        curve_ax.set_yscale("log", nonpositive="clip")
        curve_ax.set_ylim(0.1, 1.0)
        curve_ax.yaxis.set_major_locator(FixedLocator([0.1, 1.0]))
        curve_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        curve_ax.yaxis.set_minor_formatter(NullFormatter())
        _style_experiment_axis(curve_ax)
    inset.set_xlim(0.0, 500.0)
    inset.tick_params(axis="both", labelsize=5.2, pad=1.0)
    ax.set_title(title, pad=3.0, fontsize=9.5)
    ax.set_xlabel("Environment step")
    if ylabel:
        ax.set_ylabel("Predictive R2")


def _asset_plot_active_vs_baselines(output_path: Path) -> Path:
    sources = [
        _ExperimentSuiteSource(ref.suite_id, ref.label, ref.session_root / "tracks" / ref.suite_id)
        for ref in _figures.GROUPS["simple_system_identification"]
    ]
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    plt_module = load_plotting(output_path, apply_style=_apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(1, len(sources), figsize=(7.25, 2.35), squeeze=False)
    short_labels = {"Damped pendulum": "Pendulum", "Gated Duffing": "Gated"}
    for idx, source in enumerate(sources):
        label = short_labels.get(source.label, source.label)
        _asset_plot_r2_curves(
            axes[0, idx],
            source.suite_dir,
            _ASSET_MATCHED_POLICIES,
            title=f"{chr(65 + idx)}. {label}: recovery",
            ylabel=idx == 0,
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
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.75)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_bottleneck_sources() -> list[_ExperimentSuiteSource]:
    return [
        _ExperimentSuiteSource(
            "gated_duffing",
            "Nominal",
            _suite_dir("simple_system_identification", "gated_duffing"),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_observation_bottleneck_mild",
            "SNR -10",
            _suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_observation_bottleneck_mild",
            ),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_observation_bottleneck_strong",
            "SNR -15",
            _suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_observation_bottleneck_strong",
            ),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_action_bottleneck_mild",
            "Act. 0.75",
            _suite_dir("observation_action_bottleneck", "gated_duffing_action_bottleneck_mild"),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_action_bottleneck_strong",
            "Act. 0.50",
            _suite_dir("observation_action_bottleneck", "gated_duffing_action_bottleneck_strong"),
        ),
    ]


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
    policy_id = "adaptive"
    nominal = _asset_first_record("exp02_hard", "exp02_hard_gated_duffing", policy_id)
    obs_mismatch = _asset_first_record(
        "exp07_mismatch_stress",
        "exp07_gated_duffing_observation_mismatch_strong",
        policy_id,
    )
    param_mismatch = _asset_first_record(
        "exp08_parameter_mismatch_stress",
        "exp08_gated_duffing_parameter_mismatch_strong",
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
        (40, _policy_color("adaptive"), "early plan"),
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
        (
            "parameter_update_reason",
            "max_interval",
            "interval update",
            _policy_color("active_state_variance"),
        ),
        (
            "parameter_update_reason",
            "block_eig",
            "block-EIG update",
            _policy_color("active_planning"),
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
        ("Nominal", _policy_color("adaptive")),
        ("Obs. mismatch", _policy_color("active_myopic")),
        ("Param. mismatch", _policy_color("active_state_variance")),
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
        color=_policy_color("active_planning"),
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


def _asset_plot_method_comparison(
    output_path: Path,
    *,
    figure_title: str,
    sources: Sequence[_ExperimentSuiteSource],
    policy_ids: Sequence[str],
) -> Path:
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    threshold = 0.95
    metric_rows: list[dict[str, Any]] = []
    for source in sources:
        for policy_id in policy_ids:
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
    fig = plt_module.figure(figsize=(7.25, 4.55))
    gs = fig.add_gridspec(2, len(sources), height_ratios=[1.18, 1.0])
    curve_axes = [fig.add_subplot(gs[0, idx]) for idx in range(len(sources))]
    ax_bar = fig.add_subplot(gs[1, :])
    for source_idx, source in enumerate(sources):
        _asset_plot_r2_curves(
            curve_axes[source_idx],
            source.suite_dir,
            policy_ids,
            title=f"{chr(65 + source_idx)}. {source.label}: recovery",
            ylabel=source_idx == 0,
        )

    x = np.arange(len(policy_ids), dtype=np.float64) * 1.24
    width = 0.15
    offsets = (np.arange(len(sources), dtype=np.float64) - (len(sources) - 1) / 2.0) * 0.18
    x_labels = [_asset_policy_label(policy_id) for policy_id in policy_ids]
    bar_colors = (_experiment_C_STROKE, "#6F6A62", _experiment_C_NEUTRAL_LIGHT)
    for source_idx, source in enumerate(sources):
        source_rows = [row for row in metric_rows if row["experiment"] == source.exp_id]
        row_by_policy = {str(row["policy_id"]): row for row in source_rows}
        final_r2 = []
        final_sem = []
        for policy_id in policy_ids:
            row = row_by_policy[policy_id]
            value = row["trajectory_r2_mean"]
            final_r2.append(np.nan if value is None else float(value))
            final_sem.append(
                0.0 if row["trajectory_r2_sem"] is None else float(row["trajectory_r2_sem"])
            )
        color = bar_colors[source_idx % len(bar_colors)]
        ax_bar.bar(
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
    ax_bar.set_ylabel("Final predictive R2")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_title(f"{chr(65 + len(sources))}. {figure_title}: final performance")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(x_labels, rotation=34, ha="right", fontsize=6.0)
    ax_bar.legend(loc="upper left", fontsize=6.0, ncol=len(sources))
    _style_manuscript_axis(ax_bar, grid_axis="y")
    handles, labels = curve_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(4, len(policy_ids)),
        fontsize=6.2,
        columnspacing=0.9,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=0.75, h_pad=1.0)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_plot_objective_ablation(output_path: Path) -> Path:
    sources = [
        _ExperimentSuiteSource(source.exp_id, label, source.suite_dir)
        for source, label in zip(
            _experiment_objective_sources(),
            ("Nominal", "Asymmetric", "Challenging"),
        )
    ]
    return _asset_plot_method_comparison(
        output_path,
        figure_title="Objective ablation",
        sources=sources,
        policy_ids=_experiment_OBJECTIVE_POLICIES,
    )


def _asset_plot_constraints(output_path: Path) -> list[Path]:
    bottleneck_sources = _asset_bottleneck_sources()
    figures = (
        ("snr", "Observation SNR", tuple(bottleneck_sources[:3])),
        (
            "asymmetry",
            "Asymmetry",
            (
                _ExperimentSuiteSource(
                    "gated_duffing",
                    "Nominal",
                    _suite_dir("simple_system_identification", "gated_duffing"),
                ),
                _ExperimentSuiteSource(
                    "gated_duffing_asymmetric",
                    "Asymmetric",
                    _suite_dir("observation_action_bottleneck", "gated_duffing_asymmetric"),
                ),
            ),
        ),
        ("action", "Action budget", (bottleneck_sources[0], *bottleneck_sources[3:])),
    )
    written: list[Path] = []
    for suffix, figure_title, sources in figures:
        figure_path = output_path.with_name(
            f"{output_path.stem}_{suffix}{output_path.suffix}"
        )
        written.append(
            _asset_plot_method_comparison(
                figure_path,
                figure_title=figure_title,
                sources=sources,
                policy_ids=_ASSET_MATCHED_POLICIES,
            )
        )
    return written


def _asset_plot_eig_components(output_path: Path) -> Path:
    from experiments.eig_1d_example import main as _eig_1d_main

    return _eig_1d_main(["--output", str(output_path)])


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
        default=None,
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

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else _latest_session(_figures._TBME_RESULTS_DIR) / "assets"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_groups = set(group_ids)
    asset_specs = [
        (
            "tbme_eig_planning_components.pdf",
            set(),
            _asset_plot_eig_components,
        ),
        (
            "tbme_fig03_active_vs_baselines.pdf",
            {"simple_system_identification"},
            _asset_plot_active_vs_baselines,
        ),
        (
            "tbme_fig04_constraints.pdf",
            {"simple_system_identification", "observation_action_bottleneck"},
            _asset_plot_constraints,
        ),
        (
            "tbme_fig06_objective_ablation.pdf",
            {"objective_ablation"},
            _asset_plot_objective_ablation,
        ),
    ]
    written: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for filename, required_groups, plotter in asset_specs:
        if not required_groups.issubset(selected_groups):
            missing = required_groups - selected_groups
            skipped.append((filename, "missing groups " + ", ".join(sorted(missing))))
            continue
        try:
            result = plotter(output_dir / filename)
            if isinstance(result, Path):
                written.append(result)
            else:
                written.extend(result)
        except RuntimeError as exc:
            if "No trajectory R2 curves available" not in str(exc):
                raise
            skipped.append((filename, str(exc)))

    lines = [
        "TBME manuscript asset assembly",
        "",
        "Generated assets:",
        *[_asset_display_path(path) for path in written],
        "",
    ]
    if skipped:
        lines += [
            "Skipped assets:",
            *[f"{filename}: {reason}" for filename, reason in skipped],
            "",
        ]
    lines.append("Component roots:")
    for group_id in group_ids:
        for ref in _figures.GROUPS[group_id]:
            suite_dir = ref.session_root / "tracks" / ref.suite_id
            lines.append(_asset_display_path(suite_dir / "summary" / "figures"))
            lines.append(_asset_display_path(suite_dir / "experiment" / "figures"))
    manifest = output_dir / "tbme_assets_manifest.txt"
    _write_text(manifest, "\n".join(lines) + "\n")
    for path in written:
        print(path)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(assets_main())
