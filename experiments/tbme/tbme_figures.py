#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from actdyn.environment.vectorfield import ResidualDynamicsCallable
from actdyn.utils.experiment_runtime import (
    read_trace_csv,
    safe_float as _safe_float,
    write_trace_csv,
)
from actdyn.utils.figure_io import (
    load_plotting,
    parse_figure_formats,
    sample_sem,
    save_figure,
)
from actdyn.utils.plotting import (
    apply_manuscript_figure_style,
    compute_vector_field,
    style_manuscript_axis,
)

from ..experiment_definitions import get_environment_preset, get_experiment_spec
from ..experiment_io import (
    experiment_env_slug,
    expected_loglinear_rate_hz,
    find_nested_metadata_paths,
    get_environment_preset_from_metadata,
    load_json,
    reconstruct_loglinear_rate_model,
)
from ..visualize import (
    plot_final_value_by_policy,
    plot_information_colormap,
    plot_metric_over_cpu_time,
    plot_metric_over_steps,
    plot_neuron_tuning_curve_colormap,
)
from .run_tbme_experiments import configure_tbme_catalogs, shared_tbme_group_suites
from .tbme_io import (
    load_planned_trace,
    planned_xy_cycle_for_step,
    read_embedding_trace,
    read_xy_trace as _read_xy_trace,
    trace_path as _tbme_trace_path,
    true_dynamics_from_metadata,
)

configure_tbme_catalogs()


# Shared configuration
_TBME_STROKE_COLOR = "#3A3A3A"
_TBME_GRID_COLOR = "#DDD7CE"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_TBME_RESULTS_DIR = _RESULTS_ROOT / "tbme" / "tracks"


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
    policy_ids: tuple[str, ...] = ()


def _tbme_label(env_preset_id: str) -> str:
    env_preset = get_environment_preset(env_preset_id)
    label = str(getattr(env_preset, "system_label", None) or env_preset.system_id)
    return label.replace("Asymmetric basin", "Gated Duffing")


def _build_groups(results_dir: Path) -> dict[str, list[SuiteRef]]:
    session_root = _latest_session(results_dir)
    groups: dict[str, list[SuiteRef]] = {}
    for group_name, entries in shared_tbme_group_suites().items():
        refs: list[SuiteRef] = []
        for entry in entries:
            env_preset_id = str(entry["env_preset_id"])
            refs.append(
                SuiteRef(
                    str(entry["suite_id"]),
                    _tbme_label(env_preset_id),
                    session_root,
                    experiment_env_slug(env_preset_id),
                    tuple(str(policy_id) for policy_id in entry["policy_ids"]),
                )
            )
        groups[group_name] = refs
    return groups


GROUPS: dict[str, list[SuiteRef]] = _build_groups(_TBME_RESULTS_DIR)


def _set_tbme_results_dir(results_dir: Path) -> None:
    global _TBME_RESULTS_DIR, GROUPS
    _TBME_RESULTS_DIR = Path(results_dir).resolve()
    GROUPS = _build_groups(_TBME_RESULTS_DIR)


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
    "adaptive": "Adaptive cadence",
    "adaptive_async_realtime": "Adaptive async realtime",
    "adaptive_async_anytime": "Adaptive async anytime",
    "adaptive_state_fixed_update": "Adaptive state fixed update",
    "adaptive_state": "Adaptive state error",
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
    "adaptive",
    "adaptive_async_realtime",
    "adaptive_async_anytime",
    "adaptive_state_fixed_update",
    "adaptive_state",
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
    "active_planning": "#5DADE2",
    "active_planning_u1_r1_h40": "#85C1E9",
    "active_planning_u5_r5_h40": "#73C6B6",
    "active_planning_u1_r5_h40": "#82E0AA",
    "active_planning_u10_r10_h40": "#BB8FCE",
    "active_planning_u5_r10_h40": "#76D7C4",
    "active_planning_u20_r20_h40": "#5DADE2",
    "active_planning_u5_r20_h40": "#AED6F1",
    "active_planning_u10_r20_h40": "#7FB3D5",
    "adaptive": "#F1948A",
    "adaptive_async_realtime": "#C85C5C",
    "adaptive_async_anytime": "#8E4B7D",
    "adaptive_state_fixed_update": "#D7BDE2",
    "adaptive_state": "#F8C471",
    "active_fully_observable_u20_r20_h40": "#82E0AA",
    "active_e_optimality_u20_r20_h40": "#BB8FCE",
    "active_state_information_u20_r20_h40": "#F7DC6F",
    "active_dynamics_u20_r20_h40": "#76D7C4",
    "active_sampling_variance_u20_r20_h40": "#D2B48C",
    "active_myopic": "#F5B041",
    "prbs": "#45B8AC",
    "random": "#9EA7AD",
    "flex": "#AF7AC5",
    "flex_true_state": "#58D68D",
    "ensemble": "#58D68D",
    "rhc": "#F06292",
}
FALLBACK_COLORS = (
    "#5DADE2",
    "#F1948A",
    "#58D68D",
    "#AF7AC5",
    "#F5B041",
    "#9EA7AD",
    "#45B8AC",
    "#F7DC6F",
)


# Shared TBME helpers


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


def _r2_threshold_suffix(threshold: float) -> str:
    return f"{float(threshold):.2f}".replace(".", "p")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    write_trace_csv(path, list(rows), list(fields))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        out.append(path)
        seen.add(path)
    return out


# TBME figure layout functions

def plot_r2_threshold_stacked_bars(
    output_path: Path,
    *,
    group_name: str,
    refs: Sequence[Any],
    threshold_rows: list[dict[str, object]],
    thresholds: Sequence[float],
    field_prefix: str,
    ylabel: str,
    title_metric: str,
    log_y: bool,
    threshold_suffix: Callable[[float], str],
    safe_float: Callable[[object], float | None],
    threshold_segments: Callable[[dict[str, object], str], tuple[list[float], bool]],
    threshold_value_penalty: Callable[[list[dict[str, object]], str], float],
    policy_threshold_sort_key: Callable[[str, Sequence[Any], dict[tuple[str, str], dict[str, object]], str, float], Any],
    short_policy_label: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    stroke_color: str,
    neutral_color: str,
    neutral_light: str,
    neutral_fill: str,
    segment_colors: Sequence[str],
) -> Path | None:
    """Plot stacked bars for first steps or CPU time to TBME R2 thresholds."""
    if not threshold_rows:
        return None
    plt_module = load_plotting(output_path, apply_style=apply_style, path_is_file=True)
    if plt_module is None:
        return None
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    row_by_key = {(str(row["suite_id"]), str(row["policy_id"])): row for row in threshold_rows}
    missing_penalty = threshold_value_penalty(threshold_rows, field_prefix)
    policy_ids = sorted(
        {str(row["policy_id"]) for row in threshold_rows},
        key=lambda policy_id: policy_threshold_sort_key(
            policy_id, refs, row_by_key, field_prefix, missing_penalty
        ),
    )
    if not policy_ids:
        return None
    positive_values = [
        value
        for row in threshold_rows
        for threshold in thresholds
        if (value := safe_float(row.get(f"{field_prefix}_{threshold_suffix(float(threshold))}")))
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
    fig, ax = plt_module.subplots(figsize=(fig_width, 3.45))

    for env_idx, ref in enumerate(refs):
        base = env_idx * (n_methods + group_gap)
        group_centers.append(base + (n_methods - 1) / 2.0)
        if env_idx > 0:
            ax.axvline(base - group_gap / 2.0, color=neutral_light, linewidth=0.7, alpha=0.85)
        if env_idx % 2 == 1:
            ax.axvspan(base - 0.62, base + n_methods - 0.38, color=neutral_fill, alpha=0.52)
        for method_idx, policy_id in enumerate(policy_ids):
            x = base + method_idx
            x_positions.append(x)
            x_labels.append(short_policy_label(policy_id))
            row = row_by_key.get((str(ref.suite_id), policy_id), {})
            segments, reached_all = threshold_segments(row, field_prefix)
            bottom = 0.0
            for seg_idx, segment in enumerate(segments):
                if segment <= 0.0:
                    continue
                ax.bar(
                    x,
                    segment,
                    width=bar_width,
                    bottom=bottom,
                    color=segment_colors[seg_idx],
                    edgecolor=stroke_color,
                    linewidth=0.35,
                    zorder=3,
                )
                bottom += segment
            max_height = max(max_height, bottom)
            if bottom == 0.0:
                ax.plot(
                    [x - bar_width / 2.0, x + bar_width / 2.0],
                    [log_floor, log_floor],
                    color=neutral_color,
                    linewidth=0.7,
                    zorder=4,
                )
            if not reached_all:
                ax.scatter(
                    [x],
                    [bottom if bottom > 0.0 else log_floor],
                    marker="x",
                    s=12,
                    color=stroke_color,
                    linewidths=0.6,
                    zorder=5,
                )

    for center, ref in zip(group_centers, refs, strict=True):
        ax.text(
            center,
            -0.31,
            str(ref.label),
            ha="center",
            va="top",
            color=stroke_color,
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
    ax.grid(axis="y", color=neutral_light, linewidth=0.35, alpha=0.38, zorder=1)
    for spine in ax.spines.values():
        spine.set_color(stroke_color)
        spine.set_linewidth(0.55)
    legend_handles = [
        Patch(facecolor=segment_colors[0], edgecolor=stroke_color, linewidth=0.35, label="0 -> 0.90"),
        Patch(facecolor=segment_colors[1], edgecolor=stroke_color, linewidth=0.35, label="0.90 -> 0.95"),
        Patch(facecolor=segment_colors[2], edgecolor=stroke_color, linewidth=0.35, label="0.95 -> 0.99"),
        Line2D(
            [0],
            [0],
            color=stroke_color,
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
    return save_figure(fig, output_path, plt_module=plt_module)


def plot_schedule_threshold_pareto(
    output_path: Path,
    *,
    rows: list[dict[str, object]],
    env_labels: Sequence[str],
    policy_ids: Sequence[str],
    thresholds: Sequence[float],
    threshold_suffix: Callable[[float], str],
    safe_float: Callable[[object], float | None],
    short_policy_label: Callable[[str], str],
    threshold_point_colors: Mapping[float, str],
    apply_style: Callable[[Any], None] | None,
    stroke_color: str,
    neutral_light: str,
    white_color: str = "#FFFFFF",
) -> Path | None:
    """Plot CPU-time versus sample-efficiency points for schedule thresholds."""
    if not rows:
        return None
    plt_module = load_plotting(output_path, apply_style=apply_style, path_is_file=True)
    if plt_module is None:
        return None
    from matplotlib.lines import Line2D

    policy_offsets = {
        policy_id: ((idx % 4) - 1.5, (idx // 4) - 0.5) for idx, policy_id in enumerate(policy_ids)
    }
    fig, axes = plt_module.subplots(1, len(env_labels), figsize=(8.2, 2.95), sharey=True)
    if len(env_labels) == 1:
        axes = [axes]

    max_step_seen = 1.0
    for ax, env_label in zip(axes, env_labels, strict=True):
        env_rows = [row for row in rows if str(row["suite_label"]) == env_label]
        plotted_for_policy: dict[str, tuple[float, float, float]] = {}
        max_cpu_seen = 1.0
        for row in env_rows:
            policy_id = str(row["policy_id"])
            marker = "D" if policy_id == "active_myopic" else "o"
            for threshold in thresholds:
                suffix = threshold_suffix(float(threshold))
                step = safe_float(row.get(f"step_to_r2_{suffix}"))
                cpu_time = safe_float(row.get(f"cpu_time_sec_to_r2_{suffix}"))
                if step is None or cpu_time is None:
                    continue
                ax.scatter(
                    cpu_time,
                    step,
                    s=24 if policy_id != "active_myopic" else 30,
                    marker=marker,
                    facecolor=threshold_point_colors[float(threshold)],
                    edgecolor=stroke_color,
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
                    plotted_for_policy[policy_id] = (cpu_time, step, float(threshold))
        for policy_id, (cpu_time, step, _threshold) in plotted_for_policy.items():
            dx, dy = policy_offsets.get(policy_id, (0.0, 0.0))
            ax.annotate(
                short_policy_label(policy_id),
                (cpu_time, step),
                xytext=(4.0 + 3.0 * dx, 3.0 + 3.0 * dy),
                textcoords="offset points",
                fontsize=5.8,
                color=stroke_color,
                ha="left",
                va="bottom",
                bbox={"facecolor": white_color, "edgecolor": "none", "alpha": 0.72, "pad": 0.25},
            )
        ax.set_title(env_label, fontsize=8.0, pad=3.0)
        ax.set_xlabel("CPU time (sec)")
        ax.set_xlim(left=0.0, right=max_cpu_seen * 1.13)
        ax.grid(color=neutral_light, linewidth=0.32, alpha=0.36)
        for spine in ax.spines.values():
            spine.set_color(stroke_color)
            spine.set_linewidth(0.55)
        ax.tick_params(width=0.45, length=2.0, colors=stroke_color)
    axes[0].set_ylabel("Environment steps")
    axes[0].set_ylim(bottom=0.0, top=max_step_seen * 1.16)
    fig.suptitle("Exp03 schedule Pareto: time and steps to trajectory R2 thresholds", y=0.99)
    threshold_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=threshold_point_colors[float(threshold)],
            markeredgecolor=stroke_color,
            markeredgewidth=0.45,
            markersize=5.0,
            label=f"R2 {threshold:.2f}",
        )
        for threshold in thresholds
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=white_color,
            markeredgecolor=stroke_color,
            markersize=5.0,
            label="Active schedule",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor=white_color,
            markeredgecolor=stroke_color,
            markersize=5.0,
            label="Myopic",
        ),
    ]
    fig.legend(
        handles=[*threshold_handles, *method_handles],
        loc="upper center",
        ncol=5,
        fontsize=6.4,
        bbox_to_anchor=(0.5, 0.905),
        columnspacing=0.9,
        handlelength=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.73, bottom=0.18, wspace=0.22)
    return save_figure(fig, output_path, plt_module=plt_module)


def _tbme_trajectory_layout(n_panels: int) -> tuple[int, int, tuple[float, float]]:
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


def _tbme_trajectory_plot_limit(
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


def _tbme_trajectory_seed_color_map(
    plt_module: Any, seeds: list[int]
) -> dict[int, tuple[float, float, float, float]]:
    if not seeds:
        return {}
    cmap = plt_module.get_cmap("turbo")
    denom = max(len(seeds) - 1, 1)
    return {seed: cmap(idx / denom) for idx, seed in enumerate(sorted(seeds))}


def _tbme_format_trajectory_axis(
    ax: Any,
    plot_lim: float,
    *,
    title: str,
    stroke_color: str,
    neutral_light: str,
) -> None:
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8.0, color=stroke_color, pad=2.0)
    ax.set_xlabel("x", labelpad=1.5)
    ax.set_ylabel("v", labelpad=1.5)
    ax.grid(color=neutral_light, linewidth=0.28, alpha=0.28)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(stroke_color)


def _tbme_plot_vectorfield_background(
    ax: Any,
    dyn_true: Any,
    plot_lim: float,
    *,
    neutral_light: str,
) -> None:
    x_grid, y_grid, u_grid, v_grid = compute_vector_field(
        dyn_true,
        x_range=plot_lim,
        n_grid=36,
        is_residual=True,
        device="cpu",
    )
    ax.streamplot(
        x_grid.cpu().numpy(),
        y_grid.cpu().numpy(),
        u_grid.cpu().numpy(),
        v_grid.cpu().numpy(),
        color=neutral_light,
        linewidth=0.34,
        density=1.35,
        arrowsize=0.55,
        zorder=1,
    )


def plot_trajectory_overlay(
    output_path: Path,
    *,
    suite_name: str,
    grouped: dict[str, list[tuple[int, np.ndarray]]],
    dyn_true: Any,
    grid_lim: float,
    system_label: str,
    max_seeds: int,
    policy_sort_key: Callable[[str], Any],
    policy_label: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    stroke_color: str,
    write_color: str,
    neutral_light: str,
) -> Path | None:
    """Plot trajectory overlays on the true vector field by policy."""
    if not grouped:
        return None
    plt_module = load_plotting(output_path, apply_style=apply_style, path_is_file=True)
    if plt_module is None:
        return None
    policies = sorted(grouped, key=policy_sort_key)
    plot_lim = _tbme_trajectory_plot_limit(grouped, grid_lim)
    seeds = sorted({seed for traces in grouped.values() for seed, _traj in traces})
    seed_colors = _tbme_trajectory_seed_color_map(plt_module, seeds)
    n_rows, n_cols, figsize = _tbme_trajectory_layout(len(policies))
    fig, axes = plt_module.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True, sharey=True
    )
    for idx, policy_id in enumerate(policies):
        ax = axes[idx // n_cols, idx % n_cols]
        _tbme_plot_vectorfield_background(ax, dyn_true, plot_lim, neutral_light=neutral_light)
        traces = grouped[policy_id]
        for seed, traj in traces:
            color = seed_colors.get(seed, write_color)
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.55, alpha=0.72, zorder=3)
            ax.scatter(
                traj[0, 0],
                traj[0, 1],
                s=5.0,
                color=color,
                edgecolors="none",
                alpha=0.95,
                zorder=4,
            )
        _tbme_format_trajectory_axis(
            ax,
            plot_lim,
            title=f"{policy_label(policy_id)}  n={len(traces)}",
            stroke_color=stroke_color,
            neutral_light=neutral_light,
        )
    for idx in range(len(policies), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle(
        f"{suite_name}: trajectory overlays on true {system_label} vector field "
        f"(first {max_seeds} seeds)",
        fontsize=9.0,
        color=stroke_color,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save_figure(fig, output_path, plt_module=plt_module)


def _tbme_trajectory_histogram(
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


def _tbme_trajectory_density_cmap(plt_module: Any) -> Any:
    try:
        import seaborn as sns

        return sns.color_palette("crest", as_cmap=True)
    except Exception:
        return plt_module.get_cmap("viridis")


def plot_trajectory_density(
    output_path: Path,
    *,
    suite_name: str,
    grouped: dict[str, list[tuple[int, np.ndarray]]],
    dyn_true: Any,
    grid_lim: float,
    system_label: str,
    max_seeds: int,
    bins: int,
    policy_sort_key: Callable[[str], Any],
    policy_label: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    stroke_color: str,
    neutral_light: str,
) -> Path | None:
    """Plot trajectory sample density by policy on the true state space."""
    if not grouped:
        return None
    plt_module = load_plotting(output_path, apply_style=apply_style, path_is_file=True)
    if plt_module is None:
        return None
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    policies = sorted(grouped, key=policy_sort_key)
    plot_lim = _tbme_trajectory_plot_limit(grouped, grid_lim)
    hists = {
        policy_id: _tbme_trajectory_histogram(grouped[policy_id], plot_lim, bins)
        for policy_id in policies
    }
    max_count = max((float(np.nanmax(hist)) for hist in hists.values() if hist.size), default=1.0)
    max_log_count = float(np.log10(max(max_count, 1.0) + 1.0))
    norm = Normalize(vmin=0.0, vmax=max_log_count)
    cmap = _tbme_trajectory_density_cmap(plt_module).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    n_rows, n_cols, figsize = _tbme_trajectory_layout(len(policies))
    fig, axes = plt_module.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True, sharey=True
    )
    im = None
    for idx, policy_id in enumerate(policies):
        ax = axes[idx // n_cols, idx % n_cols]
        _tbme_plot_vectorfield_background(ax, dyn_true, plot_lim, neutral_light=neutral_light)
        counts = hists[policy_id]
        hist = np.ma.masked_where(counts <= 0.0, np.log10(counts + 1.0))
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
        _tbme_format_trajectory_axis(
            ax,
            plot_lim,
            title=f"{policy_label(policy_id)}  n={len(grouped[policy_id])}",
            stroke_color=stroke_color,
            neutral_light=neutral_light,
        )
    for idx in range(len(policies), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle(
        f"{suite_name}: trajectory density on true {system_label} state space "
        f"(first {max_seeds} seeds)",
        fontsize=9.0,
        color=stroke_color,
        y=0.995,
    )
    fig.subplots_adjust(left=0.065, right=0.895, bottom=0.075, top=0.91, wspace=0.22, hspace=0.32)
    if im is None:
        im = ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.915, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log10(1 + trajectory samples per bin)", color=stroke_color)
    cbar.outline.set_linewidth(0.45)
    return save_figure(fig, output_path, plt_module=plt_module)


# Summary output
_summary_trace_C_WRITE = "#1F4FA8"
_summary_trace_C_STROKE = "#3A3A3A"
_summary_trace_C_NEUTRAL_LIGHT = "#C8C1B8"


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
    track_root = suite_dir
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
    colormap_stems: list[str] = []
    try:
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
        colormap_stems = ["neuron_tuning_curve_colormap", "I_z_t_colormap"]
    except ImportError as exc:
        if "neurofisherSNR" not in str(exc):
            raise
        print(f"Skipping log-linear colormaps for {suite_dir}: {exc}", file=sys.stderr)
    stems = [
        f"final_{value_prefix}_by_policy",
        f"{value_prefix}_over_steps",
        f"{value_prefix}_over_cpu_time",
        "trajectory_r2_over_steps",
        "trajectory_r2_over_cpu_time",
        "parameter_covariance_trace_over_steps",
        *colormap_stems,
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
        plot_trajectory_overlay(
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
        plot_trajectory_density(
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
    track_dir = suite_dir
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
    candidates = sorted(suite_dir.glob("*/seed_*/repeat_*/run_metadata.json"))
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
    figure_formats = parse_figure_formats(str(args.figure_formats))

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


# Group overview output
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


def _group_session_root(group_name: str, refs: Sequence[SuiteRef] | None = None) -> Path:
    group_refs = list(GROUPS[group_name] if refs is None else refs)
    if not group_refs:
        return _latest_session(_TBME_RESULTS_DIR / group_name)
    session_roots = {ref.session_root for ref in group_refs}
    if len(session_roots) != 1:
        roots = ", ".join(str(path) for path in sorted(session_roots))
        raise ValueError(f"Expected one session root for {group_name}; found {roots}")
    return next(iter(session_roots))


def _overview_dir(group_name: str, refs: Sequence[SuiteRef] | None = None) -> Path:
    return _group_session_root(group_name, refs) / "overview" / group_name


def _overview_figures_dir(group_name: str, refs: Sequence[SuiteRef] | None = None) -> Path:
    return _overview_dir(group_name, refs) / "figures"


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
    policy_filter = set(ref.policy_ids) if ref.policy_ids else None
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        pid = str(row["policy_id"])
        if policy_filter is not None and pid not in policy_filter:
            continue
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
    policy_filter = set(ref.policy_ids) if ref.policy_ids else None
    if threshold_path.exists():
        rows = read_trace_csv(threshold_path)
        out: list[dict[str, object]] = []
        for row in rows:
            if policy_filter is not None and str(row["policy_id"]) not in policy_filter:
                continue
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
        if policy_filter is not None and policy_id not in policy_filter:
            continue
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
    apply_manuscript_figure_style(plt, stroke_color=_overview_C_STROKE)


def _overview_plot_schedule_threshold_pareto() -> Path | None:
    schedule_rows: list[dict[str, object]] = []
    for ref in GROUPS["scheduling"]:
        schedule_rows.extend(
            row
            for row in _overview_threshold_rows_for_suite(ref)
            if str(row["policy_id"]).startswith("active_planning")
        )
    myopic_rows: list[dict[str, object]] = []
    for ref in GROUPS["simple_system_identification"]:
        myopic_rows.extend(
            row
            for row in _overview_threshold_rows_for_suite(ref)
            if str(row["policy_id"]) == "active_myopic"
        )
    rows = [*schedule_rows, *myopic_rows]
    if not rows:
        return None

    env_labels = [ref.label for ref in GROUPS["scheduling"]]
    policy_ids = sorted({str(row["policy_id"]) for row in rows}, key=_policy_sort_key)
    out_path = (
        _overview_figures_dir("scheduling", GROUPS["scheduling"])
        / "r2_threshold_pareto_step_cpu_by_environment.pdf"
    )
    return plot_schedule_threshold_pareto(
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
    overview_dir = _overview_dir(group_name, refs)
    figures_dir = overview_dir / "figures"
    csv_path = overview_dir / "overview_table.csv"
    tex_path = overview_dir / "overview_table.tex"
    _write_csv(
        csv_path,
        rows,
        [
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
        ],
    )
    _overview_write_tex(tex_path, rows)
    written = [csv_path, tex_path]
    threshold_plots = [
        plot_r2_threshold_stacked_bars(
            figures_dir / "r2_threshold_stacked_steps_by_environment.pdf",
            group_name=group_name,
            refs=refs,
            threshold_rows=threshold_rows,
            thresholds=_overview_R2_THRESHOLDS,
            field_prefix="step_to_r2",
            ylabel="Environment steps",
            title_metric="steps",
            log_y=False,
            threshold_suffix=_r2_threshold_suffix,
            safe_float=_safe_float,
            threshold_segments=lambda row, prefix: _overview_threshold_segments(
                row, field_prefix=prefix
            ),
            threshold_value_penalty=lambda penalty_rows, prefix: _overview_threshold_value_penalty(
                penalty_rows, field_prefix=prefix
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
        ),
        plot_r2_threshold_stacked_bars(
            figures_dir / "r2_threshold_stacked_cpu_time_by_environment.pdf",
            group_name=group_name,
            refs=refs,
            threshold_rows=threshold_rows,
            thresholds=_overview_R2_THRESHOLDS,
            field_prefix="cpu_time_sec_to_r2",
            ylabel="CPU time (sec)",
            title_metric="CPU time",
            log_y=True,
            threshold_suffix=_r2_threshold_suffix,
            safe_float=_safe_float,
            threshold_segments=lambda row, prefix: _overview_threshold_segments(
                row, field_prefix=prefix
            ),
            threshold_value_penalty=lambda penalty_rows, prefix: _overview_threshold_value_penalty(
                penalty_rows, field_prefix=prefix
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
        ),
    ]
    written.extend(path for path in threshold_plots if path is not None)
    return rows, written


def _group_overview_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write group-level TBME overview tables and figures into each group session."
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

    if "scheduling" in groups:
        pareto_path = _overview_plot_schedule_threshold_pareto()
        if pareto_path is not None:
            written_by_group.setdefault("scheduling", []).append(pareto_path)

    for group_name in groups:
        written = written_by_group.get(group_name, [])
        print(f"{group_name}: {row_counts[group_name]} table rows, {len(written)} overview files")
        for path in written:
            print(path)
    return 0


# Experiment manuscript output
_experiment_C_STROKE = "#3A3A3A"
_experiment_C_NEUTRAL_LIGHT = "#C8C1B8"
_experiment_C_NEUTRAL_FILL = "#F4F1EC"
_experiment_C_GRID = "#DDD7CE"

_experiment_BOTTLENECK_POLICIES = [
    "adaptive",
    "active_planning_u20_r20_h40",
    "active_myopic",
    "ensemble",
    "prbs",
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
    "adaptive",
    "active_planning_u20_r20_h40",
    "active_myopic",
    "ensemble",
    "prbs",
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
    "true_dynamics_all",
    "per_parameter_recovery",
)
EXPERIMENT_PLOTS = _experiment_PLOTS

_experiment_OBJECTIVE_DEFINITION_PLOTS = {"objective_ablation"}
_experiment_REQUIRED_SUITES_BY_PLOT = {
    "bottleneck_sweep": (
        ("simple_system_identification", "gated_duffing"),
        ("observation_action_bottleneck", "gated_duffing_observation_bottleneck_mild"),
        ("observation_action_bottleneck", "gated_duffing_observation_bottleneck_strong"),
        ("observation_action_bottleneck", "gated_duffing_action_bottleneck_mild"),
        ("observation_action_bottleneck", "gated_duffing_action_bottleneck_strong"),
    ),
    "objective_ablation": (
        ("objective_ablation", "gated_duffing"),
        ("objective_ablation", "gated_duffing_hard"),
    ),
    "mismatch_dose_response": (
        ("simple_system_identification", "duffing"),
        ("simple_system_identification", "gated_duffing"),
        ("model_mismatch", "duffing_parameter_mismatch"),
        ("model_mismatch", "gated_duffing_parameter_mismatch"),
        ("model_mismatch", "duffing_parameter_mismatch_mild"),
        ("model_mismatch", "duffing_parameter_mismatch_strong"),
        ("model_mismatch", "gated_duffing_parameter_mismatch_mild"),
        ("model_mismatch", "gated_duffing_parameter_mismatch_strong"),
    ),
}


def _experiment_artifact_paths(
    suite_dirs: Sequence[Path],
    *,
    subdir: str,
    filename: str,
) -> list[Path]:
    return _unique_paths(
        suite_dir / "experiment" / subdir / filename for suite_dir in suite_dirs
    )


def _experiment_write_csv_artifacts(
    suite_dirs: Sequence[Path],
    *,
    filename: str,
    rows: Iterable[dict[str, Any]],
    fields: Sequence[str],
) -> list[Path]:
    paths = _experiment_artifact_paths(suite_dirs, subdir="tables", filename=filename)
    row_list = list(rows)
    for path in paths:
        _write_csv(path, row_list, fields)
    return paths


def _experiment_write_text_artifacts(
    suite_dirs: Sequence[Path],
    *,
    filename: str,
    text: str,
) -> list[Path]:
    paths = _experiment_artifact_paths(suite_dirs, subdir="tables", filename=filename)
    for path in paths:
        _write_text(path, text)
    return paths


def _experiment_copy_artifact(source_path: Path, paths: Sequence[Path]) -> list[Path]:
    for path in paths:
        if path == source_path:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, path)
    return list(paths)


def _style_experiment_axis(ax: Any) -> None:
    _style_manuscript_axis(ax, grid_alpha=0.55)


def plot_bottleneck_sweep(
    output_path: Path,
    *,
    sources: Sequence[Any],
    rows: list[dict[str, Any]],
    policy_ids: Sequence[str],
    policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
) -> Path:
    """Plot final prediction and threshold steps for bottleneck conditions."""
    plt_module = load_plotting(output_path, apply_style=apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(1, 2, figsize=(7.8, 2.95), sharex=True)
    x = np.arange(len(sources), dtype=np.float64)
    offsets = np.linspace(-0.30, 0.30, len(policy_ids))
    max_step = 1.0
    finite_r2: list[float] = []
    for idx, policy_id in enumerate(policy_ids):
        color = policy_color(policy_id)
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
            r2_y.append(np.nan if match["trajectory_r2_mean"] is None else match["trajectory_r2_mean"])
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
            label=policy_label(policy_id),
        )
        finite_r2.extend(float(v) for v in r2_y if np.isfinite(v))
        axes[1].plot(
            xpos,
            step_y,
            marker="o",
            color=color,
            linewidth=1.0,
            markersize=3.4,
            label=policy_label(policy_id),
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
        style_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([source.label for source in sources], rotation=22, ha="right")
    axes[0].set_ylabel("Final prediction R2")
    axes[0].set_ylim(min(-0.1, min(finite_r2) - 0.05) if finite_r2 else -0.1, 1.05)
    axes[0].set_title("A. Prediction under bottlenecks")
    axes[1].set_ylabel("Steps to prediction R2 >= 0.90")
    axes[1].set_ylim(0.0, max_step * 1.15)
    axes[1].set_title("B. Predictive sample efficiency")
    axes[1].legend(loc="upper left", fontsize=6.6, ncol=1)
    fig.tight_layout(w_pad=1.1)
    return save_figure(fig, output_path, plt_module=plt_module)


def plot_objective_ablation(
    output_path: Path,
    *,
    sources: Sequence[Any],
    metric_rows: list[dict[str, Any]],
    curves_by_source: dict[str, dict[str, list[dict[str, float]]]],
    policy_ids: Sequence[str],
    policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
    stroke_color: str,
    neutral_light: str,
) -> Path:
    """Plot objective-ablation threshold bars and prediction-R2 recovery curves."""
    plt_module = load_plotting(output_path, apply_style=apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(len(sources), 2, figsize=(7.15, 5.05), squeeze=False)
    x = np.arange(len(policy_ids), dtype=np.float64)
    x_labels = [policy_label(policy_id) for policy_id in policy_ids]
    letters = ["A", "B", "C", "D"]
    for source_idx, source in enumerate(sources):
        row_metrics = [row for row in metric_rows if row["experiment"] == source.exp_id]
        bars = [
            np.nan if row["step_to_r2_0p95"] is None else row["step_to_r2_0p95"]
            for row in row_metrics
        ]
        colors = [policy_color(str(row["policy_id"])) for row in row_metrics]
        ax_bar = axes[source_idx, 0]
        ax_curve = axes[source_idx, 1]
        ax_bar.bar(x, bars, color=colors, edgecolor=stroke_color, linewidth=0.45)
        finite_steps = [float(v) for v in bars if np.isfinite(v)]
        max_step = max(finite_steps) if finite_steps else 1.0
        missing_x = [float(x[idx]) for idx, value in enumerate(bars) if not np.isfinite(value)]
        if missing_x:
            ax_bar.scatter(
                missing_x,
                [max_step * 1.05 for _ in missing_x],
                marker="x",
                s=15,
                color=stroke_color,
                linewidths=0.8,
            )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(x_labels, rotation=35, ha="right")
        ax_bar.set_ylabel("Steps to prediction R2 >= 0.95")
        ax_bar.set_ylim(0.0, max_step * 1.18)
        ax_bar.set_title(f"{letters[2 * source_idx]}. {source.label}: threshold")
        style_axis(ax_bar)

        curves = curves_by_source[source.exp_id]
        for policy_id in policy_ids:
            curve_rows = curves.get(policy_id, [])
            if not curve_rows:
                continue
            steps = np.asarray([row["step"] for row in curve_rows], dtype=np.float64)
            values = np.asarray([row["value"] for row in curve_rows], dtype=np.float64)
            sem = np.asarray([row["sem"] for row in curve_rows], dtype=np.float64)
            color = policy_color(policy_id)
            ax_curve.plot(steps, values, color=color, linewidth=1.0, label=policy_label(policy_id))
            if np.any(sem > 0):
                ax_curve.fill_between(
                    steps, values - sem, values + sem, color=color, alpha=0.14, linewidth=0
                )
        ax_curve.axhline(0.95, color=neutral_light, linestyle="--", linewidth=0.7)
        ax_curve.set_xlabel("Environment step")
        ax_curve.set_ylabel("Prediction R2")
        ax_curve.set_ylim(-0.1, 1.05)
        ax_curve.set_title(f"{letters[2 * source_idx + 1]}. {source.label}: recovery")
        style_axis(ax_curve)
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
    return save_figure(fig, output_path, plt_module=plt_module)


def plot_mismatch_dose_response(
    output_path: Path,
    *,
    rows: list[dict[str, Any]],
    policy_ids: Sequence[str],
    policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
) -> Path:
    """Plot final prediction R2 as a function of model-mismatch dose."""
    plt_module = load_plotting(output_path, apply_style=apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    dose_order = ["none", "mild", "medium", "strong"]
    dose_labels = ["None", "Mild", "Medium", "Strong"]
    x = np.arange(len(dose_order), dtype=np.float64)
    fig, axes = plt_module.subplots(1, 2, figsize=(7.1, 2.9), sharey=False)
    for ax, family in zip(axes, ["Duffing", "Gated Duffing"], strict=True):
        family_rows = [row for row in rows if row["family"] == family]
        for policy_id in policy_ids:
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
                color=policy_color(policy_id),
                label=policy_label(policy_id),
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
        ax.set_ylim(min(-0.1, min(finite_family_r2) - 0.05) if finite_family_r2 else -0.1, 1.05)
        ax.set_title(f"{family} mismatch dose-response")
        style_axis(ax)
    axes[1].legend(loc="upper left", fontsize=6.4)
    fig.tight_layout(w_pad=1.0)
    return save_figure(fig, output_path, plt_module=plt_module)


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
            ax.plot(steps, means_arr, color=color, linewidth=1.0, label=short_policy_label(policy_id))
            ax.fill_between(
                steps,
                means_arr - sems_arr,
                means_arr + sems_arr,
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )
        ax.axhline(float(true_params[param_idx]), color=stroke_color, linewidth=0.8, linestyle="--")
        ax.set_title(f"{chr(65 + param_idx)}. {names[param_idx]}")
        ax.set_ylabel("Estimate")
        style_axis(ax)
    axes[1, 0].set_xlabel("Environment step")
    axes[1, 1].set_xlabel("Environment step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=6.4, bbox_to_anchor=(0.5, 1.015))
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
    return float(np.mean(values)), sample_sem(values), len(values)


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


def _experiment_plot_bottleneck_sweep() -> list[Path]:
    sources = [
        _ExperimentSuiteSource(
            "gated_duffing",
            "Nominal",
            _suite_dir("simple_system_identification", "gated_duffing"),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_observation_bottleneck_mild",
            "Obs. mild",
            _suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_observation_bottleneck_mild",
            ),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_observation_bottleneck_strong",
            "Obs. strong",
            _suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_observation_bottleneck_strong",
            ),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_action_bottleneck_mild",
            "Action mild",
            _suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_action_bottleneck_mild",
            ),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_action_bottleneck_strong",
            "Action strong",
            _suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_action_bottleneck_strong",
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

    suite_dirs = [source.suite_dir for source in sources]
    csv_paths = _experiment_write_csv_artifacts(
        suite_dirs,
        filename="tbme_experiment_bottleneck_sweep.csv",
        rows=rows,
        fields=[
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
    figure_paths = _experiment_artifact_paths(
        suite_dirs,
        subdir="figures",
        filename="tbme_experiment_bottleneck_sweep.pdf",
    )
    figure_path = plot_bottleneck_sweep(
        figure_paths[0],
        sources=sources,
        rows=rows,
        policy_ids=_experiment_BOTTLENECK_POLICIES,
        policy_label=_experiment_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_experiment_axis,
    )
    return [*_experiment_copy_artifact(figure_path, figure_paths), *csv_paths]


def _experiment_objective_sources() -> list[_ExperimentSuiteSource]:
    return [
        _ExperimentSuiteSource(
            "gated_duffing",
            "Nominal gated Duffing",
            _suite_dir("objective_ablation", "gated_duffing"),
        ),
        _ExperimentSuiteSource(
            "gated_duffing_hard",
            "Hard gated Duffing",
            _suite_dir("objective_ablation", "gated_duffing_hard"),
        ),
    ]


def _experiment_write_objective_definition_tables(suite_dirs: Sequence[Path]) -> list[Path]:
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
    written = _experiment_write_csv_artifacts(
        suite_dirs,
        filename="tbme_experiment_objective_ablation_objectives.csv",
        rows=rows,
        fields=[
            "policy_id",
            "policy_label",
            "objective_name",
            "objective_formula",
            "objective_notes",
        ],
    )

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
    written.extend(
        _experiment_write_text_artifacts(
            suite_dirs,
            filename="tbme_experiment_objective_ablation_objectives.tex",
            text="\n".join(lines) + "\n",
        )
    )
    return written


def _experiment_plot_objective_ablation() -> list[Path]:
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

    suite_dirs = [source.suite_dir for source in sources]
    csv_paths = _experiment_write_csv_artifacts(
        suite_dirs,
        filename="tbme_experiment_objective_ablation.csv",
        rows=metric_rows,
        fields=[
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
    figure_paths = _experiment_artifact_paths(
        suite_dirs,
        subdir="figures",
        filename="tbme_experiment_objective_ablation_gated_duffing.pdf",
    )
    figure_path = plot_objective_ablation(
        figure_paths[0],
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
    return [*_experiment_copy_artifact(figure_path, figure_paths), *csv_paths]


def _experiment_dose_sources() -> list[_ExperimentSuiteSource]:
    return [
        _ExperimentSuiteSource(
            "duffing",
            "None",
            _suite_dir("simple_system_identification", "duffing"),
            dose="none",
            family="Duffing",
        ),
        _ExperimentSuiteSource(
            "duffing_parameter_mismatch_mild",
            "Mild",
            _suite_dir("model_mismatch", "duffing_parameter_mismatch_mild"),
            dose="mild",
            family="Duffing",
        ),
        _ExperimentSuiteSource(
            "duffing_parameter_mismatch",
            "Medium",
            _suite_dir("model_mismatch", "duffing_parameter_mismatch"),
            dose="medium",
            family="Duffing",
        ),
        _ExperimentSuiteSource(
            "duffing_parameter_mismatch_strong",
            "Strong",
            _suite_dir("model_mismatch", "duffing_parameter_mismatch_strong"),
            dose="strong",
            family="Duffing",
        ),
        _ExperimentSuiteSource(
            "gated_duffing",
            "None",
            _suite_dir("simple_system_identification", "gated_duffing"),
            dose="none",
            family="Gated Duffing",
        ),
        _ExperimentSuiteSource(
            "gated_duffing_parameter_mismatch_mild",
            "Mild",
            _suite_dir("model_mismatch", "gated_duffing_parameter_mismatch_mild"),
            dose="mild",
            family="Gated Duffing",
        ),
        _ExperimentSuiteSource(
            "gated_duffing_parameter_mismatch",
            "Medium",
            _suite_dir("model_mismatch", "gated_duffing_parameter_mismatch"),
            dose="medium",
            family="Gated Duffing",
        ),
        _ExperimentSuiteSource(
            "gated_duffing_parameter_mismatch_strong",
            "Strong",
            _suite_dir("model_mismatch", "gated_duffing_parameter_mismatch_strong"),
            dose="strong",
            family="Gated Duffing",
        ),
    ]


def _experiment_plot_mismatch_dose_response() -> list[Path]:
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

    suite_dirs = [source.suite_dir for source in sources]
    csv_paths = _experiment_write_csv_artifacts(
        suite_dirs,
        filename="tbme_experiment_mismatch_dose_response.csv",
        rows=rows,
        fields=[
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
    figure_paths = _experiment_artifact_paths(
        suite_dirs,
        subdir="figures",
        filename="tbme_experiment_mismatch_dose_response.pdf",
    )
    figure_path = plot_mismatch_dose_response(
        figure_paths[0],
        rows=rows,
        policy_ids=_experiment_DOSE_POLICIES,
        policy_label=_experiment_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_experiment_axis,
    )
    return [*_experiment_copy_artifact(figure_path, figure_paths), *csv_paths]


def _experiment_collect_records(
    suite_dir: Path,
    policy_ids: Sequence[str],
    *,
    max_seeds: int | None = None,
    completed_only: bool = False,
) -> list[_ExperimentRunRecord]:
    records: list[_ExperimentRunRecord] = []
    for policy_id in sorted(policy_ids, key=_policy_sort_key):
        policy_dir = suite_dir / policy_id
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
        ref.session_root / ref.suite_id
        for group_id in group_ids
        for ref in GROUPS[group_id]
    )


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
    return _tbme_trace_path(record.run_dir, record.metadata, metadata_key, fallback_name)


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
    loading_seed = metadata.get("observation_loading_seed")
    if loading_seed is None:
        loading_seed = metadata.get("seed", 0)
    return (
        int(loading_seed),
        int(metadata.get("loading_snr_trajectory_seed", 0)),
        str(metadata.get("env_preset_id", "")),
        int(metadata.get("observation_dim", 20)),
        int(metadata.get("latent_dim", 2)),
        float(metadata.get("dt", 0.01)),
        float(metadata.get("mean_firing_rate_target", 10.0)),
        float(metadata.get("max_firing_rate_target", 100.0)),
        _safe_float(metadata.get("loading_target_snr_db")),
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
        ("tbme_asymmetric_basin", "Gated Duffing"),
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
    for panel_idx, (ax, (title, x_np, y_np, u_np, v_np, log_speed)) in enumerate(
        zip(axes, fields)
    ):
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
    policy_ids = ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "flex", "prbs"]
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
        sem=_sem,
        short_policy_label=_experiment_short_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
        stroke_color=_experiment_C_STROKE,
    )
    return _experiment_copy_artifact(figure_path, figure_paths)


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


def assets_main(argv: list[str] | None = None) -> int:
    """Run TBME manuscript asset assembly via the split asset module."""
    from .tbme_assets import assets_main as _assets_main

    return _assets_main(argv)
