"""Per-parameter recovery figure family for gated-Duffing dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from actdyn.utils.figure_io import load_plotting, sample_sem, save_figure

from ..tbme_io import read_embedding_trace
from . import artifacts, theme
from .groups import suite_dir
from .records import collect_records, record_trace_path

RECOVERY_POLICIES = [
    "active_planning",
    "active_myopic",
    "active_state_variance",
    "flex",
    "prbs",
]


def aggregate_parameter_traces(
    suite_dir_path: Path,
    policy_ids: Sequence[str],
    *,
    max_seeds: int,
    stride: int,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], np.ndarray]:
    records = collect_records(suite_dir_path, policy_ids, max_seeds=max_seeds)
    if not records:
        raise RuntimeError(f"No records found under {suite_dir_path}")
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
        path = record_trace_path(
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


def generate(max_seeds: int) -> list[Path]:
    """Aggregate embedding traces and render the recovery figure."""
    suite_dir_path = suite_dir("simple_system_identification", "gated_duffing")
    traces, true_params = aggregate_parameter_traces(
        suite_dir_path,
        RECOVERY_POLICIES,
        max_seeds=max_seeds,
        stride=20,
    )
    if true_params.size < 4:
        return []
    figure_paths = artifacts.artifact_paths(
        [suite_dir_path],
        subdir="figures",
        filename="tbme_experiment_gated_duffing_parameter_recovery.pdf",
    )
    figure_path = plot_per_parameter_recovery(
        figure_paths[0],
        traces=traces,
        true_params=true_params,
        policy_ids=RECOVERY_POLICIES,
        sem=sample_sem,
        short_policy_label=theme.extended_policy_label,
        policy_color=theme.policy_color,
        apply_style=theme.apply_style,
        style_axis=theme.style_axis,
        stroke_color=theme.STROKE_COLOR,
    )
    return artifacts.copy_artifact(figure_path, figure_paths)
