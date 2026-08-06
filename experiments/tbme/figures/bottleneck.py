"""Bottleneck-sweep figure family.

Final prediction R2 and threshold-crossing steps across observation/action
bottleneck conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from actdyn.utils.figure_io import load_plotting, save_figure

from . import artifacts, data, theme
from .groups import SuiteSource, suite_dir

BOTTLENECK_POLICIES = [
    "adaptive",
    "active_planning",
    "active_myopic",
    "active_state_variance",
    "prbs",
]

REQUIRED_SUITES = (
    ("simple_system_identification", "gated_duffing"),
    ("observation_action_bottleneck", "gated_duffing_observation_bottleneck_mild"),
    ("observation_action_bottleneck", "gated_duffing_observation_bottleneck_strong"),
    ("observation_action_bottleneck", "gated_duffing_action_bottleneck_mild"),
    ("observation_action_bottleneck", "gated_duffing_action_bottleneck_strong"),
)


def bottleneck_sources() -> list[SuiteSource]:
    return [
        SuiteSource(
            "gated_duffing",
            "Default",
            suite_dir("simple_system_identification", "gated_duffing"),
        ),
        SuiteSource(
            "gated_duffing_observation_bottleneck_mild",
            "Obs. mild",
            suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_observation_bottleneck_mild",
            ),
        ),
        SuiteSource(
            "gated_duffing_observation_bottleneck_strong",
            "Obs. strong",
            suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_observation_bottleneck_strong",
            ),
        ),
        SuiteSource(
            "gated_duffing_action_bottleneck_mild",
            "Action mild",
            suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_action_bottleneck_mild",
            ),
        ),
        SuiteSource(
            "gated_duffing_action_bottleneck_strong",
            "Action strong",
            suite_dir(
                "observation_action_bottleneck",
                "gated_duffing_action_bottleneck_strong",
            ),
        ),
    ]


@dataclass(frozen=True)
class BottleneckData:
    """Prepared inputs consumed by both the figure and its CSV sidecar."""

    sources: list[SuiteSource]
    rows: list[dict[str, Any]]
    policy_ids: Sequence[str] = field(default_factory=lambda: list(BOTTLENECK_POLICIES))


def prepare() -> BottleneckData:
    sources = bottleneck_sources()
    rows: list[dict[str, Any]] = []
    for source in sources:
        for policy_id in BOTTLENECK_POLICIES:
            r2, r2_sem, n_r2 = data.metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            step_to_r2 = data.r2_threshold_step(source.suite_dir, policy_id, threshold=0.90)
            rows.append(
                {
                    "experiment": source.exp_id,
                    "condition": source.label,
                    "policy_id": policy_id,
                    "policy_label": theme.short_policy_label(policy_id),
                    "trajectory_r2_mean": r2,
                    "trajectory_r2_sem": r2_sem,
                    "step_to_r2_0p90": step_to_r2,
                    "n_r2": n_r2,
                }
            )
    return BottleneckData(sources=sources, rows=rows)


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


def generate() -> list[Path]:
    """Load the bottleneck suites, render the figure, and write CSV sidecars."""
    prepared = prepare()
    suite_dirs = [source.suite_dir for source in prepared.sources]
    csv_paths = artifacts.write_csv_artifacts(
        suite_dirs,
        filename="tbme_experiment_bottleneck_sweep.csv",
        rows=prepared.rows,
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
    figure_paths = artifacts.artifact_paths(
        suite_dirs,
        subdir="figures",
        filename="tbme_experiment_bottleneck_sweep.pdf",
    )
    figure_path = plot_bottleneck_sweep(
        figure_paths[0],
        sources=prepared.sources,
        rows=prepared.rows,
        policy_ids=prepared.policy_ids,
        policy_label=theme.short_policy_label,
        policy_color=theme.policy_color,
        apply_style=theme.apply_style,
        style_axis=theme.style_experiment_axis,
    )
    return [*artifacts.copy_artifact(figure_path, figure_paths), *csv_paths]
