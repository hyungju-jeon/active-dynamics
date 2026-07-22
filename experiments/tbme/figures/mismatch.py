"""Mismatch dose-response figure family.

Final prediction R2 as a function of model-mismatch dose for the Duffing and
gated-Duffing families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from actdyn.utils.figure_io import load_plotting, save_figure

from . import artifacts, data, theme
from .groups import SuiteSource, suite_dir

DOSE_POLICIES = [
    "adaptive",
    "active_planning",
    "active_myopic",
    "active_state_variance",
    "prbs",
]

REQUIRED_SUITES = (
    ("simple_system_identification", "duffing"),
    ("simple_system_identification", "gated_duffing"),
    ("model_mismatch", "duffing_parameter_mismatch"),
    ("model_mismatch", "gated_duffing_parameter_mismatch"),
    ("model_mismatch", "duffing_parameter_mismatch_mild"),
    ("model_mismatch", "duffing_parameter_mismatch_strong"),
    ("model_mismatch", "gated_duffing_parameter_mismatch_mild"),
    ("model_mismatch", "gated_duffing_parameter_mismatch_strong"),
)


def dose_sources() -> list[SuiteSource]:
    return [
        SuiteSource(
            "duffing",
            "None",
            suite_dir("simple_system_identification", "duffing"),
            dose="none",
            family="Duffing",
        ),
        SuiteSource(
            "duffing_parameter_mismatch_mild",
            "Mild",
            suite_dir("model_mismatch", "duffing_parameter_mismatch_mild"),
            dose="mild",
            family="Duffing",
        ),
        SuiteSource(
            "duffing_parameter_mismatch",
            "Medium",
            suite_dir("model_mismatch", "duffing_parameter_mismatch"),
            dose="medium",
            family="Duffing",
        ),
        SuiteSource(
            "duffing_parameter_mismatch_strong",
            "Strong",
            suite_dir("model_mismatch", "duffing_parameter_mismatch_strong"),
            dose="strong",
            family="Duffing",
        ),
        SuiteSource(
            "gated_duffing",
            "None",
            suite_dir("simple_system_identification", "gated_duffing"),
            dose="none",
            family="Gated Duffing",
        ),
        SuiteSource(
            "gated_duffing_parameter_mismatch_mild",
            "Mild",
            suite_dir("model_mismatch", "gated_duffing_parameter_mismatch_mild"),
            dose="mild",
            family="Gated Duffing",
        ),
        SuiteSource(
            "gated_duffing_parameter_mismatch",
            "Medium",
            suite_dir("model_mismatch", "gated_duffing_parameter_mismatch"),
            dose="medium",
            family="Gated Duffing",
        ),
        SuiteSource(
            "gated_duffing_parameter_mismatch_strong",
            "Strong",
            suite_dir("model_mismatch", "gated_duffing_parameter_mismatch_strong"),
            dose="strong",
            family="Gated Duffing",
        ),
    ]


@dataclass(frozen=True)
class DoseResponseData:
    """Prepared inputs consumed by both the figure and its CSV sidecar."""

    sources: list[SuiteSource]
    rows: list[dict[str, Any]]
    policy_ids: Sequence[str] = field(default_factory=lambda: list(DOSE_POLICIES))


def prepare() -> DoseResponseData:
    sources = dose_sources()
    rows: list[dict[str, Any]] = []
    for source in sources:
        for policy_id in DOSE_POLICIES:
            err, err_sem, n_err = data.metric_mean_sem(
                source.suite_dir, policy_id, "value_final_mean"
            )
            r2, r2_sem, n_r2 = data.metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            rows.append(
                {
                    "family": source.family,
                    "dose": source.dose,
                    "dose_label": source.label,
                    "experiment": source.exp_id,
                    "policy_id": policy_id,
                    "policy_label": theme.short_policy_label(policy_id),
                    "parameter_error_mean": err,
                    "parameter_error_sem": err_sem,
                    "trajectory_r2_mean": r2,
                    "trajectory_r2_sem": r2_sem,
                    "n_error": n_err,
                    "n_r2": n_r2,
                }
            )
    return DoseResponseData(sources=sources, rows=rows)


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


def generate() -> list[Path]:
    """Load the mismatch suites, render the figure, and write CSV sidecars."""
    prepared = prepare()
    suite_dirs = [source.suite_dir for source in prepared.sources]
    csv_paths = artifacts.write_csv_artifacts(
        suite_dirs,
        filename="tbme_experiment_mismatch_dose_response.csv",
        rows=prepared.rows,
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
    figure_paths = artifacts.artifact_paths(
        suite_dirs,
        subdir="figures",
        filename="tbme_experiment_mismatch_dose_response.pdf",
    )
    figure_path = plot_mismatch_dose_response(
        figure_paths[0],
        rows=prepared.rows,
        policy_ids=prepared.policy_ids,
        policy_label=theme.short_policy_label,
        policy_color=theme.policy_color,
        apply_style=theme.apply_style,
        style_axis=theme.style_experiment_axis,
    )
    return [*artifacts.copy_artifact(figure_path, figure_paths), *csv_paths]
