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
from .groups import SuiteSource, groups, suite_dir

DOSE_POLICIES = [
    "adaptive",
    "active_planning",
    "active_myopic",
    "active_state_variance",
    "prbs",
]

# Dose grid the figure understands; suites are matched against the catalog so
# retired doses/families simply drop out instead of failing suite resolution.
_DOSES = (("mild", "Mild"), ("", "Medium"), ("strong", "Strong"))
_FAMILIES = (("duffing", "Duffing"), ("gated_duffing", "Gated Duffing"))


def _catalog_suite_ids(group_name: str) -> set[str]:
    return {ref.suite_id for ref in groups().get(group_name, [])}


def _dose_suite_keys() -> list[tuple[str, str, str, str, str]]:
    """Return (group, suite_id, family_label, dose, dose_label) for suites in the catalog."""
    baseline_ids = _catalog_suite_ids("simple_system_identification")
    mismatch_ids = _catalog_suite_ids("model_mismatch")
    keys: list[tuple[str, str, str, str, str]] = []
    for family_id, family_label in _FAMILIES:
        if family_id in baseline_ids:
            keys.append(
                ("simple_system_identification", family_id, family_label, "none", "None")
            )
        for dose_suffix, dose_label in _DOSES:
            suite_id = f"{family_id}_parameter_mismatch"
            if dose_suffix:
                suite_id = f"{suite_id}_{dose_suffix}"
            if suite_id in mismatch_ids:
                keys.append(
                    ("model_mismatch", suite_id, family_label, dose_suffix or "medium", dose_label)
                )
    return keys


def required_suites() -> tuple[tuple[str, str], ...]:
    return tuple((group, suite_id) for group, suite_id, *_rest in _dose_suite_keys())


def dose_sources() -> list[SuiteSource]:
    sources = [
        SuiteSource(
            suite_id,
            dose_label,
            suite_dir(group, suite_id),
            dose=dose,
            family=family_label,
        )
        for group, suite_id, family_label, dose, dose_label in _dose_suite_keys()
    ]
    if not sources:
        raise RuntimeError(
            "No mismatch dose-response suites found in the catalog "
            "(simple_system_identification / model_mismatch groups)"
        )
    return sources


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
    canonical = [("none", "None"), ("mild", "Mild"), ("medium", "Medium"), ("strong", "Strong")]
    doses_present = {str(row["dose"]) for row in rows}
    dose_order = [dose for dose, _label in canonical if dose in doses_present]
    dose_labels = [label for dose, label in canonical if dose in doses_present]
    families: list[str] = []
    for row in rows:
        family = str(row["family"])
        if family not in families:
            families.append(family)
    x = np.arange(len(dose_order), dtype=np.float64)
    fig, axes = plt_module.subplots(
        1, len(families), figsize=(3.55 * len(families), 2.9), sharey=False, squeeze=False
    )
    axes = axes.ravel()
    for ax, family in zip(axes, families, strict=True):
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
    axes[-1].legend(loc="upper left", fontsize=6.4)
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
