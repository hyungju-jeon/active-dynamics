"""Objective-ablation figure family.

Pipeline: :func:`prepare` reads the ablation suites into an
:class:`AblationData`, :func:`plot_objective_ablation` renders it, and
:func:`generate` orchestrates prepare → plot → sidecar tables so the figure
and its CSVs always come from the same prepared data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from actdyn.utils.figure_io import load_plotting, save_figure

from . import artifacts, data, theme
from .groups import SuiteSource, suite_dir

OBJECTIVE_POLICIES = [
    "active_planning",
    "active_fully_observable",
    "active_state_information",
    "active_dynamics",
    "active_observation_variance",
    "active_e_optimality",
    "active_state_variance",
    "prbs",
]

OBJECTIVE_DEFINITIONS = [
    {
        "policy_id": "active_planning",
        "objective_name": "Parameter EIG",
        "objective_formula": (
            r"$J(u_{0:H-1})=\frac{1}{2}\log\det(I+P_\theta "
            r"\sum_{i=0}^{H-1}\gamma^i \Delta\Lambda_i)$, "
            r"$\Delta\Lambda_i=S_i^\top(I+P_i^- I_{z,i})^{-1}I_{z,i}S_i$"
        ),
        "objective_notes": "Main objective; partial-observation attenuation uses the predicted latent covariance.",
    },
    {
        "policy_id": "active_fully_observable",
        "objective_name": "Full-observable EIG",
        "objective_formula": (
            r"$J(u_{0:H-1})=\frac{1}{2}\log\det(I+P_\theta "
            r"\sum_{i=0}^{H-1}\gamma^i S_i^\top I_{z,i}S_i)$"
        ),
        "objective_notes": "Ablation that removes partial-observation attenuation.",
    },
    {
        "policy_id": "active_state_information",
        "objective_name": "State information",
        "objective_formula": (
            r"$J(u_{0:H-1})=\sum_{i=0}^{H-1}\gamma^i "
            r"\log\det\operatorname{chol}(I+P_i^- I_{z,i})$"
        ),
        "objective_notes": "Scores latent-state observability, not parameter sensitivity.",
    },
    {
        "policy_id": "active_dynamics",
        "objective_name": "Dynamics sensitivity",
        "objective_formula": (
            r"$J(u_{0:H-1})=\sum_{i=0}^{H-1}\gamma^i " r"\operatorname{tr}(S_i^\top P_i^- S_i)$"
        ),
        "objective_notes": "Scores predicted state sensitivity to parameters without the decoder Fisher term.",
    },
    {
        "policy_id": "active_observation_variance",
        "objective_name": "Observation variance",
        "objective_formula": (
            r"$J(u_{0:H-1})=\sum_{i=0}^{H-1}\gamma^i "
            r"\sum_j\log(1+\operatorname{Var}_{\theta\sim q(\theta)}"
            r"[\lambda_j(z_i(\theta))])$"
        ),
        "objective_notes": "Monte Carlo objective using posterior samples and decoded observation rates.",
    },
    {
        "policy_id": "active_e_optimality",
        "objective_name": "E-optimality",
        "objective_formula": (
            r"$J(u_{0:H-1})=\lambda_{\min}(P_\theta " r"\sum_{i=0}^{H-1}\gamma^i \Delta\Lambda_i)$"
        ),
        "objective_notes": "Maximizes the least-informed parameter direction.",
    },
    {
        "policy_id": "active_state_variance",
        "objective_name": "State variance",
        "objective_formula": (
            r"$J(u_{0:H-1})=\sum_{i=0}^{H-1}\gamma^i "
            r"\sum_d \operatorname{Var}_{\theta\sim q(\theta)}[z_{i,d}(\theta)]$"
        ),
        "objective_notes": "Latent-state disagreement baseline; it is not a Fisher-information objective.",
    },
    {
        "policy_id": "prbs",
        "objective_name": "PRBS",
        "objective_formula": r"Preset pseudo-random binary excitation sequence.",
        "objective_notes": "Passive baseline with no model-based acquisition optimization.",
    },
]

REQUIRED_SUITES = (
    ("objective_ablation", "gated_duffing"),
    ("objective_ablation", "gated_duffing_asymmetric"),
    ("objective_ablation", "gated_duffing_challenging"),
)


def objective_sources() -> list[SuiteSource]:
    return [
        SuiteSource(
            "gated_duffing",
            "Default gated Duffing",
            suite_dir("objective_ablation", "gated_duffing"),
        ),
        SuiteSource(
            "gated_duffing_asymmetric",
            "Asymmetric loading",
            suite_dir("objective_ablation", "gated_duffing_asymmetric"),
        ),
        SuiteSource(
            "gated_duffing_challenging",
            "Challenging gated Duffing",
            suite_dir("objective_ablation", "gated_duffing_challenging"),
        ),
    ]


@dataclass(frozen=True)
class AblationData:
    """Prepared inputs consumed by both the figure and its CSV sidecar."""

    sources: list[SuiteSource]
    metric_rows: list[dict[str, Any]]
    curves_by_source: dict[str, dict[str, list[dict[str, float]]]]
    threshold: float = 0.95
    policy_ids: Sequence[str] = field(default_factory=lambda: list(OBJECTIVE_POLICIES))


def prepare(threshold: float = 0.95) -> AblationData:
    sources = objective_sources()
    metric_rows: list[dict[str, Any]] = []
    curves_by_source: dict[str, dict[str, list[dict[str, float]]]] = {}
    for source in sources:
        curves_by_source[source.exp_id] = data.curve_rows(
            source.suite_dir,
            "trajectory_r2_over_steps.csv",
            "trajectory_r2_mean",
        )
        for policy_id in OBJECTIVE_POLICIES:
            err, err_sem, n_err = data.metric_mean_sem(
                source.suite_dir, policy_id, "value_final_mean"
            )
            r2, r2_sem, n_r2 = data.metric_mean_sem(
                source.suite_dir, policy_id, "trajectory_r2_final_mean"
            )
            step_to_r2, cpu_time_to_r2, r2_at_threshold = data.r2_threshold_times(
                source.suite_dir,
                policy_id,
                threshold,
            )
            metric_rows.append(
                {
                    "experiment": source.exp_id,
                    "condition": source.label,
                    "policy_id": policy_id,
                    "policy_label": theme.short_policy_label(policy_id),
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
    return AblationData(
        sources=sources,
        metric_rows=metric_rows,
        curves_by_source=curves_by_source,
        threshold=threshold,
    )


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
    letters = [chr(ord("A") + idx) for idx in range(2 * len(sources))]
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
                    steps,
                    values - sem,
                    values + sem,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
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


def _escape_tex(text: object) -> str:
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_definition_tables(suite_dirs: Sequence[Path]) -> list[Path]:
    rows = [
        {
            "policy_id": str(row["policy_id"]),
            "policy_label": theme.short_policy_label(str(row["policy_id"])),
            "objective_name": str(row["objective_name"]),
            "objective_formula": str(row["objective_formula"]),
            "objective_notes": str(row["objective_notes"]),
        }
        for row in OBJECTIVE_DEFINITIONS
    ]
    written = artifacts.write_csv_artifacts(
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
                    _escape_tex(row["policy_label"]),
                    _escape_tex(row["objective_name"]),
                    str(row["objective_formula"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    written.extend(
        artifacts.write_text_artifacts(
            suite_dirs,
            filename="tbme_experiment_objective_ablation_objectives.tex",
            text="\n".join(lines) + "\n",
        )
    )
    return written


def generate() -> list[Path]:
    """Load the ablation suites, render the figure, and write CSV sidecars."""
    prepared = prepare()
    suite_dirs = [source.suite_dir for source in prepared.sources]
    csv_paths = artifacts.write_csv_artifacts(
        suite_dirs,
        filename="tbme_experiment_objective_ablation.csv",
        rows=prepared.metric_rows,
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
    figure_paths = artifacts.artifact_paths(
        suite_dirs,
        subdir="figures",
        filename="tbme_experiment_objective_ablation_gated_duffing.pdf",
    )
    figure_path = plot_objective_ablation(
        figure_paths[0],
        sources=prepared.sources,
        metric_rows=prepared.metric_rows,
        curves_by_source=prepared.curves_by_source,
        policy_ids=prepared.policy_ids,
        policy_label=theme.short_policy_label,
        policy_color=theme.policy_color,
        apply_style=theme.apply_style,
        style_axis=theme.style_experiment_axis,
        stroke_color=theme.STROKE_COLOR,
        neutral_light=theme.NEUTRAL_LIGHT,
    )
    return [*artifacts.copy_artifact(figure_path, figure_paths), *csv_paths]
