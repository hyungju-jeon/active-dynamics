#!/usr/bin/env python3
"""Manuscript asset assembly for the TBME figures package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from actdyn.utils.experiment_runtime import read_trace_csv, safe_float as _safe_float
from actdyn.utils.figure_io import load_plotting

from ...experiment_io import (
    find_nested_metadata_paths,
    get_environment_preset_from_metadata,
    load_json,
)
from . import groups as _groups_mod
from .ablation import (
    OBJECTIVE_POLICIES as _experiment_OBJECTIVE_POLICIES,
    objective_sources as _experiment_objective_sources,
)
from .artifacts import (
    write_csv as _write_csv,
    write_text as _write_text,
)
from .data import (
    metric_mean_sem as _experiment_metric_mean_sem,
    metric_values as _experiment_metric_values,
    r2_threshold_step as _experiment_r2_threshold_step,
    r2_threshold_times as _experiment_r2_threshold_times,
)
from .gates import (
    COMPOUND_POLICY_ORDER as _COMPOUND_POLICY_ORDER,
    compound_summary_rows as _compound_summary_rows,
    compound_trace_records as _compound_trace_records,
    plot_neutral_vector_field,
)
from .groups import SuiteSource as _ExperimentSuiteSource, suite_dir as _suite_dir
from .information import make_information_grid as _experiment_make_information_grid
from .records import (
    RunRecord as _ExperimentRunRecord,
    load_xy_trace as _experiment_load_xy_trace,
)
from .theme import (
    NEUTRAL_FILL as _experiment_C_NEUTRAL_FILL,
    NEUTRAL_LIGHT as _experiment_C_NEUTRAL_LIGHT,
    STROKE_COLOR as _experiment_C_STROKE,
    apply_style as _apply_style,
    extended_policy_label as _experiment_short_policy_label,
    policy_color as _policy_color,
    style_axis as _style_manuscript_axis,
    style_experiment_axis as _style_experiment_axis,
)
from .groups import (
    REPO_ROOT as _REPO_ROOT,
    RESULTS_ROOT as _RESULTS_ROOT,
)
from ..tbme_io import (
    load_planned_trace,
    planned_xy_cycle_for_step,
    true_dynamics_from_metadata,
)

# Manuscript asset assembly
def save_figure(fig: Any, output_path: Path, *, plt_module: Any) -> Path:
    """Export both vectors without cropping away the intended physical size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for legend in fig.legends:
        texts.extend(legend.get_texts())
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        texts.extend([ax.xaxis.label, ax.yaxis.label, ax.title])
        # Locators also create ticks outside the limits; those are not rendered.
        for locations, labels, limits in (
            (ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
            (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim()),
        ):
            lo, hi = sorted(limits)
            texts.extend(label for value, label in zip(locations, labels) if lo <= value <= hi)
        texts.extend(ax.texts)
        if ax.get_legend() is not None:
            texts.extend(ax.get_legend().get_texts())
    outside = []
    for text in texts:
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(text.get_text())
    if outside:
        raise RuntimeError(f"Text outside figure canvas in {output_path}: {outside}")
    output_path.with_suffix('.audit.json').write_text(json.dumps({
        'size_inches': fig.get_size_inches().tolist(), 'axes': len(fig.axes),
        'r2_axes': [dict(ylabel=ax.get_ylabel(), ylim=list(ax.get_ylim()),
                         references=[dict(label=line.get_label(), value=float(line.get_ydata()[0]),
                                          linestyle=line.get_linestyle())
                                     for line in ax.lines
                                     if line.get_label().startswith('true-model reference')])
                    for ax in fig.axes if ax.get_gid() == 'vf-roll-r2'],
        'text_outside_canvas': outside,
    }, indent=2))
    with plt_module.rc_context({"savefig.bbox": None}):
        fig.savefig(output_path, bbox_inches=None)
        if output_path.suffix != ".svg":
            fig.savefig(output_path.with_suffix(".svg"), bbox_inches=None)
    plt_module.close(fig)
    return output_path


_POLICY_LABELS = {
    "adaptive": "PALDI",
    "adaptive_async_anytime": "Async PALDI(anytime)",
    "adaptive_async_realtime": "Async PALDI",
    "active_planning": "Fixed PALDI",
    "active_myopic": "Myopic",
    "prbs": "PRBS",
    "random": "Random",
    "active_fully_observable": "Unatten.",
    "active_state_information": "State Information",
    "active_dynamics": "Dyn. sens.",
    "active_e_optimality": "E-opt.",
    "active_observation_variance": "Obs. var.",
    "active_state_variance": "State var.",
    "flex": "FLEX",
    "flex_filter": "FLEX upstream / filtered",
    "flex_true": "FLEX upstream / true",
    "flex_rollback": "FLEX",
    "rhc": "RHC-US",
    "off_policy": "Uncontrolled",
}
_ASSET_MATCHED_POLICIES = [
    "adaptive",
    "active_myopic",
    "flex_rollback",
    "rhc",
    "prbs",
    "random",
]
_ASSET_R2_CEILING_REPEATS = 48
_ASSET_R2_SUMMARIES = ("mean_sem", "median_iqr")
# The appendix variant figure separates the two FLEX adaptations: the state fed to
# the parameter update, and whether the acceptance test guards it.
_ASSET_FLEX_POLICIES = ("flex_true", "flex_filter", "flex_rollback")
_ASSET_FLEX_LABELS = {
    "flex_true": "FLEX (true)",
    "flex_filter": "FLEX (EKF)",
    "flex_rollback": "FLEX (EKF+stable)",
}
# Values and whiskers below the requested display range retain their CSV values
# and receive a clipping marker at the lower axis limit.
_ASSET_FLEX_BAR_YLIM = (0.25, 1.0)


def _asset_policy_label(
    policy_id: str, policy_labels: Mapping[str, str] | None = None
) -> str:
    if policy_labels is not None and policy_id in policy_labels:
        return policy_labels[policy_id]
    return _POLICY_LABELS.get(policy_id, _experiment_short_policy_label(policy_id))


# Aliases that keep a policy visually identified with its counterpart elsewhere in
# the manuscript. Safe because no asset figure draws both members of a pair.
_ASSET_COLOR_ALIASES = {
    # The rollback-stabilized baseline still reads as FLEX.
    "flex_rollback": "flex",
    # The full p-EIG objective carries the PALDI color of the other figures.
    "active_planning": "adaptive",
}


def _asset_baseline_policy_color(policy_id: str) -> str:
    return _policy_color(_ASSET_COLOR_ALIASES.get(policy_id, policy_id))


def _asset_parse_r2_summaries(raw: str) -> list[str]:
    summaries = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(summaries) - set(_ASSET_R2_SUMMARIES))
    if unknown:
        expected = ", ".join(_ASSET_R2_SUMMARIES)
        raise ValueError(
            f"Unknown R2 summary set(s): {', '.join(unknown)}. Expected: {expected}"
        )
    if not summaries:
        raise ValueError("At least one R2 summary set is required")
    return list(dict.fromkeys(summaries))


# Shared manuscript font rule for every asset figure: Helvetica, bold 10 panel
# indices, 8 pt titles/axis labels, 6 pt tick values.
_ASSET_FONT_STACK = ("Helvetica", "Nimbus Sans", "TeX Gyre Heros", "Arial", "DejaVu Sans")
_ASSET_PANEL_LABEL_SIZE = 10.0
_ASSET_TITLE_SIZE = 8.0
_ASSET_LABEL_SIZE = 8.0
_ASSET_TICK_SIZE = 6.0
_ASSET_PREDICTIVE_R2_LABEL = r"$R^2_{\mathrm{VF}\text{-}\mathrm{roll}}$"
_ASSET_FINAL_R2_LABEL = _ASSET_PREDICTIVE_R2_LABEL
_ASSET_SINGLE_COLUMN_WIDTH = 3.5


def _apply_asset_style(plt_module: Any | None = None) -> None:
    if plt_module is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_module
    _apply_style(plt_module)
    plt_module.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(_ASSET_FONT_STACK),
            "mathtext.fontset": "dejavusans",
            "font.size": _ASSET_TICK_SIZE,
            "axes.titlesize": _ASSET_TITLE_SIZE,
            "axes.labelsize": _ASSET_LABEL_SIZE,
            "xtick.labelsize": _ASSET_TICK_SIZE,
            "ytick.labelsize": _ASSET_TICK_SIZE,
            "legend.fontsize": _ASSET_TICK_SIZE,
        }
    )


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


def _asset_first_suite_metadata(suite_dir: Path) -> dict[str, Any] | None:
    for policy_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        if policy_dir.name == "summary":
            continue
        for seed_dir in sorted(policy_dir.glob("seed_*")):
            for metadata_path in find_nested_metadata_paths(seed_dir):
                metadata = load_json(metadata_path)
                if metadata.get("status") in {None, "", "completed"}:
                    return metadata
    return None


def _asset_true_model_r2_ceiling(
    suite_dir: Path,
    *,
    r2_summary: str = "mean_sem",
) -> float | None:
    _asset_parse_r2_summaries(r2_summary)
    metadata = _asset_first_suite_metadata(suite_dir)
    if metadata is None:
        return None
    env_preset = get_environment_preset_from_metadata(metadata)
    true_embedding_raw = (
        metadata.get("embedding_true")
        or metadata.get("true_embedding")
        or env_preset.true_embedding_vector()
    )
    true_embedding = np.asarray(true_embedding_raw, dtype=np.float32).reshape(-1)
    if true_embedding.size == 0:
        return None
    state_noise = _safe_float(metadata.get("trajectory_eval_state_noise"))
    if state_noise is None:
        state_noise = env_preset.trajectory_eval_state_noise
    if state_noise is None:
        state_noise = _safe_float(metadata.get("state_noise"))
    if state_noise is None:
        state_noise = float(env_preset.state_noise)
    if state_noise <= 0.0:
        return 1.0

    import torch
    from actdyn.utils.validation import trajectory_r2_vectorfield_many

    repeats = int(_ASSET_R2_CEILING_REPEATS)
    r2_values = trajectory_r2_vectorfield_many(
        e_estimates=torch.as_tensor(
            np.repeat(true_embedding.reshape(1, -1), repeats, axis=0),
            dtype=torch.float32,
        ),
        e_true=torch.as_tensor(true_embedding, dtype=torch.float32),
        true_dynamics_type=str(
            metadata.get("dynamics_type") or env_preset.resolved_dynamics_type()
        ),
        true_full_params=np.asarray(
            metadata.get("true_params_full") or env_preset.resolved_true_params(),
            dtype=np.float32,
        ),
        estimator_dynamics_type=str(
            metadata.get("dynamics_type") or env_preset.resolved_dynamics_type()
        ),
        estimator_full_params=np.asarray(
            metadata.get("true_params_full") or env_preset.resolved_true_params(),
            dtype=np.float32,
        ),
        true_min_embedding_dim=int(
            metadata.get("min_embedding_dim") or env_preset.resolved_min_embedding_dim()
        ),
        estimator_min_embedding_dim=int(
            metadata.get("min_embedding_dim") or env_preset.resolved_min_embedding_dim()
        ),
        dt=float(env_preset.dt),
        dynamics_alpha=float(metadata.get("dynamics_alpha") or env_preset.dynamics_alpha),
        horizon=int(metadata.get("trajectory_eval_horizon") or 200),
        n_starts=int(metadata.get("trajectory_eval_samples") or 100),
        rng=np.random.default_rng(104729),
        device="cpu",
        state_noise=state_noise,
        state_dim=len(metadata.get("state_low") or env_preset.state_low),
        state_low=metadata.get("trajectory_eval_state_low", env_preset.trajectory_eval_state_low),
        state_high=metadata.get("trajectory_eval_state_high", env_preset.trajectory_eval_state_high),
        state_indices=metadata.get("trajectory_eval_state_indices", env_preset.trajectory_eval_state_indices),
        coordinate_balanced=bool(metadata.get("trajectory_eval_coordinate_balanced",
                                             env_preset.trajectory_eval_coordinate_balanced)),
    )
    finite = r2_values[np.isfinite(r2_values)]
    if finite.size == 0:
        return None
    if r2_summary == "median_iqr":
        return float(np.median(finite))
    return float(np.mean(finite))


def _asset_r2_curve_rows(
    suite_dir: Path,
    *,
    r2_summary: str,
) -> dict[str, list[dict[str, float]]]:
    """Read one explicit R2 center-and-band summary from the suite CSV."""
    _asset_parse_r2_summaries(r2_summary)
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in read_trace_csv(suite_dir / "summary" / "trajectory_r2_over_steps.csv"):
        policy_id = str(row.get("policy_id", ""))
        step = _safe_float(row.get("step"))
        cpu_time_sec = _safe_float(row.get("cpu_time_sec_mean"))
        if r2_summary == "median_iqr":
            center = _safe_float(row.get("value_median"))
            lower = _safe_float(row.get("value_q25"))
            upper = _safe_float(row.get("value_q75"))
        else:
            center = _safe_float(row.get("trajectory_r2_mean"))
            sem = _safe_float(row.get("value_sem"))
            lower = None if center is None else center - (0.0 if sem is None else sem)
            upper = None if center is None else center + (0.0 if sem is None else sem)
        if not policy_id or step is None or center is None:
            continue
        grouped.setdefault(policy_id, []).append(
            {
                "step": step,
                "center": center,
                "lower": center if lower is None else lower,
                "upper": center if upper is None else upper,
                "cpu_time_sec": np.nan if cpu_time_sec is None else cpu_time_sec,
            }
        )
    for policy_rows in grouped.values():
        policy_rows.sort(key=lambda row: row["step"])
    return grouped


def _asset_plot_r2_curves(
    ax: Any,
    suite_dir: Path,
    policy_ids: Sequence[str],
    *,
    title: str,
    panel_label: str,
    ylabel: bool,
    r2_summary: str,
    show_inset: bool = False,
    xlabel: bool = True,
    policy_labels: Mapping[str, str] | None = None,
    ylim: tuple[float, float] = (0.25, 1.0),
    title_pad: float = 3.0,
) -> None:
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

    curves = _asset_r2_curve_rows(suite_dir, r2_summary=r2_summary)
    curve_series = []
    for policy_id in policy_ids:
        rows = curves.get(policy_id, [])
        if not rows:
            continue
        steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
        values = np.asarray([row["center"] for row in rows], dtype=np.float64)
        lower = np.asarray([row["lower"] for row in rows], dtype=np.float64)
        upper = np.asarray([row["upper"] for row in rows], dtype=np.float64)
        color = _asset_baseline_policy_color(policy_id)
        curve_series.append(
            (steps, values, lower, upper, color, _asset_policy_label(policy_id, policy_labels))
        )

    if not curve_series:
        raise RuntimeError(
            f"No trajectory R2 curves available for {r2_summary} in {suite_dir / 'summary'}"
        )

    r2_ceiling = _asset_true_model_r2_ceiling(suite_dir, r2_summary=r2_summary)
    curve_axes = [(ax, 0.95, 0.10, True)]
    inset = None
    if show_inset:
        inset = ax.inset_axes([0.55, 0.13, 0.40, 0.40])
        curve_axes.append((inset, 0.65, 0.08, False))
    for curve_ax, linewidth, alpha, labels in curve_axes:
        for steps, values, lower, upper, color, label in curve_series:
            curve_ax.plot(
                steps,
                values,
                color=color,
                linewidth=linewidth,
                label=label if labels else None,
            )
            curve_ax.fill_between(
                steps,
                lower,
                upper,
                color=color,
                alpha=alpha,
                linewidth=0.0,
            )
        if r2_ceiling is not None:
            curve_ax.axhline(
                r2_ceiling,
                color=_experiment_C_NEUTRAL_LIGHT,
                linestyle=":",
                linewidth=0.65,
                zorder=5,
                clip_on=False,
                label="true-model reference",
            )
        curve_ax.set_xlim(left=0.0)
        curve_ax.set_yscale("log", nonpositive="clip")
        curve_ax.set_ylim(*ylim)
        curve_ax.set_gid("vf-roll-r2")
        curve_ax.yaxis.set_major_locator(FixedLocator([ylim[0], 1.0]))
        curve_ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        curve_ax.yaxis.set_minor_formatter(NullFormatter())
        _style_experiment_axis(curve_ax)
    if inset is not None:
        inset.set_xlim(0.0, 250.0)
        inset.tick_params(axis="both", labelsize=_ASSET_TICK_SIZE, pad=1.0)
    ax.set_title(
        panel_label, loc="left", fontweight="bold", fontsize=_ASSET_PANEL_LABEL_SIZE, pad=title_pad
    )
    ax.set_title(title, loc="center", fontsize=_ASSET_TITLE_SIZE, pad=title_pad)
    if xlabel:
        ax.set_xlabel("Environment steps")
    if ylabel:
        ax.set_ylabel(_ASSET_PREDICTIVE_R2_LABEL)


def _asset_plot_active_vs_baselines(output_path: Path, *, r2_summary: str) -> Path:
    sources = [
        _ExperimentSuiteSource(ref.suite_id, ref.label, ref.results_root / "tracks" / ref.suite_id)
        for ref in _groups_mod.groups()["simple_system_identification"]
    ]
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(
        1, len(sources), figsize=(252.0 / 72.27, 1.55), squeeze=False, sharey=True
    )
    for idx, source in enumerate(sources):
        _asset_plot_r2_curves(
            axes[0, idx],
            source.suite_dir,
            _ASSET_MATCHED_POLICIES,
            title="",
            panel_label="",
            ylabel=idx == 0,
            xlabel=False,
            r2_summary=r2_summary,
            ylim=(0.25, 1.0),
            title_pad=1.0,
        )
        axes[0, idx].set_xticks([0, 1000, 2000])
        # Keep endpoint labels inside each panel to allow narrower gaps.
        axes[0, idx].get_xticklabels()[0].set_ha("left")
        axes[0, idx].get_xticklabels()[-1].set_ha("right")
        axes[0, idx].tick_params(axis="both", which="both", pad=1.0)
        axes[0, idx].yaxis.labelpad = 1.0
        if idx == 0:
            axes[0, idx].set_ylabel(_ASSET_PREDICTIVE_R2_LABEL)
            # Place the label beside the spine, not outside the widest tick label.
            axes[0, idx].yaxis.set_label_coords(-0.105, 0.5)
        axes[0, idx].annotate(
            chr(65 + idx), (0, 1), xycoords="axes fraction",
            xytext=(-7.2, 1.2), textcoords="offset points", ha="left", va="bottom",
            fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = ["true" if label == "true-model reference" else label for label in labels]
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.985, 0.98, 0.0),
        mode="expand",
        ncol=len(handles),
        fontsize=_ASSET_TICK_SIZE,
        columnspacing=0.5,
        handlelength=0.9,
        handletextpad=0.3,
        borderaxespad=0.0,
        borderpad=0.1,
        labelspacing=0.2,
    )
    fig.supxlabel("Environment steps", y=0.045, fontsize=_ASSET_LABEL_SIZE)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.24, top=0.78, wspace=0.045)
    return save_figure(fig, output_path, plt_module=plt_module)


# Presets of the benchmark geometry figures (Wilson-Cowan: `wilson_cowan.py`).
_MECHANICAL_ENVS = (
    ("tbme_duffing", "duffing", "Duffing"),
    ("tbme_damped_pendulum", "damped_pendulum", "Damped Pendulum"),
)
_TRAJ_COLORS = ("#E8963A", "#D1382C", "#2E6FB0")
# Grid sizes, example rollouts (no input), and the NeuroFisherSNR loading calibration.
_GEOMETRY = {
    "n_grid_field": 41,
    "n_grid_map": 81,
    "steps": 500,
    "n_trajectories": 3,
    "seed": 0,
    "snr_trajectories": 100,
    "snr_trajectory_length": 200,
}


def _style_map_axis(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _draw_phase_portrait(ax: Any, preset: Any) -> None:
    """True vector field of ``preset`` with example trajectories without input."""
    from actdyn.utils.plotting import plot_vector_field

    from .diagnostics import simulate_trajectories, true_dynamics

    plot_lim = float(preset.resolved_plot_limit())
    plot_vector_field(
        true_dynamics(preset),
        ax=ax,
        x_range=plot_lim,
        n_grid=_GEOMETRY["n_grid_field"],
        is_residual=True,
        device="cpu",
        streamplot_kwargs={"arrowsize": 0.35, "linewidth": 0.22},
    )
    trajectories = simulate_trajectories(
        preset,
        n_trajectories=_GEOMETRY["n_trajectories"],
        steps=_GEOMETRY["steps"],
        seed=_GEOMETRY["seed"],
    )
    for idx, traj in enumerate(trajectories):
        color = _TRAJ_COLORS[idx % len(_TRAJ_COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.15, alpha=0.92,
                solid_capstyle="round", zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], s=11, color=color, edgecolor="white", linewidth=0.3, zorder=4)
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_aspect("equal", adjustable="box")
    _style_map_axis(ax)


def _draw_sensitivity_maps(fig: Any, axes: Sequence[Any], cbar_ax: Any, presets: Sequence[Any]) -> None:
    """Parameter-sensitivity magnitude ||dv/dtheta||_F of each preset, one shared color scale."""
    from .diagnostics import finite_limits, parameter_sensitivity_grid

    maps = []
    for preset in presets:
        plot_lim = float(preset.resolved_plot_limit())
        _sx, _sy, sens = parameter_sensitivity_grid(preset, plot_lim=plot_lim, n_grid=_GEOMETRY["n_grid_map"])
        maps.append((sens, plot_lim))
    vmin, vmax = finite_limits(np.concatenate([sens.reshape(-1) for sens, _ in maps]))
    image = None
    for ax, (sens, plot_lim) in zip(axes, maps):
        image = ax.imshow(sens, origin="lower", extent=[-plot_lim, plot_lim, -plot_lim, plot_lim],
                          cmap="magma", vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="equal")
        _style_map_axis(ax)
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=_ASSET_TICK_SIZE, width=0.4, length=2.0)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label(r"$\|\partial\mathbf{v}/\partial\theta\|_F$", fontsize=_ASSET_LABEL_SIZE)


def _block_span(axes: Sequence[Any]) -> tuple[float, float, float]:
    """Left, right, and top edges (figure fraction) of a group of axes, after a draw."""
    positions = [ax.get_position() for ax in axes]
    return min(pos.x0 for pos in positions), max(pos.x1 for pos in positions), max(pos.y1 for pos in positions)


def _draw_for_positions(fig: Any) -> None:
    # `aspect="equal"` squares (and centers) image axes only at draw time; draw once
    # before reading positions to place titles and letters on the real boxes.
    try:
        fig.draw_without_rendering()
    except AttributeError:
        fig.canvas.draw()


def _figure_letter(fig: Any, letter: str, x: float, y: float) -> None:
    fig.text(x, y, letter, ha="left", va="bottom", fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold")


def _r2_legend_entries(ax: Any) -> tuple[list[Any], list[str]]:
    handles, labels = ax.get_legend_handles_labels()
    return handles, ["true" if label == "true-model reference" else label for label in labels]


def _asset_plot_wilson_cowan_benchmark(output_path: Path) -> Path:
    """Wilson-Cowan benchmark figure of the main text (Fig. 3); see `wilson_cowan.py`."""
    from .wilson_cowan import generate_wilson_cowan_figure

    return generate_wilson_cowan_figure(output_path)


def _asset_plot_mechanical_benchmarks(output_path: Path, *, r2_summary: str) -> Path:
    """Duffing and damped-pendulum benchmarks of the Supplementary.

    (A) vector fields with example trajectories; (B) parameter sensitivity on one
    shared color scale; (C) R2_VF-roll against environment steps for the matched
    policies, center and band given by ``r2_summary``.
    """
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    configure_tbme_catalogs(suite_entries={})
    suites = [_suite_dir("simple_system_identification", suite_id) for _env, suite_id, _label in _MECHANICAL_ENVS]
    _asset_require_suite_dirs(suites)
    presets = [get_environment_preset(env_id) for env_id, _suite, _label in _MECHANICAL_ENVS]
    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")

    fig = plt_module.figure(figsize=(516.0 / 72.27, 1.70))
    map_width = 0.11
    map_height = map_width * fig.get_figwidth() / fig.get_figheight()
    y = 0.21

    a_axes = [fig.add_axes([left, y, map_width, map_height]) for left in (0.03, 0.155)]
    for ax, preset, (_env, _suite, label) in zip(a_axes, presets, _MECHANICAL_ENVS):
        _draw_phase_portrait(ax, preset)
        ax.set_title(label, fontsize=_ASSET_TITLE_SIZE, pad=3.0)
    b_axes = [fig.add_axes([left, y, map_width, map_height]) for left in (0.33, 0.455)]
    _draw_sensitivity_maps(fig, b_axes, fig.add_axes([0.455 + map_width + 0.006, y, 0.010, map_height]), presets)
    for ax, (_env, _suite, label) in zip(b_axes, _MECHANICAL_ENVS):
        ax.set_title(label, fontsize=_ASSET_TITLE_SIZE, pad=3.0)

    c_axes = []
    for idx, (left, suite, (_env, _suite, label)) in enumerate(zip((0.705, 0.855), suites, _MECHANICAL_ENVS)):
        ax = fig.add_axes([left, y, 0.135, map_height], sharey=c_axes[0] if c_axes else None)
        _asset_plot_r2_curves(ax, suite, _ASSET_MATCHED_POLICIES, title=label, panel_label="", ylabel=idx == 0,
                              xlabel=False, r2_summary=r2_summary, ylim=(0.25, 1.0), title_pad=3.0)
        ax.title.set_size(8.0)
        ax.set_xticks([0, 1000, 2000])
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)
        c_axes.append(ax)
    fig.text(0.5 * (c_axes[0].get_position().x0 + c_axes[-1].get_position().x1), 0.015, "Environment steps",
             ha="center", va="bottom", fontsize=_ASSET_TITLE_SIZE)
    fig.legend(*_r2_legend_entries(c_axes[0]), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7,
               frameon=False, fontsize=_ASSET_TICK_SIZE, handlelength=1.0, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0.1)

    _draw_for_positions(fig)
    a_left, a_right, a_top = _block_span(a_axes)
    b_left, b_right, b_top = _block_span(b_axes)
    c_left, _c_right, c_top = _block_span(c_axes)
    fig.text(0.5 * (a_left + a_right), a_top + 0.11, "Vector Field", ha="center", va="bottom", fontsize=_ASSET_TITLE_SIZE)
    fig.text(0.5 * (b_left + b_right), b_top + 0.11, "Parameter Sensitivity", ha="center", va="bottom",
             fontsize=_ASSET_TITLE_SIZE)
    for letter, x, top_edge in (("A", a_left - 0.025, a_top), ("B", b_left - 0.025, b_top),
                                ("C", c_left - 0.05, c_top)):
        _figure_letter(fig, letter, x, top_edge + 0.11)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_bottleneck_sources() -> list[_ExperimentSuiteSource]:
    return [
        _ExperimentSuiteSource(
            "wilson_cowan",
            "Default",
            _suite_dir("simple_system_identification", "wilson_cowan"),
        ),
        _ExperimentSuiteSource(
            "wilson_cowan_observation_bottleneck_mild",
            "SNR -10",
            _suite_dir(
                "observation_action_bottleneck",
                "wilson_cowan_observation_bottleneck_mild",
            ),
        ),
        _ExperimentSuiteSource(
            "wilson_cowan_observation_bottleneck_strong",
            "SNR -15",
            _suite_dir(
                "observation_action_bottleneck",
                "wilson_cowan_observation_bottleneck_strong",
            ),
        ),
        _ExperimentSuiteSource(
            "wilson_cowan_action_bottleneck_mild",
            "Act. 0.75",
            _suite_dir("observation_action_bottleneck", "wilson_cowan_action_bottleneck_mild"),
        ),
        _ExperimentSuiteSource(
            "wilson_cowan_action_bottleneck_strong",
            "Act. 0.50",
            _suite_dir("observation_action_bottleneck", "wilson_cowan_action_bottleneck_strong"),
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
    Default = _asset_first_record("exp02_hard", "exp02_hard_gated_duffing", policy_id)
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

    Default_traj = _experiment_load_xy_trace(Default)
    panel_abs = _asset_trace_abs(Default, Default_traj)
    x_axis, y_axis, logdet_grid = _experiment_make_information_grid(
        Default.metadata,
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
        "Default": _asset_read_information(Default),
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
        true_dynamics_from_metadata(Default.metadata),
        grid_lim=panel_abs,
        n_grid=24,
        arrowsize=0.58,
        stroke_color=_experiment_C_STROKE,
    )
    ax.plot(
        Default_traj[:, 0],
        Default_traj[:, 1],
        color=_experiment_C_STROKE,
        linewidth=0.75,
        alpha=0.72,
        label="executed",
        zorder=4,
    )
    planned_trace = load_planned_trace(Default.run_dir, Default.metadata)
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
    zoom_points = [Default_traj[:, :2], *planned_paths]
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
    ax.legend(loc="lower right", fontsize=_ASSET_TICK_SIZE, framealpha=0.78, borderpad=0.25)

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
                for row in info_rows["Default"]
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
    ax.set_yticklabels([label for _field, _value, label, _color in event_specs], fontsize=_ASSET_TICK_SIZE)
    ax.set_xlim(0.0, 2000.0)
    ax.set_ylim(-0.65, len(event_specs) - 0.35)
    ax.set_title("B. Adaptive cadence event timeline")
    ax.set_xlabel("Environment step")
    _style_manuscript_axis(ax, grid_axis="x")

    mismatch_specs = [
        ("Default", _policy_color("adaptive")),
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
    ax.legend(loc="upper right", fontsize=_ASSET_TICK_SIZE)
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
    ax.legend(loc="upper left", fontsize=_ASSET_TICK_SIZE)
    _style_manuscript_axis(ax, grid_axis="y")

    fig.tight_layout(w_pad=0.95, h_pad=0.95)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_method_csv_fields(r2_summary: str) -> list[str]:
    _asset_parse_r2_summaries(r2_summary)
    r2_fields = (
        ["trajectory_r2_median", "trajectory_r2_q25", "trajectory_r2_q75"]
        if r2_summary == "median_iqr"
        else ["trajectory_r2_mean", "trajectory_r2_sem"]
    )
    return [
        "experiment",
        "condition",
        "policy_id",
        "policy_label",
        *r2_fields,
        "step_to_r2_0p95",
        "cpu_time_sec_to_r2_0p95",
        "r2_at_0p95",
        "parameter_error_mean",
        "parameter_error_sem",
        "n_error",
        "n_r2",
        "n_total",
        "n_r2_nonfinite",
        "r2_nonfinite_rate",
    ]


def _asset_write_method_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    r2_summary: str,
) -> None:
    """Write public method metrics without plotting-only R2 band fields."""
    fields = _asset_method_csv_fields(r2_summary)
    _write_csv(
        path,
        ({field: row.get(field) for field in fields} for row in rows),
        fields,
    )


def _asset_final_r2_summary(
    suite_dir: Path,
    policy_id: str,
    *,
    r2_summary: str,
) -> tuple[float | None, float | None, float | None, int]:
    """Return the final R2 center, lower band, upper band, and sample count."""
    _asset_parse_r2_summaries(r2_summary)
    if r2_summary == "mean_sem":
        center, sem, count = _experiment_metric_mean_sem(
            suite_dir,
            policy_id,
            "trajectory_r2_final_mean",
        )
        if center is None:
            return None, None, None, count
        return center, center - sem, center + sem, count

    values = np.asarray(
        _experiment_metric_values(suite_dir, policy_id, "trajectory_r2_final_mean"),
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, None, 0
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)),
        int(values.size),
    )


def _asset_r2_threshold_times(
    suite_dir: Path,
    policy_id: str,
    threshold: float,
    *,
    r2_summary: str,
) -> tuple[float | None, float | None, float | None]:
    if r2_summary == "mean_sem":
        return _experiment_r2_threshold_times(suite_dir, policy_id, threshold)
    curves = _asset_r2_curve_rows(suite_dir, r2_summary=r2_summary)
    if not curves:
        raise RuntimeError(
            f"No trajectory R2 curves available for {r2_summary} in {suite_dir / 'summary'}"
        )
    for row in curves.get(policy_id, []):
        if row["center"] < threshold:
            continue
        cpu_time_sec = row["cpu_time_sec"]
        return (
            row["step"],
            None if not np.isfinite(cpu_time_sec) else cpu_time_sec,
            row["center"],
        )
    return None, None, None


def _asset_method_metric_rows(
    sources: Sequence[_ExperimentSuiteSource],
    policy_ids: Sequence[str],
    *,
    r2_summary: str,
) -> list[dict[str, Any]]:
    threshold = 0.95
    metric_rows: list[dict[str, Any]] = []
    for source in sources:
        completed_rows = [
            row
            for row in read_trace_csv(source.suite_dir / "summary" / "metrics.csv")
            if row.get("status") in {None, "", "completed"}
        ]
        for policy_id in policy_ids:
            err, err_sem, n_err = _experiment_metric_mean_sem(
                source.suite_dir,
                policy_id,
                "value_final_mean",
            )
            r2, r2_lower, r2_upper, n_r2 = _asset_final_r2_summary(
                source.suite_dir,
                policy_id,
                r2_summary=r2_summary,
            )
            step_to_r2, cpu_time_to_r2, r2_at_threshold = _asset_r2_threshold_times(
                source.suite_dir,
                policy_id,
                threshold,
                r2_summary=r2_summary,
            )
            n_total = sum(row.get("policy_id") == policy_id for row in completed_rows)
            n_r2_nonfinite = max(0, n_total - n_r2)
            row = {
                "experiment": source.exp_id,
                "condition": source.label,
                "policy_id": policy_id,
                "policy_label": _asset_policy_label(policy_id),
                "parameter_error_mean": err,
                "parameter_error_sem": err_sem,
                "_trajectory_r2_center": r2,
                "_trajectory_r2_lower": r2_lower,
                "_trajectory_r2_upper": r2_upper,
                "step_to_r2_0p95": step_to_r2,
                "cpu_time_sec_to_r2_0p95": cpu_time_to_r2,
                "r2_at_0p95": r2_at_threshold,
                "n_error": n_err,
                "n_r2": n_r2,
                "n_total": n_total,
                "n_r2_nonfinite": n_r2_nonfinite,
                "r2_nonfinite_rate": (
                    float(n_r2_nonfinite) / float(n_total) if n_total else None
                ),
            }
            if r2_summary == "median_iqr":
                row.update(
                    {
                        "trajectory_r2_median": r2,
                        "trajectory_r2_q25": r2_lower,
                        "trajectory_r2_q75": r2_upper,
                    }
                )
            else:
                row.update(
                    {
                        "trajectory_r2_mean": r2,
                        "trajectory_r2_sem": (
                            None if r2 is None or r2_lower is None else r2 - r2_lower
                        ),
                    }
                )
            metric_rows.append(row)
    return metric_rows


def _asset_method_final_r2(
    metric_rows: Sequence[Mapping[str, Any]],
    exp_id: str,
    policy_id: str,
) -> tuple[float, float, float]:
    for row in metric_rows:
        if row["experiment"] == exp_id and str(row["policy_id"]) == policy_id:
            value = row["_trajectory_r2_center"]
            lower = row["_trajectory_r2_lower"]
            upper = row["_trajectory_r2_upper"]
            return (
                np.nan if value is None else float(value),
                np.nan if lower is None else float(lower),
                np.nan if upper is None else float(upper),
            )
    return np.nan, np.nan, np.nan


def _asset_plot_recovery_curves(
    output_path: Path,
    *,
    sources: Sequence[_ExperimentSuiteSource],
    policy_ids: Sequence[str],
    r2_summary: str,
    single_column: bool = False,
    policy_labels: Mapping[str, str] | None = None,
) -> Path:
    """Standalone R^2 recovery-curve panels (one per condition).

    Conditions run across a double-column row by default; ``single_column``
    stacks them down a 3.5 in column instead, sharing one x axis.
    """
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    n_source = len(sources)
    if single_column:
        legend_ncol = 3
        legend_rows = int(np.ceil((len(policy_ids) + 1) / legend_ncol))
        legend_height = 0.16 * legend_rows + 0.08
        fig_height = 1.45 * n_source + 0.45 + legend_height
        fig, axes = plt_module.subplots(
            n_source,
            1,
            figsize=(_ASSET_SINGLE_COLUMN_WIDTH, fig_height),
            squeeze=False,
            sharex=True,
        )
        panel_axes = [axes[idx, 0] for idx in range(n_source)]
    else:
        legend_ncol = len(policy_ids) + 1
        fig_height = 2.35
        fig, axes = plt_module.subplots(
            1, n_source, figsize=(2.42 * n_source, fig_height), squeeze=False
        )
        panel_axes = [axes[0, idx] for idx in range(n_source)]
    for idx, source in enumerate(sources):
        _asset_plot_r2_curves(
            panel_axes[idx],
            source.suite_dir,
            policy_ids,
            title=f"{source.label}: recovery",
            panel_label=chr(65 + idx),
            ylabel=single_column or idx == 0,
            xlabel=idx == n_source - 1 if single_column else True,
            r2_summary=r2_summary,
            policy_labels=policy_labels,
        )
    handles, labels = panel_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=legend_ncol,
        fontsize=_ASSET_TICK_SIZE,
        columnspacing=0.9,
        handlelength=1.4,
    )
    top = 1.0 - (legend_height / fig_height) if single_column else 0.90
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top), w_pad=0.75, h_pad=0.7)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_plot_final_bar(
    output_path: Path,
    *,
    sources: Sequence[_ExperimentSuiteSource],
    policy_ids: Sequence[str],
    metric_rows: Sequence[Mapping[str, Any]],
    single_column: bool = False,
    short: bool = False,
    ylim: tuple[float, float] = (0.25, 1.0),
    r2_summary: str = "mean_sem",
    policy_labels: Mapping[str, str] | None = None,
    policy_legend: bool = True,
    ax: Any | None = None,
    cond_alphas: Sequence[float] | None = None,
    cond_hatches: Sequence[str] | None = None,
    cond_linestyles: Sequence[str] | None = None,
) -> Path:
    """Standalone final-performance bars, colored by policy with per-condition shade.

    Width tracks the policy count by default; ``single_column`` pins it to the
    3.5 in manuscript column instead, and ``short`` trims the axes to the flatter
    manuscript proportion. Bars and whiskers past ``ylim`` are drawn clipped, with
    a caret at the floor marking the ones that run off the bottom.
    ``cond_alphas``, ``cond_hatches``, and ``cond_linestyles`` override the
    per-condition face shade and add hatching or a dashed edge, for conditions
    that vary along different axes in one panel.
    """
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")

    active_policies = [
        policy_id
        for policy_id in policy_ids
        if any(
            np.isfinite(_asset_method_final_r2(metric_rows, source.exp_id, policy_id)[0])
            for source in sources
        )
    ]
    n_cond = len(sources)
    n_policy = len(active_policies)
    cond_alpha = np.linspace(1.0, 0.32, n_cond) if n_cond > 1 else np.array([1.0], dtype=np.float64)
    if cond_alphas is not None:
        cond_alpha = np.asarray(cond_alphas, dtype=np.float64)
    hatches = list(cond_hatches) if cond_hatches is not None else [""] * n_cond
    edge_styles = list(cond_linestyles) if cond_linestyles is not None else ["-"] * n_cond

    # Without the policy legend the x tick labels carry the policy names, so the
    # condition legend takes the strip above the axes instead of sitting inside it.
    n_legend = n_policy if policy_legend else n_cond + 1
    legend_ncol = min(n_legend, 4 if policy_legend else 2) if single_column else n_legend
    legend_rows = int(np.ceil(n_legend / max(legend_ncol, 1)))
    fig_width = _ASSET_SINGLE_COLUMN_WIDTH if single_column else 1.6 + 0.5 * max(n_policy, 1)
    # A wrapped policy legend needs its own strip above the axes, not axes height.
    # One row reserves 8% of the default figure, matching the unwrapped layout.
    legend_height = 0.188 + 0.16 * (legend_rows - 1)
    fig_height = (1.55 if short else 2.35) + 0.16 * (legend_rows - 1)
    standalone = ax is None
    if standalone:
        fig, ax = plt_module.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure
    y_floor, y_top = float(ylim[0]), float(ylim[1])
    x = np.arange(n_policy, dtype=np.float64)
    bar_width = 0.8 / max(n_cond, 1)
    clipped_x: list[float] = []
    for cond_idx, source in enumerate(sources):
        offset = (cond_idx - (n_cond - 1) / 2.0) * bar_width
        values, lower_errors, upper_errors, faces, edges = [], [], [], [], []
        for policy_idx, policy_id in enumerate(active_policies):
            value, lower, upper = _asset_method_final_r2(
                metric_rows, source.exp_id, policy_id
            )
            values.append(value)
            lower_errors.append(0.0 if not np.isfinite(lower) else max(0.0, value - lower))
            upper_errors.append(0.0 if not np.isfinite(upper) else max(0.0, upper - value))
            color = _asset_baseline_policy_color(policy_id)
            faces.append(mcolors.to_rgba(color, alpha=float(cond_alpha[cond_idx])))
            edges.append(color)
            if min(value, lower if np.isfinite(lower) else value) < y_floor:
                clipped_x.append(float(x[policy_idx] + offset))
        ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            yerr=np.asarray([lower_errors, upper_errors], dtype=np.float64),
            color=faces,
            edgecolor=edges,
            hatch=hatches[cond_idx] or None,
            linestyle=edge_styles[cond_idx],
            linewidth=0.6 if edge_styles[cond_idx] == "-" else 0.9,
            capsize=1.6,
            error_kw={"elinewidth": 0.6, "capthick": 0.6},
        )
        reference = _asset_true_model_r2_ceiling(source.suite_dir, r2_summary=r2_summary)
        if reference is not None:
            ax.axhline(reference, color=_experiment_C_NEUTRAL_LIGHT,
                       alpha=float(cond_alpha[cond_idx]), linestyle=":", linewidth=0.8,
                       zorder=5, clip_on=False,
                       label=f"true-model reference ({source.label})")

    ax.set_ylabel(_ASSET_FINAL_R2_LABEL)
    ax.set_gid("vf-roll-r2")
    ax.set_ylim(y_floor, y_top)
    if y_floor < 0.0:
        ax.set_yticks(np.arange(y_floor, y_top + 1e-9, 0.5))
        ax.axhline(0.0, color=_experiment_C_STROKE, linewidth=0.5)
        # Carets mark bars whose value or lower band runs past the axis floor;
        # the exact numbers stay in the companion CSV.
    if clipped_x:
        ax.plot(
            clipped_x,
            np.full(len(clipped_x), y_floor),
            marker="v",
            linestyle="none",
            markersize=2.2,
            color=_experiment_C_STROKE,
            clip_on=False,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_asset_policy_label(policy_id, policy_labels) for policy_id in active_policies],
        rotation=30,
        ha="right",
    )
    _style_manuscript_axis(ax, grid_axis="y")

    cond_handles = [
        Patch(
            facecolor=mcolors.to_rgba(_experiment_C_STROKE, alpha=float(cond_alpha[cond_idx])),
            edgecolor=_experiment_C_STROKE,
            hatch=hatches[cond_idx] or None,
            linestyle=edge_styles[cond_idx],
            linewidth=0.5 if edge_styles[cond_idx] == "-" else 0.8,
        )
        for cond_idx in range(n_cond)
    ]
    cond_labels = [source.label for source in sources]
    if any(line.get_label().startswith("true-model reference") for line in ax.lines):
        cond_handles.append(Line2D([0], [0], color=_experiment_C_NEUTRAL_LIGHT,
                                   linestyle=":", linewidth=0.8))
        cond_labels.append("true-model reference")
    if policy_legend:
        policy_handles = [
            Line2D([0], [0], color=_asset_baseline_policy_color(policy_id), linewidth=1.6)
            for policy_id in active_policies
        ]
        if standalone:
            fig.legend(
                policy_handles,
                [_asset_policy_label(policy_id, policy_labels) for policy_id in active_policies],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=legend_ncol,
                fontsize=_ASSET_TICK_SIZE,
                columnspacing=1.0,
                handlelength=1.4,
            )
        ax.legend(
            cond_handles,
            cond_labels,
            loc="upper left",
            fontsize=_ASSET_TICK_SIZE,
            ncol=min(len(cond_handles), max(1, int(fig_width / 1.5))),
            handlelength=1.2,
            borderpad=0.3,
            columnspacing=1.0,
        )
    else:
        fig.legend(
            cond_handles,
            cond_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=legend_ncol,
            fontsize=_ASSET_TICK_SIZE,
            columnspacing=1.0,
            handlelength=1.2,
        )
    if not standalone:
        return output_path
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - legend_height / fig_height))
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_plot_objective_ablation(output_path: Path, *, r2_summary: str) -> list[Path]:
    """Default/asymmetric ablation assets: final-R2 bars and recovery curves."""
    condition_labels = {
        "wilson_cowan": "Default",
        "wilson_cowan_asymmetric": "Asymmetric",
    }
    sources = [
        _ExperimentSuiteSource(
            source.exp_id, condition_labels[source.exp_id], source.suite_dir
        )
        for source in _experiment_objective_sources()
        if source.exp_id in condition_labels
    ]
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    metric_rows = _asset_method_metric_rows(
        sources,
        _experiment_OBJECTIVE_POLICIES,
        r2_summary=r2_summary,
    )
    _asset_write_method_csv(
        output_path.with_suffix(".csv"),
        metric_rows,
        r2_summary=r2_summary,
    )
    recovery_path = output_path.with_name(f"{output_path.stem}_recovery{output_path.suffix}")
    return [
        _asset_plot_final_bar(
            output_path,
            sources=sources,
            policy_ids=_experiment_OBJECTIVE_POLICIES,
            metric_rows=metric_rows,
            r2_summary=r2_summary,
            single_column=True,
            ylim=(0.0, 1.18),  # headroom for the condition legend above the bars
        ),
        _asset_plot_recovery_curves(
            recovery_path,
            sources=sources,
            policy_ids=_experiment_OBJECTIVE_POLICIES,
            r2_summary=r2_summary,
            single_column=True,
        ),
    ]


# Designed three-gate objective diagnostic (compact Poisson observations).
# The suite lives in the shared result tracks (objective_ablation group); the
# asset reads the raw run traces because its panels need per-seed occupancy and
# final-value quantiles that the suite summary does not carry.
_ASSET_TRI_GATE_EXP_ID = "three_gate_diagnostic"
_ASSET_TRI_GATE_LABELS = {
    "compound_active_planning": "PALDI",
    "compound_active_fully_observable": "Unatten.",
    "compound_active_e_optimality": "E-opt.",
    "compound_active_state_information": "State Information",
    "compound_active_dynamics": "Dyn. sens. (trace)",
    "compound_active_dynamics_logdet": "Dyn. sens.",
    "compound_active_observation_variance": "Obs. var.",
    "compound_active_state_variance": "State var.",
    "prbs": "PRBS",
    "random": "Random",
}
_ASSET_TRI_GATE_CENTERS = (-0.5, -0.1, 0.3)
_ASSET_TRI_GATE_WIDTH = 0.1
# Policies present in the suite but kept out of the polished manuscript figure:
# the nonadaptive PRBS control and the trace dynamics-sensitivity variant (the
# paper reports the rank-aware logdet form as "Dynamics sensitivity").
_ASSET_TRI_GATE_EXCLUDED_POLICIES = frozenset(
    {"prbs", "compound_active_dynamics"}
)
# Exemplar seed for the trajectory panels, chosen by ranking matched seeds on
# occupancy contrast: PALDI holds gate F while the unattenuated objective
# abandons F for gate N, with every panel showing its policy's modal behavior.
# Population occupancy statistics live in the main diagnostic figure.
_ASSET_TRI_GATE_EXEMPLAR_SEED = 90
_ASSET_TRI_GATE_REST_CENTER = -1.0
_ASSET_TRI_GATE_REST_CUTOFF = -0.75
_ASSET_TRI_GATE_R2_YLIM = (0.25, 1.0)
# Gate identity colors couple the occupancy stacks (panel B) to the selector
# traces (panel C); muted tones of the manuscript palette, labeled in dark text.
_ASSET_TRI_GATE_GATE_COLORS = (
    ("rest_fraction", "Rest", "#D5CFC6"),
    ("gate_A_fraction", "N: confounded", "#6FAE97"),
    ("gate_B_fraction", "B: weak, balanced", "#9A8BCB"),
    ("gate_M_fraction", "F: full rank", "#DD8F85"),
)


def _asset_tri_gate_assignment_bands(top: float) -> list[tuple[float, float, str]]:
    """Gate assignment regions (midpoint boundaries), tiling the axis gap-free.

    These are the occupancy-classification regions, not the Gaussian gate
    support; the gate width stays w = 0.1 in the dynamics.
    """
    gate_a, gate_b, gate_m = _ASSET_TRI_GATE_CENTERS
    mid_ab = 0.5 * (gate_a + gate_b)
    mid_bm = 0.5 * (gate_b + gate_m)
    colors = [color for _key, _label, color in _ASSET_TRI_GATE_GATE_COLORS[1:]]
    return [
        (_ASSET_TRI_GATE_REST_CUTOFF, mid_ab, colors[0]),
        (mid_ab, mid_bm, colors[1]),
        (mid_bm, float(top), colors[2]),
    ]


# Tri-gate objective colors from the shared pastel manuscript palette. The shared
# unattenuated green is near the state-variance green, so the unattenuated
# objective takes the palette's blue; the state-information yellow is one step
# darker so its trace stays visible on white.
_ASSET_TRI_GATE_POLICY_COLORS = {
    "compound_active_planning": "#F1948A",
    "compound_active_fully_observable": "#5DADE2",
    "compound_active_e_optimality": "#BB8FCE",
    "compound_active_state_information": "#EBC237",
    "compound_active_dynamics": "#A3E4D7",
    "compound_active_dynamics_logdet": "#76D7C4",
    "compound_active_observation_variance": "#D2B48C",
    "compound_active_state_variance": "#58D68D",
    "random": "#9EA7AD",
}


def _asset_shade(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    """Darker shade of ``color``: RGB scaled by ``factor`` (lines and edges on pastel fills)."""
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _asset_tri_gate_policy_color(policy_id: str) -> str:
    """Distinguishable per-objective color for the overlaid tri-gate panels."""
    color = _ASSET_TRI_GATE_POLICY_COLORS.get(policy_id)
    if color is not None:
        return color
    base = policy_id.removeprefix("compound_")
    return _asset_baseline_policy_color(
        "adaptive" if base == "active_planning" else base
    )


def _asset_plot_gate_diagnostic(
    output_path: Path,
    *,
    r2_summary: str,
    result_roots: Sequence[Path],
    exemplar_seed: int = _ASSET_TRI_GATE_EXEMPLAR_SEED,
    exp_id: str = _ASSET_TRI_GATE_EXP_ID,
) -> Path:
    """Manuscript figure for the designed three-gate objective diagnostic.

    ``exp_id`` selects the suite whose runs are read (``three_gate_diagnostic``
    or a retuned variant such as ``three_gate_tradeoff`` sharing its gates).

    Single row: (A) final rollout R2 per objective, (B) selector occupancy, (C)
    every objective's exemplar selector trace overlaid on the gate assignment
    bands. The nonadaptive PRBS control and the trace dynamics-sensitivity
    variant are omitted (the paper reports the rank-aware logdet form as
    "Dyn. sens.").
    """
    _asset_parse_r2_summaries(r2_summary)
    records = _compound_trace_records(result_roots, exp_id=exp_id)
    if not records:
        roots_text = ", ".join(str(root) for root in result_roots)
        raise RuntimeError(
            f"No trajectory R2 curves available for {exp_id} in {roots_text}"
        )
    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")

    summary_rows = [
        row
        for row in _compound_summary_rows(
            records,
            gate_centers=_ASSET_TRI_GATE_CENTERS,
            rest_cutoff=_ASSET_TRI_GATE_REST_CUTOFF,
        )
        if str(row["policy_id"]) not in _ASSET_TRI_GATE_EXCLUDED_POLICIES
    ]
    _write_csv(
        output_path.with_suffix(".csv"),
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
    exemplar_by_policy = {
        record.policy_id: record
        for record in records
        if record.seed == int(exemplar_seed)
    }
    line_styles = {
        "compound_active_planning": "-",
        "compound_active_fully_observable": "--",
        "compound_active_e_optimality": "-.",
        "compound_active_state_information": ":",
        "compound_active_dynamics_logdet": (0, (5, 1, 1, 1)),
        "compound_active_observation_variance": (0, (3, 1, 1, 1, 1, 1)),
        "compound_active_state_variance": (0, (2, 2)),
        "random": (0, (6, 3)),
    }

    # IEEEtran journal text width is 43 picas (516 TeX points).
    manuscript_width_in = 516.0 / 72.27
    fig, axis_grid = plt_module.subplots(
        1, 3, figsize=(manuscript_width_in, 2.1),
        gridspec_kw={"width_ratios": (2.5, 2.5, 5.0)},
    )
    axes = list(axis_grid.ravel())
    x = np.arange(len(summary_rows), dtype=np.float64)

    # A: final rollout R2, one bar per objective.
    ax = axes[0]
    if r2_summary == "median_iqr":
        r2_center = np.asarray(
            [row["trajectory_r2_median"] for row in summary_rows], dtype=np.float64
        )
        r2_yerr = np.vstack(
            (
                r2_center
                - np.asarray(
                    [row["trajectory_r2_q25"] for row in summary_rows], dtype=np.float64
                ),
                np.asarray(
                    [row["trajectory_r2_q75"] for row in summary_rows], dtype=np.float64
                )
                - r2_center,
            )
        )
    else:
        r2_center = np.asarray(
            [row["trajectory_r2_mean"] for row in summary_rows], dtype=np.float64
        )
        r2_yerr = np.asarray(
            [row["trajectory_r2_sem"] for row in summary_rows], dtype=np.float64
        )
    bar_colors = [
        _asset_tri_gate_policy_color(str(row["policy_id"])) for row in summary_rows
    ]
    ax.bar(
        x,
        r2_center,
        yerr=r2_yerr,
        color=bar_colors,
        edgecolor=[_asset_shade(color) for color in bar_colors],
        linewidth=0.7,
        capsize=1.6,
        error_kw={"elinewidth": 0.6, "capthick": 0.6},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_ASSET_TRI_GATE_LABELS[str(row["policy_id"])] for row in summary_rows],
        rotation=30,
        ha="right",
    )
    reference = _asset_true_model_r2_ceiling(records[0].run_dir.parents[2], r2_summary=r2_summary)
    if reference is not None:
        ax.axhline(reference, color=_experiment_C_NEUTRAL_LIGHT, linestyle=":",
                   linewidth=0.8, zorder=5, clip_on=False, label="true-model reference")
    lower = r2_center - (r2_yerr[0] if r2_yerr.ndim == 2 else r2_yerr)
    clipped = lower < _ASSET_TRI_GATE_R2_YLIM[0]
    if clipped.any():
        ax.plot(x[clipped], np.full(clipped.sum(), _ASSET_TRI_GATE_R2_YLIM[0]),
                marker="v", linestyle="none", markersize=2.2,
                color=_experiment_C_STROKE, clip_on=False)
    ax.set_ylim(*_ASSET_TRI_GATE_R2_YLIM)
    ax.set_gid("vf-roll-r2")
    ax.set_ylabel(_ASSET_FINAL_R2_LABEL)
    _style_experiment_axis(ax)
    ax.set_title(
        "A", loc="left", fontweight="bold", fontsize=_ASSET_PANEL_LABEL_SIZE, pad=3.0
    )

    # B: selector occupancy stacks, one per objective.
    ax = axes[1]
    bottom = np.zeros(len(summary_rows), dtype=np.float64)
    for key, label, color in _ASSET_TRI_GATE_GATE_COLORS:
        value = np.asarray([row[key] for row in summary_rows], dtype=np.float64)
        ax.bar(x, value, bottom=bottom, width=0.72, color=color, edgecolor="white", linewidth=0.4)
        # Direct labels keep the occupancy comparison readable in grayscale.
        for column, fraction in enumerate(value):
            if fraction >= 0.25:
                is_rest = key == "rest_fraction"
                ax.text(
                    x[column],
                    bottom[column] + fraction / 2,
                    "rest" if is_rest else label.split(":")[0],
                    ha="center",
                    va="center",
                    fontsize=_ASSET_TICK_SIZE,
                    color=_experiment_C_STROKE,
                    rotation=90 if is_rest else 0,
                )
        bottom += value
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_ASSET_TRI_GATE_LABELS[str(row["policy_id"])] for row in summary_rows],
        rotation=30,
        ha="right",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of steps")
    _style_experiment_axis(ax)
    ax.set_title(
        "B", loc="left", fontweight="bold", fontsize=_ASSET_PANEL_LABEL_SIZE, pad=3.0
    )

    # C: every objective's exemplar selector trace overlaid on the gate bands,
    # so dwell-at-N, dwell-at-B, and reach-and-hold-F behaviors read against the
    # shared gate assignment regions.
    ax = axes[2]
    rest_color = _ASSET_TRI_GATE_GATE_COLORS[0][2]
    y_bottom, y_top = -1.2, 0.62
    ax.axhspan(
        y_bottom, _ASSET_TRI_GATE_REST_CUTOFF, color=rest_color, alpha=0.16, linewidth=0.0
    )
    for low, high, color in _asset_tri_gate_assignment_bands(y_top):
        ax.axhspan(low, high, color=color, alpha=0.13, linewidth=0.0)
    ax.axhline(
        _ASSET_TRI_GATE_REST_CENTER, color=rest_color, linestyle="--", linewidth=0.6
    )
    max_steps = 1
    for policy_id in _COMPOUND_POLICY_ORDER:
        if policy_id in _ASSET_TRI_GATE_EXCLUDED_POLICIES:
            continue
        record = exemplar_by_policy.get(policy_id)
        if record is None:
            continue
        rows = read_trace_csv(record.run_dir / "state_action_trace.csv")
        selector = np.asarray([float(row["true_x"]) for row in rows], dtype=np.float64)
        max_steps = max(max_steps, selector.size)
        is_paldi = policy_id == "compound_active_planning"
        ax.plot(
            np.arange(selector.size, dtype=np.float64),
            selector,
            color=_asset_shade(_asset_tri_gate_policy_color(policy_id), 0.9 if is_paldi else 0.8),
            linestyle=line_styles[policy_id],
            linewidth=1.4 if is_paldi else 0.75,
            alpha=1.0 if is_paldi else 0.9,
            zorder=3 if is_paldi else 2,
        )
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(0.0, float(max_steps))
    for (_key, label, color), center in zip(
        _ASSET_TRI_GATE_GATE_COLORS[1:], _ASSET_TRI_GATE_CENTERS, strict=True
    ):
        ax.text(
            1.02,
            center,
            label.split(":")[0],
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=_ASSET_TICK_SIZE,
            fontweight="bold",
            color=_asset_shade(color, 0.75),
        )
    ax.text(
        1.02,
        _ASSET_TRI_GATE_REST_CENTER,
        "rest",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=_ASSET_TICK_SIZE,
        color=_experiment_C_STROKE,
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Selector $r$")
    _style_experiment_axis(ax)
    ax.set_title(
        "C", loc="left", fontweight="bold", fontsize=_ASSET_PANEL_LABEL_SIZE, pad=3.0
    )

    from matplotlib.lines import Line2D

    legend_policies = [str(row["policy_id"]) for row in summary_rows]
    fig.legend(
        [
            Line2D(
                [0], [0],
                color=_asset_shade(_asset_tri_gate_policy_color(policy_id), 0.8),
                linestyle=line_styles[policy_id],
                linewidth=1.6,
            )
            for policy_id in legend_policies
        ],
        [_ASSET_TRI_GATE_LABELS[policy_id] for policy_id in legend_policies],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(legend_policies),
        fontsize=_ASSET_TICK_SIZE,
        columnspacing=0.9,
        handlelength=2.6,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9), w_pad=1.1)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_plot_gate_diagnostic_trajectories(
    output_path: Path,
    *,
    result_roots: Sequence[Path],
    exemplar_seed: int = _ASSET_TRI_GATE_EXEMPLAR_SEED,
    exp_id: str = _ASSET_TRI_GATE_EXP_ID,
) -> Path:
    """Appendix companion: one exemplar selector trace per acquisition objective.

    Gate bands (center +/- one gate width) and the rest line reuse the gate
    identity colors of the main diagnostic figure, so dwell-at-N, dwell-at-B,
    and reach-and-hold-F behaviors are visible directly.
    """
    records = [
        record
        for record in _compound_trace_records(result_roots, exp_id=exp_id)
        if record.seed == int(exemplar_seed)
    ]
    if not records:
        roots_text = ", ".join(str(root) for root in result_roots)
        raise RuntimeError(
            f"No trajectory R2 curves available for {exp_id} "
            f"seed {exemplar_seed} in {roots_text}"
        )
    by_policy = {record.policy_id: record for record in records}
    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")

    policy_ids = [
        policy_id
        for policy_id in _COMPOUND_POLICY_ORDER
        if policy_id in by_policy
        and policy_id != "compound_active_dynamics"
    ]
    n_col = 3
    n_row = int(np.ceil(len(policy_ids) / n_col))
    fig, axes = plt_module.subplots(
        n_row,
        n_col,
        figsize=(_ASSET_SINGLE_COLUMN_WIDTH, 1.05 * n_row + 0.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    rest_color = _ASSET_TRI_GATE_GATE_COLORS[0][2]
    y_bottom, y_top = -1.2, 0.62
    for idx, policy_id in enumerate(policy_ids):
        ax = axes[idx // n_col, idx % n_col]
        ax.axhspan(
            y_bottom,
            _ASSET_TRI_GATE_REST_CUTOFF,
            color=rest_color,
            alpha=0.22,
            linewidth=0.0,
        )
        for low, high, color in _asset_tri_gate_assignment_bands(y_top):
            ax.axhspan(low, high, color=color, alpha=0.14, linewidth=0.0)
        ax.axhline(
            _ASSET_TRI_GATE_REST_CENTER, color=rest_color, linestyle="--", linewidth=0.6
        )
        rows = read_trace_csv(by_policy[policy_id].run_dir / "state_action_trace.csv")
        selector = np.asarray([float(row["true_x"]) for row in rows], dtype=np.float64)
        ax.plot(
            np.arange(selector.size, dtype=np.float64),
            selector,
            color=_asset_tri_gate_policy_color(policy_id),
            linewidth=0.55,
        )
        ax.set_ylim(y_bottom, y_top)
        ax.set_xlim(0.0, float(max(selector.size, 1)))
        ax.set_title(
            _ASSET_TRI_GATE_LABELS[policy_id], fontsize=_ASSET_TITLE_SIZE, pad=2.0
        )
        _style_experiment_axis(ax)
        ax.tick_params(axis="both", labelsize=_ASSET_TICK_SIZE, pad=1.0)
    for idx in range(len(policy_ids), n_row * n_col):
        axes[idx // n_col, idx % n_col].set_visible(False)
    # Keep gate labels on a visible panel when fewer than three policies exist.
    right_ax = axes[0, min(n_col, len(policy_ids)) - 1]
    for (_key, label, color), center in zip(
        _ASSET_TRI_GATE_GATE_COLORS[1:], _ASSET_TRI_GATE_CENTERS, strict=True
    ):
        right_ax.text(
            1.03,
            center,
            label.split(":")[0],
            transform=right_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=_ASSET_TICK_SIZE,
            fontweight="bold",
            color=color,
        )
    right_ax.text(
        1.03,
        _ASSET_TRI_GATE_REST_CENTER,
        "rest",
        transform=right_ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=_ASSET_TICK_SIZE,
        color=_experiment_C_STROKE,
    )
    axes[n_row - 1, n_col // 2].set_xlabel("Environment steps")
    axes[n_row // 2, 0].set_ylabel("Selector $r$")
    fig.tight_layout(w_pad=0.5, h_pad=0.7)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_flex_groups() -> tuple[tuple[str, tuple[_ExperimentSuiteSource, ...]], ...]:
    """Group the FLEX-variant suites into the three manuscript panels."""
    display_titles = {
        "duffing": "Duffing",
        "damped_pendulum": "Damped Pendulum",
        "wilson_cowan": "Wilson-Cowan",
        "wilson_cowan_asymmetric": "Asymmetric",
        "wilson_cowan_challenging": "Challenging",
        "wilson_cowan_observation_bottleneck_mild": "SNR -10 dB",
        "wilson_cowan_observation_bottleneck_strong": "SNR -15 dB",
    }
    sources = {
        ref.suite_id: _ExperimentSuiteSource(
            ref.suite_id,
            display_titles.get(ref.suite_id, ref.label),
            ref.results_root / "tracks" / ref.suite_id,
        )
        for ref in _groups_mod.groups()["flex_comparison"]
    }
    grouped = (
        ("baseline", ("duffing", "damped_pendulum", "wilson_cowan")),
        ("hard", ("wilson_cowan_asymmetric", "wilson_cowan_challenging")),
        (
            "snr",
            (
                "wilson_cowan_observation_bottleneck_mild",
                "wilson_cowan_observation_bottleneck_strong",
            ),
        ),
    )
    missing = sorted(
        suite_id
        for _suffix, suite_ids in grouped
        for suite_id in suite_ids
        if suite_id not in sources
    )
    if missing:
        raise RuntimeError(
            "flex_comparison group is missing suite(s): " + ", ".join(missing)
        )
    return tuple(
        (suffix, tuple(sources[suite_id] for suite_id in suite_ids))
        for suffix, suite_ids in grouped
    )


def _asset_plot_flex_comparison(
    output_path: Path, *, r2_summary: str,
    skipped: list[tuple[str, str]] | None = None,
) -> list[Path]:
    """FLEX state-source/update variants: short final-R2 bars plus recovery curves.

    One bar figure and one recovery figure per condition group, matching how the
    constraints panels are assembled for the manuscript.
    """
    written: list[Path] = []
    for suffix, sources in _asset_flex_groups():
        if skipped is not None:
            available = []
            for source in sources:
                rows = read_trace_csv(source.suite_dir / "summary" / "metrics.csv")
                if any(row.get("policy_id") in _ASSET_FLEX_POLICIES for row in rows):
                    available.append(source)
                else:
                    skipped.append((str(source.suite_dir), "No FLEX runs; condition omitted from FLEX assets"))
            sources = tuple(available)
            if not sources:
                continue
        _asset_require_suite_dirs([source.suite_dir for source in sources])
        metric_rows = _asset_method_metric_rows(
            sources,
            _ASSET_FLEX_POLICIES,
            r2_summary=r2_summary,
        )
        for row in metric_rows:
            row["policy_label"] = _ASSET_FLEX_LABELS[str(row["policy_id"])]
        bar_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
        curves_path = output_path.with_name(
            f"{output_path.stem}_{suffix}_recovery{output_path.suffix}"
        )
        _asset_write_method_csv(
            bar_path.with_suffix(".csv"),
            metric_rows,
            r2_summary=r2_summary,
        )
        written.append(
            _asset_plot_final_bar(
                bar_path,
                sources=sources,
                policy_ids=_ASSET_FLEX_POLICIES,
                metric_rows=metric_rows,
                r2_summary=r2_summary,
                single_column=True,
                short=True,
                ylim=_ASSET_FLEX_BAR_YLIM,
                policy_labels=_ASSET_FLEX_LABELS,
                policy_legend=False,
            )
        )
        written.append(
            _asset_plot_recovery_curves(
                curves_path,
                sources=sources,
                policy_ids=_ASSET_FLEX_POLICIES,
                r2_summary=r2_summary,
                policy_labels=_ASSET_FLEX_LABELS,
                single_column=len(sources) == 1,
            )
        )
    return written


def _asset_plot_flex_combined(output_path: Path, *, r2_summary: str) -> Path:
    """Six-condition FLEX recovery panel from the configured saved summaries."""
    sources = [source for _, group in _asset_flex_groups() for source in group
               if source.exp_id != "wilson_cowan_challenging"]
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    plt = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt.subplots(2, 3, figsize=(516 / 72.27, 4.4))
    for idx, (ax, source) in enumerate(zip(axes.flat, sources, strict=True)):
        curves = _asset_r2_curve_rows(source.suite_dir, r2_summary=r2_summary)
        for policy in _ASSET_FLEX_POLICIES:
            rows = curves.get(policy, [])
            if not rows:
                raise RuntimeError(f"Missing {policy} curves in {source.suite_dir}")
            steps = [row["step"] for row in rows]
            color = _asset_baseline_policy_color(policy)
            ax.plot(steps, [row["center"] for row in rows], color=color,
                    linewidth=.9, label=_ASSET_FLEX_LABELS[policy])
            ax.fill_between(steps, [row["lower"] for row in rows],
                            [row["upper"] for row in rows], color=color,
                            alpha=.1, linewidth=0)
        ax.set_yscale("symlog", linthresh=.1)
        ax.set_xlim(left=0)
        ax.set_xlabel("Environment steps")
        if idx % 3 == 0:
            ax.set_ylabel(_ASSET_PREDICTIVE_R2_LABEL)
        ax.set_title(chr(65 + idx), loc="left", fontweight="bold")
        ax.set_title(source.label, fontsize=_ASSET_TITLE_SIZE)
        _style_experiment_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=_ASSET_TICK_SIZE)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=.8, h_pad=.8)
    return save_figure(fig, output_path, plt_module=plt)


def _asset_plot_constraints_combined(
    output_path: Path, *,
    curve_source: _ExperimentSuiteSource,
    bar_sources: Sequence[_ExperimentSuiteSource],
    bar_rows: Sequence[Mapping[str, Any]],
    r2_summary: str,
) -> Path:
    """Wilson-Cowan identification at the manuscript's 516 pt text width.

    (A) R2_VF-roll against environment steps under the default observations;
    (B) final R2_VF-roll at the default, lower SNR, and biased loading, one panel:
    SNR levels shaded dark to light, biased loading as a diagonal-hatch fill.
    """
    from matplotlib.lines import Line2D

    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(1, 2, figsize=(516.0 / 72.27, 1.8),
                                    gridspec_kw={"width_ratios": [0.62, 1.55]})
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.79, wspace=0.16)
    _asset_plot_r2_curves(axes[0], curve_source.suite_dir, _ASSET_MATCHED_POLICIES, title="", panel_label="",
                          ylabel=True, xlabel=True, r2_summary=r2_summary, ylim=(0.25, 1.0), title_pad=1.0)
    axes[0].set_xticks([0, 1000, 2000])
    axes[0].xaxis.labelpad = 1.0
    axes[0].yaxis.labelpad = 0.0
    axes[0].text(-0.2, 1.10, "A", transform=axes[0].transAxes, fontsize=_ASSET_PANEL_LABEL_SIZE,
                 fontweight="bold")
    ax = axes[1]
    _asset_plot_final_bar(
        output_path, sources=bar_sources, policy_ids=_ASSET_MATCHED_POLICIES,
        metric_rows=bar_rows, r2_summary=r2_summary, ylim=(0.0, 1.0), ax=ax,
        cond_alphas=(1.0, 0.55, 0.22, 0.0), cond_hatches=("", "", "", "//////"),
    )
    legend = ax.get_legend()
    ax.legend(legend.legend_handles,
              [text.get_text().replace("true-model reference", "True model")
               for text in legend.get_texts()],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(legend.legend_handles), frameon=False,
              fontsize=_ASSET_TICK_SIZE, handlelength=1.0, borderpad=0.1, borderaxespad=0.1, columnspacing=0.8)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel("")
    ax.text(-0.06, 1.10, "B", transform=ax.transAxes, fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold")
    fig.legend(
        [Line2D([0], [0], color=_asset_baseline_policy_color(policy), linewidth=1.6)
         for policy in _ASSET_MATCHED_POLICIES],
        [_asset_policy_label(policy) for policy in _ASSET_MATCHED_POLICIES],
        loc="upper left", bbox_to_anchor=(0.075, 1.015), ncol=6,
        fontsize=_ASSET_TICK_SIZE, columnspacing=1.0, handlelength=1.4,
    )
    _asset_write_method_csv(output_path.with_suffix(".csv"), bar_rows, r2_summary=r2_summary)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_plot_constraints(
    output_path: Path, *, r2_summary: str,
    skipped: list[tuple[str, str]] | None = None,
) -> list[Path]:
    bottleneck_sources = _asset_bottleneck_sources()
    figures = (
        ("snr", "Observation SNR", tuple(bottleneck_sources[:3])),
        (
            "asymmetry",
            "Asymmetry",
            (
                _ExperimentSuiteSource(
                    "wilson_cowan",
                    "Default",
                    _suite_dir("simple_system_identification", "wilson_cowan"),
                ),
                _ExperimentSuiteSource(
                    "wilson_cowan_asymmetric",
                    "Asymmetric",
                    _suite_dir("observation_action_bottleneck", "wilson_cowan_asymmetric"),
                ),
            ),
        ),
        ("action", "Action budget", (bottleneck_sources[0], *bottleneck_sources[3:])),
    )
    written: list[Path] = []
    observation_panels = []
    for suffix, _figure_title, sources in figures:
        try:
            _asset_require_suite_dirs([source.suite_dir for source in sources])
        except FileNotFoundError as exc:
            if skipped is None:
                raise
            skipped.append((str(output_path.with_stem(f"{output_path.stem}_{suffix}")), str(exc)))
            continue
        metric_rows = _asset_method_metric_rows(
            sources,
            _ASSET_MATCHED_POLICIES,
            r2_summary=r2_summary,
        )
        if suffix in {"snr", "asymmetry"}:
            observation_panels.append((sources, metric_rows))
        bar_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
        curves_path = output_path.with_name(
            f"{output_path.stem}_{suffix}_recovery{output_path.suffix}"
        )
        _asset_write_method_csv(
            bar_path.with_suffix(".csv"),
            metric_rows,
            r2_summary=r2_summary,
        )
        # The rollback-stabilized FLEX variant is safe to retain in final-R2 bars.
        bar_policies = list(_ASSET_MATCHED_POLICIES)
        written.append(
            _asset_plot_final_bar(
                bar_path,
                sources=sources,
                policy_ids=bar_policies,
                metric_rows=metric_rows,
                r2_summary=r2_summary,
            )
        )
        written.append(
            _asset_plot_recovery_curves(
                curves_path,
                sources=sources,
                policy_ids=_ASSET_MATCHED_POLICIES,
                r2_summary=r2_summary,
            )
        )
    if len(observation_panels) == 2:
        # One bar panel: default and lower SNR, then biased loading (the default is shared).
        (snr_sources, snr_rows), (asym_sources, asym_rows) = observation_panels
        biased = [source for source in asym_sources if source.exp_id == "wilson_cowan_asymmetric"]
        bar_sources = [*snr_sources, *(_ExperimentSuiteSource(s.exp_id, "Biased", s.suite_dir) for s in biased)]
        bar_rows = [*snr_rows, *(row for row in asym_rows if row["experiment"] == "wilson_cowan_asymmetric")]
        written.append(_asset_plot_constraints_combined(
            output_path, curve_source=bottleneck_sources[0], bar_sources=bar_sources, bar_rows=bar_rows,
            r2_summary=r2_summary,
        ))
    return written


def _asset_plot_eig_components(output_path: Path, *, results_dir: Path) -> list[Path]:
    """Render both column widths from the saved scalar arrays and parameters."""
    from experiments.eig_1d_example import _apply_eig_style, build_figure

    results_dir = results_dir.resolve()
    array_path = results_dir / "figure_mechanistic.npz"
    metadata_path = results_dir / "figure_mechanistic.json"
    metadata = json.loads(metadata_path.read_text())
    with np.load(array_path, allow_pickle=False) as saved:
        curve = {key: saved[key] for key in saved.files}
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (array_path, metadata_path)}
    plt = load_plotting(output_path, apply_style=_apply_eig_style,
                        path_is_file=True, use_agg=True)
    if plt is None:
        raise RuntimeError("Matplotlib is required for the mechanistic figure")
    written = []
    for column in ("double", "single"):
        path = (output_path if column == "double" else
                output_path.with_stem(f"{output_path.stem}_single"))
        fig = build_figure(curve, **{key: metadata[key] for key in
                           ("theta_mean", "theta_var", "c", "b", "dt")},
                           plt=plt, single_column=column == "single")
        written.append(save_figure(fig, path, plt_module=plt))
        shutil.copyfile(array_path, path.with_suffix(".npz"))
        path.with_suffix(".json").write_text(json.dumps(
            metadata | {"output": str(path), "column": column,
                        "source_sha256": inputs, "rendered_from_saved_arrays": True},
            indent=2) + "\n")
        caption = results_dir / "caption.tex"
        if caption.is_file():
            shutil.copyfile(caption, path.with_suffix(".caption.tex"))
    return written


# Neural-circuit benchmarks (Wilson-Cowan, Wong-Wang): identification curves plus
# the Wong-Wang decision-reversal score from basin_switch_*_over_steps.csv.
_NEURAL_SUITES = (("wilson_cowan", "Wilson-Cowan"), ("wong_wang", "Wong-Wang"))
_NEURAL_POLICIES = ("adaptive", "active_myopic", "prbs", "random")
_NEURAL_TRAJ_POLICIES = ("adaptive", "active_myopic", "random")
_NEURAL_TRAJ_STEPS = 2000
# Third seed of the suite: the Wilson-Cowan trial that starts in the down-state basin.
_NEURAL_TRAJ_SEED_INDEX = 2
_NEURAL_PHASE_LIM = 2.5
_NEURAL_R2_YLIM = (0.1, 1.0)


def _asset_switch_curve_rows(
    suite_dir: Path, value_col: str
) -> dict[str, list[tuple[float, float, float]]]:
    """Read (step, mean, sem) rows of one basin-switch summary column per policy."""
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for row in read_trace_csv(suite_dir / "summary" / f"{value_col}_over_steps.csv"):
        policy_id = str(row.get("policy_id", ""))
        step = _safe_float(row.get("step"))
        center = _safe_float(row.get(f"{value_col}_mean"))
        sem = _safe_float(row.get("value_sem"))
        if not policy_id or step is None or center is None:
            continue
        grouped.setdefault(policy_id, []).append((step, center, 0.0 if sem is None else sem))
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
    return grouped


def _asset_plot_neural_phase(
    ax: Any,
    suite_dir: Path,
    *,
    title: str,
    panel_label: str,
    markers: Sequence[tuple[np.ndarray, str]] = (),
) -> None:
    """Neutral vector field with executed latent trajectories of two policies."""
    from .diagnostics import true_dynamics as _true_dynamics
    from .records import collect_records as _collect_records

    metadata = _asset_first_suite_metadata(suite_dir)
    if metadata is None:
        raise RuntimeError(f"No run metadata found under {suite_dir}")
    env_preset = get_environment_preset_from_metadata(metadata)
    plot_neutral_vector_field(
        ax,
        _true_dynamics(env_preset),
        grid_lim=_NEURAL_PHASE_LIM,
        n_grid=41,
        arrowsize=0.35,
        stroke_color=_experiment_C_STROKE,
    )
    for policy_id in _NEURAL_TRAJ_POLICIES:
        records = _collect_records(suite_dir, [policy_id], completed_only=True)
        if len(records) <= _NEURAL_TRAJ_SEED_INDEX:
            continue
        record = records[_NEURAL_TRAJ_SEED_INDEX]
        traj = _experiment_load_xy_trace(record)[: _NEURAL_TRAJ_STEPS]
        if traj.shape[0] == 0:
            continue
        color = _asset_baseline_policy_color(policy_id)
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.45, alpha=0.85, zorder=3,
                solid_capstyle="round")
        ax.scatter(traj[0, 0], traj[0, 1], s=9, color=color, edgecolor="white",
                   linewidth=0.3, zorder=4)
    for point, label in markers:
        ax.scatter(point[0], point[1], s=16, marker="s", facecolor="white",
                   edgecolor=_experiment_C_STROKE, linewidth=0.6, zorder=5)
        ax.annotate(label, (point[0], point[1]), xytext=(3.0, 2.0), textcoords="offset points",
                    fontsize=_ASSET_TICK_SIZE, color=_experiment_C_STROKE, zorder=6)
    ax.set_xlim(-_NEURAL_PHASE_LIM, _NEURAL_PHASE_LIM)
    ax.set_ylim(-_NEURAL_PHASE_LIM, _NEURAL_PHASE_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_title(title, fontsize=_ASSET_TITLE_SIZE, pad=2.5)
    ax.annotate(
        panel_label, (0, 1), xycoords="axes fraction",
        xytext=(-7.2, 1.2), textcoords="offset points", ha="left", va="bottom",
        fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold",
    )


def _asset_plot_neural_circuits(output_path: Path, *, r2_summary: str) -> Path:
    """Neural-circuit benchmark figure: phase portraits, R2 curves, decision reversal.

    Panels A/C draw the true vector fields with the executed latent trajectory
    of the seed at index ``_NEURAL_TRAJ_SEED_INDEX`` for PALDI, Myopic, and Random over
    the first ``_NEURAL_TRAJ_STEPS`` steps. Panels B/D reuse the manuscript R2 curves. Panel E reads the
    Wong-Wang basin-switch summaries: control energy ``dt * sum ||u_t||^2`` of
    the input planned on the current estimate (mean +/- SEM over seeds), the
    same quantity planned on the true parameters (dotted), and the fraction of
    seeds whose planned input reversed the decision (top strip).
    """
    from matplotlib.ticker import FixedLocator, FormatStrFormatter

    suite_dirs = {
        "wilson_cowan": _suite_dir("simple_system_identification", "wilson_cowan"),
        "wong_wang": _suite_dir("neural_circuits", "wong_wang"),
    }
    _asset_require_suite_dirs(list(suite_dirs.values()))
    ww_dir = suite_dirs["wong_wang"]
    ww_metadata = _asset_first_suite_metadata(ww_dir)
    ww_preset = get_environment_preset_from_metadata(ww_metadata)
    source = np.asarray(ww_preset.basin_switch_source, dtype=np.float64)
    target = np.asarray(ww_preset.basin_switch_target, dtype=np.float64)

    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig = plt_module.figure(figsize=(516.0 / 72.27, 1.95))
    gs = fig.add_gridspec(
        1, 5, width_ratios=[1.0, 1.05, 1.0, 1.05, 1.05], wspace=0.32,
        left=0.045, right=0.985, top=0.80, bottom=0.20,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3], sharey=ax_b)
    e_gs = gs[0, 4].subgridspec(2, 1, height_ratios=[0.32, 1.0], hspace=0.12)
    ax_e_top = fig.add_subplot(e_gs[0, 0])
    ax_e = fig.add_subplot(e_gs[1, 0], sharex=ax_e_top)

    _asset_plot_neural_phase(ax_a, suite_dirs["wilson_cowan"], title="Wilson-Cowan", panel_label="A")
    _asset_plot_neural_phase(
        ax_c, ww_dir, title="Wong-Wang", panel_label="C",
        markers=((source, "choice 1"), (target, "choice 2")),
    )
    for ax, suite_id, label, ylabel in ((ax_b, "wilson_cowan", "B", True), (ax_d, "wong_wang", "D", False)):
        _asset_plot_r2_curves(
            ax, suite_dirs[suite_id], _NEURAL_POLICIES, title="", panel_label="",
            ylabel=ylabel, xlabel=True, r2_summary=r2_summary, ylim=_NEURAL_R2_YLIM, title_pad=1.0,
            show_inset=suite_id == "wong_wang",
        )
        ax.set_xticks([0, 1000, 2000])
        ax.tick_params(axis="both", which="both", pad=1.0)
        ax.annotate(
            label, (0, 1), xycoords="axes fraction",
            xytext=(-7.2, 1.2), textcoords="offset points", ha="left", va="bottom",
            fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold",
        )
    ax_d.tick_params(axis="y", labelleft=False)
    ax_b.yaxis.labelpad = 1.0

    # Panel E: decision reversal planned on the running estimate.
    energy = _asset_switch_curve_rows(ww_dir, "basin_switch_energy")
    success = _asset_switch_curve_rows(ww_dir, "basin_switch_success")
    oracle = _asset_switch_curve_rows(ww_dir, "basin_switch_energy_oracle")
    if not energy:
        raise RuntimeError(f"No basin-switch summary under {ww_dir / 'summary'}")
    for policy_id in _NEURAL_POLICIES:
        rows = energy.get(policy_id, [])
        if not rows:
            continue
        steps = np.asarray([r[0] for r in rows]); center = np.asarray([r[1] for r in rows])
        sem = np.asarray([r[2] for r in rows])
        color = _asset_baseline_policy_color(policy_id)
        ax_e.plot(steps, center, color=color, linewidth=0.95, marker="o", markersize=1.8)
        ax_e.fill_between(steps, center - sem, center + sem, color=color, alpha=0.10, linewidth=0.0)
        s_rows = success.get(policy_id, [])
        if s_rows:
            ax_e_top.plot(
                [r[0] for r in s_rows], [r[1] for r in s_rows],
                color=color, linewidth=0.95, drawstyle="steps-post",
            )
    oracle_values = [r[1] for rows in oracle.values() for r in rows]
    if oracle_values:
        ax_e.axhline(
            float(np.mean(oracle_values)), color=_experiment_C_NEUTRAL_LIGHT, linestyle=":",
            linewidth=0.65, zorder=5, label="true-model reference",
        )
    ax_e.set_xlim(left=0.0)
    ax_e.set_xticks([0, 1000, 2000])
    ax_e.set_ylim(bottom=0.0)
    ax_e.set_ylabel("Switch energy")
    ax_e.set_xlabel("Environment steps")
    ax_e.tick_params(axis="both", which="both", pad=1.0)
    ax_e.yaxis.labelpad = 1.0
    _style_experiment_axis(ax_e)
    ax_e_top.set_ylim(-0.08, 1.08)
    ax_e_top.yaxis.set_major_locator(FixedLocator([0.0, 1.0]))
    ax_e_top.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax_e_top.set_ylabel("Success", fontsize=_ASSET_TICK_SIZE)
    ax_e_top.yaxis.labelpad = 1.0
    ax_e_top.tick_params(axis="x", labelbottom=False, length=0)
    ax_e_top.tick_params(axis="y", pad=1.0)
    _style_experiment_axis(ax_e_top)
    ax_e_top.annotate(
        "E", (0, 1), xycoords="axes fraction",
        xytext=(-7.2, 1.2), textcoords="offset points", ha="left", va="bottom",
        fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold",
    )

    handles, labels = ax_b.get_legend_handles_labels()
    labels = ["true" if label == "true-model reference" else label for label in labels]
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
        fontsize=_ASSET_TICK_SIZE, columnspacing=0.6, handlelength=0.9, handletextpad=0.3,
        borderaxespad=0.0, borderpad=0.1, labelspacing=0.2,
    )
    return save_figure(fig, output_path, plt_module=plt_module)


def _assets_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare TBME manuscript asset assembly outputs.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(_groups_mod.groups()),
        help="Comma-separated TBME groups to scan for component figures.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Result folder containing tracks/ directly; assets/ is written here by default.",
    )
    parser.add_argument(
        "--mechanistic-results-dir", type=Path,
        default=_RESULTS_ROOT / "scalar_final_q005_20260912",
        help="Saved scalar result folder containing figure_mechanistic.npz and .json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for assembled manuscript assets.",
    )
    parser.add_argument(
        "--r2-summaries",
        type=str,
        default=",".join(_ASSET_R2_SUMMARIES),
        help=(
            "Comma-separated R2 summary sets to generate: mean_sem and/or "
            "median_iqr. Median/IQR variants are written under median_iqr/."
        ),
    )
    parser.add_argument(
        "--tri-gate-root",
        type=Path,
        default=None,
        help=(
            "Root holding the SimpleTriGate diagnostic runs (searched "
            f"recursively). Defaults to the {_ASSET_TRI_GATE_EXP_ID} suite in "
            "the result tracks."
        ),
    )
    parser.add_argument(
        "--tri-gate-exp-id",
        type=str,
        default=_ASSET_TRI_GATE_EXP_ID,
        help="Suite id of the three-gate runs under --tri-gate-root (same gate geometry).",
    )
    parser.add_argument(
        "--tri-gate-exemplar-seed",
        type=int,
        default=_ASSET_TRI_GATE_EXEMPLAR_SEED,
        help="Seed drawn in the three-gate exemplar selector panels.",
    )
    return parser


def assets_main(argv: list[str] | None = None) -> int:
    """Generate TBME manuscript asset figures from existing result summaries."""
    args = _assets_build_parser().parse_args(argv)
    if args.results_dir is not None:
        _groups_mod.set_results_dir(args.results_dir)
    group_ids = [item.strip() for item in str(args.groups).split(",") if item.strip()]
    unknown = sorted(set(group_ids) - set(_groups_mod.groups()))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")
    if not group_ids:
        raise ValueError("At least one TBME group is required")
    r2_summaries = _asset_parse_r2_summaries(args.r2_summaries)
    tri_gate_root = (
        Path(args.tri_gate_root)
        if args.tri_gate_root is not None
        else _suite_dir("objective_ablation", str(args.tri_gate_exp_id))
    )

    tri_gate_kwargs = {
        "exp_id": str(args.tri_gate_exp_id),
        "exemplar_seed": int(args.tri_gate_exemplar_seed),
    }
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else _groups_mod.results_dir() / "assets"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_groups = set(group_ids)
    skipped: list[tuple[str, str]] = []
    asset_specs: list[tuple[Path, set[str], Any, dict[str, str]]] = [
        (
            output_dir / "tbme_fig_mechanistic.pdf",
            set(),
            _asset_plot_eig_components,
            {"results_dir": args.mechanistic_results_dir},
        ),
        (
            output_dir / "tbme_fig_wilson_cowan_benchmark.pdf",
            set(),
            _asset_plot_wilson_cowan_benchmark,
            {},
        ),
        (
            output_dir / "tbme_fig_gate_diagnostic_trajectories.pdf",
            {"objective_ablation"},
            _asset_plot_gate_diagnostic_trajectories,
            {"result_roots": (tri_gate_root,), **tri_gate_kwargs},
        ),
    ]
    for r2_summary in r2_summaries:
        r2_output_dir = output_dir if r2_summary == "mean_sem" else output_dir / "median_iqr"
        kwargs = {"r2_summary": r2_summary}
        asset_specs.extend(
            [
                (
                    r2_output_dir / "tbme_fig_active_vs_baselines.pdf",
                    {"simple_system_identification"},
                    _asset_plot_active_vs_baselines,
                    kwargs,
                ),
                (
                    r2_output_dir / "tbme_fig_mechanical_benchmarks.pdf",
                    {"simple_system_identification"},
                    _asset_plot_mechanical_benchmarks,
                    kwargs,
                ),
                (
                    r2_output_dir / "tbme_fig_neural_circuits.pdf",
                    {"neural_circuits"},
                    _asset_plot_neural_circuits,
                    kwargs,
                ),
                (
                    r2_output_dir / "tbme_fig_constraints.pdf",
                    {"simple_system_identification", "observation_action_bottleneck"},
                    _asset_plot_constraints,
                    {**kwargs, "skipped": skipped},
                ),
                (
                    r2_output_dir / "tbme_fig_objective_ablation.pdf",
                    {"objective_ablation"},
                    _asset_plot_objective_ablation,
                    kwargs,
                ),
                (
                    r2_output_dir / "tbme_fig_flex_comparison.pdf",
                    {"flex_comparison"},
                    _asset_plot_flex_comparison,
                    {**kwargs, "skipped": skipped},
                ),
                (
                    r2_output_dir / "tbme_fig_flex_comparison_combined.pdf",
                    {"flex_comparison"},
                    _asset_plot_flex_combined,
                    kwargs,
                ),
                (
                    r2_output_dir / "tbme_fig_gate_diagnostic.pdf",
                    {"objective_ablation"},
                    _asset_plot_gate_diagnostic,
                    {**kwargs, "result_roots": (tri_gate_root,), **tri_gate_kwargs},
                ),
            ]
        )
    written: list[Path] = []
    for output_path, required_groups, plotter, kwargs in asset_specs:
        if not required_groups.issubset(selected_groups):
            missing = required_groups - selected_groups
            skipped.append(
                (_asset_display_path(output_path), "missing groups " + ", ".join(sorted(missing)))
            )
            continue
        try:
            result = plotter(output_path, **kwargs)
            if isinstance(result, Path):
                written.append(result)
            else:
                written.extend(result)
        except RuntimeError as exc:
            if "No trajectory R2 curves available" not in str(exc):
                raise
            skipped.append((_asset_display_path(output_path), str(exc)))

    lines = [
        "TBME manuscript asset assembly",
        "",
        f"Result folder: {_groups_mod.results_dir()}",
        f"Three-gate input: {tri_gate_root.resolve()}",
        f"Three-gate experiment: {args.tri_gate_exp_id}",
        f"Mechanistic input: {args.mechanistic_results_dir.resolve()}",
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
        for ref in _groups_mod.groups()[group_id]:
            suite_dir = ref.results_root / "tracks" / ref.suite_id
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
