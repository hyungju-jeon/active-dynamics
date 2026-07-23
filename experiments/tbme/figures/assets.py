#!/usr/bin/env python3
"""Manuscript asset assembly for the TBME figures package."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from actdyn.utils.experiment_runtime import read_trace_csv, safe_float as _safe_float
from actdyn.utils.figure_io import load_plotting, save_figure

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
    latest_session as _latest_session,
)
from ..tbme_io import (
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
    "flex_filter": "FLEX upstream / filtered",
    "flex_true": "FLEX upstream / true",
    "flex_rollback": "FLEX",
    "rhc": "RHC-US",
    "off_policy": "Off-policy",
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
# FLEX variants lose whole seeds to the unguarded update, so their bars need room
# below zero; whiskers past this floor are drawn as clipped.
_ASSET_FLEX_BAR_YLIM = (-1.0, 1.0)


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
_ASSET_PREDICTIVE_R2_LABEL = "Predictive R²"
_ASSET_FINAL_R2_LABEL = "Final predictive R²"
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
                linestyle="--",
                linewidth=0.65,
                label="true-model max" if ylabel and labels else None,
            )
        curve_ax.set_xlim(left=0.0)
        curve_ax.set_yscale("log", nonpositive="clip")
        curve_ax.set_ylim(0.25, 1.05)
        curve_ax.yaxis.set_major_locator(FixedLocator([0.25, 1.0]))
        curve_ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        curve_ax.yaxis.set_minor_formatter(NullFormatter())
        _style_experiment_axis(curve_ax)
    if inset is not None:
        inset.set_xlim(0.0, 250.0)
        inset.tick_params(axis="both", labelsize=5.2, pad=1.0)
    ax.set_title(
        panel_label, loc="left", fontweight="bold", fontsize=_ASSET_PANEL_LABEL_SIZE, pad=3.0
    )
    ax.set_title(title, loc="center", fontsize=_ASSET_TITLE_SIZE, pad=3.0)
    if xlabel:
        ax.set_xlabel("Environment steps")
    if ylabel:
        ax.set_ylabel(_ASSET_PREDICTIVE_R2_LABEL)


def _asset_plot_active_vs_baselines(output_path: Path, *, r2_summary: str) -> Path:
    sources = [
        _ExperimentSuiteSource(ref.suite_id, ref.label, ref.session_root / "tracks" / ref.suite_id)
        for ref in _groups_mod.groups()["simple_system_identification"]
    ]
    _asset_require_suite_dirs([source.suite_dir for source in sources])
    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(1, len(sources), figsize=(7.25, 2.35), squeeze=False)
    display_titles = {
        "duffing oscillator": "Duffing",
        "damped pendulum": "Damped Pendulum",
        "Gated Duffing": "Gated Duffing",
    }
    for idx, source in enumerate(sources):
        title = display_titles.get(source.label, source.label)
        _asset_plot_r2_curves(
            axes[0, idx],
            source.suite_dir,
            _ASSET_MATCHED_POLICIES,
            title=title,
            panel_label=chr(65 + idx),
            ylabel=idx == 0,
            r2_summary=r2_summary,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(_ASSET_MATCHED_POLICIES) + 1,
        fontsize=_ASSET_TICK_SIZE,
        columnspacing=0.9,
        handlelength=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.75)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_nice_ceiling(value: float) -> float:
    import math

    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        return 10.0
    return float(math.ceil(v / 10.0) * 10.0)


# Environment groupings for the composite dynamics/observation figure.
_DYNAMICS_FULL_PHASE_ENVS = (
    ("tbme_duffing", "Duffing"),
    ("tbme_damped_pendulum", "Damped Pendulum"),
    ("tbme_gated_duffing", "Gated Duffing"),
)
_DYNAMICS_FULL_FISHER_ENVS = (
    "tbme_gated_duffing",
    "tbme_gated_duffing_asymmetric",
)
_DYNAMICS_FULL_SNR_ENVS = (
    "tbme_gated_duffing",
    "tbme_gated_duffing_observation_bottleneck_strong",
)
_DYNAMICS_FULL_TRAJ_COLORS = ("#E8963A", "#D1382C", "#2E6FB0")


def _asset_plot_dynamics_full(output_path: Path) -> Path:
    """Composite dynamics/observation diagnostics figure (manuscript figure 2)."""
    from actdyn.utils.plotting import plot_vector_field
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tbme.run_tbme_experiments import configure_tbme_catalogs

    from .diagnostics import (
        finite_limits as _finite_limits,
        loading_model as _loading_model,
        parameter_sensitivity_grid as _parameter_sensitivity_grid,
        rate_hz as _rate_hz,
        simulate_trajectories as _simulate_trajectories,
        state_information_grid as _state_information_grid,
        true_dynamics as _true_dynamics,
    )

    configure_tbme_catalogs(suite_entries={})
    plt_module = load_plotting(output_path, apply_style=_apply_asset_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")

    n_grid_field = 41
    n_grid_map = 81
    steps = 500
    n_trajectories = 3
    seed = 0
    snr_trajectories = 100
    snr_trajectory_length = 200

    phase_presets = [get_environment_preset(env_id) for env_id, _ in _DYNAMICS_FULL_PHASE_ENVS]

    fig = plt_module.figure(figsize=(7.52, 2.56))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.0], hspace=0.32, left=0.045, right=0.985, top=0.91, bottom=0.11
    )
    # Top row as one grid so the A and B maps share an identical cell size and
    # inter-panel gap (3 A cells | spacer | 3 B cells | colorbar).
    top = outer[0].subgridspec(
        1,
        8,
        width_ratios=[1.0, 1.0, 1.0, 0.55, 1.0, 1.0, 1.0, 0.08],
        wspace=0.16,
    )
    a_cols = (0, 1, 2)
    b_cols = (4, 5, 6)
    b_cbar_col = 7
    # Keep C compact (left) and let D run wider across the bottom row.
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.9, 1.62], wspace=0.13)

    # Panel A: phase portraits with executed trajectories.
    a_axes = []
    for col, preset in enumerate(phase_presets):
        ax = fig.add_subplot(top[0, a_cols[col]])
        plot_lim = float(preset.resolved_plot_limit())
        dynamics = _true_dynamics(preset)
        plot_vector_field(
            dynamics,
            ax=ax,
            x_range=plot_lim,
            n_grid=n_grid_field,
            is_residual=True,
            device="cpu",
            streamplot_kwargs={"arrowsize": 0.35, "linewidth": 0.22},
        )
        trajectories = _simulate_trajectories(
            preset,
            n_trajectories=n_trajectories,
            steps=steps,
            seed=seed,
        )
        for idx, traj in enumerate(trajectories):
            color = _DYNAMICS_FULL_TRAJ_COLORS[idx % len(_DYNAMICS_FULL_TRAJ_COLORS)]
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=color,
                linewidth=1.15,
                alpha=0.92,
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(
                traj[0, 0],
                traj[0, 1],
                s=11,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=4,
            )
        ax.set_xlim(-plot_lim, plot_lim)
        ax.set_ylim(-plot_lim, plot_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(_DYNAMICS_FULL_PHASE_ENVS[col][1], fontsize=8.0, pad=2.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        a_axes.append(ax)

    # Panel B: parameter-sensitivity heatmaps sharing one colorbar.
    sens_maps = []
    for preset in phase_presets:
        plot_lim = float(preset.resolved_plot_limit())
        _sx, _sy, sens = _parameter_sensitivity_grid(preset, plot_lim=plot_lim, n_grid=n_grid_map)
        sens_maps.append((sens, plot_lim))
    svmin, svmax = _finite_limits(np.concatenate([s.reshape(-1) for s, _ in sens_maps]))
    b_axes = []
    im_sens = None
    for col, (sens, plot_lim) in enumerate(sens_maps):
        ax = fig.add_subplot(top[0, b_cols[col]])
        im_sens = ax.imshow(
            sens,
            origin="lower",
            extent=[-plot_lim, plot_lim, -plot_lim, plot_lim],
            cmap="magma",
            vmin=svmin,
            vmax=svmax,
            interpolation="bilinear",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        b_axes.append(ax)
    cbar_ax = fig.add_subplot(top[0, b_cbar_col])
    cbar = fig.colorbar(im_sens, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6.0, width=0.4, length=2.0)
    cbar.outline.set_linewidth(0.4)

    # Panel C: state Fisher information (per-map log scale) with loading-vector insets.
    from matplotlib.colors import LogNorm

    c_gs = bottom[0].subgridspec(1, 5, width_ratios=[1.0, 0.05, 0.16, 1.0, 0.05], wspace=0.10)
    c_map_cols = (0, 3)
    c_axes = []
    for idx, env_id in enumerate(_DYNAMICS_FULL_FISHER_ENVS):
        preset = get_environment_preset(env_id)
        plot_lim = float(preset.resolved_plot_limit())
        weights, bias, dt = _loading_model(
            preset,
            snr_trajectories=snr_trajectories,
            snr_trajectory_length=snr_trajectory_length,
        )
        _ix, _iy, info = _state_information_grid(
            weights,
            bias,
            dt=dt,
            plot_lim=plot_lim,
            n_grid=n_grid_map,
        )
        # `_state_information_grid` returns log-det; exponentiate for a log-scale colorbar.
        det = np.exp(np.clip(info, -50.0, 50.0))
        finite_det = det[np.isfinite(det) & (det > 0.0)]
        vmin = float(np.percentile(finite_det, 1.0))
        vmax = float(np.percentile(finite_det, 99.0))
        if vmax <= vmin:
            vmax = vmin * 10.0 + 1e-12
        ax = fig.add_subplot(c_gs[0, c_map_cols[idx]])
        im_info = ax.imshow(
            det,
            origin="lower",
            extent=[-plot_lim, plot_lim, -plot_lim, plot_lim],
            cmap="plasma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="bilinear",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        inset = ax.inset_axes([0.62, 0.05, 0.34, 0.34])
        inset.axhline(0.0, color="#8A8A8A", linewidth=0.4, zorder=1)
        inset.axvline(0.0, color="#8A8A8A", linewidth=0.4, zorder=1)
        for row in weights:
            inset.plot(
                [0.0, row[0]],
                [0.0, row[1]],
                color="#606060",
                alpha=0.30,
                linewidth=0.3,
                zorder=2,
            )
        inset.scatter(
            weights[:, 0],
            weights[:, 1],
            c=np.linspace(0.15, 0.95, weights.shape[0]),
            cmap="magma",
            s=6,
            alpha=0.95,
            linewidths=0.0,
            zorder=3,
        )
        span = 1.12 * float(np.max(np.abs(weights))) if weights.size else 1.0
        inset.set_xlim(-span, span)
        inset.set_ylim(-span, span)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_linewidth(0.4)
        cbar_ax = fig.add_subplot(c_gs[0, c_map_cols[idx] + 1])
        cbar = fig.colorbar(im_info, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6.0, width=0.4, length=2.0)
        cbar.outline.set_linewidth(0.4)
        c_axes.append(ax)

    # Panel D: observation rates + spike rasters at two SNR levels.
    d_gs = bottom[1].subgridspec(1, 2, wspace=0.24)
    d_rate_axes = []
    for col, env_id in enumerate(_DYNAMICS_FULL_SNR_ENVS):
        preset = get_environment_preset(env_id)
        weights, bias, dt = _loading_model(
            preset,
            snr_trajectories=snr_trajectories,
            snr_trajectory_length=snr_trajectory_length,
        )
        traj = _simulate_trajectories(preset, n_trajectories=1, steps=steps, seed=seed)[0]
        rate_hz = _rate_hz(traj, weights=weights, bias=bias)
        mean_counts = np.clip(rate_hz * dt, 1e-8, 1e8)
        observations = np.random.default_rng(int(seed)).poisson(mean_counts).astype(np.float32)

        stack = d_gs[0, col].subgridspec(2, 1, height_ratios=[1.0, 3.0], hspace=0.10)
        ax_rate = fig.add_subplot(stack[0, 0])
        ax_rast = fig.add_subplot(stack[1, 0], sharex=ax_rate)
        time = np.arange(rate_hz.shape[0])
        for neuron_idx in range(rate_hz.shape[1]):
            ax_rate.plot(time, rate_hz[:, neuron_idx], linewidth=0.4, alpha=0.7)
        ymax = _asset_nice_ceiling(float(np.nanmax(rate_hz)))
        ax_rate.set_xlim(0.0, float(steps))
        ax_rate.set_ylim(0.0, ymax)
        ax_rate.set_yticks([0.0, ymax])
        ax_rate.set_yticklabels(["0", f"{int(ymax)} Hz"], fontsize=6.0)
        ax_rate.tick_params(axis="x", labelbottom=False)
        target_snr = getattr(preset, "loading_target_snr_db", None)
        snr_title = "SNR" if target_snr is None else f"SNR : {int(round(float(target_snr)))} dB"
        ax_rate.set_title(snr_title, fontsize=8.0, pad=3.0)
        for spine in ax_rate.spines.values():
            spine.set_linewidth(0.5)

        spike_steps, spike_neurons = np.nonzero(observations > 0)
        if spike_steps.size:
            ax_rast.vlines(
                spike_steps.astype(np.float32),
                spike_neurons.astype(np.float32) - 0.4,
                spike_neurons.astype(np.float32) + 0.4,
                color="#222222",
                linewidth=0.3,
            )
        ax_rast.set_xlim(0.0, float(steps))
        ax_rast.set_ylim(-0.5, observations.shape[1] - 0.5)
        ax_rast.set_xticks([0.0, float(steps)])
        ax_rast.set_xticklabels(["0", str(int(steps))], fontsize=6.0)
        ax_rast.set_xlabel("step", fontsize=8.0)
        ax_rast.set_yticks([])
        if col == 0:
            ax_rast.set_ylabel("neuron", fontsize=8.0)
        for spine in ax_rast.spines.values():
            spine.set_linewidth(0.5)
        d_rate_axes.append(ax_rate)

    # `aspect="equal"` squares (and centers) the image axes only at draw time, so
    # render once before reading positions to place titles/letters on the real boxes.
    try:
        fig.draw_without_rendering()
    except AttributeError:
        fig.canvas.draw()

    # Group titles and bold panel letters, positioned from axis geometry.
    def _block_span(axes: Sequence[Any]) -> tuple[float, float, float]:
        positions = [ax.get_position() for ax in axes]
        left = min(pos.x0 for pos in positions)
        right = max(pos.x1 for pos in positions)
        top_edge = max(pos.y1 for pos in positions)
        return left, right, top_edge

    a_left, a_right, a_top = _block_span(a_axes)
    b_left, b_right, b_top = _block_span(b_axes)
    c_left, c_right, c_top = _block_span(c_axes)
    d_left, d_right, d_top = _block_span(d_rate_axes)

    fig.text(
        0.5 * (b_left + b_right),
        b_top + 0.045,
        r"$\|df/d\theta\|_F$",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.text(
        0.5 * (c_left + c_right),
        c_top + 0.045,
        "State Fisher Information",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    for letter, left, top_edge in (
        ("A", a_left, a_top),
        ("B", b_left, b_top),
        ("C", c_left, c_top),
        ("D", d_left, d_top),
    ):
        fig.text(
            left - 0.028,
            top_edge + 0.04,
            letter,
            ha="left",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
        )

    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_bottleneck_sources() -> list[_ExperimentSuiteSource]:
    return [
        _ExperimentSuiteSource(
            "gated_duffing",
            "Default",
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
    ax.set_yticklabels([label for _field, _value, label, _color in event_specs], fontsize=6.3)
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
        bbox_to_anchor=(0.5, 1.01),
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
    ylim: tuple[float, float] = (0.0, 1.0),
    policy_labels: Mapping[str, str] | None = None,
    policy_legend: bool = True,
) -> Path:
    """Standalone final-performance bars, colored by policy with per-condition shade.

    Width tracks the policy count by default; ``single_column`` pins it to the
    3.5 in manuscript column instead, and ``short`` trims the axes to the flatter
    manuscript proportion. Bars and whiskers past ``ylim`` are drawn clipped, with
    a caret at the floor marking the ones that run off the bottom.
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

    # Without the policy legend the x tick labels carry the policy names, so the
    # condition legend takes the strip above the axes instead of sitting inside it.
    n_legend = n_policy if policy_legend else n_cond
    legend_ncol = min(n_legend, 4) if single_column else n_legend
    legend_rows = int(np.ceil(n_legend / max(legend_ncol, 1)))
    fig_width = _ASSET_SINGLE_COLUMN_WIDTH if single_column else 1.6 + 0.5 * max(n_policy, 1)
    # A wrapped policy legend needs its own strip above the axes, not axes height.
    # One row reserves 8% of the default figure, matching the unwrapped layout.
    legend_height = 0.188 + 0.16 * (legend_rows - 1)
    fig_height = (1.55 if short else 2.35) + 0.16 * (legend_rows - 1)
    fig, ax = plt_module.subplots(figsize=(fig_width, fig_height))
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
            linewidth=0.6,
            capsize=1.6,
            error_kw={"elinewidth": 0.6, "capthick": 0.6},
        )

    ax.set_ylabel(_ASSET_FINAL_R2_LABEL)
    ax.set_ylim(y_floor, y_top)
    if y_floor < 0.0:
        ax.set_yticks(np.arange(y_floor, y_top + 1e-9, 0.5))
        ax.axhline(0.0, color=_experiment_C_STROKE, linewidth=0.5)
        # Carets mark bars whose value or lower band runs past the axis floor;
        # the exact numbers stay in the companion CSV.
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
            linewidth=0.5,
        )
        for cond_idx in range(n_cond)
    ]
    cond_labels = [source.label for source in sources]
    if policy_legend:
        policy_handles = [
            Line2D([0], [0], color=_asset_baseline_policy_color(policy_id), linewidth=1.6)
            for policy_id in active_policies
        ]
        fig.legend(
            policy_handles,
            [_asset_policy_label(policy_id, policy_labels) for policy_id in active_policies],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
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
            ncol=min(n_cond, 3),
            handlelength=1.2,
            borderpad=0.3,
            columnspacing=1.0,
        )
    else:
        fig.legend(
            cond_handles,
            cond_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=legend_ncol,
            fontsize=_ASSET_TICK_SIZE,
            columnspacing=1.0,
            handlelength=1.2,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - legend_height / fig_height))
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_plot_objective_ablation(output_path: Path, *, r2_summary: str) -> list[Path]:
    """Single-column ablation assets: final-R2 bars plus a stacked recovery figure."""
    sources = [
        _ExperimentSuiteSource(source.exp_id, label, source.suite_dir)
        for source, label in zip(
            _experiment_objective_sources(),
            ("Default", "Asymmetric", "Challenging"),
        )
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
            single_column=True,
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
# The suite lives in the shared session tracks (objective_ablation group); the
# asset reads the raw run traces because its panels need per-seed occupancy and
# final-value quantiles that the suite summary does not carry.
_ASSET_TRI_GATE_EXP_ID = "three_gate_diagnostic"
_ASSET_TRI_GATE_LABELS = {
    "compound_active_planning": "PALDI",
    "compound_active_fully_observable": "Full obs.",
    "compound_active_e_optimality": "E-opt.",
    "compound_active_state_information": "State info",
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
# occupancy contrast: PALDI holds gate F while the fully observed objective
# abandons F for gate N, with every panel showing its policy's modal behavior.
# Population occupancy statistics live in the main diagnostic figure.
_ASSET_TRI_GATE_EXEMPLAR_SEED = 90
_ASSET_TRI_GATE_REST_CENTER = -1.0
_ASSET_TRI_GATE_REST_CUTOFF = -0.75
_ASSET_TRI_GATE_R2_YLIM = (-0.2, 1.0)
# Gate identity colors couple the occupancy stacks (panel B) to the selector
# traces (panel C); they are deliberately darker than the pastel policy palette.
_ASSET_TRI_GATE_GATE_COLORS = (
    ("rest_fraction", "Rest", "#C8CDD1"),
    ("gate_A_fraction", "N: confounded", "#2F7D5B"),
    ("gate_B_fraction", "B: weak, balanced", "#6C5FB8"),
    ("gate_M_fraction", "F: full rank", "#C4564E"),
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


# Dedicated qualitative palette for the tri-gate objectives: the shared
# manuscript colors put two objectives in near-identical greens, so this figure
# spreads the hues for legibility when eight traces overlay in one panel. PALDI
# keeps its warm identity; Random keeps a neutral gray.
_ASSET_TRI_GATE_POLICY_COLORS = {
    "compound_active_planning": "#D1495B",
    "compound_active_fully_observable": "#2E6FB8",
    "compound_active_e_optimality": "#944FC7",
    "compound_active_state_information": "#E8A33D",
    "compound_active_dynamics": "#AA4499",
    "compound_active_dynamics_logdet": "#17A398",
    "compound_active_observation_variance": "#8C5A3B",
    "compound_active_state_variance": "#4CAF50",
    "random": "#7C868D",
}


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
) -> Path:
    """Manuscript figure for the designed three-gate objective diagnostic.

    Single row: (A) final rollout R2 per objective, (B) selector occupancy, (C)
    every objective's exemplar selector trace overlaid on the gate assignment
    bands. The nonadaptive PRBS control and the trace dynamics-sensitivity
    variant are omitted (the paper reports the rank-aware logdet form as
    "Dyn. sens.").
    """
    _asset_parse_r2_summaries(r2_summary)
    records = _compound_trace_records(result_roots, exp_id=_ASSET_TRI_GATE_EXP_ID)
    if not records:
        roots_text = ", ".join(str(root) for root in result_roots)
        raise RuntimeError(
            f"No trajectory R2 curves available for {_ASSET_TRI_GATE_EXP_ID} in {roots_text}"
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

    fig, axis_grid = plt_module.subplots(
        1, 3, figsize=(7.25, 2.1), gridspec_kw={"width_ratios": (2.5, 2.5, 5.0)}
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
        edgecolor=bar_colors,
        linewidth=0.6,
        capsize=1.6,
        error_kw={"elinewidth": 0.6, "capthick": 0.6},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_ASSET_TRI_GATE_LABELS[str(row["policy_id"])] for row in summary_rows],
        rotation=30,
        ha="right",
    )
    ax.set_ylim(0.0, _ASSET_TRI_GATE_R2_YLIM[1])
    ax.set_ylabel(_ASSET_FINAL_R2_LABEL)
    _style_experiment_axis(ax)
    ax.set_title(
        "A", loc="left", fontweight="bold", fontsize=_ASSET_PANEL_LABEL_SIZE, pad=3.0
    )
    ax.set_title("Rollout recovery", loc="center", fontsize=_ASSET_TITLE_SIZE, pad=3.0)

    # B: selector occupancy stacks, one per objective.
    ax = axes[1]
    bottom = np.zeros(len(summary_rows), dtype=np.float64)
    for key, _label, color in _ASSET_TRI_GATE_GATE_COLORS:
        value = np.asarray([row[key] for row in summary_rows], dtype=np.float64)
        ax.bar(x, value, bottom=bottom, width=0.72, color=color)
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
    ax.set_title("Selector occupancy", loc="center", fontsize=_ASSET_TITLE_SIZE, pad=3.0)

    # C: every objective's exemplar selector trace overlaid on the gate bands,
    # so dwell-at-N, dwell-at-B, and reach-and-hold-F behaviors read against the
    # shared gate assignment regions.
    ax = axes[2]
    rest_color = _ASSET_TRI_GATE_GATE_COLORS[0][2]
    y_bottom, y_top = -1.2, 0.62
    ax.axhspan(
        y_bottom, _ASSET_TRI_GATE_REST_CUTOFF, color=rest_color, alpha=0.22, linewidth=0.0
    )
    for low, high, color in _asset_tri_gate_assignment_bands(y_top):
        ax.axhspan(low, high, color=color, alpha=0.14, linewidth=0.0)
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
            color=_asset_tri_gate_policy_color(policy_id),
            linewidth=1.1 if is_paldi else 0.5,
            alpha=1.0 if is_paldi else 0.7,
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
            color=color,
        )
    ax.text(
        1.02,
        _ASSET_TRI_GATE_REST_CENTER,
        "rest",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=5.2,
        color=_experiment_C_STROKE,
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Selector $r$")
    _style_experiment_axis(ax)
    ax.set_title(
        "C", loc="left", fontweight="bold", fontsize=_ASSET_PANEL_LABEL_SIZE, pad=3.0
    )
    ax.set_title("Exemplar selector traces", loc="center", fontsize=_ASSET_TITLE_SIZE, pad=3.0)

    from matplotlib.lines import Line2D

    legend_policies = [str(row["policy_id"]) for row in summary_rows]
    fig.legend(
        [
            Line2D([0], [0], color=_asset_tri_gate_policy_color(policy_id), linewidth=1.6)
            for policy_id in legend_policies
        ],
        [_ASSET_TRI_GATE_LABELS[policy_id] for policy_id in legend_policies],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_policies),
        fontsize=_ASSET_TICK_SIZE,
        columnspacing=0.9,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9), w_pad=1.1)
    return save_figure(fig, output_path, plt_module=plt_module)


def _asset_plot_gate_diagnostic_trajectories(
    output_path: Path,
    *,
    result_roots: Sequence[Path],
    exemplar_seed: int = _ASSET_TRI_GATE_EXEMPLAR_SEED,
) -> Path:
    """Appendix companion: one exemplar selector trace per acquisition objective.

    Gate bands (center +/- one gate width) and the rest line reuse the gate
    identity colors of the main diagnostic figure, so dwell-at-N, dwell-at-B,
    and reach-and-hold-F behaviors are visible directly.
    """
    records = [
        record
        for record in _compound_trace_records(result_roots, exp_id=_ASSET_TRI_GATE_EXP_ID)
        if record.seed == int(exemplar_seed)
    ]
    if not records:
        roots_text = ", ".join(str(root) for root in result_roots)
        raise RuntimeError(
            f"No trajectory R2 curves available for {_ASSET_TRI_GATE_EXP_ID} "
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
        ax.tick_params(axis="both", labelsize=5.2, pad=1.0)
    for idx in range(len(policy_ids), n_row * n_col):
        axes[idx // n_col, idx % n_col].set_visible(False)
    # Gate letters ride the right edge of the last column, keyed by band color.
    right_ax = axes[0, n_col - 1]
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
        fontsize=5.2,
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
        "gated_duffing": "Gated Duffing",
        "gated_duffing_asymmetric": "Asymmetric",
        "gated_duffing_challenging": "Challenging",
        "gated_duffing_observation_bottleneck_mild": "SNR -10 dB",
        "gated_duffing_observation_bottleneck_strong": "SNR -15 dB",
    }
    sources = {
        ref.suite_id: _ExperimentSuiteSource(
            ref.suite_id,
            display_titles.get(ref.suite_id, ref.label),
            ref.session_root / "tracks" / ref.suite_id,
        )
        for ref in _groups_mod.groups()["flex_comparison"]
    }
    grouped = (
        ("baseline", ("duffing", "damped_pendulum", "gated_duffing")),
        ("hard", ("gated_duffing_asymmetric", "gated_duffing_challenging")),
        (
            "snr",
            (
                "gated_duffing_observation_bottleneck_mild",
                "gated_duffing_observation_bottleneck_strong",
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


def _asset_plot_flex_comparison(output_path: Path, *, r2_summary: str) -> list[Path]:
    """FLEX state-source/update variants: short final-R2 bars plus recovery curves.

    One bar figure and one recovery figure per condition group, matching how the
    constraints panels are assembled for the manuscript.
    """
    written: list[Path] = []
    for suffix, sources in _asset_flex_groups():
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
            )
        )
    return written


def _asset_plot_constraints(output_path: Path, *, r2_summary: str) -> list[Path]:
    bottleneck_sources = _asset_bottleneck_sources()
    figures = (
        ("snr", "Observation SNR", tuple(bottleneck_sources[:3])),
        (
            "asymmetry",
            "Asymmetry",
            (
                _ExperimentSuiteSource(
                    "gated_duffing",
                    "Default",
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
    for suffix, _figure_title, sources in figures:
        _asset_require_suite_dirs([source.suite_dir for source in sources])
        metric_rows = _asset_method_metric_rows(
            sources,
            _ASSET_MATCHED_POLICIES,
            r2_summary=r2_summary,
        )
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
    return written


def _asset_plot_eig_components(output_path: Path) -> list[Path]:
    """Write the EIG components figure at both manuscript column widths."""
    from experiments.eig_1d_example import main as _eig_1d_main

    single_path = output_path.with_name(f"{output_path.stem}_single{output_path.suffix}")
    return [
        _eig_1d_main(["--output", str(output_path), "--column", "double"]),
        _eig_1d_main(["--output", str(single_path), "--column", "single"]),
    ]


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
        help="TBME results root. Defaults to results/tbme.",
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
            "the session tracks."
        ),
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
        else _suite_dir("objective_ablation", _ASSET_TRI_GATE_EXP_ID)
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else _groups_mod.session_root() / "assets"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_groups = set(group_ids)
    asset_specs: list[tuple[Path, set[str], Any, dict[str, str]]] = [
        (
            output_dir / "tbme_fig_mechanistic.pdf",
            set(),
            _asset_plot_eig_components,
            {},
        ),
        (
            output_dir / "tbme_fig_dynamics_full.pdf",
            set(),
            _asset_plot_dynamics_full,
            {},
        ),
        (
            output_dir / "tbme_fig_gate_diagnostic_trajectories.pdf",
            {"objective_ablation"},
            _asset_plot_gate_diagnostic_trajectories,
            {"result_roots": (tri_gate_root,)},
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
                    r2_output_dir / "tbme_fig_constraints.pdf",
                    {"simple_system_identification", "observation_action_bottleneck"},
                    _asset_plot_constraints,
                    kwargs,
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
                    kwargs,
                ),
                (
                    r2_output_dir / "tbme_fig_gate_diagnostic.pdf",
                    {"objective_ablation"},
                    _asset_plot_gate_diagnostic,
                    {**kwargs, "result_roots": (tri_gate_root,)},
                ),
            ]
        )
    written: list[Path] = []
    skipped: list[tuple[str, str]] = []
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
