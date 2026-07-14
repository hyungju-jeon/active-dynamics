#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from actdyn.utils.experiment_runtime import write_trace_csv
from actdyn.utils.figure_io import (
    load_plotting,
    save_figure_formats,
)
from actdyn.utils.plotting import (
    apply_manuscript_figure_style,
    plot_vector_field,
    style_manuscript_axis,
)

from ..experiment_definitions import get_environment_preset
from ..experiment_io import experiment_env_slug
from .run_tbme_experiments import configure_tbme_catalogs, shared_tbme_group_suites

configure_tbme_catalogs()


# Shared configuration
_TBME_STROKE_COLOR = "#3A3A3A"
_TBME_GRID_COLOR = "#DDD7CE"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_TBME_RESULTS_DIR = _RESULTS_ROOT / "tbme"


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
    return str(getattr(env_preset, "system_label", None) or env_preset.system_id)


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
    "adaptive": "PALDI",
    "adaptive_async_anytime": "Async PALDI",
    "adaptive_async_realtime": "Async PALDI (zero-fill)",
    "active_planning": "Fixed PALDI",
    "active_myopic": "Myopic",
    "active_fully_observable": "Full obs.",
    "active_state_information": "State info",
    "active_dynamics": "Dyn. sens.",
    "active_e_optimality": "E-opt.",
    "active_observation_variance": "Obs. var.",
    "active_state_variance": "State var.",
    "prbs": "PRBS",
    "random": "Random",
    "flex": "FLEX",
    "flex_true_state": "FLEX true state",
    "flex_filter": "FLEX upstream / filtered",
    "flex_true": "FLEX upstream / true",
    "flex_rollback": "FLEX rollback / filtered",
    "rhc": "RHC-US",
    "off_policy": "Off-policy",
    "active_planning_u1_r1_h40": "Planning u1/r1",
    "active_planning_u5_r5_h40": "Planning u5/r5",
    "active_planning_u1_r5_h40": "Planning u1/r5",
    "active_planning_u10_r10_h40": "Planning u10/r10",
    "active_planning_u5_r10_h40": "Planning u5/r10",
    "active_planning_u5_r20_h40": "Planning u5/r20",
    "active_planning_u10_r20_h40": "Planning u10/r20",
}

POLICY_ORDER = [
    "active_planning",
    "active_planning_u1_r1_h40",
    "active_planning_u1_r5_h40",
    "active_planning_u5_r5_h40",
    "active_planning_u5_r10_h40",
    "active_planning_u10_r10_h40",
    "active_planning_u5_r20_h40",
    "active_planning_u10_r20_h40",
    "adaptive",
    "adaptive_async_realtime",
    "adaptive_async_anytime",
    "adaptive_state_fixed_update",
    "adaptive_state",
    "active_fully_observable",
    "active_e_optimality",
    "active_state_information",
    "active_dynamics",
    "active_observation_variance",
    "active_state_variance",
    "active_myopic",
    "prbs",
    "random",
    "flex",
    "flex_true_state",
    "flex_filter",
    "flex_true",
    "flex_rollback",
    "rhc",
]

POLICY_COLORS = {
    "active_planning": "#5DADE2",
    "active_planning_u1_r1_h40": "#85C1E9",
    "active_planning_u5_r5_h40": "#73C6B6",
    "active_planning_u1_r5_h40": "#82E0AA",
    "active_planning_u10_r10_h40": "#BB8FCE",
    "active_planning_u5_r10_h40": "#76D7C4",
    "active_planning_u5_r20_h40": "#AED6F1",
    "active_planning_u10_r20_h40": "#7FB3D5",
    "adaptive": "#F1948A",
    "adaptive_async_realtime": "#C85C5C",
    "adaptive_async_anytime": "#8E4B7D",
    "adaptive_state_fixed_update": "#D7BDE2",
    "adaptive_state": "#F8C471",
    "active_fully_observable": "#82E0AA",
    "active_e_optimality": "#BB8FCE",
    "active_state_information": "#F7DC6F",
    "active_dynamics": "#76D7C4",
    "active_observation_variance": "#D2B48C",
    "active_myopic": "#F5B041",
    "prbs": "#45B8AC",
    "random": "#9EA7AD",
    "flex": "#AF7AC5",
    "flex_true_state": "#58D68D",
    "flex_filter": "#2E86C1",
    "flex_true": "#239B56",
    "flex_rollback": "#CB4335",
    "active_state_variance": "#58D68D",
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
            return ref.session_root / "tracks" / ref.suite_id
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


def plot_trajectory_overlay(
    figures_dir: Path,
    *,
    output_stem: str,
    figure_formats: Sequence[str],
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
) -> list[Path]:
    """Plot trajectory overlays on the true vector field by policy."""
    if not grouped:
        return []
    plt_module = load_plotting(figures_dir, apply_style=apply_style)
    if plt_module is None:
        return []
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
        plot_vector_field(
            dyn_true,
            ax=ax,
            x_range=plot_lim,
            n_grid=36,
            is_residual=True,
            device="cpu",
            streamplot_kwargs={
                "color": neutral_light,
                "linewidth": 0.34,
                "density": 1.35,
                "arrowsize": 0.55,
                "zorder": 1,
            },
        )
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
    return save_figure_formats(
        fig,
        figures_dir / output_stem,
        figure_formats,
        plt_module=plt_module,
    )


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
    figures_dir: Path,
    *,
    output_stem: str,
    figure_formats: Sequence[str],
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
) -> list[Path]:
    """Plot trajectory sample density by policy on the true state space."""
    if not grouped:
        return []
    plt_module = load_plotting(figures_dir, apply_style=apply_style)
    if plt_module is None:
        return []
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
        plot_vector_field(
            dyn_true,
            ax=ax,
            x_range=plot_lim,
            n_grid=36,
            is_residual=True,
            device="cpu",
            streamplot_kwargs={
                "color": neutral_light,
                "linewidth": 0.34,
                "density": 1.35,
                "arrowsize": 0.55,
                "zorder": 1,
            },
        )
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
    return save_figure_formats(
        fig,
        figures_dir / output_stem,
        figure_formats,
        plt_module=plt_module,
    )


def summary_main(argv: list[str] | None = None) -> int:
    """Run TBME summary figure generation via the split summary module."""
    from .tbme_figures_summary import summary_main as _summary_main

    return _summary_main(argv)


def experiment_main(argv: list[str] | None = None) -> int:
    """Run TBME experiment-level figure generation via the split experiment module."""
    from .tbme_figures_experiment import experiment_main as _experiment_main

    return _experiment_main(argv)


def assets_main(argv: list[str] | None = None) -> int:
    """Run TBME manuscript asset assembly via the split asset module."""
    from .tbme_figures_assets import assets_main as _assets_main

    return _assets_main(argv)


def diagnostics_main(argv: list[str] | None = None) -> int:
    """Run TBME environment diagnostics via the split diagnostics module."""
    from .tbme_figures_diagnostics import main as _diagnostics_main

    return _diagnostics_main(argv)
