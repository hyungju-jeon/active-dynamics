"""Trajectory overlay and density figure builders.

Per-policy phase-space panels over the true vector field, colored by seed
(overlay) or by visit density (density).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from actdyn.utils.figure_io import load_plotting, save_figure_formats
from actdyn.visualize import plot_vector_field


def trajectory_layout(n_panels: int) -> tuple[int, int, tuple[float, float]]:
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


def trajectory_plot_limit(
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


def trajectory_seed_color_map(
    plt_module: Any, seeds: list[int]
) -> dict[int, tuple[float, float, float, float]]:
    if not seeds:
        return {}
    cmap = plt_module.get_cmap("turbo")
    denom = max(len(seeds) - 1, 1)
    return {seed: cmap(idx / denom) for idx, seed in enumerate(sorted(seeds))}


def format_trajectory_axis(
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
    plot_lim = trajectory_plot_limit(grouped, grid_lim)
    seeds = sorted({seed for traces in grouped.values() for seed, _traj in traces})
    seed_colors = trajectory_seed_color_map(plt_module, seeds)
    n_rows, n_cols, figsize = trajectory_layout(len(policies))
    fig, axes = plt_module.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True, sharey=True
    )
    for idx, policy_id in enumerate(policies):
        ax = axes[idx // n_cols, idx % n_cols]
        if dyn_true is not None:
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
        format_trajectory_axis(
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


def trajectory_histogram(
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


def trajectory_density_cmap(plt_module: Any) -> Any:
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
    plot_lim = trajectory_plot_limit(grouped, grid_lim)
    hists = {
        policy_id: trajectory_histogram(grouped[policy_id], plot_lim, bins)
        for policy_id in policies
    }
    max_count = max((float(np.nanmax(hist)) for hist in hists.values() if hist.size), default=1.0)
    max_log_count = float(np.log10(max(max_count, 1.0) + 1.0))
    norm = Normalize(vmin=0.0, vmax=max_log_count)
    cmap = trajectory_density_cmap(plt_module).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    n_rows, n_cols, figsize = trajectory_layout(len(policies))
    fig, axes = plt_module.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True, sharey=True
    )
    im = None
    for idx, policy_id in enumerate(policies):
        ax = axes[idx // n_cols, idx % n_cols]
        if dyn_true is not None:
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
        format_trajectory_axis(
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
