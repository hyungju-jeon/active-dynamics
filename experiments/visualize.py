#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from actdyn.utils.experiment_runtime import safe_float as _safe_float
from actdyn.utils.figure_io import load_plotting, sample_sem, save_figure_formats


def _identity_policy_sort_key(policy_id: str) -> str:
    return policy_id


def _default_policy_label(policy_id: str) -> str:
    return policy_id.replace("_", " ")


def _default_policy_color(_policy_id: str, fallback_idx: int) -> str:
    return f"C{fallback_idx}"


def _seed_reference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: dict[int, dict[str, Any]] = {}
    for record in sorted(
        records,
        key=lambda rec: (int(rec["seed"]), str(rec["policy_id"]), str(rec["run_dir"])),
    ):
        refs.setdefault(int(record["seed"]), record)
    return [refs[seed] for seed in sorted(refs)]


def _format_count_title(template: str | None, count: int) -> str | None:
    if template is None:
        return None
    return template.format(n_seeds=count)


def plot_final_value_by_policy(
    figures_dir: Path,
    *,
    rows: list[dict[str, Any]],
    ylabel: str,
    title: str,
    output_stem: str,
    figure_formats: Sequence[str],
    policy_sort_key: Callable[[str], Any] = _identity_policy_sort_key,
    policy_label: Callable[[str], str] = _default_policy_label,
    policy_color: Callable[[str, int], str] = _default_policy_color,
    apply_style: Callable[[Any], None] | None = None,
    style_axis: Callable[..., None] | None = None,
    stroke_color: str = "black",
) -> None:
    """Plot final per-policy summary values from a metrics CSV."""
    if not rows:
        return
    plt_module = load_plotting(figures_dir, apply_style=apply_style)
    if plt_module is None:
        return

    policy_ids = sorted({str(row["policy_id"]) for row in rows}, key=policy_sort_key)
    means = []
    sems = []
    for policy_id in policy_ids:
        vals = [
            _safe_float(row.get("value_final_mean"))
            for row in rows
            if str(row.get("policy_id")) == policy_id
        ]
        nums = [value for value in vals if value is not None]
        means.append(float(np.mean(nums)) if nums else np.nan)
        sems.append(sample_sem(nums))

    fig, ax = plt_module.subplots(figsize=(5.8, 3.0))
    x = np.arange(len(policy_ids), dtype=np.float64)
    colors = [policy_color(policy_id, idx) for idx, policy_id in enumerate(policy_ids)]
    ax.bar(
        x,
        means,
        yerr=sems,
        capsize=2.5,
        color=colors,
        edgecolor=stroke_color,
        linewidth=0.45,
        error_kw={
            "elinewidth": 0.55,
            "ecolor": stroke_color,
            "capthick": 0.55,
        },
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [policy_label(policy_id) for policy_id in policy_ids], rotation=28, ha="right"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if style_axis is not None:
        style_axis(ax, grid_axis="y", grid_alpha=0.38)
    fig.tight_layout()
    save_figure_formats(fig, figures_dir / output_stem, figure_formats, plt_module=plt_module)


def plot_metric_over_steps(
    figures_dir: Path,
    *,
    rows: list[dict[str, Any]],
    ylabel: str,
    title: str,
    output_stem: str,
    figure_formats: Sequence[str],
    policy_sort_key: Callable[[str], Any] = _identity_policy_sort_key,
    policy_label: Callable[[str], str] = _default_policy_label,
    policy_color: Callable[[str, int], str] = _default_policy_color,
    apply_style: Callable[[Any], None] | None = None,
    style_axis: Callable[..., None] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot a per-policy mean/SEM metric against environment steps."""
    if not rows:
        return
    plt_module = load_plotting(figures_dir, apply_style=apply_style)
    if plt_module is None:
        return

    fig, ax = plt_module.subplots(figsize=(5.85, 3.15))
    policy_ids = sorted({str(row["policy_id"]) for row in rows}, key=policy_sort_key)
    for idx, policy_id in enumerate(policy_ids):
        series = [row for row in rows if str(row.get("policy_id")) == policy_id]
        series.sort(key=lambda row: int(row["step"]))
        xs = [int(row["step"]) for row in series]
        ys = [float(row["value_mean"]) for row in series]
        sem = [float(row["value_sem"]) for row in series]
        color = policy_color(policy_id, idx)
        ax.plot(xs, ys, label=policy_label(policy_id), color=color, linewidth=1.05)
        ax.fill_between(
            xs,
            np.asarray(ys) - np.asarray(sem),
            np.asarray(ys) + np.asarray(sem),
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )
    ax.set_xlabel("Environment Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if style_axis is not None:
        style_axis(ax)
    ax.legend(loc="best", fontsize=6.3, ncol=2, columnspacing=0.8, handlelength=1.5)
    fig.tight_layout()
    save_figure_formats(fig, figures_dir / output_stem, figure_formats, plt_module=plt_module)


def plot_metric_over_cpu_time(
    figures_dir: Path,
    *,
    rows: list[dict[str, Any]],
    ylabel: str,
    title: str,
    output_stem: str,
    figure_formats: Sequence[str],
    policy_sort_key: Callable[[str], Any] = _identity_policy_sort_key,
    policy_label: Callable[[str], str] = _default_policy_label,
    policy_color: Callable[[str, int], str] = _default_policy_color,
    apply_style: Callable[[Any], None] | None = None,
    style_axis: Callable[..., None] | None = None,
) -> None:
    """Plot a per-policy mean/SEM metric against CPU time."""
    if not rows:
        return
    plt_module = load_plotting(figures_dir, apply_style=apply_style)
    if plt_module is None:
        return

    fig, ax = plt_module.subplots(figsize=(5.85, 3.15))
    policy_ids = sorted({str(row["policy_id"]) for row in rows}, key=policy_sort_key)
    for idx, policy_id in enumerate(policy_ids):
        series = [
            row
            for row in rows
            if (
                str(row.get("policy_id")) == policy_id
                and _safe_float(row.get("cpu_time_sec_mean")) is not None
            )
        ]
        series.sort(key=lambda row: int(row["step"]))
        xs = [float(row["cpu_time_sec_mean"]) for row in series]
        ys = [float(row["value_mean"]) for row in series]
        sem = [float(row["value_sem"]) for row in series]
        color = policy_color(policy_id, idx)
        ax.plot(xs, ys, label=policy_label(policy_id), color=color, linewidth=1.05)
        ax.fill_between(
            xs,
            np.asarray(ys) - np.asarray(sem),
            np.asarray(ys) + np.asarray(sem),
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )
    ax.set_xlabel("CPU Time (sec)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if style_axis is not None:
        style_axis(ax)
    ax.legend(loc="best", fontsize=6.3, ncol=2, columnspacing=0.8, handlelength=1.5)
    fig.tight_layout()
    save_figure_formats(fig, figures_dir / output_stem, figure_formats, plt_module=plt_module)


def plot_neuron_tuning_curve_colormap(
    figures_dir: Path,
    *,
    records: list[dict[str, Any]],
    figure_formats: Sequence[str],
    get_environment_preset_from_metadata: Callable[[dict[str, Any]], Any],
    reconstruct_loglinear_rate_model: Callable[..., tuple[Any, Any, Any]],
    expected_loglinear_rate_hz: Callable[..., Any],
    apply_style: Callable[[Any], None] | None = None,
    style_axis: Callable[..., None] | None = None,
    style_colorbar: Callable[[Any], None] | None = None,
    output_stem: str = "neuron_tuning_curve_colormap",
    axis_labels: tuple[str, str] = ("z[0]", "z[1]"),
    colorbar_label: str = "Total firing rate",
    title_template: str | None = "Total firing rate (mean over {n_seeds} seeds)",
    contour_percentile: float | None = 90.0,
) -> None:
    """Plot the mean total firing-rate map for one reference run per seed.

    The plotted latent grid has shape ``(n_grid, n_grid)``. If
    ``contour_percentile`` is not None, one contour marks that percentile of
    the finite total firing-rate values.
    """
    seed_refs = _seed_reference_records(records)
    if not seed_refs:
        return
    plt_module = load_plotting(figures_dir, apply_style=apply_style)
    if plt_module is None:
        return

    env_preset = get_environment_preset_from_metadata(dict(seed_refs[0]["metadata"]))
    grid_lim = float(env_preset.resolved_plot_limit())
    n_grid = 121
    axis = np.linspace(-grid_lim, grid_lim, n_grid, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    latent = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    maps: list[np.ndarray] = []
    for ref in seed_refs:
        metadata = dict(ref["metadata"])
        weights, bias, _dt = reconstruct_loglinear_rate_model(
            metadata,
            obs_dim=metadata.get("observation_dim"),
            latent_dim=metadata.get("latent_dim"),
        )
        rate_hz = expected_loglinear_rate_hz(latent, weights=weights, bias=bias)
        maps.append(np.sum(rate_hz, axis=1).reshape(n_grid, n_grid))
    heat = np.mean(np.stack(maps, axis=0), axis=0)

    finite = heat[np.isfinite(heat)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    contour_level = None
    if contour_percentile is not None:
        if not 0.0 <= contour_percentile <= 100.0:
            raise ValueError("contour_percentile must be in [0, 100]")
        contour_level = float(np.percentile(finite, contour_percentile))
        finite_min = float(np.min(finite))
        finite_max = float(np.max(finite))
        if not np.isfinite(contour_level) or not (finite_min < contour_level < finite_max):
            contour_level = None

    fig, ax = plt_module.subplots(figsize=(3.35, 3.05))
    im = ax.imshow(
        heat,
        aspect="equal",
        origin="lower",
        extent=[-grid_lim, grid_lim, -grid_lim, grid_lim],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if contour_level is not None:
        ax.contour(
            axis,
            axis,
            np.ma.masked_invalid(heat),
            levels=[contour_level],
            colors="white",
            linewidths=0.6,
            linestyles="-",
        )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    if style_colorbar is not None:
        style_colorbar(cbar)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")
    title = _format_count_title(title_template, len(seed_refs))
    if title is not None:
        ax.set_title(title)
    if style_axis is not None:
        style_axis(ax)
    fig.tight_layout()
    save_figure_formats(fig, figures_dir / output_stem, figure_formats, plt_module=plt_module)


def plot_information_colormap(
    figures_dir: Path,
    *,
    records: list[dict[str, Any]],
    figure_formats: Sequence[str],
    get_environment_preset_from_metadata: Callable[[dict[str, Any]], Any],
    reconstruct_loglinear_rate_model: Callable[..., tuple[Any, Any, Any]],
    expected_loglinear_rate_hz: Callable[..., Any],
    apply_style: Callable[[Any], None] | None = None,
    style_axis: Callable[..., None] | None = None,
    style_colorbar: Callable[[Any], None] | None = None,
    output_stem: str = "information_colormap",
    axis_labels: tuple[str, str] = ("z[0]", "z[1]"),
    colorbar_label: str = "log det information",
    title_template: str | None = "log det information (mean over {n_seeds} seeds)",
) -> None:
    """Plot the mean log-determinant Fisher information map."""
    seed_refs = _seed_reference_records(records)
    if not seed_refs:
        return
    plt_module = load_plotting(figures_dir, apply_style=apply_style)
    if plt_module is None:
        return

    env_preset = get_environment_preset_from_metadata(dict(seed_refs[0]["metadata"]))
    grid_lim = float(env_preset.resolved_plot_limit())
    n_grid = 121
    axis = np.linspace(-grid_lim, grid_lim, n_grid, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    latent = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    eye = np.eye(2, dtype=np.float32)

    maps: list[np.ndarray] = []
    for ref in seed_refs:
        metadata = dict(ref["metadata"])
        weights, bias, dt = reconstruct_loglinear_rate_model(
            metadata,
            obs_dim=metadata.get("observation_dim"),
            latent_dim=metadata.get("latent_dim"),
        )
        rate_hz = expected_loglinear_rate_hz(latent, weights=weights, bias=bias)
        mean_counts = np.clip(rate_hz * float(dt), 1e-8, 1e8)
        info_mats = np.einsum("nd,di,dj->nij", mean_counts, weights, weights, optimize=True)
        info_mats = info_mats + 1e-9 * eye[None, :, :]
        sign, logabsdet = np.linalg.slogdet(info_mats)
        logdet = np.where(sign > 0.0, logabsdet, np.nan).reshape(n_grid, n_grid)
        maps.append(logdet.astype(np.float32))
    matrix = np.nanmean(np.stack(maps, axis=0), axis=0)

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, ax = plt_module.subplots(figsize=(3.35, 3.05))
    im = ax.imshow(
        matrix,
        aspect="equal",
        origin="lower",
        extent=[-grid_lim, grid_lim, -grid_lim, grid_lim],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    if style_colorbar is not None:
        style_colorbar(cbar)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")
    title = _format_count_title(title_template, len(seed_refs))
    if title is not None:
        ax.set_title(title)
    if style_axis is not None:
        style_axis(ax)
    fig.tight_layout()
    save_figure_formats(fig, figures_dir / output_stem, figure_formats, plt_module=plt_module)
