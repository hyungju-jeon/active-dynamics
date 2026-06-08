#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SUPPORTED_FIGURE_FORMATS = frozenset({".pdf", ".png", ".svg"})


def _parse_figure_formats(raw: str) -> tuple[str, ...]:
    """Parse comma-separated output formats for experiment figure files."""
    formats: list[str] = []
    for item in str(raw).split(","):
        fmt = item.strip().lower()
        if not fmt:
            continue
        if not fmt.startswith("."):
            fmt = f".{fmt}"
        if fmt not in SUPPORTED_FIGURE_FORMATS:
            expected = ", ".join(sorted(SUPPORTED_FIGURE_FORMATS))
            raise ValueError(f"Unsupported figure format {item!r}. Expected one of: {expected}")
        if fmt not in formats:
            formats.append(fmt)
    return tuple(formats) if formats else (".pdf",)


def _save_figure(fig: Any, stem_path: Path, figure_formats: Sequence[str]) -> None:
    """Save a Matplotlib figure under one stem using validated extensions."""
    for fmt in figure_formats:
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if fmt == ".png":
            save_kwargs["dpi"] = 300
        fig.savefig(stem_path.with_suffix(fmt), **save_kwargs)


def load_plotting(
    figures_dir: Path,
    *,
    apply_style: Callable[[Any], None] | None = None,
) -> Any | None:
    """Load Matplotlib lazily and create the output directory."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt_module
    except Exception:
        return None
    if apply_style is not None:
        apply_style(plt_module)
    return plt_module


def _safe_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _sample_sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


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
        sems.append(_sample_sem(nums))

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
    _save_figure(fig, figures_dir / output_stem, figure_formats)
    plt_module.close(fig)


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
    _save_figure(fig, figures_dir / output_stem, figure_formats)
    plt_module.close(fig)


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
    _save_figure(fig, figures_dir / output_stem, figure_formats)
    plt_module.close(fig)


def plot_neuron_tuning_curve_colormap(
    figures_dir: Path,
    *,
    records: list[dict[str, Any]],
    figure_formats: Sequence[str],
    get_environment_preset_from_metadata: Callable[[dict[str, Any]], Any],
    reconstruct_loglinear_rate_model: Callable[[dict[str, Any]], tuple[Any, Any, Any]],
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
        weights, bias, _dt = reconstruct_loglinear_rate_model(metadata)
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
    _save_figure(fig, figures_dir / output_stem, figure_formats)
    plt_module.close(fig)


def plot_information_colormap(
    figures_dir: Path,
    *,
    records: list[dict[str, Any]],
    figure_formats: Sequence[str],
    get_environment_preset_from_metadata: Callable[[dict[str, Any]], Any],
    reconstruct_loglinear_rate_model: Callable[[dict[str, Any]], tuple[Any, Any, Any]],
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
        weights, bias, dt = reconstruct_loglinear_rate_model(metadata)
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
    _save_figure(fig, figures_dir / output_stem, figure_formats)
    plt_module.close(fig)


def _load_pdf_plotting(output_path: Path, apply_style: Callable[[Any], None] | None) -> Any | None:
    """Load Matplotlib with an Agg backend for one PDF output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_module
    except Exception:
        return None
    if apply_style is not None:
        apply_style(plt_module)
    return plt_module


def _save_pdf(fig: Any, plt_module: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt_module.close(fig)
    return output_path


def apply_tbme_asset_plot_style(plt_module: Any, *, stroke_color: str = "#3A3A3A") -> None:
    """Apply compact TBME overview figure defaults."""
    plt_module.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 7.8,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": stroke_color,
            "axes.linewidth": 0.55,
            "axes.labelcolor": stroke_color,
            "xtick.color": stroke_color,
            "ytick.color": stroke_color,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def plot_tbme_r2_threshold_stacked_bars(
    output_path: Path,
    *,
    group_name: str,
    refs: Sequence[Any],
    threshold_rows: list[dict[str, object]],
    thresholds: Sequence[float],
    field_prefix: str,
    ylabel: str,
    title_metric: str,
    log_y: bool,
    threshold_suffix: Callable[[float], str],
    safe_float: Callable[[object], float | None],
    threshold_segments: Callable[[dict[str, object], str], tuple[list[float], bool]],
    threshold_value_penalty: Callable[[list[dict[str, object]], str], float],
    policy_threshold_sort_key: Callable[[str, Sequence[Any], dict[tuple[str, str], dict[str, object]], str, float], Any],
    short_policy_label: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    stroke_color: str,
    neutral_color: str,
    neutral_light: str,
    neutral_fill: str,
    segment_colors: Sequence[str],
) -> Path | None:
    """Plot stacked bars for first steps or CPU time to TBME R2 thresholds."""
    if not threshold_rows:
        return None
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        return None
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    row_by_key = {(str(row["suite_id"]), str(row["policy_id"])): row for row in threshold_rows}
    missing_penalty = threshold_value_penalty(threshold_rows, field_prefix)
    policy_ids = sorted(
        {str(row["policy_id"]) for row in threshold_rows},
        key=lambda policy_id: policy_threshold_sort_key(
            policy_id, refs, row_by_key, field_prefix, missing_penalty
        ),
    )
    if not policy_ids:
        return None
    positive_values = [
        value
        for row in threshold_rows
        for threshold in thresholds
        if (value := safe_float(row.get(f"{field_prefix}_{threshold_suffix(float(threshold))}")))
        is not None
        and value > 0.0
    ]
    log_floor = min(positive_values) * 0.72 if log_y and positive_values else 0.0
    n_methods = len(policy_ids)
    group_gap = 3.0
    bar_width = 0.48
    x_positions: list[float] = []
    x_labels: list[str] = []
    group_centers: list[float] = []
    max_height = 1.0
    fig_width = max(6.8, 0.42 * n_methods * len(refs) + 0.35 * len(refs) + 2.2)
    fig, ax = plt_module.subplots(figsize=(fig_width, 3.45))

    for env_idx, ref in enumerate(refs):
        base = env_idx * (n_methods + group_gap)
        group_centers.append(base + (n_methods - 1) / 2.0)
        if env_idx > 0:
            ax.axvline(base - group_gap / 2.0, color=neutral_light, linewidth=0.7, alpha=0.85)
        if env_idx % 2 == 1:
            ax.axvspan(base - 0.62, base + n_methods - 0.38, color=neutral_fill, alpha=0.52)
        for method_idx, policy_id in enumerate(policy_ids):
            x = base + method_idx
            x_positions.append(x)
            x_labels.append(short_policy_label(policy_id))
            row = row_by_key.get((str(ref.suite_id), policy_id), {})
            segments, reached_all = threshold_segments(row, field_prefix)
            bottom = 0.0
            for seg_idx, segment in enumerate(segments):
                if segment <= 0.0:
                    continue
                ax.bar(
                    x,
                    segment,
                    width=bar_width,
                    bottom=bottom,
                    color=segment_colors[seg_idx],
                    edgecolor=stroke_color,
                    linewidth=0.35,
                    zorder=3,
                )
                bottom += segment
            max_height = max(max_height, bottom)
            if bottom == 0.0:
                ax.plot(
                    [x - bar_width / 2.0, x + bar_width / 2.0],
                    [log_floor, log_floor],
                    color=neutral_color,
                    linewidth=0.7,
                    zorder=4,
                )
            if not reached_all:
                ax.scatter(
                    [x],
                    [bottom if bottom > 0.0 else log_floor],
                    marker="x",
                    s=12,
                    color=stroke_color,
                    linewidths=0.6,
                    zorder=5,
                )

    for center, ref in zip(group_centers, refs, strict=True):
        ax.text(
            center,
            -0.31,
            str(ref.label),
            ha="center",
            va="top",
            color=stroke_color,
            transform=ax.get_xaxis_transform(),
            fontsize=7.5,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=5.8)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{group_name}: {title_metric} to trajectory R2 thresholds", pad=6.0)
    ax.set_xlim(min(x_positions) - 0.8, max(x_positions) + 0.8)
    if log_y and positive_values:
        ax.set_yscale("log")
        ax.set_ylim(log_floor, max_height * 1.45)
    else:
        ax.set_ylim(0.0, max_height * 1.16)
    ax.grid(axis="y", color=neutral_light, linewidth=0.35, alpha=0.38, zorder=1)
    for spine in ax.spines.values():
        spine.set_color(stroke_color)
        spine.set_linewidth(0.55)
    legend_handles = [
        Patch(facecolor=segment_colors[0], edgecolor=stroke_color, linewidth=0.35, label="0 -> 0.90"),
        Patch(facecolor=segment_colors[1], edgecolor=stroke_color, linewidth=0.35, label="0.90 -> 0.95"),
        Patch(facecolor=segment_colors[2], edgecolor=stroke_color, linewidth=0.35, label="0.95 -> 0.99"),
        Line2D(
            [0],
            [0],
            color=stroke_color,
            marker="x",
            linestyle="None",
            markersize=4.5,
            markeredgewidth=0.7,
            label="threshold not reached",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        ncol=1,
        fontsize=6.5,
        handlelength=1.4,
    )
    fig.subplots_adjust(left=0.075, right=0.83, top=0.88, bottom=0.37)
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_schedule_threshold_pareto(
    output_path: Path,
    *,
    rows: list[dict[str, object]],
    env_labels: Sequence[str],
    policy_ids: Sequence[str],
    thresholds: Sequence[float],
    threshold_suffix: Callable[[float], str],
    safe_float: Callable[[object], float | None],
    short_policy_label: Callable[[str], str],
    threshold_point_colors: Mapping[float, str],
    apply_style: Callable[[Any], None] | None,
    stroke_color: str,
    neutral_light: str,
    white_color: str = "#FFFFFF",
) -> Path | None:
    """Plot CPU-time versus sample-efficiency points for schedule thresholds."""
    if not rows:
        return None
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        return None
    from matplotlib.lines import Line2D

    policy_offsets = {
        policy_id: ((idx % 4) - 1.5, (idx // 4) - 0.5) for idx, policy_id in enumerate(policy_ids)
    }
    fig, axes = plt_module.subplots(1, len(env_labels), figsize=(8.2, 2.95), sharey=True)
    if len(env_labels) == 1:
        axes = [axes]

    max_step_seen = 1.0
    for ax, env_label in zip(axes, env_labels, strict=True):
        env_rows = [row for row in rows if str(row["suite_label"]) == env_label]
        plotted_for_policy: dict[str, tuple[float, float, float]] = {}
        max_cpu_seen = 1.0
        for row in env_rows:
            policy_id = str(row["policy_id"])
            marker = "D" if policy_id == "active_myopic" else "o"
            for threshold in thresholds:
                suffix = threshold_suffix(float(threshold))
                step = safe_float(row.get(f"step_to_r2_{suffix}"))
                cpu_time = safe_float(row.get(f"cpu_time_sec_to_r2_{suffix}"))
                if step is None or cpu_time is None:
                    continue
                ax.scatter(
                    cpu_time,
                    step,
                    s=24 if policy_id != "active_myopic" else 30,
                    marker=marker,
                    facecolor=threshold_point_colors[float(threshold)],
                    edgecolor=stroke_color,
                    linewidth=0.45,
                    alpha=0.92,
                    zorder=4,
                )
                max_step_seen = max(max_step_seen, step)
                max_cpu_seen = max(max_cpu_seen, cpu_time)
                if (
                    policy_id not in plotted_for_policy
                    or threshold > plotted_for_policy[policy_id][2]
                ):
                    plotted_for_policy[policy_id] = (cpu_time, step, float(threshold))
        for policy_id, (cpu_time, step, _threshold) in plotted_for_policy.items():
            dx, dy = policy_offsets.get(policy_id, (0.0, 0.0))
            ax.annotate(
                short_policy_label(policy_id),
                (cpu_time, step),
                xytext=(4.0 + 3.0 * dx, 3.0 + 3.0 * dy),
                textcoords="offset points",
                fontsize=5.8,
                color=stroke_color,
                ha="left",
                va="bottom",
                bbox={"facecolor": white_color, "edgecolor": "none", "alpha": 0.72, "pad": 0.25},
            )
        ax.set_title(env_label, fontsize=8.0, pad=3.0)
        ax.set_xlabel("CPU time (sec)")
        ax.set_xlim(left=0.0, right=max_cpu_seen * 1.13)
        ax.grid(color=neutral_light, linewidth=0.32, alpha=0.36)
        for spine in ax.spines.values():
            spine.set_color(stroke_color)
            spine.set_linewidth(0.55)
        ax.tick_params(width=0.45, length=2.0, colors=stroke_color)
    axes[0].set_ylabel("Environment steps")
    axes[0].set_ylim(bottom=0.0, top=max_step_seen * 1.16)
    fig.suptitle("Exp03 schedule Pareto: time and steps to trajectory R2 thresholds", y=0.99)
    threshold_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=threshold_point_colors[float(threshold)],
            markeredgecolor=stroke_color,
            markeredgewidth=0.45,
            markersize=5.0,
            label=f"R2 {threshold:.2f}",
        )
        for threshold in thresholds
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=white_color,
            markeredgecolor=stroke_color,
            markersize=5.0,
            label="Active schedule",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor=white_color,
            markeredgecolor=stroke_color,
            markersize=5.0,
            label="Myopic",
        ),
    ]
    fig.legend(
        handles=[*threshold_handles, *method_handles],
        loc="upper center",
        ncol=5,
        fontsize=6.4,
        bbox_to_anchor=(0.5, 0.905),
        columnspacing=0.9,
        handlelength=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.73, bottom=0.18, wspace=0.22)
    return _save_pdf(fig, plt_module, output_path)


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


def _tbme_plot_vectorfield_background(
    ax: Any,
    dyn_true: Any,
    plot_lim: float,
    *,
    neutral_light: str,
) -> None:
    from actdyn.utils.plotting import compute_vector_field

    x_grid, y_grid, u_grid, v_grid = compute_vector_field(
        dyn_true,
        x_range=plot_lim,
        n_grid=36,
        is_residual=True,
        device="cpu",
    )
    ax.streamplot(
        x_grid.cpu().numpy(),
        y_grid.cpu().numpy(),
        u_grid.cpu().numpy(),
        v_grid.cpu().numpy(),
        color=neutral_light,
        linewidth=0.34,
        density=1.35,
        arrowsize=0.55,
        zorder=1,
    )


def plot_tbme_trajectory_overlay(
    output_path: Path,
    *,
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
) -> Path | None:
    """Plot trajectory overlays on the true vector field by policy."""
    if not grouped:
        return None
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        return None
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
        _tbme_plot_vectorfield_background(ax, dyn_true, plot_lim, neutral_light=neutral_light)
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
    return _save_pdf(fig, plt_module, output_path)


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


def plot_tbme_trajectory_density(
    output_path: Path,
    *,
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
) -> Path | None:
    """Plot trajectory sample density by policy on the true state space."""
    if not grouped:
        return None
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        return None
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
        _tbme_plot_vectorfield_background(ax, dyn_true, plot_lim, neutral_light=neutral_light)
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
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_bottleneck_sweep(
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
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(1, 2, figsize=(7.1, 2.95), sharex=True)
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
            r2_y.append(np.nan if match["trajectory_r2_mean"] is None else match["trajectory_r2_mean"])
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
        ax.set_xticklabels([source.label for source in sources], rotation=18, ha="right")
    axes[0].set_ylabel("Final prediction R2")
    axes[0].set_ylim(min(-0.1, min(finite_r2) - 0.05) if finite_r2 else -0.1, 1.05)
    axes[0].set_title("A. Prediction under bottlenecks")
    axes[1].set_ylabel("Steps to prediction R2 >= 0.90")
    axes[1].set_ylim(0.0, max_step * 1.15)
    axes[1].set_title("B. Predictive sample efficiency")
    axes[1].legend(loc="upper left", fontsize=6.6, ncol=1)
    fig.tight_layout(w_pad=1.1)
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_objective_ablation(
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
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(len(sources), 2, figsize=(7.15, 5.05), squeeze=False)
    x = np.arange(len(policy_ids), dtype=np.float64)
    x_labels = [policy_label(policy_id) for policy_id in policy_ids]
    letters = ["A", "B", "C", "D"]
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
                    steps, values - sem, values + sem, color=color, alpha=0.14, linewidth=0
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
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_mismatch_dose_response(
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
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    dose_order = ["none", "mild", "medium", "strong"]
    dose_labels = ["None", "Mild", "Medium", "Strong"]
    x = np.arange(len(dose_order), dtype=np.float64)
    fig, axes = plt_module.subplots(1, 2, figsize=(7.1, 2.9), sharey=False)
    for ax, family in zip(axes, ["Duffing", "Asymmetric basin"], strict=True):
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
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_downstream_control(
    output_path: Path,
    *,
    rows: list[dict[str, Any]],
    policy_ids: Sequence[str],
    sem: Callable[[Sequence[float]], float],
    policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
    stroke_color: str,
    neutral_light: str,
    neutral_fill: str,
) -> Path:
    """Plot downstream control utility and its relation to prediction quality."""
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    grouped: dict[str, list[dict[str, Any]]] = {
        policy_id: [row for row in rows if row["policy_id"] == policy_id]
        for policy_id in policy_ids
    }
    summary_rows: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        items = grouped.get(policy_id, [])
        if not items:
            continue
        costs = [float(row["relative_control_cost"]) for row in items]
        param = [
            float(row["parameter_error_final"])
            for row in items
            if row.get("parameter_error_final") is not None
        ]
        r2 = [
            float(row["trajectory_r2_final"])
            for row in items
            if row.get("trajectory_r2_final") is not None
        ]
        summary_rows.append(
            {
                "policy_id": policy_id,
                "policy_label": policy_label(policy_id) if policy_id != "oracle_true_model" else "Oracle",
                "relative_control_cost_mean": float(np.mean(costs)),
                "relative_control_cost_sem": sem(costs),
                "parameter_error_mean": float(np.mean(param)) if param else None,
                "trajectory_r2_mean": float(np.mean(r2)) if r2 else None,
                "n": len(costs),
            }
        )

    fig, axes = plt_module.subplots(1, 2, figsize=(7.1, 2.9))
    x = np.arange(len(summary_rows), dtype=np.float64)
    y = [float(row["relative_control_cost_mean"]) for row in summary_rows]
    yerr = [float(row["relative_control_cost_sem"]) for row in summary_rows]
    colors = [
        neutral_fill if row["policy_id"] == "oracle_true_model" else policy_color(str(row["policy_id"]))
        for row in summary_rows
    ]
    axes[0].bar(x, y, yerr=yerr, color=colors, edgecolor=stroke_color, linewidth=0.45, capsize=2.0)
    axes[0].axhline(1.0, color=neutral_light, linestyle="--", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(row["policy_label"]) for row in summary_rows], rotation=35, ha="right")
    axes[0].set_ylabel("Control cost / oracle")
    axes[0].set_title("A. Downstream control utility")
    style_axis(axes[0])

    for row in rows:
        if row["policy_id"] == "oracle_true_model":
            continue
        r2 = row.get("trajectory_r2_final")
        cost = row.get("relative_control_cost")
        if r2 is None or cost is None:
            continue
        axes[1].scatter(
            [float(r2)],
            [float(cost)],
            s=22,
            color=policy_color(str(row["policy_id"])),
            edgecolor=stroke_color,
            linewidth=0.35,
            label=policy_label(str(row["policy_id"])),
        )
    handles, labels = axes[1].get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        dedup.setdefault(label, handle)
    axes[1].set_xlabel("Final prediction R2")
    axes[1].set_ylabel("Control cost / oracle")
    axes[1].set_title("B. Prediction quality vs control cost")
    axes[1].legend(dedup.values(), dedup.keys(), fontsize=6.0, loc="upper left")
    style_axis(axes[1])
    fig.tight_layout(w_pad=1.0)
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_true_dynamics_all(
    output_path: Path,
    *,
    fields: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    grid_lim: float,
    apply_style: Callable[[Any], None] | None,
    stroke_color: str,
) -> Path:
    """Plot the true vector fields for the TBME synthetic systems."""
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    from actdyn.utils.plotting import decorate_phase_space_axis
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    finite_speed = np.concatenate([arr[np.isfinite(arr)].reshape(-1) for *_rest, arr in fields])
    vmax = float(np.percentile(finite_speed, 98.0)) if finite_speed.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    fig = plt_module.figure(figsize=(7.25, 4.05))
    gs = fig.add_gridspec(2, 7, wspace=0.36, hspace=0.42, width_ratios=[1, 1, 1, 1, 1, 1, 0.08])
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    cax = fig.add_subplot(gs[:, 6])
    for panel_idx, (ax, (title, x_np, y_np, u_np, v_np, log_speed)) in enumerate(zip(axes, fields)):
        ax.pcolormesh(
            x_np,
            y_np,
            log_speed,
            cmap="viridis",
            norm=norm,
            shading="auto",
            alpha=0.82,
            rasterized=True,
            zorder=0,
        )
        ax.streamplot(
            x_np,
            y_np,
            u_np,
            v_np,
            color=stroke_color,
            linewidth=0.38,
            density=1.25,
            arrowsize=0.62,
            zorder=2,
        )
        decorate_phase_space_axis(
            ax,
            xlim=(-grid_lim, grid_lim),
            ylim=(-grid_lim, grid_lim),
            title=f"{chr(ord('A') + panel_idx)}. {title}",
            xlabel="x",
            ylabel="v",
            grid_alpha=0.20,
        )
        ax.set_xticks([-6, 0, 6])
        ax.set_yticks([-6, 0, 6])

    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\log(1 + \|f(z)\|)$")
    cbar.outline.set_linewidth(0.45)
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_neutral_vector_field(
    ax: Any,
    dynamics: Any,
    *,
    grid_lim: float,
    n_grid: int,
    arrowsize: float,
    stroke_color: str,
) -> None:
    """Draw a neutral TBME vector field background on an existing axis."""
    from actdyn.utils.plotting import compute_vector_field
    from matplotlib.colors import to_rgba

    x_grid, y_grid, u_grid, v_grid = compute_vector_field(
        dynamics,
        x_range=grid_lim,
        n_grid=n_grid,
        is_residual=True,
        device="cpu",
    )
    ax.streamplot(
        x_grid.cpu().numpy(),
        y_grid.cpu().numpy(),
        u_grid.cpu().numpy(),
        v_grid.cpu().numpy(),
        color=to_rgba(stroke_color, 0.42),
        linewidth=0.34,
        density=1.55,
        arrowsize=arrowsize,
        zorder=1,
    )


def plot_tbme_asymmetric_basin_mechanism(
    output_path: Path,
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    logdet_grid: np.ndarray,
    info_threshold: float,
    info_vmin: float,
    info_vmax: float,
    panel_min: float,
    panel_max: float,
    true_dynamics: Any,
    traces_by_policy: Mapping[str, Sequence[np.ndarray]],
    policy_ids: Sequence[str],
    informative_fraction: Mapping[str, Sequence[float]],
    coverage_fraction: Mapping[str, Sequence[float]],
    final_r2: Mapping[str, Sequence[float]],
    sem: Callable[[Sequence[float]], float],
    short_policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
    stroke_color: str,
) -> Path:
    """Plot asymmetric-basin mechanism diagnostics."""
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    from actdyn.utils.plotting import decorate_phase_space_axis
    from matplotlib.lines import Line2D

    fig, axes = plt_module.subplots(2, 2, figsize=(7.05, 5.75))
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
        alpha=0.72,
    )
    ax.contour(
        x_axis,
        y_axis,
        logdet_grid,
        levels=[info_threshold],
        colors=[stroke_color],
        linewidths=0.7,
        linestyles="--",
    )
    plot_tbme_neutral_vector_field(
        ax,
        true_dynamics,
        n_grid=28,
        grid_lim=panel_max,
        arrowsize=0.70,
        stroke_color=stroke_color,
    )
    highlighted = ["active_planning_u20_r20_h40", "active_myopic", "flex", "prbs"]
    for policy_id in highlighted:
        for traj in traces_by_policy.get(policy_id, [])[:8]:
            ax.plot(traj[:, 0], traj[:, 1], color=policy_color(policy_id), linewidth=0.55, alpha=0.68)
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
    cbar.set_label("mean log det(I_z)")
    cbar.outline.set_linewidth(0.45)
    decorate_phase_space_axis(
        ax,
        xlim=(panel_min, panel_max),
        ylim=(panel_min, panel_max),
        title="A. Hard asymmetric-basin information and vector field",
        grid_alpha=0.20,
    )
    style_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], color=policy_color(policy_id), linewidth=0.9, label=short_policy_label(policy_id))
            for policy_id in highlighted
        ],
        loc="lower right",
        fontsize=5.8,
        frameon=True,
        framealpha=0.78,
        borderpad=0.25,
    )

    panels = [
        (axes[0, 1], informative_fraction, "B. Occupancy of high-information states", "Fraction of samples"),
        (axes[1, 0], coverage_fraction, "C. State-space coverage", "Visited-bin fraction"),
        (axes[1, 1], final_r2, "D. Endpoint prediction", "Final prediction R2"),
    ]
    labels = [short_policy_label(policy_id) for policy_id in policy_ids]
    x = np.arange(len(policy_ids), dtype=np.float64)
    for ax_i, data, title, ylabel in panels:
        means = []
        errors = []
        for policy_id in policy_ids:
            vals = [float(v) for v in data.get(policy_id, []) if math.isfinite(float(v))]
            means.append(float(np.mean(vals)) if vals else np.nan)
            errors.append(sem(vals))
        ax_i.bar(
            x,
            means,
            yerr=errors,
            color=[policy_color(policy_id) for policy_id in policy_ids],
            edgecolor=stroke_color,
            linewidth=0.45,
            capsize=2.3,
            error_kw={"elinewidth": 0.55, "ecolor": stroke_color, "capthick": 0.55},
        )
        ax_i.set_xticks(x)
        ax_i.set_xticklabels(labels, rotation=25, ha="right")
        ax_i.set_ylabel(ylabel)
        ax_i.set_title(title)
        style_axis(ax_i, grid_axis="y")
    axes[1, 1].set_ylim(-0.05, 1.05)
    fig.suptitle(
        "Hard asymmetric-basin mechanism: information geometry, coverage, and prediction",
        y=0.995,
    )
    fig.tight_layout()
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_learned_vectorfield_snapshots(
    output_path: Path,
    *,
    seed: int,
    row_ids: Sequence[str],
    checkpoints: Sequence[int],
    dynamics_by_cell: Mapping[tuple[str, int], Any],
    traces_by_cell: Mapping[tuple[str, int], np.ndarray],
    plot_abs: float,
    short_policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    stroke_color: str,
    neutral_fill: str,
    grid_color: str,
) -> Path:
    """Plot true and learned vector-field snapshots for a shared seed."""
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(
        len(row_ids), len(checkpoints), figsize=(7.25, 8.85), sharex=True, sharey=True
    )
    for row_idx, row_id in enumerate(row_ids):
        color = stroke_color if row_id == "true" else policy_color(row_id)
        for col_idx, checkpoint in enumerate(checkpoints):
            ax = axes[row_idx, col_idx]
            dynamics = dynamics_by_cell[(row_id, int(checkpoint))]
            plot_tbme_neutral_vector_field(
                ax,
                dynamics,
                grid_lim=plot_abs,
                n_grid=22,
                arrowsize=0.58,
                stroke_color=stroke_color,
            )
            traj = traces_by_cell.get((row_id, int(checkpoint)), np.empty((0, 2), dtype=np.float32))
            if traj.size:
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.8, alpha=0.92, zorder=4)
                ax.scatter(
                    [traj[-1, 0]],
                    [traj[-1, 1]],
                    s=13,
                    color=color,
                    edgecolor=stroke_color,
                    linewidth=0.35,
                    zorder=5,
                )
                ax.scatter(
                    [traj[0, 0]],
                    [traj[0, 1]],
                    s=9,
                    color=neutral_fill,
                    edgecolor=stroke_color,
                    linewidth=0.3,
                    zorder=5,
                )
            ax.set_xlim(-plot_abs, plot_abs)
            ax.set_ylim(-plot_abs, plot_abs)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color=grid_color, linewidth=0.28, alpha=0.25)
            for spine in ax.spines.values():
                spine.set_color(stroke_color)
                spine.set_linewidth(0.48)
            ax.tick_params(width=0.4, length=1.6, labelsize=5.8)
            if row_idx == 0:
                ax.set_title(f"step {checkpoint}", fontsize=7.4, pad=2.0)
            ylabel = "True" if row_id == "true" else short_policy_label(row_id)
            ax.set_ylabel(ylabel if col_idx == 0 else "", fontsize=7.2)
            ax.set_xlabel("x" if row_idx == len(row_ids) - 1 else "")
    fig.suptitle(f"Hard asymmetric-basin true and learned vector fields, seed {seed}", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), w_pad=0.25, h_pad=0.45)
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_sample_efficiency_thresholds(
    output_path: Path,
    *,
    values: Sequence[tuple[str, str, float | None, float]],
    suite_labels: Sequence[str],
    policy_ids: Sequence[str],
    short_policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
    stroke_color: str,
) -> Path:
    """Plot first environment step to predictive-accuracy thresholds."""
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    max_step = max((float(step) for _suite, _policy, step, _threshold in values if step is not None), default=1.0)
    fig, ax = plt_module.subplots(figsize=(7.1, 3.55))
    width = 0.12
    group_x = np.arange(len(suite_labels), dtype=np.float64)
    for idx, policy_id in enumerate(policy_ids):
        xs = group_x + (idx - (len(policy_ids) - 1) / 2.0) * width
        heights = []
        missing_x = []
        missing_y = []
        for suite_label in suite_labels:
            match = [v for v in values if v[0] == suite_label and v[1] == policy_id]
            step = match[0][2] if match else None
            if step is None:
                heights.append(0.0)
                missing_x.append(xs[len(heights) - 1])
                missing_y.append(max_step * 1.04)
            else:
                heights.append(float(step))
        ax.bar(
            xs,
            heights,
            width=width * 0.92,
            color=policy_color(policy_id),
            edgecolor=stroke_color,
            linewidth=0.35,
            label=short_policy_label(policy_id),
        )
        if missing_x:
            ax.scatter(missing_x, missing_y, marker="x", s=13, color=policy_color(policy_id), linewidths=0.75)
    ax.set_xticks(group_x)
    ax.set_xticklabels(suite_labels, rotation=18, ha="right")
    ax.set_ylabel("Environment steps")
    ax.set_title("Steps to predictive-accuracy thresholds")
    ax.text(
        0.02,
        0.97,
        "Base and hard bars use R2 >= 0.95; mismatch bars use R2 >= 0.90. x = threshold not reached.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=stroke_color,
    )
    ax.set_ylim(0.0, max_step * 1.18)
    style_axis(ax, grid_axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=6.3)
    fig.tight_layout()
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_compute_accuracy_pareto(
    output_path: Path,
    *,
    schedule_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    focus_policies: Sequence[str],
    short_policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
    stroke_color: str,
) -> Path:
    """Plot prediction-cost Pareto views for schedules and policy families."""
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(1, 2, figsize=(7.1, 3.15))

    markers = {"Duffing": "o", "Damped pendulum": "s", "Asymmetric basin": "^"}
    ax = axes[0]
    for row in schedule_rows:
        policy_id = str(row["policy_id"])
        suite_label = str(row["suite_label"])
        ax.scatter(
            float(row["runtime_sec"]),
            float(row["trajectory_r2"]),
            s=30,
            marker=markers.get(suite_label, "o"),
            color=policy_color(policy_id),
            edgecolor=stroke_color,
            linewidth=0.35,
            alpha=0.9,
        )
        if policy_id in {
            "active_planning_u20_r20_h40",
            "active_planning_u10_r20_h40",
            "active_planning_u1_r1_h40",
        }:
            ax.annotate(
                policy_id.replace("active_planning_", "").replace("_h40", ""),
                (float(row["runtime_sec"]), float(row["trajectory_r2"])),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.8,
                color=stroke_color,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Runtime per run (sec, log scale)")
    ax.set_ylabel("Final prediction R2")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("A. Planning schedule prediction-cost tradeoff")
    style_axis(ax)

    group_markers = {
        "Duffing": "o",
        "Damped pendulum": "s",
        "Asymmetric basin": "^",
        "Duffing hard": "D",
        "Asymmetric basin hard": "P",
        "Damped pendulum hard": "X",
        "Duffing parameter mismatch": "v",
        "Asymmetric basin parameter mismatch": ">",
    }
    ax = axes[1]
    for row in group_rows:
        policy_id = str(row["policy_id"])
        if policy_id not in focus_policies:
            continue
        suite_label = str(row["suite_label"])
        ax.scatter(
            float(row["runtime_sec"]),
            float(row["trajectory_r2"]),
            s=31,
            marker=group_markers.get(suite_label, "o"),
            color=policy_color(policy_id),
            edgecolor=stroke_color,
            linewidth=0.35,
            alpha=0.86,
            label=short_policy_label(policy_id),
        )
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.set_xscale("log")
    ax.set_xlabel("Runtime per run (sec, log scale)")
    ax.set_ylabel("Final prediction R2")
    ax.set_ylim(-0.25, 1.05)
    ax.set_title("B. Policy-level prediction-cost tradeoff")
    style_axis(ax)
    ax.legend([h for h, _l in unique], [l for _h, l in unique], fontsize=6.0, loc="upper right")
    fig.tight_layout()
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_per_parameter_recovery(
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
    """Plot per-parameter recovery traces for asymmetric-basin dynamics."""
    plt_module = _load_pdf_plotting(output_path, apply_style)
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
            ax.plot(steps, means_arr, color=color, linewidth=1.0, label=short_policy_label(policy_id))
            ax.fill_between(
                steps,
                means_arr - sems_arr,
                means_arr + sems_arr,
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )
        ax.axhline(float(true_params[param_idx]), color=stroke_color, linewidth=0.8, linestyle="--")
        ax.set_title(f"{chr(65 + param_idx)}. {names[param_idx]}")
        ax.set_ylabel("Estimate")
        style_axis(ax)
    axes[1, 0].set_xlabel("Environment step")
    axes[1, 1].set_xlabel("Environment step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=6.4, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Asymmetric-basin per-parameter recovery", y=1.06)
    fig.tight_layout()
    return _save_pdf(fig, plt_module, output_path)


def plot_tbme_information_learning_coupling(
    output_path: Path,
    *,
    points: Mapping[str, Sequence[tuple[float, float, float]]],
    policy_ids: Sequence[str],
    short_policy_label: Callable[[str], str],
    policy_color: Callable[[str], str],
    apply_style: Callable[[Any], None] | None,
    style_axis: Callable[..., None],
    stroke_color: str,
) -> Path:
    """Plot cumulative information against endpoint prediction and R2 improvement."""
    plt_module = _load_pdf_plotting(output_path, apply_style)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    fig, axes = plt_module.subplots(1, 2, figsize=(7.1, 3.1))
    for policy_id in policy_ids:
        vals = points.get(policy_id, [])
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        x = np.log10(1.0 + arr[:, 0])
        axes[0].scatter(x, arr[:, 1], s=13, color=policy_color(policy_id), alpha=0.38, edgecolors="none")
        axes[0].scatter(
            [float(np.median(x))],
            [float(np.median(arr[:, 1]))],
            s=42,
            color=policy_color(policy_id),
            edgecolor=stroke_color,
            linewidth=0.45,
            label=short_policy_label(policy_id),
        )
        axes[1].scatter(x, arr[:, 2], s=13, color=policy_color(policy_id), alpha=0.38, edgecolors="none")
        axes[1].scatter(
            [float(np.median(x))],
            [float(np.median(arr[:, 2]))],
            s=42,
            color=policy_color(policy_id),
            edgecolor=stroke_color,
            linewidth=0.45,
        )
    axes[0].set_title("A. Information versus endpoint prediction")
    axes[0].set_xlabel("log10(1 + cumulative I_theta)")
    axes[0].set_ylabel("Final prediction R2")
    axes[0].set_ylim(-0.1, 1.05)
    axes[1].set_title("B. Information versus R2 improvement")
    axes[1].set_xlabel("log10(1 + cumulative I_theta)")
    axes[1].set_ylabel("Final minus initial prediction R2")
    for ax in axes:
        style_axis(ax)
    axes[0].legend(fontsize=6.0, loc="best")
    fig.tight_layout()
    return _save_pdf(fig, plt_module, output_path)
