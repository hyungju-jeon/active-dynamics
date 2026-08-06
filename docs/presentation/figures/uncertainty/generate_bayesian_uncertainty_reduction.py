from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BASENAME = "bayesian_uncertainty_reduction"
DEFAULT_DPI = 300
DEFAULT_FORMATS = "svg,png"
SUPPORTED_FORMATS = {"png", "svg", "pdf"}
SURFACE_ALPHA = 0.94
MIN_SURFACE_ALPHA = 0.24
DENSITY_TRUNCATION_RATIO = 0.08
X_LIMITS = (-1.85, 1.85)
Y_LIMITS = (-1.10, 1.10)
X_GRID_TICKS = np.linspace(-1.3, 1.3, 4)
Y_GRID_TICKS = np.linspace(-0.85, 0.85, 5)

COLORS = {
    "prior_surface": "#9fb9d3",
    "prior_wire": "#6e8eaf",
    "posterior_surface": "#a7c9b2",
    "posterior_wire": "#6a9479",
    "arrow": "#5a5a5a",
    "text": "#222222",
    "axis": "#7a7a7a",
}


def get_distribution_parameters() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prior = {
        "mean": np.array([-0.95, 0.0], dtype=float),
        "covariance": np.array([[0.34, 0.05], [0.05, 0.22]], dtype=float),
    }
    posterior = {
        "mean": np.array([0.95, 0.0], dtype=float),
        "covariance": np.array([[0.12, 0.02], [0.02, 0.08]], dtype=float),
    }
    return prior, posterior


def _gaussian_density(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    grid = np.stack([xx - mean[0], yy - mean[1]], axis=-1)
    inv_cov = np.linalg.inv(covariance)
    exponent = np.einsum("...i,ij,...j->...", grid, inv_cov, grid)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(covariance)))
    return norm * np.exp(-0.5 * exponent)


def compute_density_grids(
    xlim: tuple[float, float] = X_LIMITS,
    ylim: tuple[float, float] = Y_LIMITS,
    n_grid: int = 120,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    xs = np.linspace(xlim[0], xlim[1], n_grid)
    ys = np.linspace(ylim[0], ylim[1], n_grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    prior, posterior = get_distribution_parameters()
    prior_density = _gaussian_density(xx, yy, mean=prior["mean"], covariance=prior["covariance"])
    posterior_density = _gaussian_density(
        xx, yy, mean=posterior["mean"], covariance=posterior["covariance"]
    )
    prior_density = np.where(
        prior_density >= DENSITY_TRUNCATION_RATIO * np.max(prior_density),
        prior_density,
        np.nan,
    )
    posterior_density = np.where(
        posterior_density >= DENSITY_TRUNCATION_RATIO * np.max(posterior_density),
        posterior_density,
        np.nan,
    )
    return (xx, yy, prior_density), (xx, yy, posterior_density)


def _style_3d_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=15, fontweight="bold", color=COLORS["text"], pad=10)
    ax.set_xlabel(r"$\theta_1$", fontsize=12, color=COLORS["text"], labelpad=8)
    ax.set_ylabel(r"$\theta_2$", fontsize=12, color=COLORS["text"], labelpad=8)
    ax.set_zlabel("")
    ax.set_xticks(X_GRID_TICKS)
    ax.set_yticks(Y_GRID_TICKS)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticks([])
    ax.tick_params(axis="x", length=0, pad=-2)
    ax.tick_params(axis="y", length=0, pad=-2)
    ax.view_init(elev=28, azim=-58)
    ax.grid(False)
    ax.set_box_aspect((1.35, 1.15, 0.78))

    for axis in (ax.xaxis, ax.yaxis):
        axis.line.set_color(COLORS["axis"])
        axis.line.set_linewidth(0.7)
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["linewidth"] = 0.0

    ax.zaxis.pane.set_facecolor((0.96, 0.97, 0.98, 0.8))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.0
    ax.zaxis.line.set_linewidth(0.0)
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


def _draw_xy_floor_grid(ax) -> None:
    for index, x_value in enumerate(X_GRID_TICKS):
        line = ax.plot(
            [x_value, x_value],
            [Y_LIMITS[0], Y_LIMITS[1]],
            [0.0, 0.0],
            color="#d0d0d0",
            linewidth=0.65,
            alpha=0.9,
            zorder=0,
        )[0]
        line.set_gid(f"xy-grid-x-{index}")

    for index, y_value in enumerate(Y_GRID_TICKS):
        line = ax.plot(
            [X_LIMITS[0], X_LIMITS[1]],
            [y_value, y_value],
            [0.0, 0.0],
            color="#d0d0d0",
            linewidth=0.65,
            alpha=0.9,
            zorder=0,
        )[0]
        line.set_gid(f"xy-grid-y-{index}")


def _compute_surface_facecolors(zz: np.ndarray, face: str) -> np.ndarray:
    corners = np.stack(
        [zz[:-1, :-1], zz[1:, :-1], zz[:-1, 1:], zz[1:, 1:]],
        axis=-1,
    )
    valid_counts = np.sum(~np.isnan(corners), axis=-1)
    cell_density = np.full(valid_counts.shape, np.nan, dtype=float)
    valid_cells = valid_counts > 0
    cell_density[valid_cells] = np.nansum(corners, axis=-1)[valid_cells] / valid_counts[valid_cells]

    normalized = np.zeros_like(cell_density)
    if np.any(valid_cells):
        density_min = float(np.nanmin(cell_density))
        density_max = float(np.nanmax(cell_density))
        density_span = density_max - density_min
        if density_span > 0.0:
            normalized[valid_cells] = (cell_density[valid_cells] - density_min) / density_span
        else:
            normalized[valid_cells] = 1.0

    base_rgb = np.array(mcolors.to_rgb(face), dtype=float)
    light_rgb = 1.0 - 0.78 * (1.0 - base_rgb)
    rgb = light_rgb + normalized[..., None] * (base_rgb - light_rgb)

    alpha = np.zeros_like(cell_density)
    alpha[valid_cells] = MIN_SURFACE_ALPHA + (SURFACE_ALPHA - MIN_SURFACE_ALPHA) * normalized[valid_cells]

    facecolors = np.zeros(cell_density.shape + (4,), dtype=float)
    facecolors[..., :3] = rgb
    facecolors[..., 3] = alpha
    return facecolors


def _draw_distribution(ax, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, *, face: str, gid: str) -> None:
    masked = np.ma.masked_invalid(zz)
    surface = ax.plot_surface(
        xx,
        yy,
        masked,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        facecolors=_compute_surface_facecolors(zz, face),
        shade=False,
    )
    surface.set_gid(gid)
    surface.set_edgecolor((0.0, 0.0, 0.0, 0.0))


def _add_floor_contour(ax, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, *, wire: str, gid: str) -> None:
    contour = ax.contour(
        xx,
        yy,
        np.ma.masked_invalid(zz),
        zdir="z",
        offset=0.0,
        levels=5,
        colors=wire,
        linewidths=0.75,
        alpha=0.72,
    )
    for collection in contour.collections:
        collection.set_gid(gid)


def _add_mean_shift_arrow(
    fig,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=COLORS["arrow"],
        linewidth=1.3,
        mutation_scale=10,
        transform=fig.transFigure,
    )
    arrow.set_gid("mean-shift-arrow")
    fig.add_artist(arrow)


def _add_observation_arrow(fig) -> None:
    arrow = FancyArrowPatch(
        (0.36, 0.77),
        (0.75, 0.75),
        arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.30",
        color=COLORS["arrow"],
        linewidth=1.3,
        mutation_scale=9,
        transform=fig.transFigure,
    )
    arrow.set_gid("observation-arrow")
    fig.add_artist(arrow)

def build_figure(figsize: tuple[float, float] = (4.97, 3.76)) -> Figure:
    prior_grid, posterior_grid = compute_density_grids()
    xx, yy, prior_density = prior_grid
    _, _, posterior_density = posterior_grid

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    fig.patch.set_facecolor("white")

    _draw_distribution(
        fig.add_subplot(1, 1, 1, projection="3d"),
        xx,
        yy,
        prior_density,
        face=COLORS["prior_surface"],
        gid="prior-surface",
    )
    ax = fig.axes[0]
    _draw_distribution(
        ax,
        xx,
        yy,
        posterior_density,
        face=COLORS["posterior_surface"],
        gid="posterior-surface",
    )
    _add_floor_contour(ax, xx, yy, prior_density, wire=COLORS["prior_wire"], gid="prior-contour")
    _add_floor_contour(ax, xx, yy, posterior_density, wire=COLORS["posterior_wire"], gid="posterior-contour")

    z_max = float(max(np.nanmax(prior_density), np.nanmax(posterior_density)) * 1.08)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_zlim(0.0, z_max)

    _style_3d_axes(ax, "")
    _draw_xy_floor_grid(ax)
    ax.text2D(
        0.15,
        0.70,
        r"Prior $p(\theta)$",
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=COLORS["prior_surface"],
    )
    ax.text2D(
        0.72,
        0.70,
        r"Posterior $p(\theta \mid y_t)$",
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=COLORS["posterior_surface"],
    )

    _add_observation_arrow(fig)
    fig.text(
        0.44,
        0.82,
        "Observation",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=COLORS["text"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    _add_mean_shift_arrow(fig, start=(0.41, 0.40), end=(0.64, 0.35))

    ax.set_position([0.06, 0.00, 0.88, 0.76])
    return fig


def _parse_formats(formats: str | Iterable[str]) -> list[str]:
    if isinstance(formats, str):
        requested = [part.strip().lower() for part in formats.split(",") if part.strip()]
    else:
        requested = [str(part).strip().lower() for part in formats if str(part).strip()]

    if not requested:
        raise ValueError("No output formats requested.")

    unsupported = sorted(set(requested) - SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(
            f"Unsupported format(s): {', '.join(unsupported)}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )
    return requested


def save_figure(
    fig: Figure,
    outdir: str | Path,
    basename: str = DEFAULT_BASENAME,
    formats: str | Iterable[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    selected_formats = _parse_formats(formats)

    saved_paths: list[Path] = []
    for file_format in selected_formats:
        output_path = outdir_path / f"{basename}.{file_format}"
        if file_format == "svg":
            with plt.rc_context({"svg.fonttype": "none"}):
                fig.savefig(output_path, dpi=dpi)
        else:
            fig.savefig(output_path, dpi=dpi)
        saved_paths.append(output_path)
    return saved_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a two-panel Bayesian uncertainty reduction figure."
    )
    parser.add_argument(
        "--outdir",
        default="docs/presentation/figures/uncertainty",
        help="Directory where figure files are written.",
    )
    parser.add_argument(
        "--basename",
        default=DEFAULT_BASENAME,
        help="Base filename for saved outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Raster DPI used for PNG export.",
    )
    parser.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help="Comma-separated output formats. Supported: png, svg, pdf.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    fig = build_figure()
    try:
        output_paths = save_figure(
            fig=fig,
            outdir=args.outdir,
            basename=args.basename,
            formats=args.formats,
            dpi=args.dpi,
        )
    finally:
        plt.close(fig)

    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
