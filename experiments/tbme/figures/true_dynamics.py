"""True-dynamics overview figure: vector fields of the TBME synthetic systems."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from actdyn.environment.vectorfield import ResidualDynamicsCallable
from actdyn.utils.figure_io import load_plotting, save_figure
from actdyn.visualize import compute_vector_field, decorate_phase_space_axis

from ...experiment_definitions import get_environment_preset
from . import artifacts, theme

PANEL_SPECS = [
    ("tbme_duffing", "Duffing"),
    ("tbme_damped_pendulum", "Damped pendulum"),
    ("tbme_gated_duffing", "Gated Duffing"),
]


def generate(suite_dirs: Sequence[Path]) -> list[Path]:
    """Plot the true vector fields for the TBME synthetic systems."""
    figure_paths = artifacts.artifact_paths(
        suite_dirs,
        subdir="figures",
        filename="tbme_experiment_true_dynamics_all.pdf",
    )
    output_path = figure_paths[0]
    plt_module = load_plotting(output_path, apply_style=theme.apply_style, path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    grid_lim = 6.0
    fields = []
    for preset_id, title in PANEL_SPECS:
        env_preset = get_environment_preset(preset_id)
        theta_true = env_preset.true_embedding_vector()
        dynamics = ResidualDynamicsCallable(
            dynamics_type=env_preset.resolved_dynamics_type(),
            dyn_params=env_preset.params_from_embedding(theta_true),
            dynamics_alpha=float(env_preset.dynamics_alpha),
            device="cpu",
        )
        x_grid, y_grid, u_grid, v_grid = compute_vector_field(
            dynamics,
            x_range=grid_lim,
            n_grid=53,
            is_residual=True,
            device="cpu",
        )
        x_np = x_grid.cpu().numpy()
        y_np = y_grid.cpu().numpy()
        u_np = np.nan_to_num(u_grid.cpu().numpy(), nan=0.0, posinf=1e6, neginf=-1e6)
        v_np = np.nan_to_num(v_grid.cpu().numpy(), nan=0.0, posinf=1e6, neginf=-1e6)
        speed = np.hypot(u_np, v_np)
        log_speed = np.log1p(np.nan_to_num(speed, nan=0.0, posinf=1e6, neginf=0.0))
        fields.append((title, x_np, y_np, u_np, v_np, log_speed))

    finite_speed = np.concatenate([arr[np.isfinite(arr)].reshape(-1) for *_rest, arr in fields])
    vmax = float(np.percentile(finite_speed, 98.0)) if finite_speed.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    panel_title_size = 12.0
    label_size = 10.0

    fig = plt_module.figure(figsize=(7.25, 2.35))
    gs = fig.add_gridspec(
        1,
        4,
        wspace=0.05,
        width_ratios=[1, 1, 1, 0.08],
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]
    cax = fig.add_subplot(gs[0, 3])
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
            color=theme.STROKE_COLOR,
            linewidth=0.38,
            density=1.25,
            arrowsize=0.62,
            zorder=2,
        )
        decorate_phase_space_axis(
            ax,
            xlim=(-grid_lim, grid_lim),
            ylim=(-grid_lim, grid_lim),
            title=title,
            xlabel=r"$z_1$" if panel_idx == 0 else "",
            ylabel=r"$z_2$" if panel_idx == 0 else "",
            grid_alpha=0.20,
        )
        ax.title.set_fontsize(panel_title_size)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.set_xticks([-6, 0, 6])
        ax.set_yticks([-6, 0, 6])
        ax.tick_params(labelbottom=False, labelleft=False)

    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\log(1 + \|f(z)\|)$")
    cbar.ax.yaxis.label.set_fontsize(label_size)
    cbar.ax.tick_params(labelright=False, labelleft=False, length=0)
    cbar.outline.set_linewidth(0.45)
    fig.canvas.draw()
    panel_pos = axes[-1].get_position()
    cbar_pos = cax.get_position()
    cax.set_position([cbar_pos.x0, panel_pos.y0, cbar_pos.width, panel_pos.height])
    figure_path = save_figure(
        fig,
        output_path,
        plt_module=plt_module,
    )
    return artifacts.copy_artifact(figure_path, figure_paths)
