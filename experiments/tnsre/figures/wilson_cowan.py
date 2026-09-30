"""Wilson-Cowan benchmark figure of the main text (Fig. 3): latent dynamics and observation model.

Left block, the latent dynamics:

* (A) circuit: excitatory and inhibitory populations, the four unknown weights, the two
  input channels, and the Poisson readout.
* (B) phase portrait in the latent coordinates z = 8([E, I] - 1/4): streamlines, nullclines,
  stable states and saddle, the saddle's stable manifold (separatrix) with the down-state
  basin shaded, the region of R2 start states, the region that holds 90% of passive
  (no-input) activity, and the example trajectory of (F).
* (C) per-weight parameter sensitivity ||dv/dw|| on one color scale, with the passive region
  and separatrix from (B).

Right block, one column per observation condition of the degraded-observation study
(uniform loading at -5 dB "Default", uniform at -10 dB, biased at -5 dB):

* (D) loading directions (rows of C), one compass per condition on one scale.
* (E) det of the local state Fisher information, one log color scale, with the same separatrix.
* (F) spike trains of the 20 neurons (sorted by loading angle) along the trajectory in
  (B); shading marks the input pulse that switches it from the down to the up state.

All quantities come from the environment presets; no experiment results are read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from actdyn.utils.figure_io import load_plotting

from .assets import (
    _ASSET_LABEL_SIZE,
    _ASSET_PANEL_LABEL_SIZE,
    _ASSET_TICK_SIZE,
    _apply_asset_style,
    save_figure,
)
from .spiking_sessions import POOL_COLORS
from .theme import NEUTRAL_FILL, STROKE_COLOR, style_experiment_axis

ENV_DEFAULT = "tbme_wilson_cowan"  # uniform loading, -5 dB
ENV_LOW_SNR = "tbme_wilson_cowan_observation_bottleneck_mild"  # uniform loading, -10 dB
ENV_BIASED = "tbme_wilson_cowan_asymmetric"  # biased loading, -5 dB
CONDITIONS = ((ENV_DEFAULT, "Default"), (ENV_LOW_SNR, "SNR -10"), (ENV_BIASED, "Biased"))
E_COLOR, I_COLOR = POOL_COLORS[1], POOL_COLORS[0]
WEIGHT_LABELS = (r"$w_{EE}$", r"$w_{EI}$", r"$w_{IE}$", r"$w_{II}$")
MAP_CMAP = "magma"
PULSE_SHADE = "#C6C0B8"
MAP_LIM = 3.0  # shared B/C/E domain includes the full path; z = -2 and 2 are rates 0 and 1/2
FIGURE_WIDTH = 516.0 / 72.27
FIGURE_HEIGHT = 3.1
# Layout in inches. Left block: A and B on top, the C strip below. Right block: a 3 x 3 matrix
# with one column per observation condition and rows D (loadings), E (Fisher), F (spikes).
LEFT = {"top": 2.89, "a_left": 0.15, "a_width": 1.5, "b_left": 2.12, "b_size": 1.4,
        "c_bottom": 0.42, "c_size": 0.7}
RIGHT = {"x0": 4.10, "col": 0.88, "gap": 0.06, "d": (2.42, 0.50), "e": (1.36, 0.88),
         "f_raster": (0.28, 0.60)}
# Example trajectory of (B) and (F): start in the down state; (start, stop, u_E, u_I) pulses.
# From the down state this pulse reaches the up state in 39 of 40 noise seeds (Wilson-Cowan 1972
# parameters, checked 2026-09-29; seed 0 is one of the 39).
# 600 steps keep single spikes visible in (F); longer windows merge into solid bands.
EXAMPLE = {
    "steps": 600,
    "seed": 0,
    "pulses": ((150, 250, 2.0, -2.0),),
}
PASSIVE = {"n_trajectories": 50, "steps": 3000, "mass": 0.9}
GRID = {"field": 41, "map": 121, "basin": 161}  # retain 0.05 spacing on the wider maps
LOADING_SNR = {"snr_trajectories": 100, "snr_trajectory_length": 200}


# ------------------------------------------------------------------ model helpers
def _preset(env_id: str) -> Any:
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    configure_tbme_catalogs(suite_entries={})
    return get_environment_preset(env_id)


class _Model:
    """True Wilson-Cowan drift v(z) and one environment step z + dt (v(z) + u), batched over rows."""

    def __init__(self, preset: Any) -> None:
        self.preset = preset
        self.kind = preset.resolved_dynamics_type()
        self.theta = np.asarray(preset.resolved_true_params(), dtype=np.float64)
        self.dt = float(preset.dt)
        self.q = float(preset.state_noise)
        self.clip = float(preset.resolved_plot_limit())

    def drift(self, z: np.ndarray) -> np.ndarray:
        from actdyn.environment.vectorfield import residual_np

        z = np.asarray(z, dtype=np.float64)
        return residual_np(self.kind, z.reshape(-1, 2), self.theta, dynamics_alpha=1.0).reshape(z.shape)

    def step(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.clip(z + self.dt * (self.drift(z) + u), -self.clip, self.clip)

    def jacobian(self, z: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """d v / d z at one state ``z`` (2,), central differences."""
        jac = np.empty((2, 2))
        for i in range(2):
            e = np.zeros(2)
            e[i] = h
            jac[:, i] = (self.drift(z + e) - self.drift(z - e)) / (2.0 * h)
        return jac


def _fixed_points(model: _Model) -> list[tuple[np.ndarray, bool]]:
    """Fixed points inside the map, each with ``stable`` (all eigenvalues of dv/dz negative)."""
    from scipy.optimize import fsolve

    found: list[np.ndarray] = []
    for x0 in np.linspace(-2.2, 2.2, 9):
        for y0 in np.linspace(-2.2, 2.2, 9):
            z = fsolve(lambda s: model.drift(s), [x0, y0], xtol=1e-12)
            if np.max(np.abs(z)) < MAP_LIM and np.linalg.norm(model.drift(z)) < 1e-5:  # drift is float32 inside
                if all(np.linalg.norm(z - f) > 1e-3 for f in found):
                    found.append(z)
    return [(z, bool(np.all(np.linalg.eigvals(model.jacobian(z)).real < 0.0))) for z in found]


def _separatrix(model: _Model, saddle: np.ndarray, *, limit: float = MAP_LIM) -> list[np.ndarray]:
    """Both branches of the saddle's stable manifold, integrated backward in time (RK4)."""
    evals, evecs = np.linalg.eig(model.jacobian(saddle))
    direction = np.real(evecs[:, int(np.argmin(evals.real))])
    branches = []
    for sign in (1.0, -1.0):
        z = saddle + sign * 1e-4 * direction
        path = [z]
        for _ in range(20000):
            k1 = -model.drift(z)
            k2 = -model.drift(z + 0.0025 * k1)
            k3 = -model.drift(z + 0.0025 * k2)
            k4 = -model.drift(z + 0.005 * k3)
            z = z + 0.005 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            path.append(z)
            if np.max(np.abs(z)) > limit + 0.2:
                break
        branches.append(np.array(path))
    return branches


def _down_basin(model: _Model, *, limit: float = MAP_LIM) -> tuple[np.ndarray, np.ndarray]:
    """Grid axis and mask (n, n) of states whose noise-free trajectory ends in the down state."""
    axis = np.linspace(-limit, limit, GRID["basin"])
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    z = np.stack([xx.ravel(), yy.ravel()], axis=1)
    for _ in range(2000):
        z = model.step(z, np.zeros_like(z))
    return axis, (z[:, 0] < 0.0).reshape(xx.shape)


def _simulate(model: _Model, z0: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy trajectory (T, 2) from ``z0`` under inputs ``u`` (T, 2); ``u[t]`` acts on the step to t + 1."""
    noise = np.sqrt(model.q * model.dt)
    z = np.empty_like(u)
    z[0] = z0
    for t in range(1, u.shape[0]):
        z[t] = np.clip(model.step(z[t - 1], u[t - 1]) + rng.normal(scale=noise, size=2), -model.clip, model.clip)
    return z


def _example(model: _Model, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.zeros((int(EXAMPLE["steps"]), 2))
    for start, stop, u_e, u_i in EXAMPLE["pulses"]:
        u[start:stop] = (u_e, u_i)
    return _simulate(model, down, u, np.random.default_rng(int(EXAMPLE["seed"]))), u


def _passive_density(model: _Model) -> tuple[np.ndarray, np.ndarray, float]:
    """Smoothed histogram of passive states and the level that encloses ``PASSIVE['mass']`` of them.

    Starts are the preset's initial-state distribution (uniform on [-2, 2]^2); no input.
    """
    from scipy.ndimage import gaussian_filter

    z = np.stack([model.preset.sample_initial_state(s) for s in range(int(PASSIVE["n_trajectories"]))])
    z = z.astype(np.float64)
    rng = np.random.default_rng(0)
    noise = np.sqrt(model.q * model.dt)
    samples = []
    for _ in range(int(PASSIVE["steps"])):
        z = np.clip(model.step(z, np.zeros_like(z)) + rng.normal(scale=noise, size=z.shape), -model.clip, model.clip)
        samples.append(z.copy())
    samples = np.concatenate(samples)
    edges = np.linspace(-MAP_LIM, MAP_LIM, GRID["map"])
    hist, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[edges, edges])
    density = gaussian_filter(hist.T, sigma=1.5)
    ordered = np.sort(density.ravel())[::-1]
    level = float(ordered[np.searchsorted(np.cumsum(ordered) / ordered.sum(), float(PASSIVE["mass"]))])
    return 0.5 * (edges[:-1] + edges[1:]), density, level


def _weight_sensitivity(preset: Any) -> np.ndarray:
    """||dv/dw_j|| for each weight j on the map grid, shape (4, n, n)."""
    import torch

    from actdyn.environment.vectorfield import jacobian_embedding_torch

    axis = np.linspace(-MAP_LIM, MAP_LIM, GRID["map"], dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    states = torch.as_tensor(np.stack([xx.ravel(), yy.ravel()], axis=1))
    theta = torch.as_tensor(preset.true_embedding_vector(), dtype=torch.float32)
    jac = jacobian_embedding_torch(
        preset.resolved_dynamics_type(), states, theta.reshape(1, -1).expand(states.shape[0], -1),
        full_params=preset.resolved_true_params(), min_embedding_dim=preset.resolved_min_embedding_dim(),
        dynamics_alpha=float(preset.dynamics_alpha),
    ).detach().numpy()
    return np.linalg.norm(jac, axis=1).T.reshape(4, *xx.shape)


def _loading(env_id: str) -> tuple[np.ndarray, np.ndarray, float]:
    from .diagnostics import loading_model

    return loading_model(_preset(env_id), **LOADING_SNR)


def _fisher_det(weights: np.ndarray, bias: np.ndarray, dt: float) -> np.ndarray:
    from .diagnostics import state_information_grid

    _x, _y, logdet = state_information_grid(weights, bias, dt=dt, plot_lim=MAP_LIM, n_grid=GRID["map"])
    return np.exp(logdet)


# ------------------------------------------------------------------ drawing helpers
def _inch_axes(fig: Any, x: float, y: float, w: float, h: float) -> Any:
    return fig.add_axes([x / FIGURE_WIDTH, y / FIGURE_HEIGHT, w / FIGURE_WIDTH, h / FIGURE_HEIGHT])


def _map_axis(ax: Any, *, xlabels: bool = True, ylabels: bool = True, limit: float = MAP_LIM) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xticks([-limit, 0, limit])
    ax.set_yticks([-limit, 0, limit])
    ax.tick_params(labelsize=_ASSET_TICK_SIZE, length=2.0, width=0.4, pad=1.5,
                   labelbottom=xlabels, labelleft=ylabels)
    ticks = ax.xaxis.get_major_ticks()
    ticks[0].label1.set_ha("left")
    ticks[-1].label1.set_ha("right")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _mark_fixed_points(ax: Any, fixed: list[tuple[np.ndarray, bool]], *, size: float = 3.2,
                       edge: str = STROKE_COLOR) -> None:
    for z, stable in fixed:
        ax.plot(z[0], z[1], marker="o", ms=size, ls="none", zorder=6, markeredgewidth=0.6,
                markerfacecolor=STROKE_COLOR if stable else "white", markeredgecolor=edge if stable else STROKE_COLOR)


def _map_label(ax: Any, text: str) -> None:
    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=_ASSET_TICK_SIZE,
            color="white", zorder=7)


def _map_separatrix(ax: Any, branches: list[np.ndarray]) -> None:
    """Overlay the phase portrait's stable manifold with contrast on dark and light maps."""
    from matplotlib import patheffects

    for branch in branches:
        ax.plot(branch[:, 0], branch[:, 1], color=STROKE_COLOR, lw=0.65, ls="--", zorder=5,
                gid="separatrix", path_effects=[patheffects.Stroke(linewidth=1.35, foreground="white"),
                                               patheffects.Normal()])


def _colorbar(fig: Any, image: Any, cax: Any, label: str, *, log: bool = False) -> Any:
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=_ASSET_TICK_SIZE, width=0.4, length=2.0, pad=1.5)
    cbar.outline.set_linewidth(0.4)
    if log:
        from matplotlib.ticker import LogLocator

        cbar.locator = LogLocator(base=10, numticks=4)
        cbar.update_ticks()
        cbar.minorticks_off()
    cbar.set_label(label, fontsize=_ASSET_LABEL_SIZE, labelpad=2.0)
    return cbar


def _draw_circuit(ax: Any) -> None:
    """E and I populations with the four weights, inputs u_E and u_I, and the Poisson readout."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    pops = {"E": (2.6, 7.2, E_COLOR), "I": (7.4, 7.2, I_COLOR)}
    for name, (x, y, color) in pops.items():
        ax.add_patch(Circle((x, y), 1.3, facecolor=color, alpha=0.18, edgecolor=color, lw=1.0))
        ax.text(x, y, name, ha="center", va="center", fontsize=_ASSET_LABEL_SIZE, color=STROKE_COLOR)
        ax.annotate("", xy=(x, y + 1.35), xytext=(x, y + 3.4),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=STROKE_COLOR, mutation_scale=7))
        ax.text(x, y + 3.5, f"$u_{name}$", ha="center", va="bottom", fontsize=_ASSET_LABEL_SIZE)
    # Self connections: w_EE excites E, w_II inhibits I.
    ax.add_patch(FancyArrowPatch((1.45, 7.9), (1.45, 6.5), connectionstyle="arc3,rad=1.5", arrowstyle="-|>",
                                 mutation_scale=6, lw=0.8, color=E_COLOR))
    ax.text(0.7, 9.0, WEIGHT_LABELS[0], ha="center", va="bottom", fontsize=_ASSET_TICK_SIZE)
    ax.add_patch(FancyArrowPatch((8.55, 6.5), (8.55, 7.9), connectionstyle="arc3,rad=1.5", arrowstyle="-[",
                                 mutation_scale=3, lw=0.8, color=I_COLOR))
    ax.text(9.3, 9.0, WEIGHT_LABELS[3], ha="center", va="bottom", fontsize=_ASSET_TICK_SIZE)
    # E -> I excitation above, I -> E inhibition below.
    ax.add_patch(FancyArrowPatch((3.7, 8.0), (6.3, 8.0), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=6, lw=0.8, color=E_COLOR))
    ax.text(5.0, 9.0, WEIGHT_LABELS[2], ha="center", va="bottom", fontsize=_ASSET_TICK_SIZE)
    ax.add_patch(FancyArrowPatch((6.3, 6.4), (3.7, 6.4), connectionstyle="arc3,rad=-0.35", arrowstyle="-[",
                                 mutation_scale=3, lw=0.8, color=I_COLOR))
    ax.text(5.0, 5.4, WEIGHT_LABELS[1], ha="center", va="top", fontsize=_ASSET_TICK_SIZE)
    # Readout: both populations drive 20 Poisson neurons through the loadings C.
    ax.add_patch(FancyBboxPatch((0.8, 0.6), 8.4, 1.6, boxstyle="round,pad=0.1,rounding_size=0.4",
                                facecolor=NEUTRAL_FILL, edgecolor=STROKE_COLOR, lw=0.6))
    ax.text(5.0, 1.4, "20 Poisson neurons", ha="center", va="center", fontsize=_ASSET_TICK_SIZE)
    for x in (2.6, 7.4):
        ax.annotate("", xy=(x + (5.0 - x) * 0.35, 2.3), xytext=(x, 5.85),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=STROKE_COLOR, mutation_scale=6))
    ax.text(5.0, 3.6, r"$\mathbf{C}$", ha="center", va="center", fontsize=_ASSET_TICK_SIZE)


# ------------------------------------------------------------------ figure
def _passive_contour(ax: Any, centers: np.ndarray, density: np.ndarray, level: float, color: str) -> None:
    ax.contour(centers, centers, density, levels=[level], colors=[color], linewidths=0.5, linestyles="--")


def _letter(fig: Any, letter: str, x: float, y: float) -> None:
    fig.text(x / FIGURE_WIDTH, y / FIGURE_HEIGHT, letter, ha="left", va="bottom", fontsize=_ASSET_PANEL_LABEL_SIZE,
             fontweight="bold")


def _title(fig: Any, text: str, left: float, right: float, y: float) -> None:
    fig.text(0.5 * (left + right) / FIGURE_WIDTH, y / FIGURE_HEIGHT, text, ha="center", va="bottom",
             fontsize=_ASSET_LABEL_SIZE)


def generate_wilson_cowan_figure(output: Path) -> Path:
    model = _Model(_preset(ENV_DEFAULT))
    fixed = _fixed_points(model)
    saddle = next(z for z, stable in fixed if not stable)
    separatrix = _separatrix(model, saddle, limit=MAP_LIM)
    down = min((z for z, stable in fixed if stable), key=lambda z: z[0])
    up = max((z for z, stable in fixed if stable), key=lambda z: z[0])
    traj, _u = _example(model, down)
    centers, density, level = _passive_density(model)
    loadings = {env: _loading(env) for env, _label in CONDITIONS}
    preset = model.preset

    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    top, b_size = LEFT["top"], LEFT["b_size"]

    # (A) circuit.
    ax = _inch_axes(fig, LEFT["a_left"], top - 1.45, LEFT["a_width"], 1.5)
    _draw_circuit(ax)
    _letter(fig, "A", 0.02, top + 0.05)
    _title(fig, "Circuit", LEFT["a_left"], LEFT["a_left"] + LEFT["a_width"], top + 0.05)

    # (B) phase portrait.
    b_left = LEFT["b_left"]
    ax = _inch_axes(fig, b_left, top - b_size, b_size, b_size)
    axis, basin = _down_basin(model, limit=MAP_LIM)
    ax.contourf(axis, axis, basin.astype(float), levels=[0.5, 1.5], colors=[NEUTRAL_FILL], zorder=0)
    field_axis = np.linspace(-MAP_LIM, MAP_LIM, GRID["field"])
    fx, fy = np.meshgrid(field_axis, field_axis, indexing="xy")
    v = model.drift(np.stack([fx.ravel(), fy.ravel()], axis=1)).reshape(*fx.shape, 2)
    ax.streamplot(field_axis, field_axis, v[..., 0], v[..., 1], color="#B3ADA6", linewidth=0.35, density=0.8,
                  arrowsize=0.45, zorder=1)
    ax.contour(field_axis, field_axis, v[..., 0], levels=[0.0], colors=[E_COLOR], linewidths=0.9, zorder=2)
    ax.contour(field_axis, field_axis, v[..., 1], levels=[0.0], colors=[I_COLOR], linewidths=0.9, zorder=2)
    for branch in separatrix:
        ax.plot(branch[:, 0], branch[:, 1], color=STROKE_COLOR, lw=0.7, ls="--", zorder=3)
    low, high = np.asarray(preset.trajectory_eval_state_low), np.asarray(preset.trajectory_eval_state_high)
    ax.plot([low[0], high[0], high[0], low[0], low[0]], [low[1], low[1], high[1], high[1], low[1]],
            color=STROKE_COLOR, lw=0.5, ls=":", zorder=3)
    _passive_contour(ax, centers, density, level, "#8A847C")
    # A white outline separates the measured path from the nullclines and flow.
    from matplotlib import patheffects

    ax.plot(traj[:, 0], traj[:, 1], color="#181818", lw=1.1, zorder=4,
            path_effects=[patheffects.Stroke(linewidth=2.1, foreground="white"), patheffects.Normal()])
    for step in (210, 330):
        ax.annotate("", xy=traj[step + 12], xytext=traj[step], zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="#181818", lw=0.9, mutation_scale=6))
    _mark_fixed_points(ax, fixed, size=3.6, edge="white")
    for z, text, dx, dy, ha in ((down, "down", 0.2, -0.32, "left"), (saddle, "saddle", 0.15, -0.38, "left"),
                                (up, "up", -0.2, -0.3, "right")):
        ax.text(z[0] + dx, z[1] + dy, text, ha=ha, va="center", fontsize=_ASSET_TICK_SIZE, zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))
    _map_axis(ax)
    ax.set_xlabel(r"$z_1$ (E)", fontsize=_ASSET_LABEL_SIZE, labelpad=1.0)
    ax.set_ylabel(r"$z_2$ (I)", fontsize=_ASSET_LABEL_SIZE, labelpad=0.5)
    _letter(fig, "B", b_left - 0.4, top + 0.05)
    _title(fig, "Phase portrait", b_left, b_left + b_size, top + 0.05)

    # (C) per-weight sensitivity, one strip.
    sens = _weight_sensitivity(preset)
    vmax = float(np.percentile(sens, 99.5))
    c_left, c_size, c_gap, c_bottom = 0.4, LEFT["c_size"], 0.06, LEFT["c_bottom"]
    c_right = c_left + 4 * c_size + 3 * c_gap
    map_axes = []
    for j in range(4):
        ax = _inch_axes(fig, c_left + (c_size + c_gap) * j, c_bottom, c_size, c_size)
        if map_axes:
            ax.sharex(map_axes[0])
            ax.sharey(map_axes[0])
        map_axes.append(ax)
        image = ax.imshow(sens[j], origin="lower", extent=[-MAP_LIM, MAP_LIM, -MAP_LIM, MAP_LIM], cmap=MAP_CMAP,
                          vmin=0.0, vmax=vmax, interpolation="bilinear")
        _passive_contour(ax, centers, density, level, "white")
        _map_separatrix(ax, separatrix)
        _mark_fixed_points(ax, fixed, size=2.4, edge="white")
        _map_axis(ax, ylabels=j == 0)
        _map_label(ax, WEIGHT_LABELS[j])
        if j == 0:
            ax.set_ylabel(r"$z_2$ (I)", fontsize=_ASSET_LABEL_SIZE, labelpad=0.5)
            ax.set_xlabel(r"$z_1$ (E)", fontsize=_ASSET_LABEL_SIZE)
            ax.xaxis.set_label_coords(0.5 * (c_left + c_right) / FIGURE_WIDTH, 0.03 / FIGURE_HEIGHT,
                                     transform=fig.transFigure)
            ax.xaxis.label.set(ha="center", va="bottom")
    _colorbar(fig, image, _inch_axes(fig, c_right + 0.06, c_bottom, 0.045, c_size),
              r"$\|\partial\mathbf{v}/\partial w\|$")
    _letter(fig, "C", 0.02, c_bottom + c_size + 0.05)
    _title(fig, "Parameter sensitivity", c_left, c_right, c_bottom + c_size + 0.05)

    # Right block: one column per observation condition.
    cols = [RIGHT["x0"] + (RIGHT["col"] + RIGHT["gap"]) * k for k in range(len(CONDITIONS))]
    r_left, r_right = cols[0], cols[-1] + RIGHT["col"]
    letter_x = r_left - 0.42

    # (D) loading directions, one compass per condition on one scale.
    d_bottom, d_height = RIGHT["d"]
    reach = max(float(np.max(np.linalg.norm(loadings[env][0], axis=1))) for env, _l in CONDITIONS)
    for k, (env, label) in enumerate(CONDITIONS):
        ax = _inch_axes(fig, cols[k], d_bottom, RIGHT["col"], d_height)
        ax.add_patch(plt.Circle((0, 0), reach, fill=False, lw=0.4, color="#D5D0C9"))
        ax.plot([-1.08 * reach, 1.08 * reach], [0, 0], color="#D5D0C9", lw=0.4)
        ax.plot([0, 0], [-1.08 * reach, 1.08 * reach], color="#D5D0C9", lw=0.4)
        for w in loadings[env][0]:
            ax.plot([0, w[0]], [0, w[1]], color="#4A4A4A", lw=0.7)
        ax.set_xlim(-1.25 * reach, 1.25 * reach)
        ax.set_ylim(-1.1 * reach, 1.1 * reach)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(1.1 * reach, 0.0, r"$z_1$", ha="left", va="center", fontsize=_ASSET_TICK_SIZE)
        ax.text(0.12 * reach, 0.72 * reach, r"$z_2$", ha="left", va="bottom", fontsize=_ASSET_TICK_SIZE)
        _title(fig, label, cols[k], cols[k] + RIGHT["col"], top + 0.05)
    fig.text((r_left - 0.2) / FIGURE_WIDTH, (d_bottom + 0.5 * d_height) / FIGURE_HEIGHT, "loadings",
             ha="center", va="center", rotation=90, fontsize=_ASSET_LABEL_SIZE)
    _letter(fig, "D", letter_x, top + 0.05)

    # (E) state Fisher information.
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    e_bottom, e_size = RIGHT["e"]
    dets = [_fisher_det(*loadings[env]) for env, _label in CONDITIONS]
    pooled = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in dets])
    norm = LogNorm(vmin=float(np.percentile(pooled, 1.0)), vmax=float(np.percentile(pooled, 99.0)))
    # Keep one logarithmic scale; omit magma's near-black end so low information
    # and the contours remain visible. This does not rescale individual panels.
    fisher_cmap = LinearSegmentedColormap.from_list(
        "magma_visible_low", plt.get_cmap(MAP_CMAP)(np.linspace(0.18, 1.0, 256)))
    for k, det in enumerate(dets):
        ax = _inch_axes(fig, cols[k], e_bottom, e_size, e_size)
        ax.sharex(map_axes[0])
        ax.sharey(map_axes[0])
        image = ax.imshow(det, origin="lower", extent=[-MAP_LIM, MAP_LIM, -MAP_LIM, MAP_LIM], cmap=fisher_cmap,
                          norm=norm, interpolation="nearest")
        _passive_contour(ax, centers, density, level, "white")
        _map_separatrix(ax, separatrix)
        _mark_fixed_points(ax, fixed, size=2.8, edge="white")
        _map_axis(ax, ylabels=k == 0)
        ax.tick_params(pad=0.8)
        if k == 0:
            ax.set_ylabel(r"$z_2$ (I)", fontsize=_ASSET_LABEL_SIZE, labelpad=0.5)
    cax = _inch_axes(fig, r_right + 0.02, e_bottom, 0.03, e_size)
    cbar = _colorbar(fig, image, cax, "", log=True)
    # Compact decimal tick labels preserve the logarithmic scale while leaving
    # more width for the three aligned observation columns.
    cbar.set_ticks([0.01, 0.1, 1.0], labels=["0.01", "0.1", "1"])
    cax.tick_params(axis="y", length=1.5, pad=0.5)
    # Label the colorbar itself, using the shared row title to avoid a second
    # vertical label at the page edge.
    cax.set_ylabel(r"State Fisher information $\det\mathbf{I}_z$", rotation=0)
    cax.yaxis.set_label_coords(0.5 * (r_left + r_right) / FIGURE_WIDTH,
                             (e_bottom + e_size + 0.04) / FIGURE_HEIGHT, transform=fig.transFigure)
    cax.yaxis.label.set(ha="center", va="bottom")
    _letter(fig, "E", letter_x, e_bottom + e_size + 0.04)

    # (F) spike trains along the shared trajectory drawn prominently in (B).
    fr_bottom, fr_height = RIGHT["f_raster"]
    raster_axes = []
    for k, (env, _label) in enumerate(CONDITIONS):
        weights, bias, dt = loadings[env]
        rng = np.random.default_rng(int(EXAMPLE["seed"]) + 1 + k)
        counts = rng.poisson(np.clip(dt * np.exp(traj @ weights.T + bias), 0.0, 1e6))
        order = np.argsort(np.arctan2(weights[:, 1], weights[:, 0]))
        ax_r = _inch_axes(fig, cols[k], fr_bottom, RIGHT["col"], fr_height)
        if raster_axes:
            ax_r.sharex(raster_axes[0])
            ax_r.sharey(raster_axes[0])
        raster_axes.append(ax_r)
        steps, neurons = np.nonzero(counts[:, order] > 0)
        ax_r.vlines(steps, neurons - 0.28, neurons + 0.28, color=STROKE_COLOR, lw=0.18, rasterized=True)
        ax_r.set_ylim(-0.5, weights.shape[0] - 0.5)
        ax_r.set_yticks([0, weights.shape[0] - 1], labels=["1", str(weights.shape[0])])
        for start, stop, _ue, _ui in EXAMPLE["pulses"]:
            ax_r.axvspan(start, stop, color=PULSE_SHADE, lw=0, zorder=0)
            ax_r.plot([start, stop], [1.025, 1.025], color=STROKE_COLOR, lw=2,
                      transform=ax_r.get_xaxis_transform(), clip_on=False)
            ax_r.text(0.5 * (start + stop), 1.055, "pulse", ha="center", va="bottom",
                      transform=ax_r.get_xaxis_transform(), fontsize=_ASSET_TICK_SIZE)
        style_experiment_axis(ax_r)
        ax_r.tick_params(pad=1.0)
        ax_r.set_xlim(0, traj.shape[0])
        ax_r.set_xticks([0, 300, 600])
        # Edge labels point inward so neighbouring columns' "600" and "0" do not meet.
        ax_r.get_xticklabels()[0].set_ha("left")
        ax_r.get_xticklabels()[-1].set_ha("right")
        if k == 0:
            ax_r.set_ylabel("neuron", fontsize=_ASSET_LABEL_SIZE, labelpad=1.0)
            ax_r.set_xlabel("Environment steps", fontsize=_ASSET_LABEL_SIZE)
            ax_r.xaxis.set_label_coords(0.5 * (r_left + r_right) / FIGURE_WIDTH, 0.03 / FIGURE_HEIGHT,
                                       transform=fig.transFigure)
            ax_r.xaxis.label.set(ha="center", va="bottom")
        else:
            ax_r.tick_params(labelleft=False)
    _letter(fig, "F", letter_x, fr_bottom + fr_height + 0.16)
    _title(fig, "Spike trains", r_left, r_right, fr_bottom + fr_height + 0.16)
    return save_figure(fig, output, plt_module=plt)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Wilson-Cowan benchmark figure (manuscript Fig. 3).")
    parser.add_argument("--output", type=Path, required=True, help="output file (.pdf, .png, or .svg)")
    args = parser.parse_args(argv)
    print(generate_wilson_cowan_figure(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
