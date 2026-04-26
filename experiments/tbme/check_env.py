# %%
from __future__ import annotations

"""Simple notebook-style check for 2D vector fields."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from actdyn.environment.vectorfield import build_vectorfield


def make_vectorfield(
    dynamics_type: str,
    params=None,
    *,
    dynamics_alpha: float = 1.0,
    device: str = "cpu",
):
    dyn_params = None
    if params is not None:
        dyn_params = torch.as_tensor(params, dtype=torch.float32, device=device)
    return build_vectorfield(
        dynamics_type=str(dynamics_type),
        dyn_params=dyn_params,
        dynamics_alpha=float(dynamics_alpha),
        device=device,
    )


def evaluate_vectorfield(
    vectorfield,
    *,
    xlim: tuple[float, float] = (-4.0, 4.0),
    ylim: tuple[float, float] = (-4.0, 4.0),
    n_grid: int = 49,
    device: str = "cpu",
):
    x_axis = torch.linspace(xlim[0], xlim[1], n_grid, device=device)
    y_axis = torch.linspace(ylim[0], ylim[1], n_grid, device=device)
    xx, yy = torch.meshgrid(x_axis, y_axis, indexing="xy")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    with torch.no_grad():
        flow = vectorfield(points).detach().cpu().numpy()
    u = flow[:, 0].reshape(xx.shape)
    v = flow[:, 1].reshape(yy.shape)
    speed = np.hypot(u, v)
    return xx.cpu().numpy(), yy.cpu().numpy(), u, v, speed


def plot_vectorfield(
    vectorfield,
    *,
    xlim: tuple[float, float] = (-4.0, 4.0),
    ylim: tuple[float, float] = (-4.0, 4.0),
    n_grid: int = 49,
    title: str | None = None,
    device: str = "cpu",
):
    x, y, u, v, speed = evaluate_vectorfield(
        vectorfield,
        xlim=xlim,
        ylim=ylim,
        n_grid=n_grid,
        device=device,
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    stream = ax.streamplot(
        x,
        y,
        u,
        v,
        color=speed,
        cmap="viridis",
        density=1.2,
        linewidth=1.0,
        arrowsize=1.0,
    )
    fig.colorbar(stream.lines, ax=ax, label="speed")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    if title is not None:
        ax.set_title(title)
    return fig, ax


def max_vectorfield_norm(
    vectorfield,
    *,
    xlim: tuple[float, float] = (-4.0, 4.0),
    ylim: tuple[float, float] = (-4.0, 4.0),
    n_grid: int = 49,
    device: str = "cpu",
):
    x, y, _u, _v, speed = evaluate_vectorfield(
        vectorfield,
        xlim=xlim,
        ylim=ylim,
        n_grid=n_grid,
        device=device,
    )
    flat_idx = int(np.argmax(speed))
    ij = np.unravel_index(flat_idx, speed.shape)
    return {
        "max_norm": float(speed[ij]),
        "argmax_xy": (float(x[ij]), float(y[ij])),
        "grid_index": (int(ij[0]), int(ij[1])),
    }


def show_figure(fig):
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        plt.show()


# %%
# %%
# Parameter order:
# - duffing: [a, b, c]
# - damped_pendulum: [damping, gravity]
# - double_integrator: [bias, damping]
# - asymmetric_basin: [a_left, b_left, a_right, b_right]
# - multi_stable: [a_nw, a_ne, a_sw, a_se, w_nw, w_ne, w_sw, w_se]
PARAMETER_EXAMPLES = {
    "duffing": [-0.5, -0.75, 0.1],  # action = 1
    "damped_pendulum": [-0.5, 1],  # action = 1
    "double_integrator": [0.15, 0.55],
    "asymmetric_basin": [-1.2, -0.8, 0.5, 0.1],  # action = 1
    "multi_stable": [1.15, -0.1, -0.2, 1.5, 1.55, -0.2, -0.4, -2.0],  # action = 1
}

DYNAMICS_TYPE = "duffing"
PARAMS = PARAMETER_EXAMPLES[DYNAMICS_TYPE]
DYNAMICS_ALPHA = 1.0
XLIM = (-3.0, 3.0)
YLIM = (-3.0, 3.0)
N_GRID = 49
DEVICE = "cpu"


vectorfield = make_vectorfield(
    DYNAMICS_TYPE,
    PARAMS,
    dynamics_alpha=DYNAMICS_ALPHA,
    device=DEVICE,
)

fig, ax = plot_vectorfield(
    vectorfield,
    xlim=XLIM,
    ylim=YLIM,
    n_grid=N_GRID,
    title=f"{DYNAMICS_TYPE}  params={PARAMS}",
    device=DEVICE,
)

max_info = max_vectorfield_norm(
    vectorfield,
    xlim=XLIM,
    ylim=YLIM,
    n_grid=N_GRID,
    device=DEVICE,
)
ax.scatter(*max_info["argmax_xy"], color="red", s=40, zorder=5, label="max norm")
ax.legend(loc="upper right")
# show_figure(fig)

print(f"Max vectorfield norm on grid: {max_info['max_norm']:.4f}")
print(f"Argmax location: {max_info['argmax_xy']}")
