"""Figure-level helpers kept for backward compatibility.

Nothing in the repo calls these today (except ``create_subplot`` via
``actdyn.utils.validation``); they predate the primitive/builder split and are
candidates for deletion once confirmed unused downstream.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from actdyn.utils.torch_utils import to_np
from actdyn.visualize.vectorfield import create_grid, plot_vector_field


@torch.no_grad()
def compute_fisher_map(
    fisher,
    x_range=2.5,
    n_grid=50,
    show_plot=False,
    ax=None,
    device="cpu",
):
    """Create a Fisher information map by computing FIM on sampled points in the grid."""
    if ax is not None:
        plt.sca(ax)
    else:
        plt.figure(figsize=(10, 8))

    xy, X, Y = create_grid(x_range=x_range, n_grid=n_grid, device=device)
    xy = xy.to(device)

    grid_dict = {"model_state": xy.unsqueeze(1)}
    fisher_map = fisher.compute(grid_dict)
    fisher_map = fisher_map.reshape(len(X), len(Y))

    if show_plot:
        plt.contourf(X.cpu(), Y.cpu(), fisher_map.cpu(), levels=10, cmap="plasma")
        plt.colorbar(label="Fisher Information")
        plt.title("Fisher Information Map")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.tight_layout()

    return fisher_map, X.cpu(), Y.cpu()


def plot_per_dimension(x, ax=None, title=None, **kwargs):
    """Plot each dimension of a 2D tensor x over time."""
    fig, axs = create_subplot(x)

    for i in range(x.shape[-1]):
        axs[i].plot(to_np(x[:, i]), **kwargs)
        axs[i].set_title(f"Dimension {i+1}")
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel("Value")
        axs[i].grid(True)

    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()


def create_subplot(x):
    """Create a grid of subplots based on the dimension of x."""
    d = x.shape[-1]
    if d % 2 == 0:
        if d % 3 == 0:
            n_cols = 3
        else:
            n_cols = 2
    else:
        n_cols = min(3, d)
    n_rows = (d + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten() if d > 1 else [axs]

    return fig, axs


def plot_embedding_error_comparison(
    unknown_results,
    known_results,
    methods=("active (k=5)", "step", "active chunk(k=20)"),
    max_steps=500,
    ax=None,
):
    """Plot mean/std embedding error for unknown vs known observation settings."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    colorset = sns.color_palette("Set1", n_colors=max(len(methods), 1))
    color_idx = 0

    for method in methods:
        if method not in unknown_results or method not in known_results:
            continue

        unknown = np.asarray(unknown_results[method])
        known = np.asarray(known_results[method])
        if unknown.size == 0 or known.size == 0:
            continue

        unknown_mean = unknown.mean(axis=0)
        unknown_std = unknown.std(axis=0)
        known_mean = known.mean(axis=0)
        known_std = known.std(axis=0)

        color = colorset[color_idx % len(colorset)]
        color_idx += 1

        ax.plot(unknown_mean, label=f"{method} (unknown obs.)", linestyle="--", color=color)
        ax.fill_between(
            np.arange(len(unknown_mean)),
            unknown_mean - unknown_std,
            unknown_mean + unknown_std,
            alpha=0.1,
            color=color,
        )

        ax.plot(known_mean, label=f"{method} (known obs.)", linestyle="-", color=color)
        ax.fill_between(
            np.arange(len(known_mean)),
            known_mean - known_std,
            known_mean + known_std,
            alpha=0.1,
            color=color,
        )

    if max_steps is not None:
        ax.set_xlim(0, max_steps)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Embedding Error Norm")
    ax.set_title("Embedding Error Norm over Environment Steps")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_current_state(
    env,
    model,
    delta_f=None,
    x=None,
    z=None,
    title=None,
):
    def plot_trajectory(x, ax):
        num_bold = min(20, x.shape[1] // 10)
        ax.plot(
            x[0, :-num_bold, 0],
            x[0, :-num_bold, 1],
            color="red",
            alpha=0.5,
            lw=1,
        )
        ax.plot(
            x[0, -num_bold:, 0],
            x[0, -num_bold:, 1],
            color="red",
            alpha=0.7,
            marker=".",
            lw=1,
        )

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs = axs.flatten()

    plot_vector_field(env.dynamics, ax=axs[0], x_range=5)
    axs[0].set_xlim(-5, 5)
    axs[0].set_ylim(-5, 5)
    axs[0].set_title("True Vector Field")
    plot_trajectory(x, axs[0])

    plot_vector_field(model.dynamics, ax=axs[1], x_range=5)
    axs[1].set_xlim(-5, 5)
    axs[1].set_ylim(-5, 5)
    axs[1].set_title("Learned Vector Field")
    plot_trajectory(z, axs[1])

    axs[2].plot(
        delta_f,
        color="red",
    )
    axs[2].set_title(r"norm($f - \hat{f}$) over time")

    if title is not None:
        fig.suptitle(title)

    return fig, axs
