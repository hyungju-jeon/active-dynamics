"""Trajectory drawing primitives (phase-space overlays, gradients, annotations)."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from actdyn.utils.torch_utils import to_np


def trace_index(trace_steps: np.ndarray, step: int) -> int:
    if trace_steps.size == 0:
        return 0
    idx = int(np.searchsorted(trace_steps, step, side="right") - 1)
    return int(np.clip(idx, 0, len(trace_steps) - 1))


def create_gradient_line(
    ax,
    data,
    base_color,
    label=None,
    alpha_start=0.2,
    alpha_end=0.95,
    linewidth=1.5,
):
    """Plot a 2D trajectory with a fading alpha gradient."""
    if isinstance(data, torch.Tensor):
        data = to_np(data)
    data = np.asarray(data)

    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[1] == 1:
            data = data[:, 0, :]
        else:
            data = data.reshape(-1, data.shape[-1])
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return None

    points = data[:, :2]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    alphas = np.linspace(alpha_start, alpha_end, len(segments))
    colors = [to_rgba(base_color, alpha) for alpha in alphas]
    line = LineCollection(segments, colors=colors, linewidths=linewidth, zorder=3)
    ax.add_collection(line)
    if label is not None:
        ax.plot([], [], color=base_color, linewidth=linewidth, label=label)
    return line


def annotate_action_arrow(
    ax,
    *,
    origin,
    action,
    max_display_len: float = 2.5,
    scale: float = 0.45,
    color: str = "white",
    width: float = 0.03,
    head_width: float = 0.28,
    zorder: int = 7,
) -> float:
    origin_xy = np.asarray(origin, dtype=float).reshape(-1)
    action_xy = np.asarray(action, dtype=float).reshape(-1)
    if origin_xy.size < 2 or action_xy.size < 2 or not np.all(np.isfinite(action_xy[:2])):
        return float("nan")
    action_xy = action_xy[:2]
    act_norm = float(np.linalg.norm(action_xy))
    if act_norm > 1e-12:
        display_len = min(float(max_display_len), float(scale) * act_norm)
        direction = action_xy / act_norm
        ax.arrow(
            float(origin_xy[0]),
            float(origin_xy[1]),
            float(display_len * direction[0]),
            float(display_len * direction[1]),
            color=color,
            width=width,
            head_width=head_width,
            length_includes_head=True,
            alpha=0.95,
            zorder=zorder,
        )
    ax.text(
        0.02,
        0.02,
        f"u=({action_xy[0]:.2f}, {action_xy[1]:.2f})  |u|={act_norm:.2f}",
        transform=ax.transAxes,
        color=color,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="black",
            alpha=0.45,
            edgecolor="none",
        ),
    )
    return act_norm


def plot_rollout_latent_comparison(
    env_state,
    model_state,
    ax=None,
    title="Latent Trajectory Comparison",
    labels=("true", "model"),
):
    """Overlay true and model latent trajectories in 2D."""
    env_xy = to_np(env_state)
    model_xy = to_np(model_state)
    if env_xy.ndim == 3:
        env_xy = env_xy[0]
    if model_xy.ndim == 3:
        model_xy = model_xy[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    ax.plot(env_xy[:, 0], env_xy[:, 1], alpha=0.7, label=labels[0])
    ax.plot(model_xy[:, 0], model_xy[:, 1], alpha=0.7, label=labels[1])
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, ax
