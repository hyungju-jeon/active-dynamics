"""Neural-observation drawing primitives (spike rasters, observation channels)."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from actdyn.utils.torch_utils import to_np


def plot_spike_train(z, y, dt, fname=None):
    if isinstance(z, torch.Tensor):
        z = to_np(z.squeeze())
    if isinstance(y, torch.Tensor):
        y = to_np(y.squeeze())

    tr = np.arange(0, z.shape[0]) * dt

    dy = y.shape[1]
    spike_times = [np.where(y[:, k] > 0)[0] * dt for k in range(dy)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    for i, st in enumerate(spike_times):
        ax1.eventplot(st, colors="black", lineoffsets=i, linelengths=0.5)

    ax1.set_ylim(-1, dy)
    ax1.set_ylabel("Neurons")
    ax1.xaxis.set_ticklabels([])

    ax2.plot(tr, z)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Latents")
    ax2.set_xlim(tr[0], tr[-1])

    plt.subplots_adjust(hspace=0.05)
    if fname is not None:
        plt.savefig(f"../figs/{fname}.pdf")


def plot_observation_channels(obs, n_channels=5, ax=None, title="Observation Channels"):
    """Plot the first ``n_channels`` observation channels over time."""
    obs_np = to_np(obs)
    if obs_np.ndim == 3:
        obs_np = obs_np[0]
    n_channels = min(n_channels, obs_np.shape[-1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.plot(obs_np[:, :n_channels])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Observation")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
