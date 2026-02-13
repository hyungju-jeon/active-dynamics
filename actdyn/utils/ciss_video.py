"""Reusable visualization helpers for CISS videos."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection


def build_trail_segments(z_xy: np.ndarray, trail_len: int) -> np.ndarray:
    """Return line segments for the trailing latent path."""
    pts = z_xy[-trail_len:]
    if len(pts) < 2:
        return np.zeros((0, 2, 2), dtype=float)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def animate_latent_and_spikes(
    z,
    spikes,
    dt,
    trail_len=100,
    raster_window=200,
    fps=30,
    save_path=None,
):
    """Animate 2D latent trajectory and recent spike raster side-by-side."""
    T, latent_dim = z.shape
    _, n_neurons = spikes.shape
    assert latent_dim == 2, "Only 2D latent supported in this viz."

    plt.rcParams.update(
        {
            "figure.facecolor": "#0f0f10",
            "axes.facecolor": "#0f0f10",
            "axes.edgecolor": "#cccccc",
            "text.color": "#cccccc",
            "axes.labelcolor": "#cccccc",
            "xtick.color": "#888888",
            "ytick.color": "#888888",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(8, 4), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.3)

    ax_latent = fig.add_subplot(gs[0, 0])
    ax_latent.set_title("Latent state", color="#cccccc", pad=6)
    ax_latent.set_xlabel("z1")
    ax_latent.set_ylabel("z2")

    margin = 0.5
    x_min, x_max = z[:, 0].min() - margin, z[:, 0].max() + margin
    y_min, y_max = z[:, 1].min() - margin, z[:, 1].max() + margin
    ax_latent.set_xlim(x_min, x_max)
    ax_latent.set_ylim(y_min, y_max)
    ax_latent.plot(z[:, 0], z[:, 1], lw=0.5, alpha=0.15, color="#bbbbbb")

    init_trail = build_trail_segments(z[:trail_len], trail_len)
    trail_lc = LineCollection(
        init_trail,
        linewidths=2.0,
        cmap="viridis",
        array=np.linspace(0.2, 1.0, max(trail_len - 1, 1)),
        alpha=0.9,
    )
    ax_latent.add_collection(trail_lc)

    (head_dot,) = ax_latent.plot(
        z[0, 0],
        z[0, 1],
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markerfacecolor="none",
        markeredgewidth=0.8,
    )
    time_text = ax_latent.text(
        0.02,
        0.95,
        "",
        transform=ax_latent.transAxes,
        ha="left",
        va="top",
        color="#cccccc",
    )

    ax_raster = fig.add_subplot(gs[0, 1])
    ax_raster.set_title("Spikes (recent)", color="#cccccc", pad=6)
    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_ylabel("Neuron")
    ax_raster.set_ylim(-0.5, n_neurons - 0.5)
    ax_raster.set_yticks([0, n_neurons - 1])
    ax_raster.set_yticklabels(["0", f"{n_neurons-1}"])

    win_dur_ms = raster_window * dt * 1000.0
    ax_raster.set_xlim(-win_dur_ms, 0.0)

    raster_scatter = ax_raster.scatter(
        [],
        [],
        s=6,
        linewidths=0,
        c="#39ff14",
        alpha=0.8,
    )

    def update(frame_idx):
        start_idx = max(0, frame_idx - trail_len + 1)
        trail_slice = z[start_idx : frame_idx + 1]
        if trail_slice.shape[0] >= 2:
            segs = np.stack([trail_slice[:-1], trail_slice[1:]], axis=1)
            trail_lc.set_segments(segs)
            trail_lc.set_array(np.linspace(0.2, 1.0, segs.shape[0]))
        else:
            trail_lc.set_segments([])

        head_dot.set_data([z[frame_idx, 0]], [z[frame_idx, 1]])
        time_text.set_text(f"t = {frame_idx * dt * 1000.0:.1f} ms")

        r_start = max(0, frame_idx - raster_window + 1)
        r_spk = spikes[r_start : frame_idx + 1]
        w = r_spk.shape[0]
        if w > 0:
            t_rel = np.arange(-w + 1, 1) * dt * 1000.0
            tt = np.repeat(t_rel[:, None], n_neurons, axis=1)
            nn = np.repeat(np.arange(n_neurons)[None, :], w, axis=0)
            mask = r_spk > 0
            if np.any(mask):
                raster_scatter.set_offsets(np.stack([tt[mask], nn[mask]], axis=1))
            else:
                raster_scatter.set_offsets(np.zeros((0, 2)))
        else:
            raster_scatter.set_offsets(np.zeros((0, 2)))

        ax_raster.set_xlim(-win_dur_ms, 0.0)
        return trail_lc, head_dot, time_text, raster_scatter

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000.0 / fps,
        blit=False,
    )

    if save_path is not None:
        anim.save(save_path, fps=fps, dpi=150, codec="h264", bitrate=-1)

    return anim


def animate_latent_trajectory(
    z,
    dt=0.01,
    fps=60,
    skip=1,
    trail=100,
    out_path="latent_trajectory.mov",
    prores=True,
):
    """Animate a 2D latent trajectory with a fading trail."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    z = np.asarray(z)
    t_total = len(z)
    frames = range(0, t_total, skip)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.grid(True, color="#333333", lw=0.5, ls="--", alpha=0.5)
    plt.tight_layout()

    init_trail = build_trail_segments(z[:trail], trail)
    trail_line = LineCollection(
        init_trail,
        linewidths=2.0,
        cmap="magma",
        array=np.linspace(0.2, 1.0, max(trail - 1, 1)),
        alpha=0.9,
    )
    ax.add_collection(trail_line)
    (head_dot,) = ax.plot(z[0, 0], z[0, 1], "o", color="#000000", markersize=6)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top", color="white", fontsize=8
    )

    def init():
        trail_line.set_segments([])
        head_dot.set_data([], [])
        time_text.set_text("")
        return trail_line, head_dot, time_text

    def update(frame_idx):
        i = min(frame_idx, t_total - 1)
        start = max(0, i - trail)
        if i - start > 1:
            seg1 = z[start:i, :]
            seg2 = z[start + 1 : i + 1, :]
            seg = np.stack([seg1, seg2], axis=1)
            trail_line.set_segments(seg)
            trail_line.set_array(np.linspace(0, 1, len(seg)))
        else:
            trail_line.set_segments([])
        head_dot.set_data([z[i, 0]], [z[i, 1]])
        time_text.set_text(f"t = {i * dt:.2f} s")
        return trail_line, head_dot, time_text

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    if prores:
        out_path = out_path.replace(".mp4", ".mov")
        writer = FFMpegWriter(
            fps=fps,
            codec="prores_ks",
            bitrate=-1,
            extra_args=["-profile:v", "3", "-pix_fmt", "yuv422p10le"],
        )
    else:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=8000,
            extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-crf", "12", "-movflags", "+faststart"],
        )

    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    return anim


def animate_spikes(
    spikes,
    dt,
    window=200,
    fps=60,
    skip=None,
    save_path="spikes.mp4",
    prores=False,
):
    """Animate spike raster in a rolling time window."""
    t_total, n_neurons = spikes.shape
    win_ms = window * dt * 1000.0

    if skip is None:
        skip = max(1, int(1.0 / (fps * dt)))

    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#FFFFFF",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "axes.labelcolor": "#000000",
            "text.color": "#000000",
        }
    )

    fig, ax = plt.subplots(figsize=(15, 5), dpi=150)
    ax.set_xlim(-win_ms, 0)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.set_title("Spike Raster", color="#333333", pad=8)
    plt.tight_layout()

    scatter = ax.scatter([], [], s=8, marker="o", c="#313131", lw=0, alpha=0.9)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, color="#313131", ha="left", va="top"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.grid(False)

    def init():
        scatter.set_offsets(np.zeros((0, 2)))
        time_text.set_text("")
        return scatter, time_text

    def update(frame_idx):
        start = max(0, frame_idx - window)
        seg = spikes[start : frame_idx + 1]
        w = seg.shape[0]
        if w == 0:
            scatter.set_offsets(np.zeros((0, 2)))
            return scatter, time_text

        t_rel = np.arange(-w + 1, 1) * dt * 1000.0
        tt = np.repeat(t_rel[:, None], n_neurons, axis=1)
        nn = np.repeat(np.arange(n_neurons)[None, :], w, axis=0)
        mask = seg > 0
        if np.any(mask):
            scatter.set_offsets(np.c_[tt[mask], nn[mask]])
        else:
            scatter.set_offsets(np.zeros((0, 2)))
        time_text.set_text("")
        return scatter, time_text

    frames = range(0, t_total, skip)
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000.0 / fps,
        blit=True,
        repeat=False,
    )

    if prores:
        save_path = save_path.replace(".mp4", ".mov")
        writer = FFMpegWriter(
            fps=fps,
            codec="prores_ks",
            bitrate=-1,
            extra_args=["-profile:v", "3", "-pix_fmt", "yuv422p10le"],
        )
    else:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=15000,
            extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-crf", "12", "-movflags", "+faststart"],
        )

    anim.save(save_path, writer=writer, dpi=150)
    plt.close(fig)
    return anim


__all__ = [
    "build_trail_segments",
    "animate_latent_and_spikes",
    "animate_latent_trajectory",
    "animate_spikes",
]
