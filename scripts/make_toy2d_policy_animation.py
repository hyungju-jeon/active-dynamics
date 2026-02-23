#!/usr/bin/env python3
"""
Multistable 2D latent dynamics + active-learning policy comparison.

This experiment is designed to illustrate:
1) Grid-placed local attractors with strong local stability,
2) Single random action is insufficient to escape initial basin,
3) Policy behavior difference: myopic single-step vs planning-based active learning.

Outputs are always written to an experiment-specific subfolder.
If experiment details (config signature) change, existing results are archived automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class Config:
    # Experiment output routing
    output_root: str = "docs/figs/experiments"
    experiment_name: str = "multistable_grid_escape_v1"

    # State-space grid for visualizations
    grid_min: float = -2.2
    grid_max: float = 2.2
    grid_n: int = 181
    stream_grid_n: int = 35

    # Dynamics + rollout horizon
    dt: float = 0.12
    steps: int = 42

    # Action set (small enough for smooth transitions)
    # Note: strong attractors require repeated designed inputs to escape.
    u_max: float = 0.24

    # Active-learning objective
    action_penalty: float = 1.0

    # Planning policy hyperparameters
    plan_horizon: int = 16
    plan_gamma: float = 0.97
    plan_num_sequences: int = 900

    # Animation
    fps: int = 3

    # Initial condition and initial parameter precision
    z0: tuple[float, float] = (-1.45, -1.35)
    precision0_diag: tuple[float, float] = (0.03, 0.02)

    # Multi-stable latent dynamics
    attractor_grid: tuple[float, float, float] = (-1.5, 0.0, 1.5)
    attractor_strength: float = 3.0
    assignment_beta: float = 5.0
    swirl: float = 0.06

    # Scalar Poisson log-linear observation model
    # y_t ~ Poisson(lambda_t), lambda_t = exp(c^T z_t + b)
    obs_c: tuple[float, float] = (0.95, 0.65)
    obs_b: float = -0.9

    # Random single-action escape test
    random_escape_trials: int = 2000
    random_relax_steps: int = 50


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def attractor_centers(cfg: Config) -> np.ndarray:
    vals = np.array(cfg.attractor_grid, dtype=float)
    return np.array([[x, y] for x in vals for y in vals], dtype=float)


def softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def phi1(z: np.ndarray) -> np.ndarray:
    z1 = z[..., 0]
    z2 = z[..., 1]
    return 0.5 + 1.1 * sigmoid(1.6 * (z1 + 0.1)) + 0.45 * sigmoid(1.4 * (z2 + 0.1))


def phi2(z: np.ndarray) -> np.ndarray:
    z1 = z[..., 0]
    z2 = z[..., 1]
    return 0.55 + 1.25 * sigmoid(-1.7 * (z1 + 0.1)) + 0.35 * sigmoid(1.3 * (z2 + 0.2))


def latent_dynamics(z: np.ndarray, cfg: Config, centers: np.ndarray) -> np.ndarray:
    """
    Multi-stable latent dynamics with attractors on a grid.

    f(z) = kappa * (soft-nearest-center(z) - z) + swirl * R z
    where R z = [-z2, z1].
    """
    if z.ndim == 1:
        d2 = np.sum((centers - z) ** 2, axis=1)
        weights = softmax_last((-cfg.assignment_beta * d2)[None, :])[0]
        target = np.sum(weights[:, None] * centers, axis=0)
        rot = cfg.swirl * np.array([-z[1], z[0]], dtype=float)
        return cfg.attractor_strength * (target - z) + rot

    # Batched shape (..., 2)
    d2 = np.sum((z[..., None, :] - centers[None, None, :, :]) ** 2, axis=-1)
    weights = softmax_last(-cfg.assignment_beta * d2)
    target = np.sum(weights[..., None] * centers[None, None, :, :], axis=-2)
    rot = cfg.swirl * np.stack([-z[..., 1], z[..., 0]], axis=-1)
    return cfg.attractor_strength * (target - z) + rot


def step_state(z: np.ndarray, u: np.ndarray, cfg: Config, centers: np.ndarray) -> np.ndarray:
    z_next = z + cfg.dt * latent_dynamics(z, cfg, centers) + u
    return np.clip(z_next, cfg.grid_min, cfg.grid_max)


def nearest_attractor_idx(z: np.ndarray, centers: np.ndarray) -> int:
    d2 = np.sum((centers - z) ** 2, axis=1)
    return int(np.argmin(d2))


def poisson_rate(z: np.ndarray, cfg: Config) -> np.ndarray:
    c = np.array(cfg.obs_c, dtype=float)
    if z.ndim == 1:
        return np.exp(c @ z + cfg.obs_b)
    return np.exp(z @ c + cfg.obs_b)


def sensitivity_matrix(z: np.ndarray, cfg: Config) -> np.ndarray:
    """One-step sensitivity approximation S_t = d z_{t+1}^- / d theta."""
    if z.ndim == 1:
        return cfg.dt * np.array([[phi1(z), 0.0], [0.0, phi2(z)]], dtype=float)

    out = np.zeros(z.shape[:-1] + (2, 2), dtype=float)
    out[..., 0, 0] = cfg.dt * phi1(z)
    out[..., 1, 1] = cfg.dt * phi2(z)
    return out


def fisher_information(z: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Scalar Poisson log-linear observation approximation:
      H_z(z) = lambda(z) * c c^T
      I_theta(z) = S(z)^T H_z(z) S(z)
    """
    c = np.array(cfg.obs_c, dtype=float)
    lam = poisson_rate(z, cfg)

    if z.ndim == 1:
        S = sensitivity_matrix(z, cfg)
        v = S.T @ c
        return lam * np.outer(v, v)

    S = sensitivity_matrix(z, cfg)
    v1 = S[..., 0, 0] * c[0]
    v2 = S[..., 1, 1] * c[1]

    I = np.zeros(z.shape[:-1] + (2, 2), dtype=float)
    I[..., 0, 0] = lam * v1 * v1
    I[..., 0, 1] = lam * v1 * v2
    I[..., 1, 0] = I[..., 0, 1]
    I[..., 1, 1] = lam * v2 * v2
    return I


def logdet2x2(M: np.ndarray) -> np.ndarray:
    sign, value = np.linalg.slogdet(M)
    if np.any(sign <= 0):
        raise ValueError("Non-positive-definite matrix encountered in logdet.")
    return value


def acquisition_value(z: np.ndarray, precision: np.ndarray, u: np.ndarray, cfg: Config) -> float:
    I = fisher_information(z, cfg)
    info_gain = float(logdet2x2(precision + I) - logdet2x2(precision))
    return info_gain - cfg.action_penalty * float(np.dot(u, u))


def acquisition_map(precision: np.ndarray, grid_points: np.ndarray, cfg: Config) -> np.ndarray:
    I = fisher_information(grid_points, cfg)
    return logdet2x2(precision + I) - logdet2x2(precision)


def action_set(cfg: Config) -> np.ndarray:
    vals = [-cfg.u_max, 0.0, cfg.u_max]
    return np.array([[ux, uy] for ux in vals for uy in vals], dtype=float)


def choose_action_myopic(
    z: np.ndarray,
    precision: np.ndarray,
    actions: np.ndarray,
    cfg: Config,
    centers: np.ndarray,
) -> np.ndarray:
    best_val = -np.inf
    best_u = actions[0]
    for u in actions:
        z_next = step_state(z, u, cfg, centers)
        val = acquisition_value(z_next, precision, u, cfg)
        if val > best_val:
            best_val = val
            best_u = u
    return best_u


def choose_action_planning(
    z: np.ndarray,
    precision: np.ndarray,
    actions: np.ndarray,
    cfg: Config,
    centers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    H-step random-shooting planning in information space.

    maximize sum_k gamma^k [Delta_info - action_penalty * ||u||^2]
    while rolling hypothetical precision updates.
    """
    n_actions = actions.shape[0]
    idx = rng.integers(0, n_actions, size=(cfg.plan_num_sequences, cfg.plan_horizon))

    best_score = -np.inf
    best_first_idx = 0

    for m in range(cfg.plan_num_sequences):
        z_roll = z.copy()
        P_roll = precision.copy()
        total = 0.0

        for k in range(cfg.plan_horizon):
            u = actions[idx[m, k]]
            z_roll = step_state(z_roll, u, cfg, centers)
            total += (cfg.plan_gamma ** k) * acquisition_value(z_roll, P_roll, u, cfg)
            P_roll = P_roll + fisher_information(z_roll, cfg)

        if total > best_score:
            best_score = total
            best_first_idx = idx[m, 0]

    return actions[best_first_idx]


def run_policy(mode: str, cfg: Config, grid_points: np.ndarray, centers: np.ndarray) -> dict[str, np.ndarray]:
    assert mode in {"myopic", "planning"}

    actions = action_set(cfg)

    z = np.array(cfg.z0, dtype=float)
    precision = np.diag(np.array(cfg.precision0_diag, dtype=float))

    traj = [z.copy()]
    precisions = [precision.copy()]
    maps = []
    argmax_points = []
    chosen_actions = []
    chosen_scores = []
    basin_idx = []

    for t in range(cfg.steps):
        A = acquisition_map(precision, grid_points, cfg)
        maps.append(A)

        ij = np.unravel_index(np.argmax(A), A.shape)
        argmax_points.append(np.array([grid_points[ij][0], grid_points[ij][1]], dtype=float))

        basin_idx.append(nearest_attractor_idx(z, centers))

        if mode == "myopic":
            u = choose_action_myopic(z, precision, actions, cfg, centers)
        else:
            rng = np.random.default_rng(100 + t)
            u = choose_action_planning(z, precision, actions, cfg, centers, rng)

        z = step_state(z, u, cfg, centers)
        score = acquisition_value(z, precision, u, cfg)
        precision = precision + fisher_information(z, cfg)

        chosen_actions.append(u.copy())
        chosen_scores.append(score)
        traj.append(z.copy())
        precisions.append(precision.copy())

    return {
        "mode": mode,
        "traj": np.asarray(traj),
        "precisions": np.asarray(precisions),
        "maps": np.asarray(maps),
        "argmax": np.asarray(argmax_points),
        "actions": np.asarray(chosen_actions),
        "scores": np.asarray(chosen_scores),
        "basin_idx": np.asarray(basin_idx),
    }


def random_single_action_escape_probability(cfg: Config, centers: np.ndarray) -> float:
    """
    Probability that one random action can move trajectory to a different attractor basin
    after passive relaxation.
    """
    rng = np.random.default_rng(0)

    z0 = np.array(cfg.z0, dtype=float)
    home = nearest_attractor_idx(z0, centers)

    escapes = 0
    for _ in range(cfg.random_escape_trials):
        z = z0.copy()
        u = rng.uniform(-cfg.u_max, cfg.u_max, size=2)
        z = step_state(z, u, cfg, centers)

        for _ in range(cfg.random_relax_steps):
            z = step_state(z, np.zeros(2, dtype=float), cfg, centers)

        if nearest_attractor_idx(z, centers) != home:
            escapes += 1

    return escapes / float(cfg.random_escape_trials)


def dynamics_field(cfg: Config, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.linspace(cfg.grid_min, cfg.grid_max, cfg.stream_grid_n)
    Xd, Yd = np.meshgrid(g, g)
    Zd = np.stack([Xd, Yd], axis=-1)
    Fd = latent_dynamics(Zd, cfg, centers)
    Ud = Fd[..., 0]
    Vd = Fd[..., 1]

    mag = np.sqrt(Ud**2 + Vd**2)
    scale = np.quantile(mag, 0.85) + 1e-9
    Ud = Ud / scale
    Vd = Vd / scale
    return Xd, Yd, Ud, Vd


def config_signature(cfg: Config) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def prepare_experiment_dir(cfg: Config) -> Path:
    root = Path(cfg.output_root)
    exp_dir = root / cfg.experiment_name

    sig = config_signature(cfg)
    meta_path = exp_dir / "config.json"

    if exp_dir.exists():
        old_sig = None
        if meta_path.exists():
            try:
                old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                old_sig = old_meta.get("signature")
            except Exception:
                old_sig = None

        if old_sig != sig:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            archive_dir = root / "archive" / f"{ts}-{cfg.experiment_name}"
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exp_dir), str(archive_dir))

    exp_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "experiment_name": cfg.experiment_name,
        "signature": sig,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return exp_dir


def add_static_background(
    ax: plt.Axes,
    Xd: np.ndarray,
    Yd: np.ndarray,
    Ud: np.ndarray,
    Vd: np.ndarray,
    centers: np.ndarray,
) -> None:
    ax.streamplot(
        Xd,
        Yd,
        Ud,
        Vd,
        color=(1.0, 1.0, 1.0, 0.32),
        density=0.8,
        linewidth=0.55,
        arrowsize=0.62,
        minlength=0.12,
        broken_streamlines=True,
        zorder=2,
    )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="#FF6D6D",
        edgecolors="black",
        s=28,
        marker="X",
        linewidths=0.5,
        zorder=5,
        label="attractors",
    )


def make_animation(
    data_myopic: dict[str, np.ndarray],
    data_planning: dict[str, np.ndarray],
    cfg: Config,
    centers: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    Ud: np.ndarray,
    Vd: np.ndarray,
    out_gif: Path,
    out_mp4: Path,
) -> None:
    maps1 = data_myopic["maps"]
    maps2 = data_planning["maps"]
    traj1 = data_myopic["traj"]
    traj2 = data_planning["traj"]
    arg1 = data_myopic["argmax"]
    arg2 = data_planning["argmax"]
    P1 = data_myopic["precisions"]
    P2 = data_planning["precisions"]
    b1 = data_myopic["basin_idx"]
    b2 = data_planning["basin_idx"]

    vmin = float(min(maps1.min(), maps2.min()))
    vmax = float(max(maps1.max(), maps2.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), constrained_layout=True)

    ims = []
    lines = []
    curs = []
    stars = []
    starts = []
    txts = []

    titles = [
        "Single-step active learning (myopic)",
        rf"Planning-based active learning (H={cfg.plan_horizon})",
    ]

    for j, ax in enumerate(axes):
        im = ax.imshow(
            maps1[0] if j == 0 else maps2[0],
            extent=[cfg.grid_min, cfg.grid_max, cfg.grid_min, cfg.grid_max],
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            animated=True,
            zorder=1,
        )
        ims.append(im)

        add_static_background(ax, Xd, Yd, Ud, Vd, centers)

        line, = ax.plot([], [], "w-", lw=1.6, alpha=0.95, zorder=6)
        lines.append(line)

        cur = ax.scatter([], [], c="#00E676", edgecolors="black", s=62, marker="o", zorder=7)
        curs.append(cur)

        star = ax.scatter([], [], c="yellow", edgecolors="black", s=92, marker="*", zorder=8)
        stars.append(star)

        start = ax.scatter([], [], c="cyan", edgecolors="black", s=56, marker="s", zorder=8)
        starts.append(start)

        txt = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5),
            zorder=9,
        )
        txts.append(txt)

        ax.set_title(titles[j])
        ax.set_xlim(cfg.grid_min, cfg.grid_max)
        ax.set_ylim(cfg.grid_min, cfg.grid_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")

    cbar = fig.colorbar(ims[1], ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label(r"acquisition $\mathcal{J}_t(\mathbf{z})$")

    fig.suptitle(
        "Multistable latent dynamics: sequential acquisition evolution and policy behavior",
        fontsize=12,
        y=1.01,
    )

    def set_scatter(sc, xy: np.ndarray) -> None:
        sc.set_offsets(np.asarray(xy, dtype=float).reshape(1, 2))

    def update(frame: int):
        ims[0].set_data(maps1[frame])
        ims[1].set_data(maps2[frame])

        lines[0].set_data(traj1[: frame + 1, 0], traj1[: frame + 1, 1])
        lines[1].set_data(traj2[: frame + 1, 0], traj2[: frame + 1, 1])

        set_scatter(curs[0], traj1[frame])
        set_scatter(curs[1], traj2[frame])

        set_scatter(stars[0], arg1[frame])
        set_scatter(stars[1], arg2[frame])

        set_scatter(starts[0], traj1[0])
        set_scatter(starts[1], traj2[0])

        txts[0].set_text(
            f"t={frame}\n"
            f"state=({traj1[frame,0]:+.2f},{traj1[frame,1]:+.2f})\n"
            f"basin={b1[frame]}\n"
            f"diag(Λ)=({P1[frame,0,0]:.2f},{P1[frame,1,1]:.2f})"
        )
        txts[1].set_text(
            f"t={frame}\n"
            f"state=({traj2[frame,0]:+.2f},{traj2[frame,1]:+.2f})\n"
            f"basin={b2[frame]}\n"
            f"diag(Λ)=({P2[frame,0,0]:.2f},{P2[frame,1,1]:.2f})"
        )

        return [*ims, *lines, *curs, *stars, *starts, *txts]

    ani = FuncAnimation(fig, update, frames=cfg.steps, interval=1000 // cfg.fps, blit=False)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_gif, writer=PillowWriter(fps=cfg.fps))

    if shutil.which("ffmpeg"):
        from matplotlib.animation import FFMpegWriter

        ani.save(out_mp4, writer=FFMpegWriter(fps=cfg.fps, bitrate=2400))

    plt.close(fig)


def make_sequence_figure(
    data_myopic: dict[str, np.ndarray],
    data_planning: dict[str, np.ndarray],
    cfg: Config,
    centers: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    Ud: np.ndarray,
    Vd: np.ndarray,
    out_png: Path,
    out_pdf: Path,
) -> None:
    maps1 = data_myopic["maps"]
    maps2 = data_planning["maps"]
    traj1 = data_myopic["traj"]
    traj2 = data_planning["traj"]
    arg1 = data_myopic["argmax"]
    arg2 = data_planning["argmax"]

    times = list(range(0, cfg.steps, 6))
    if times[-1] != cfg.steps - 1:
        times.append(cfg.steps - 1)

    vmin = float(min(maps1.min(), maps2.min()))
    vmax = float(max(maps1.max(), maps2.max()))

    fig, axes = plt.subplots(2, len(times), figsize=(2.75 * len(times), 6.2), constrained_layout=True)

    for j, t in enumerate(times):
        ax = axes[0, j]
        im = ax.imshow(
            maps1[t],
            extent=[cfg.grid_min, cfg.grid_max, cfg.grid_min, cfg.grid_max],
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=1,
        )
        add_static_background(ax, Xd, Yd, Ud, Vd, centers)
        ax.plot(traj1[: t + 1, 0], traj1[: t + 1, 1], "w-", lw=1.2, zorder=6)
        ax.scatter(traj1[0, 0], traj1[0, 1], c="cyan", edgecolors="black", s=30, marker="s", zorder=7)
        ax.scatter(traj1[t, 0], traj1[t, 1], c="#00E676", edgecolors="black", s=34, marker="o", zorder=7)
        ax.scatter(arg1[t, 0], arg1[t, 1], c="yellow", edgecolors="black", s=54, marker="*", zorder=8)
        ax.set_title(f"myopic, t={t}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(cfg.grid_min, cfg.grid_max)
        ax.set_ylim(cfg.grid_min, cfg.grid_max)
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")

        ax = axes[1, j]
        ax.imshow(
            maps2[t],
            extent=[cfg.grid_min, cfg.grid_max, cfg.grid_min, cfg.grid_max],
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=1,
        )
        add_static_background(ax, Xd, Yd, Ud, Vd, centers)
        ax.plot(traj2[: t + 1, 0], traj2[: t + 1, 1], "w-", lw=1.2, zorder=6)
        ax.scatter(traj2[0, 0], traj2[0, 1], c="cyan", edgecolors="black", s=30, marker="s", zorder=7)
        ax.scatter(traj2[t, 0], traj2[t, 1], c="#00E676", edgecolors="black", s=34, marker="o", zorder=7)
        ax.scatter(arg2[t, 0], arg2[t, 1], c="yellow", edgecolors="black", s=54, marker="*", zorder=8)
        ax.set_title(f"planning, t={t}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(cfg.grid_min, cfg.grid_max)
        ax.set_ylim(cfg.grid_min, cfg.grid_max)
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, pad=0.01)
    cbar.set_label(r"acquisition $\mathcal{J}_t(\mathbf{z})$")

    fig.suptitle(
        "Time-resolved acquisition maps on multistable latent dynamics",
        fontsize=12,
        y=1.02,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_representative_figure(
    data_myopic: dict[str, np.ndarray],
    data_planning: dict[str, np.ndarray],
    cfg: Config,
    centers: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    Ud: np.ndarray,
    Vd: np.ndarray,
    out_png: Path,
    out_pdf: Path,
) -> None:
    maps1 = data_myopic["maps"]
    maps2 = data_planning["maps"]
    traj1 = data_myopic["traj"]
    traj2 = data_planning["traj"]
    arg1 = data_myopic["argmax"]
    arg2 = data_planning["argmax"]

    t_mid = cfg.steps // 2
    vmin = float(min(maps1.min(), maps2.min()))
    vmax = float(max(maps1.max(), maps2.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    # A: trajectory over dynamics field
    ax = axes[0]
    speed = np.sqrt(Ud**2 + Vd**2)
    ax.contourf(Xd, Yd, speed, levels=20, cmap="Greys", alpha=0.65, zorder=1)
    add_static_background(ax, Xd, Yd, Ud, Vd, centers)
    ax.plot(traj1[:, 0], traj1[:, 1], color="#4DD0E1", lw=2.0, label="myopic", zorder=7)
    ax.plot(traj2[:, 0], traj2[:, 1], color="#7CFF6B", lw=2.0, label="planning", zorder=7)
    ax.scatter(traj1[0, 0], traj1[0, 1], c="cyan", edgecolors="black", s=48, marker="s", zorder=8)
    ax.scatter(traj1[-1, 0], traj1[-1, 1], c="#4DD0E1", edgecolors="black", s=48, marker="o", zorder=8)
    ax.scatter(traj2[-1, 0], traj2[-1, 1], c="#7CFF6B", edgecolors="black", s=48, marker="o", zorder=8)
    ax.set_title("Underlying multistable dynamics + policy trajectories")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_xlim(cfg.grid_min, cfg.grid_max)
    ax.set_ylim(cfg.grid_min, cfg.grid_max)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=True)

    # B: myopic acquisition at mid-time
    ax = axes[1]
    im = ax.imshow(
        maps1[t_mid],
        extent=[cfg.grid_min, cfg.grid_max, cfg.grid_min, cfg.grid_max],
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        zorder=1,
    )
    add_static_background(ax, Xd, Yd, Ud, Vd, centers)
    ax.plot(traj1[: t_mid + 1, 0], traj1[: t_mid + 1, 1], "w-", lw=1.4, zorder=7)
    ax.scatter(traj1[t_mid, 0], traj1[t_mid, 1], c="#00E676", edgecolors="black", s=50, zorder=8)
    ax.scatter(arg1[t_mid, 0], arg1[t_mid, 1], c="yellow", edgecolors="black", s=70, marker="*", zorder=8)
    ax.set_title(f"Myopic acquisition at t={t_mid}")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_xlim(cfg.grid_min, cfg.grid_max)
    ax.set_ylim(cfg.grid_min, cfg.grid_max)
    ax.set_aspect("equal", adjustable="box")

    # C: planning acquisition at mid-time
    ax = axes[2]
    ax.imshow(
        maps2[t_mid],
        extent=[cfg.grid_min, cfg.grid_max, cfg.grid_min, cfg.grid_max],
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        zorder=1,
    )
    add_static_background(ax, Xd, Yd, Ud, Vd, centers)
    ax.plot(traj2[: t_mid + 1, 0], traj2[: t_mid + 1, 1], "w-", lw=1.4, zorder=7)
    ax.scatter(traj2[t_mid, 0], traj2[t_mid, 1], c="#00E676", edgecolors="black", s=50, zorder=8)
    ax.scatter(arg2[t_mid, 0], arg2[t_mid, 1], c="yellow", edgecolors="black", s=70, marker="*", zorder=8)
    ax.set_title(f"Planning acquisition at t={t_mid}")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_xlim(cfg.grid_min, cfg.grid_max)
    ax.set_ylim(cfg.grid_min, cfg.grid_max)
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.01)
    cbar.set_label(r"acquisition $\mathcal{J}_t(\mathbf{z})$")

    fig.suptitle(
        "Representative view: strong local attractors, smooth control, and policy-dependent escape",
        fontsize=12,
        y=1.02,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    data_myopic: dict[str, np.ndarray],
    data_planning: dict[str, np.ndarray],
    cfg: Config,
    centers: np.ndarray,
    random_escape_prob: float,
    out_txt: Path,
) -> None:
    traj1 = data_myopic["traj"]
    traj2 = data_planning["traj"]
    P1 = data_myopic["precisions"]
    P2 = data_planning["precisions"]
    s1 = data_myopic["scores"]
    s2 = data_planning["scores"]
    b1 = data_myopic["basin_idx"]
    b2 = data_planning["basin_idx"]

    start_basin = nearest_attractor_idx(np.array(cfg.z0, dtype=float), centers)
    final_basin_myopic = nearest_attractor_idx(traj1[-1], centers)
    final_basin_planning = nearest_attractor_idx(traj2[-1], centers)

    def path_length(traj: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())

    center_str = ", ".join([f"{i}:{tuple(np.round(c, 3))}" for i, c in enumerate(centers)])

    lines = [
        "Multistable grid active-learning experiment summary",
        "===============================================",
        f"experiment_name = {cfg.experiment_name}",
        f"steps = {cfg.steps}",
        f"dt = {cfg.dt}, u_max = {cfg.u_max}",
        f"planning horizon = {cfg.plan_horizon}, gamma = {cfg.plan_gamma}",
        f"action_penalty = {cfg.action_penalty}",
        "",
        f"attractor_centers = {center_str}",
        f"start_state = ({cfg.z0[0]:.4f}, {cfg.z0[1]:.4f}), start_basin = {start_basin}",
        "",
        "Random single-action escape test:",
        f"escape_probability = {random_escape_prob:.6f}",
        f"random single action insufficient? {'YES' if random_escape_prob < 0.01 else 'NO'}",
        "",
        "Final state / basin:",
        f"myopic   z_T = ({traj1[-1,0]:.4f}, {traj1[-1,1]:.4f}), basin = {final_basin_myopic}",
        f"planning z_T = ({traj2[-1,0]:.4f}, {traj2[-1,1]:.4f}), basin = {final_basin_planning}",
        "",
        "Policy-level stats:",
        f"myopic   path_length = {path_length(traj1):.6f}",
        f"planning path_length = {path_length(traj2):.6f}",
        f"myopic   mean_step_score = {s1.mean():.6f}",
        f"planning mean_step_score = {s2.mean():.6f}",
        f"myopic   final diag(Lambda) = ({P1[-1,0,0]:.6f}, {P1[-1,1,1]:.6f})",
        f"planning final diag(Lambda) = ({P2[-1,0,0]:.6f}, {P2[-1,1,1]:.6f})",
        "",
        "Basin trajectory IDs per step:",
        f"myopic   = {b1.tolist()}",
        f"planning = {b2.tolist()}",
        "",
        "Per-step comparison (t, score_myopic, score_planning, state_distance):",
    ]

    for t in range(cfg.steps):
        dist = float(np.linalg.norm(traj1[t] - traj2[t]))
        lines.append(f"{t}, {s1[t]:.6f}, {s2[t]:.6f}, {dist:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = Config()

    exp_dir = prepare_experiment_dir(cfg)

    centers = attractor_centers(cfg)

    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_n)
    X, Y = np.meshgrid(grid, grid)
    grid_points = np.stack([X, Y], axis=-1)

    data_myopic = run_policy("myopic", cfg, grid_points, centers)
    data_planning = run_policy("planning", cfg, grid_points, centers)

    random_escape_prob = random_single_action_escape_probability(cfg, centers)

    Xd, Yd, Ud, Vd = dynamics_field(cfg, centers)

    out_gif = exp_dir / "multistable_policy_comparison.gif"
    out_mp4 = exp_dir / "multistable_policy_comparison.mp4"
    out_seq_png = exp_dir / "multistable_policy_sequence.png"
    out_seq_pdf = exp_dir / "multistable_policy_sequence.pdf"
    out_rep_png = exp_dir / "multistable_representative.png"
    out_rep_pdf = exp_dir / "multistable_representative.pdf"
    out_txt = exp_dir / "summary.txt"

    make_animation(data_myopic, data_planning, cfg, centers, Xd, Yd, Ud, Vd, out_gif, out_mp4)
    make_sequence_figure(data_myopic, data_planning, cfg, centers, Xd, Yd, Ud, Vd, out_seq_png, out_seq_pdf)
    make_representative_figure(data_myopic, data_planning, cfg, centers, Xd, Yd, Ud, Vd, out_rep_png, out_rep_pdf)
    write_summary(data_myopic, data_planning, cfg, centers, random_escape_prob, out_txt)

    print(f"[ok] experiment_dir: {exp_dir}")
    print(f"[ok] wrote {out_gif}")
    if out_mp4.exists():
        print(f"[ok] wrote {out_mp4}")
    else:
        print("[ok] mp4 not written (ffmpeg not available)")
    print(f"[ok] wrote {out_seq_png}")
    print(f"[ok] wrote {out_seq_pdf}")
    print(f"[ok] wrote {out_rep_png}")
    print(f"[ok] wrote {out_rep_pdf}")
    print(f"[ok] wrote {out_txt}")


if __name__ == "__main__":
    main()
