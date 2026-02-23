#!/usr/bin/env python3
"""
Toy 2D active-learning policy comparison with Poisson log-linear observations.

Generates:
- docs/figs/toy2d_policy_comparison.gif                 (side-by-side animation)
- docs/figs/toy2d_policy_comparison.mp4                 (optional, if ffmpeg exists)
- docs/figs/toy2d_policy_comparison_sequence.png/.pdf   (selected time snapshots)
- docs/figs/toy2d_policy_comparison_summary.txt

Comparison:
1) single-step myopic active learning
2) planning-based active learning (H-step random-shooting MPC over information gain)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class Config:
    grid_min: float = -2.3
    grid_max: float = 2.3
    grid_n: int = 141

    dt: float = 0.20
    steps: int = 24

    # Control/action discretization
    u_max: float = 0.30

    # Planning policy hyperparameters
    plan_horizon: int = 6
    plan_gamma: float = 0.96
    plan_num_sequences: int = 700

    # Animation
    fps: int = 2

    # Initial state and parameter precision
    z0: tuple[float, float] = (-1.6, -1.2)
    precision0_diag: tuple[float, float] = (0.08, 0.004)

    # Nominal dynamics parameters (used by the toy dynamics model)
    theta_hat: tuple[float, float] = (1.0, 1.0)

    # Two-channel Poisson log-linear observation model
    # y_{r,t} ~ Poisson(lambda_{r,t}), lambda_{r,t}=exp(c_r^T z_t + b_r)
    C: tuple[tuple[float, float], tuple[float, float]] = ((1.05, 0.15), (-0.95, 0.95))
    b: tuple[float, float] = (-0.7, -1.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def phi1(z: np.ndarray) -> np.ndarray:
    z1 = z[..., 0]
    z2 = z[..., 1]
    return 0.9 + 0.9 * sigmoid(1.6 * (z1 - 0.1)) + 0.05 * np.tanh(0.6 * z2)


def phi2(z: np.ndarray) -> np.ndarray:
    z1 = z[..., 0]
    z2 = z[..., 1]
    return 0.5 + 0.35 * sigmoid(1.5 * (z2 + 0.2)) + 1.3 * sigmoid(-1.8 * (z1 + 0.2))


def toy_dynamics(z: np.ndarray, cfg: Config) -> np.ndarray:
    """Nominal drift f(z; theta_hat) used in z_{t+1} = z_t + dt * f + u_t."""
    theta = np.array(cfg.theta_hat, dtype=float)
    p1 = phi1(z)
    p2 = phi2(z)

    z1 = z[..., 0]
    z2 = z[..., 1]
    f1 = theta[0] * p1 - 0.32 * z2
    f2 = theta[1] * p2 + 0.20 * z1
    return np.stack([f1, f2], axis=-1)


def poisson_rates(z: np.ndarray, cfg: Config) -> np.ndarray:
    """Two-channel independent Poisson rates."""
    C = np.array(cfg.C, dtype=float)
    b = np.array(cfg.b, dtype=float)
    if z.ndim == 1:
        return np.exp(C @ z + b)
    return np.exp(z @ C.T + b)


def obs_hessian(z: np.ndarray, cfg: Config) -> np.ndarray:
    """
    For independent Poisson channels with log-linear links,
    H_z = sum_r lambda_r c_r c_r^T.
    """
    C = np.array(cfg.C, dtype=float)
    lam = poisson_rates(z, cfg)

    if z.ndim == 1:
        H = np.zeros((2, 2), dtype=float)
        for r in range(C.shape[0]):
            c = C[r]
            H += lam[r] * np.outer(c, c)
        return H

    H = np.zeros(z.shape[:-1] + (2, 2), dtype=float)
    for r in range(C.shape[0]):
        c = C[r]
        H += lam[..., r][..., None, None] * (c[None, None, :, None] * c[None, None, None, :])
    return H


def sensitivity_matrix(z: np.ndarray, cfg: Config) -> np.ndarray:
    """
    One-step sensitivity approximation:
    S_t = d z_{t+1}^- / d theta ~= dt * diag(phi1(z_t), phi2(z_t)).
    """
    if z.ndim == 1:
        return cfg.dt * np.array([[phi1(z), 0.0], [0.0, phi2(z)]], dtype=float)

    out = np.zeros(z.shape[:-1] + (2, 2), dtype=float)
    out[..., 0, 0] = cfg.dt * phi1(z)
    out[..., 1, 1] = cfg.dt * phi2(z)
    return out


def fisher_information(z: np.ndarray, cfg: Config) -> np.ndarray:
    """I_theta(z) = S(z)^T H_z(z) S(z)."""
    if z.ndim == 1:
        S = sensitivity_matrix(z, cfg)
        H = obs_hessian(z, cfg)
        return S.T @ H @ S

    S = sensitivity_matrix(z, cfg)
    H = obs_hessian(z, cfg)
    return np.einsum("...ji,...jk,...kl->...il", S, H, S)


def logdet2x2(M: np.ndarray) -> np.ndarray:
    sign, value = np.linalg.slogdet(M)
    if np.any(sign <= 0):
        raise ValueError("Non-positive-definite matrix encountered in logdet.")
    return value


def acquisition_value(z: np.ndarray, precision: np.ndarray, cfg: Config) -> float:
    I = fisher_information(z, cfg)
    return float(logdet2x2(precision + I) - logdet2x2(precision))


def acquisition_map(precision: np.ndarray, grid_points: np.ndarray, cfg: Config) -> np.ndarray:
    I = fisher_information(grid_points, cfg)
    return logdet2x2(precision + I) - logdet2x2(precision)


def step_state(z: np.ndarray, u: np.ndarray, cfg: Config) -> np.ndarray:
    z_next = z + cfg.dt * toy_dynamics(z, cfg) + u
    return np.clip(z_next, cfg.grid_min, cfg.grid_max)


def action_set(cfg: Config) -> np.ndarray:
    vals = [-cfg.u_max, 0.0, cfg.u_max]
    return np.array([[ux, uy] for ux in vals for uy in vals], dtype=float)


def choose_action_myopic(z: np.ndarray, precision: np.ndarray, actions: np.ndarray, cfg: Config) -> np.ndarray:
    best_val = -np.inf
    best_u = actions[0]
    for u in actions:
        z_next = step_state(z, u, cfg)
        v = acquisition_value(z_next, precision, cfg)
        if v > best_val:
            best_val = v
            best_u = u
    return best_u


def choose_action_planning(
    z: np.ndarray,
    precision: np.ndarray,
    actions: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Random-shooting H-step planning:
    maximize sum_{k=1}^H gamma^{k-1} * DeltaJ(z_{t+k}, Lambda_{t+k-1})
    with hypothetical precision rollouts.
    """
    n_actions = actions.shape[0]
    idx = rng.integers(0, n_actions, size=(cfg.plan_num_sequences, cfg.plan_horizon))

    best_score = -np.inf
    best_first = 0

    for m in range(cfg.plan_num_sequences):
        z_roll = z.copy()
        P_roll = precision.copy()
        score = 0.0

        for k in range(cfg.plan_horizon):
            u = actions[idx[m, k]]
            z_roll = step_state(z_roll, u, cfg)
            gain = acquisition_value(z_roll, P_roll, cfg)
            score += (cfg.plan_gamma ** k) * gain
            P_roll = P_roll + fisher_information(z_roll, cfg)

        if score > best_score:
            best_score = score
            best_first = idx[m, 0]

    return actions[best_first]


def run_policy(mode: str, cfg: Config, grid_points: np.ndarray) -> dict[str, np.ndarray]:
    assert mode in {"myopic", "planning"}

    actions = action_set(cfg)

    z = np.array(cfg.z0, dtype=float)
    precision = np.diag(np.array(cfg.precision0_diag, dtype=float))

    traj = [z.copy()]
    precisions = [precision.copy()]
    maps = []
    argmax_points = []
    chosen_u = []
    chosen_gain = []

    for t in range(cfg.steps):
        A = acquisition_map(precision, grid_points, cfg)
        maps.append(A)

        ij = np.unravel_index(np.argmax(A), A.shape)
        argmax_points.append(np.array([grid_points[ij][0], grid_points[ij][1]]))

        if mode == "myopic":
            u = choose_action_myopic(z, precision, actions, cfg)
        else:
            rng = np.random.default_rng(1234 + t)
            u = choose_action_planning(z, precision, actions, cfg, rng)

        z = step_state(z, u, cfg)
        gain = acquisition_value(z, precision, cfg)
        precision = precision + fisher_information(z, cfg)

        chosen_u.append(u.copy())
        chosen_gain.append(gain)
        traj.append(z.copy())
        precisions.append(precision.copy())

    return {
        "mode": mode,
        "traj": np.asarray(traj),
        "precisions": np.asarray(precisions),
        "maps": np.asarray(maps),
        "argmax": np.asarray(argmax_points),
        "actions": np.asarray(chosen_u),
        "gains": np.asarray(chosen_gain),
    }


def make_animation(
    data_myopic: dict[str, np.ndarray],
    data_planning: dict[str, np.ndarray],
    cfg: Config,
    X: np.ndarray,
    Y: np.ndarray,
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

    vmin = float(min(maps1.min(), maps2.min()))
    vmax = float(max(maps1.max(), maps2.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), constrained_layout=True)

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
        )
        ims.append(im)

        line, = ax.plot([], [], "w-", lw=1.6, alpha=0.95)
        lines.append(line)

        cur = ax.scatter([], [], c="#00E676", edgecolors="black", s=64, marker="o", zorder=5)
        curs.append(cur)

        star = ax.scatter([], [], c="yellow", edgecolors="black", s=92, marker="*", zorder=6)
        stars.append(star)

        start = ax.scatter([], [], c="cyan", edgecolors="black", s=56, marker="s", zorder=6)
        starts.append(start)

        txt = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
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
        "Sequential acquisition-map evolution and policy-dependent behavior",
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
            f"diag(Λ)=({P1[frame,0,0]:.2f},{P1[frame,1,1]:.3f})"
        )
        txts[1].set_text(
            f"t={frame}\n"
            f"state=({traj2[frame,0]:+.2f},{traj2[frame,1]:+.2f})\n"
            f"diag(Λ)=({P2[frame,0,0]:.2f},{P2[frame,1,1]:.3f})"
        )

        artists = [*ims, *lines, *curs, *stars, *starts, *txts]
        return artists

    ani = FuncAnimation(fig, update, frames=cfg.steps, interval=1000 // cfg.fps, blit=False)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_gif, writer=PillowWriter(fps=cfg.fps))

    if shutil.which("ffmpeg"):
        from matplotlib.animation import FFMpegWriter

        ani.save(out_mp4, writer=FFMpegWriter(fps=cfg.fps, bitrate=2200))

    plt.close(fig)


def make_sequence_figure(
    data_myopic: dict[str, np.ndarray],
    data_planning: dict[str, np.ndarray],
    cfg: Config,
    out_png: Path,
    out_pdf: Path,
) -> None:
    maps1 = data_myopic["maps"]
    maps2 = data_planning["maps"]
    traj1 = data_myopic["traj"]
    traj2 = data_planning["traj"]
    arg1 = data_myopic["argmax"]
    arg2 = data_planning["argmax"]

    times = np.linspace(0, cfg.steps - 1, 6, dtype=int)

    vmin = float(min(maps1.min(), maps2.min()))
    vmax = float(max(maps1.max(), maps2.max()))

    fig, axes = plt.subplots(2, len(times), figsize=(3.1 * len(times), 6.4), constrained_layout=True)

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
        )
        ax.plot(traj1[: t + 1, 0], traj1[: t + 1, 1], "w-", lw=1.2)
        ax.scatter(traj1[0, 0], traj1[0, 1], c="cyan", edgecolors="black", s=36, marker="s")
        ax.scatter(traj1[t, 0], traj1[t, 1], c="#00E676", edgecolors="black", s=40, marker="o")
        ax.scatter(arg1[t, 0], arg1[t, 1], c="yellow", edgecolors="black", s=58, marker="*")
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
        )
        ax.plot(traj2[: t + 1, 0], traj2[: t + 1, 1], "w-", lw=1.2)
        ax.scatter(traj2[0, 0], traj2[0, 1], c="cyan", edgecolors="black", s=36, marker="s")
        ax.scatter(traj2[t, 0], traj2[t, 1], c="#00E676", edgecolors="black", s=40, marker="o")
        ax.scatter(arg2[t, 0], arg2[t, 1], c="yellow", edgecolors="black", s=58, marker="*")
        ax.set_title(f"planning, t={t}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(cfg.grid_min, cfg.grid_max)
        ax.set_ylim(cfg.grid_min, cfg.grid_max)
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, pad=0.01)
    cbar.set_label(r"acquisition $\mathcal{J}_t(\mathbf{z})$")

    fig.suptitle(
        "Time-resolved acquisition evolution: myopic vs planning-based active learning",
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
    out_txt: Path,
) -> None:
    traj1 = data_myopic["traj"]
    traj2 = data_planning["traj"]
    P1 = data_myopic["precisions"]
    P2 = data_planning["precisions"]
    g1 = data_myopic["gains"]
    g2 = data_planning["gains"]

    def path_length(traj: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())

    lines = [
        "Toy 2D Poisson active-learning policy comparison",
        "==============================================",
        f"steps = {cfg.steps}",
        f"planning horizon = {cfg.plan_horizon}, gamma = {cfg.plan_gamma}",
        "",
        "Final state:",
        f"myopic    z_T = ({traj1[-1,0]:.4f}, {traj1[-1,1]:.4f})",
        f"planning  z_T = ({traj2[-1,0]:.4f}, {traj2[-1,1]:.4f})",
        "",
        "Final precision diagonal:",
        f"myopic    diag(Lambda_T) = ({P1[-1,0,0]:.6f}, {P1[-1,1,1]:.6f})",
        f"planning  diag(Lambda_T) = ({P2[-1,0,0]:.6f}, {P2[-1,1,1]:.6f})",
        "",
        "Trajectory statistics:",
        f"myopic    path length = {path_length(traj1):.4f}",
        f"planning  path length = {path_length(traj2):.4f}",
        f"mean per-step gain (myopic)   = {g1.mean():.6f}",
        f"mean per-step gain (planning) = {g2.mean():.6f}",
        "",
        "Per-step comparison (t, gain_myopic, gain_planning, state_distance):",
    ]

    for t in range(cfg.steps):
        dist = float(np.linalg.norm(traj1[t] - traj2[t]))
        lines.append(f"{t}, {g1[t]:.6f}, {g2[t]:.6f}, {dist:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = Config()

    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_n)
    X, Y = np.meshgrid(grid, grid)
    grid_points = np.stack([X, Y], axis=-1)

    data_myopic = run_policy("myopic", cfg, grid_points)
    data_planning = run_policy("planning", cfg, grid_points)

    out_dir = Path("docs/figs")
    out_gif = out_dir / "toy2d_policy_comparison.gif"
    out_mp4 = out_dir / "toy2d_policy_comparison.mp4"
    out_png = out_dir / "toy2d_policy_comparison_sequence.png"
    out_pdf = out_dir / "toy2d_policy_comparison_sequence.pdf"
    out_txt = out_dir / "toy2d_policy_comparison_summary.txt"

    make_animation(data_myopic, data_planning, cfg, X, Y, out_gif, out_mp4)
    make_sequence_figure(data_myopic, data_planning, cfg, out_png, out_pdf)
    write_summary(data_myopic, data_planning, cfg, out_txt)

    print(f"[ok] wrote {out_gif}")
    if out_mp4.exists():
        print(f"[ok] wrote {out_mp4}")
    else:
        print("[ok] mp4 not written (ffmpeg not available)")
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_pdf}")
    print(f"[ok] wrote {out_txt}")


if __name__ == "__main__":
    main()
