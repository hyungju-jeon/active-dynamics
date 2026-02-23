#!/usr/bin/env python3
"""
Generate a qualitative 2D active-learning illustration for Poisson log-linear sensing.

Outputs:
- docs/figs/toy2d_poisson_active_learning.pdf
- docs/figs/toy2d_poisson_active_learning.png
- docs/figs/toy2d_poisson_active_learning_summary.txt

The setup is intentionally simple and designed for explanatory figures, not benchmarking.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


@dataclass
class ToyConfig:
    grid_min: float = -2.3
    grid_max: float = 2.3
    grid_n: int = 181
    dt: float = 0.22
    steps: int = 20
    move_step: float = 0.42

    # Poisson log-linear observation model: y_t ~ Poisson(lambda_t), lambda_t = exp(c^T z_t + b)
    obs_c: tuple[float, float] = (0.90, 0.28)
    obs_b: float = -0.60

    # Nominal dynamics parameter estimate used only for speed visualization
    theta_hat: tuple[float, float] = (0.90, 1.10)

    # Initial parameter precision (inverse covariance)
    # small precision means high uncertainty
    precision0_diag: tuple[float, float] = (0.06, 0.003)

    # Start state for the active-learning trajectory
    z0: tuple[float, float] = (-1.8, -1.3)

    # Which step to visualize for "later" acquisition map
    show_step: int = 16


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def phi1(z: np.ndarray) -> np.ndarray:
    z1 = z[..., 0]
    z2 = z[..., 1]
    return 0.7 + 0.9 * sigmoid(1.7 * (z1 - 0.15)) + 0.12 * np.tanh(0.8 * z2)


def phi2(z: np.ndarray) -> np.ndarray:
    z1 = z[..., 0]
    z2 = z[..., 1]
    return 0.45 + 0.45 * sigmoid(1.5 * (z2 + 0.1)) + 1.4 * sigmoid(-1.8 * (z1 + 0.1))


def poisson_rate(z: np.ndarray, c: np.ndarray, b: float) -> np.ndarray:
    if z.ndim == 1:
        return np.exp(c @ z + b)
    return np.exp(z @ c + b)


def nominal_dynamics(z: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    """
    Nominal drift used only to visualize where dynamics are fast.
    """
    p1 = phi1(z)
    p2 = phi2(z)
    z1 = z[..., 0]
    z2 = z[..., 1]
    f1 = theta_hat[0] * p1 - 0.30 * z2
    f2 = theta_hat[1] * p2 + 0.24 * z1
    return np.stack([f1, f2], axis=-1)


def sensitivity_matrix(z: np.ndarray, dt: float) -> np.ndarray:
    """
    One-step sensitivity approximation S_t = d z_{t+1}^- / d theta.
    We use a diagonal approximation for clarity.
    """
    if z.ndim == 1:
        return dt * np.array([[phi1(z), 0.0], [0.0, phi2(z)]])

    out = np.zeros(z.shape[:-1] + (2, 2), dtype=float)
    out[..., 0, 0] = dt * phi1(z)
    out[..., 1, 1] = dt * phi2(z)
    return out


def fisher_information(z: np.ndarray, cfg: ToyConfig) -> np.ndarray:
    """
    I_theta(z) = S(z)^T H_z(z) S(z),
    with scalar Poisson log-linear observation approximation
    H_z(z) = lambda(z) c c^T.
    """
    c = np.array(cfg.obs_c, dtype=float)
    lam = poisson_rate(z, c, cfg.obs_b)

    if z.ndim == 1:
        S = sensitivity_matrix(z, cfg.dt)
        v = S.T @ c
        return lam * np.outer(v, v)

    S = sensitivity_matrix(z, cfg.dt)
    v1 = S[..., 0, 0] * c[0]
    v2 = S[..., 1, 1] * c[1]

    out = np.zeros(z.shape[:-1] + (2, 2), dtype=float)
    out[..., 0, 0] = lam * v1 * v1
    out[..., 0, 1] = lam * v1 * v2
    out[..., 1, 0] = out[..., 0, 1]
    out[..., 1, 1] = lam * v2 * v2
    return out


def logdet2x2(M: np.ndarray) -> np.ndarray:
    sign, value = np.linalg.slogdet(M)
    if np.any(sign <= 0):
        raise ValueError("Encountered non-positive-definite matrix in logdet.")
    return value


def acquisition_map(precision: np.ndarray, grid_points: np.ndarray, cfg: ToyConfig) -> np.ndarray:
    fim = fisher_information(grid_points, cfg)
    return logdet2x2(precision + fim) - logdet2x2(precision)


def run_active_learning(cfg: ToyConfig) -> dict[str, np.ndarray]:
    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_n)
    X, Y = np.meshgrid(grid, grid)
    grid_points = np.stack([X, Y], axis=-1)

    precision = np.diag(np.array(cfg.precision0_diag, dtype=float))
    z = np.array(cfg.z0, dtype=float)

    trajectory = [z.copy()]
    precisions = [precision.copy()]
    argmax_points = []
    maps = []
    selected_acq = []

    for _t in range(cfg.steps):
        A = acquisition_map(precision, grid_points, cfg)
        maps.append(A)

        ij = np.unravel_index(np.argmax(A), A.shape)
        target = np.array([X[ij], Y[ij]])
        argmax_points.append(target)

        direction = target - z
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            z = z + cfg.move_step * direction / norm
        z = np.clip(z, cfg.grid_min, cfg.grid_max)

        gain = logdet2x2(precision + fisher_information(z, cfg)) - logdet2x2(precision)
        selected_acq.append(gain)

        precision = precision + fisher_information(z, cfg)

        trajectory.append(z.copy())
        precisions.append(precision.copy())

    return {
        "X": X,
        "Y": Y,
        "grid_points": grid_points,
        "maps": np.asarray(maps),
        "trajectory": np.asarray(trajectory),
        "precisions": np.asarray(precisions),
        "argmax_points": np.asarray(argmax_points),
        "selected_acq": np.asarray(selected_acq),
    }


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
    )


def plot_results(results: dict[str, np.ndarray], cfg: ToyConfig, out_pdf: Path, out_png: Path) -> None:
    X = results["X"]
    Y = results["Y"]
    Z = results["grid_points"]
    maps = results["maps"]
    traj = results["trajectory"]
    argmax_points = results["argmax_points"]
    precisions = results["precisions"]

    show_t = min(cfg.show_step, maps.shape[0] - 1)

    c_vec = np.array(cfg.obs_c, dtype=float)
    theta_hat = np.array(cfg.theta_hat, dtype=float)

    lam = poisson_rate(Z, c_vec, cfg.obs_b)
    speed = np.linalg.norm(nominal_dynamics(Z, theta_hat), axis=-1)

    A0 = maps[0]
    At = maps[show_t]
    dA = At - A0

    idx0 = np.unravel_index(np.argmax(A0), A0.shape)
    idxt = np.unravel_index(np.argmax(At), At.shape)
    p0 = np.array([X[idx0], Y[idx0]])
    pt = np.array([X[idxt], Y[idxt]])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6), constrained_layout=True)

    # A) firing-rate geometry
    ax = axes[0, 0]
    im = ax.contourf(X, Y, np.log10(lam + 1e-12), levels=24, cmap="viridis")
    ax.contour(X, Y, np.log10(lam + 1e-12), colors="white", levels=8, linewidths=0.5, alpha=0.6)
    fig.colorbar(im, ax=ax, shrink=0.88, label=r"$\log_{10}\lambda(\mathbf{z})$")
    ax.set_title("Poisson log-linear firing-rate field")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    add_panel_label(ax, "A")

    # B) speed of nominal dynamics
    ax = axes[0, 1]
    im = ax.contourf(X, Y, speed, levels=24, cmap="cividis")
    fig.colorbar(im, ax=ax, shrink=0.88, label=r"$\|f(\mathbf{z};\hat{\theta})\|_2$")
    ax.set_title("Nominal dynamics speed map")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    add_panel_label(ax, "B")

    # C) acquisition at t=0
    ax = axes[0, 2]
    im = ax.contourf(X, Y, A0, levels=24, cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.88, label=r"$\mathcal{J}_t(\mathbf{z})$")
    ax.scatter(*traj[0], c="cyan", edgecolor="black", s=55, marker="s", label="start")
    ax.scatter(*p0, c="yellow", edgecolor="black", s=90, marker="*", label=r"argmax $\mathcal{J}_0$")
    ax.set_title(r"Acquisition map at $t=0$")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.legend(loc="lower right", frameon=True)
    add_panel_label(ax, "C")

    # D) acquisition at later step
    ax = axes[1, 0]
    im = ax.contourf(X, Y, At, levels=24, cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.88, label=rf"$\mathcal{{J}}_{{{show_t}}}(\mathbf{{z}})$")
    ax.plot(traj[: show_t + 1, 0], traj[: show_t + 1, 1], "w-", lw=1.4, alpha=0.9)
    ax.scatter(*traj[show_t], c="lime", edgecolor="black", s=65, marker="o", label=rf"state at $t={show_t}$")
    ax.scatter(*pt, c="yellow", edgecolor="black", s=90, marker="*", label=rf"argmax $\mathcal{{J}}_{{{show_t}}}$")
    ax.set_title(rf"Acquisition map at $t={show_t}$")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.legend(loc="lower right", frameon=True)
    add_panel_label(ax, "D")

    # E) trajectory and moving argmax
    ax = axes[1, 1]
    im = ax.contourf(X, Y, np.log10(lam + 1e-12), levels=18, cmap="Greys", alpha=0.55)
    fig.colorbar(im, ax=ax, shrink=0.88, label=r"background: $\log_{10}\lambda(\mathbf{z})$")
    t_idx = np.arange(traj.shape[0])
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=t_idx, cmap="plasma", s=36, edgecolor="none", label="agent state")
    fig.colorbar(sc, ax=ax, shrink=0.88, label="time step")
    ax.plot(traj[:, 0], traj[:, 1], "k-", lw=1.0, alpha=0.8)
    ax.plot(argmax_points[:, 0], argmax_points[:, 1], "--", color="#00BFC4", lw=1.6, label="moving argmax")
    ax.scatter(*traj[0], c="white", edgecolor="black", s=60, marker="s")
    ax.set_title("Active-learning trajectory and moving target")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.legend(loc="lower right", frameon=True)
    add_panel_label(ax, "E")

    # F) acquisition change map
    ax = axes[1, 2]
    vmax = np.max(np.abs(dA))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.contourf(X, Y, dA, levels=25, cmap="RdBu_r", norm=norm)
    fig.colorbar(im, ax=ax, shrink=0.88, label=rf"$\mathcal{{J}}_{{{show_t}}}(\mathbf{{z}})-\mathcal{{J}}_0(\mathbf{{z}})$")
    ax.scatter(*p0, c="black", s=70, marker="*", label=r"argmax $\mathcal{J}_0$")
    ax.scatter(*pt, c="white", edgecolor="black", s=70, marker="*", label=rf"argmax $\mathcal{{J}}_{{{show_t}}}$")
    ax.set_title("How acquisition changes with posterior updates")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.legend(loc="lower right", frameon=True)
    add_panel_label(ax, "F")

    for ax in axes.ravel():
        ax.set_xlim(cfg.grid_min, cfg.grid_max)
        ax.set_ylim(cfg.grid_min, cfg.grid_max)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Toy 2D Active Learning with Poisson Log-Linear Observation and Asymmetric Information Geometry",
        fontsize=12,
        y=1.01,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: dict[str, np.ndarray], cfg: ToyConfig, out_txt: Path) -> None:
    X = results["X"]
    Y = results["Y"]
    Z = results["grid_points"]
    maps = results["maps"]
    traj = results["trajectory"]
    precisions = results["precisions"]

    show_t = min(cfg.show_step, maps.shape[0] - 1)

    c_vec = np.array(cfg.obs_c, dtype=float)
    theta_hat = np.array(cfg.theta_hat, dtype=float)
    lam = poisson_rate(Z, c_vec, cfg.obs_b)
    speed = np.linalg.norm(nominal_dynamics(Z, theta_hat), axis=-1)

    A0 = maps[0]
    At = maps[show_t]

    corr_lambda = np.corrcoef(A0.ravel(), lam.ravel())[0, 1]
    corr_speed = np.corrcoef(A0.ravel(), speed.ravel())[0, 1]

    q = np.quantile(A0, 0.95)
    mask = A0 >= q
    lambda_ratio = lam[mask].mean() / lam.mean()
    speed_ratio = speed[mask].mean() / speed.mean()

    idx0 = np.unravel_index(np.argmax(A0), A0.shape)
    idxt = np.unravel_index(np.argmax(At), At.shape)
    p0 = np.array([X[idx0], Y[idx0]])
    pt = np.array([X[idxt], Y[idxt]])

    lines = [
        "Toy 2D Poisson active-learning summary",
        "====================================",
        f"show_step = {show_t}",
        f"argmax J_0      = ({p0[0]:.3f}, {p0[1]:.3f})",
        f"argmax J_show   = ({pt[0]:.3f}, {pt[1]:.3f})",
        f"start state     = ({traj[0,0]:.3f}, {traj[0,1]:.3f})",
        f"state@show_step = ({traj[show_t,0]:.3f}, {traj[show_t,1]:.3f})",
        f"final state     = ({traj[-1,0]:.3f}, {traj[-1,1]:.3f})",
        "",
        "Qualitative informativeness diagnostics (from J_0 map):",
        f"corr(J_0, lambda) = {corr_lambda:.4f}",
        f"corr(J_0, speed)  = {corr_speed:.4f}",
        f"mean(lambda | top 5% J_0) / mean(lambda) = {lambda_ratio:.3f}",
        f"mean(speed  | top 5% J_0) / mean(speed)  = {speed_ratio:.3f}",
        "",
        "Precision diagonal trajectory:",
        "t, Lambda11, Lambda22",
    ]

    for t, P in enumerate(precisions):
        lines.append(f"{t}, {P[0,0]:.6f}, {P[1,1]:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = ToyConfig()
    results = run_active_learning(cfg)

    out_dir = Path("docs/figs")
    out_pdf = out_dir / "toy2d_poisson_active_learning.pdf"
    out_png = out_dir / "toy2d_poisson_active_learning.png"
    out_txt = out_dir / "toy2d_poisson_active_learning_summary.txt"

    plot_results(results, cfg, out_pdf, out_png)
    write_summary(results, cfg, out_txt)

    print(f"[ok] wrote {out_pdf}")
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_txt}")


if __name__ == "__main__":
    main()
