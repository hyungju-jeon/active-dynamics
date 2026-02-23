#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Config:
    output_root: str = "docs/figs/experiments"
    experiment_name: str = "circular_constant_omega_info_v1"

    grid_min: float = -2.6
    grid_max: float = 2.6
    grid_n: int = 201

    dt: float = 0.12
    omega: float = 1.4
    process_sigma: float = 0.14
    obs_sigma: float = 0.23

    # observation: y_t = g * c^T z_t + eps, eps ~ N(0, obs_sigma^2)
    obs_direction: tuple[float, float] = (0.92, 0.38)

    # qualitative radius invariance demo
    radii_demo: tuple[float, float, float] = (0.7, 1.3, 2.0)
    demo_steps: int = 130

    # active-learning policy comparison
    steps: int = 90
    z0: tuple[float, float] = (0.45, 0.35)
    u_max: float = 0.16


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
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                old_sig = old.get("signature")
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


def rot(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def step_state(z: np.ndarray, u: np.ndarray, cfg: Config) -> np.ndarray:
    R = rot(cfg.omega * cfg.dt)
    z_next = R @ z + u
    return np.clip(z_next, cfg.grid_min, cfg.grid_max)


def dynamics_info(z: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Fisher information for omega in transition model
    z_{t+1} = R(omega*dt) z_t + w, w ~ N(0, sigma_d^2 I)
    I_dyn(z) = (1/sigma_d^2) * ||d mu / d omega||^2 = dt^2 * ||z||^2 / sigma_d^2
    """
    if z.ndim == 1:
        return np.array((cfg.dt ** 2) * float(np.dot(z, z)) / (cfg.process_sigma ** 2))
    r2 = np.sum(z * z, axis=-1)
    return (cfg.dt ** 2) * r2 / (cfg.process_sigma ** 2)


def observation_info(z: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Fisher information for observation gain g in
    y_t = g * c^T z_t + v, v ~ N(0, sigma_o^2)
    I_obs(z) = (c^T z)^2 / sigma_o^2
    """
    c = np.array(cfg.obs_direction, dtype=float)
    c = c / np.linalg.norm(c)
    if z.ndim == 1:
        return np.array((float(c @ z) ** 2) / (cfg.obs_sigma ** 2))
    proj = z @ c
    return (proj ** 2) / (cfg.obs_sigma ** 2)


def estimate_omega_from_traj(traj: np.ndarray, dt: float) -> float:
    ang = np.unwrap(np.arctan2(traj[:, 1], traj[:, 0]))
    d_ang = np.diff(ang) / dt
    return float(np.mean(d_ang))


def make_radius_demo(cfg: Config) -> dict[str, np.ndarray]:
    theta0 = 0.35
    trajectories = []
    est_omegas = []
    for r in cfg.radii_demo:
        z = np.array([r * np.cos(theta0), r * np.sin(theta0)], dtype=float)
        tr = [z.copy()]
        for _ in range(cfg.demo_steps):
            z = step_state(z, np.zeros(2), cfg)
            tr.append(z.copy())
        tr = np.asarray(tr)
        trajectories.append(tr)
        est_omegas.append(estimate_omega_from_traj(tr, cfg.dt))
    return {
        "trajectories": np.asarray(trajectories),
        "radii": np.asarray(cfg.radii_demo, dtype=float),
        "estimated_omegas": np.asarray(est_omegas, dtype=float),
    }


def action_set(cfg: Config) -> np.ndarray:
    vals = [-cfg.u_max, 0.0, cfg.u_max]
    return np.array([[ux, uy] for ux in vals for uy in vals], dtype=float)


def run_policy(cfg: Config, mode: str) -> dict[str, np.ndarray]:
    assert mode in {"dynamics_greedy", "observation_greedy"}
    z = np.array(cfg.z0, dtype=float)
    traj = [z.copy()]
    dyn_hist = [float(dynamics_info(z, cfg))]
    obs_hist = [float(observation_info(z, cfg))]
    act_hist = []

    actions = action_set(cfg)
    for _ in range(cfg.steps):
        best_score = -1e18
        best_u = actions[0]
        for u in actions:
            z1 = step_state(z, u, cfg)
            if mode == "dynamics_greedy":
                score = float(dynamics_info(z1, cfg)) - 0.25 * float(np.dot(u, u))
            else:
                score = float(observation_info(z1, cfg)) - 0.25 * float(np.dot(u, u))
            if score > best_score:
                best_score = score
                best_u = u
        z = step_state(z, best_u, cfg)
        act_hist.append(best_u.copy())
        traj.append(z.copy())
        dyn_hist.append(float(dynamics_info(z, cfg)))
        obs_hist.append(float(observation_info(z, cfg)))

    return {
        "mode": mode,
        "traj": np.asarray(traj),
        "actions": np.asarray(act_hist),
        "dyn_info": np.asarray(dyn_hist),
        "obs_info": np.asarray(obs_hist),
        "cum_dyn": np.cumsum(np.asarray(dyn_hist)),
        "cum_obs": np.cumsum(np.asarray(obs_hist)),
    }


def _save_panel1_radius_invariance(cfg: Config, demo: dict[str, np.ndarray], out_path: Path) -> None:
    tr = demo["trajectories"]
    radii = demo["radii"]
    est = demo["estimated_omegas"]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), constrained_layout=True)

    ax = axes[0]
    cols = ["#1f77b4", "#2ca02c", "#d62728"]
    for i in range(len(radii)):
        ax.plot(tr[i, :, 0], tr[i, :, 1], color=cols[i], lw=2, label=f"r={radii[i]:.1f}")
        ax.scatter(tr[i, 0, 0], tr[i, 0, 1], color=cols[i], s=26)
    ax.set_title("Circular latent trajectories (same \\omega)")
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")
    ax.set_aspect("equal")
    ax.set_xlim(cfg.grid_min, cfg.grid_max)
    ax.set_ylim(cfg.grid_min, cfg.grid_max)
    ax.grid(alpha=0.2)
    ax.legend()

    ax = axes[1]
    ax.plot(radii, est, "o-", color="#6a3d9a", lw=2, ms=6, label="estimated")
    ax.axhline(cfg.omega, color="black", ls="--", lw=1.5, label="true \\omega")
    ax.set_title("Estimated angular velocity vs radius")
    ax.set_xlabel("radius")
    ax.set_ylabel("\\hat{\\omega}")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_panel2_dynamics_info(cfg: Config, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, out_path: Path) -> None:
    I_dyn = dynamics_info(Z, cfg)
    r = np.linspace(0.0, cfg.grid_max, 120)
    I_r = (cfg.dt ** 2) * r ** 2 / (cfg.process_sigma ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), constrained_layout=True)

    ax = axes[0]
    im = ax.contourf(X, Y, I_dyn, levels=26, cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.88, label="I_{dyn}(z)")
    ax.set_title("Dynamics information map")
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(r, I_r, color="#e41a1c", lw=2)
    ax.set_title("Dynamics info grows with radius")
    ax.set_xlabel("radius r")
    ax.set_ylabel("I_{dyn}(r)=dt^2r^2/\\sigma_d^2")
    ax.grid(alpha=0.25)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_panel3_observation_info(cfg: Config, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, out_path: Path) -> None:
    I_obs = observation_info(Z, cfg)
    c = np.array(cfg.obs_direction, dtype=float)
    c = c / np.linalg.norm(c)
    ang = np.linspace(0, 2 * np.pi, 240)
    r0 = 1.6
    ring = np.stack([r0 * np.cos(ang), r0 * np.sin(ang)], axis=-1)
    ring_info = observation_info(ring, cfg)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), constrained_layout=True)

    ax = axes[0]
    im = ax.contourf(X, Y, I_obs, levels=26, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.88, label="I_{obs}(z)")
    ax.arrow(0, 0, 1.25 * c[0], 1.25 * c[1], width=0.03, color="white", ec="black", length_includes_head=True)
    ax.set_title("Observation information map")
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(ang, ring_info, color="#377eb8", lw=2)
    ax.set_title(f"Observation info on ring r={r0:.1f}")
    ax.set_xlabel("angle")
    ax.set_ylabel("I_{obs}(r,\\theta)")
    ax.grid(alpha=0.25)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_panel4_policy_timecourse(cfg: Config, dyn_pol: dict[str, np.ndarray], obs_pol: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    td = dyn_pol["traj"]
    to = obs_pol["traj"]
    ax.plot(td[:, 0], td[:, 1], color="#d62728", lw=2, label="dynamics-greedy")
    ax.plot(to[:, 0], to[:, 1], color="#1f77b4", lw=2, label="observation-greedy")
    ax.scatter(td[0, 0], td[0, 1], c="yellow", edgecolors="k", s=46, marker="s", label="same start")
    ax.set_title("Policy trajectories (fair init)")
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")
    ax.set_aspect("equal")
    ax.set_xlim(cfg.grid_min, cfg.grid_max)
    ax.set_ylim(cfg.grid_min, cfg.grid_max)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    t = np.arange(cfg.steps + 1)
    ax = axes[0, 1]
    ax.plot(t, dyn_pol["cum_dyn"], color="#d62728", lw=2, label="dyn-greedy")
    ax.plot(t, obs_pol["cum_dyn"], color="#1f77b4", lw=2, label="obs-greedy")
    ax.set_title("Cumulative dynamics information")
    ax.set_xlabel("step")
    ax.set_ylabel("\\sum_{\\tau\\le t} I_{dyn}(z_\\tau)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, dyn_pol["cum_obs"], color="#d62728", lw=2, label="dyn-greedy")
    ax.plot(t, obs_pol["cum_obs"], color="#1f77b4", lw=2, label="obs-greedy")
    ax.set_title("Cumulative observation information")
    ax.set_xlabel("step")
    ax.set_ylabel("\\sum_{\\tau\\le t} I_{obs}(z_\\tau)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, dyn_pol["dyn_info"], color="#fb8072", lw=1.8, label="I_{dyn} dyn-greedy")
    ax.plot(t, dyn_pol["obs_info"], color="#fdb462", lw=1.8, label="I_{obs} dyn-greedy")
    ax.plot(t, obs_pol["dyn_info"], color="#80b1d3", lw=1.8, label="I_{dyn} obs-greedy")
    ax.plot(t, obs_pol["obs_info"], color="#8dd3c7", lw=1.8, label="I_{obs} obs-greedy")
    ax.set_title("Per-step information signals")
    ax.set_xlabel("step")
    ax.set_ylabel("information")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compose_panel_figure(panel_pngs: list[Path], out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), constrained_layout=True)
    axes = axes.flatten()
    for i, p in enumerate(panel_pngs):
        img = plt.imread(str(p))
        axes[i].imshow(img)
        axes[i].axis("off")
    fig.suptitle("Circular latent dynamics with radius-invariant angular velocity and dual information sources", fontsize=13)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(cfg: Config, demo: dict[str, np.ndarray], dyn_pol: dict[str, np.ndarray], obs_pol: dict[str, np.ndarray], out_txt: Path, metrics_path: Path) -> None:
    est = demo["estimated_omegas"]
    radii = demo["radii"]
    omega_err = np.abs(est - cfg.omega)

    metrics = {
        "experiment_name": cfg.experiment_name,
        "omega_true": cfg.omega,
        "radii_demo": radii.tolist(),
        "estimated_omega_per_radius": est.tolist(),
        "max_abs_omega_error": float(np.max(omega_err)),
        "std_estimated_omega": float(np.std(est)),
        "dynamics_info_definition": "I_dyn(z) = dt^2 * ||z||^2 / sigma_d^2 (Fisher info for omega in transition model)",
        "observation_info_definition": "I_obs(z) = (c^T z)^2 / sigma_o^2 (Fisher info for gain g in observation model)",
        "policy_init_same": bool(np.allclose(dyn_pol["traj"][0], obs_pol["traj"][0])),
        "dynamics_greedy": {
            "final_cum_dyn_info": float(dyn_pol["cum_dyn"][-1]),
            "final_cum_obs_info": float(dyn_pol["cum_obs"][-1]),
        },
        "observation_greedy": {
            "final_cum_dyn_info": float(obs_pol["cum_dyn"][-1]),
            "final_cum_obs_info": float(obs_pol["cum_obs"][-1]),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "Circular constant-omega experiment summary",
        "========================================",
        f"experiment_name: {cfg.experiment_name}",
        "",
        "Latent dynamics:",
        "z_{t+1} = R(omega*dt) z_t + u_t (+ bounded clipping)",
        "R(theta) = [[cos theta, -sin theta], [sin theta, cos theta]]",
        "",
        "Information definitions:",
        "Dynamics information (for omega): I_dyn(z) = dt^2 ||z||^2 / sigma_d^2",
        "Observation information (for gain g in y=g c^T z + v): I_obs(z) = (c^T z)^2 / sigma_o^2",
        "",
        "Radius-invariant angular velocity check:",
    ]
    for r, w in zip(radii, est):
        lines.append(f"  radius={r:.3f}, estimated_omega={w:.6f}, abs_err={abs(w-cfg.omega):.6f}")

    lines += [
        f"max_abs_omega_error: {np.max(omega_err):.6f}",
        f"std_estimated_omega: {np.std(est):.6f}",
        "",
        "Policy comparison (same initialization):",
        f"  same_start: {bool(np.allclose(dyn_pol['traj'][0], obs_pol['traj'][0]))}",
        f"  dynamics-greedy final cumulative I_dyn: {dyn_pol['cum_dyn'][-1]:.4f}",
        f"  observation-greedy final cumulative I_dyn: {obs_pol['cum_dyn'][-1]:.4f}",
        f"  dynamics-greedy final cumulative I_obs: {dyn_pol['cum_obs'][-1]:.4f}",
        f"  observation-greedy final cumulative I_obs: {obs_pol['cum_obs'][-1]:.4f}",
        "",
        "Qualitative observations:",
        "- Rotation speed estimate remains nearly constant across radii.",
        "- Dynamics information increases radially (larger radius gives stronger omega identifiability).",
        "- Observation information is anisotropic and aligned with observation direction c.",
        "- Policies emphasize different information sources over time.",
    ]

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = Config()
    exp_dir = prepare_experiment_dir(cfg)

    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_n)
    X, Y = np.meshgrid(grid, grid)
    Z = np.stack([X, Y], axis=-1)

    demo = make_radius_demo(cfg)
    dyn_pol = run_policy(cfg, "dynamics_greedy")
    obs_pol = run_policy(cfg, "observation_greedy")

    panel1 = exp_dir / "panel1_radius_invariance.png"
    panel2 = exp_dir / "panel2_dynamics_info.png"
    panel3 = exp_dir / "panel3_observation_info.png"
    panel4 = exp_dir / "panel4_policy_time_evolution.png"

    rep_png = exp_dir / "representative_circular_constant_omega_info.png"
    rep_pdf = exp_dir / "representative_circular_constant_omega_info.pdf"
    summary_txt = exp_dir / "summary.txt"
    metrics_json = exp_dir / "metrics.json"

    _save_panel1_radius_invariance(cfg, demo, panel1)
    _save_panel2_dynamics_info(cfg, X, Y, Z, panel2)
    _save_panel3_observation_info(cfg, X, Y, Z, panel3)
    _save_panel4_policy_timecourse(cfg, dyn_pol, obs_pol, panel4)

    compose_panel_figure([panel1, panel2, panel3, panel4], rep_png, rep_pdf)
    write_summary(cfg, demo, dyn_pol, obs_pol, summary_txt, metrics_json)

    print(f"[ok] experiment_dir: {exp_dir}")
    print(f"[ok] wrote {panel1}")
    print(f"[ok] wrote {panel2}")
    print(f"[ok] wrote {panel3}")
    print(f"[ok] wrote {panel4}")
    print(f"[ok] wrote {rep_png}")
    print(f"[ok] wrote {rep_pdf}")
    print(f"[ok] wrote {summary_txt}")
    print(f"[ok] wrote {metrics_json}")


if __name__ == "__main__":
    main()
