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
    experiment_name: str = "circular_constant_omega_acquisition_qualitative_v1"

    grid_min: float = -2.6
    grid_max: float = 2.6
    grid_n: int = 141

    dt: float = 0.12
    omega: float = 1.4
    process_sigma: float = 0.14
    obs_sigma: float = 0.23
    obs_center_sigma: float = 0.80

    # acquisition weights
    w_dyn: float = 1.00
    w_obs: float = 0.45
    w_nov: float = 0.25
    control_cost: float = 0.30

    novelty_sigma: float = 0.45

    z0: tuple[float, float] = (0.95, 0.25)
    steps: int = 36
    planning_horizon: int = 6
    planning_candidates: int = 320
    u_max: float = 0.16
    random_seed: int = 11


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
    z_next = rot(cfg.omega * cfg.dt) @ z + u
    return np.clip(z_next, cfg.grid_min, cfg.grid_max)


def dynamics_info(z: np.ndarray, cfg: Config) -> np.ndarray:
    if z.ndim == 1:
        return np.array((cfg.dt ** 2) * float(np.dot(z, z)) / (cfg.process_sigma ** 2))
    r2 = np.sum(z * z, axis=-1)
    return (cfg.dt ** 2) * r2 / (cfg.process_sigma ** 2)


def observation_info(z: np.ndarray, cfg: Config) -> np.ndarray:
    if z.ndim == 1:
        r2 = float(np.dot(z, z))
        return np.array(np.exp(-r2 / (cfg.obs_center_sigma ** 2)) / (cfg.obs_sigma ** 2))
    r2 = np.sum(z * z, axis=-1)
    return np.exp(-r2 / (cfg.obs_center_sigma ** 2)) / (cfg.obs_sigma ** 2)


def novelty_term(z: np.ndarray, visited: np.ndarray, cfg: Config) -> float:
    if visited.size == 0:
        return 1.0
    d2 = np.sum((visited - z[None, :]) ** 2, axis=1)
    return float(1.0 - np.exp(-np.min(d2) / (2.0 * cfg.novelty_sigma ** 2)))


def action_set(cfg: Config) -> np.ndarray:
    vals = np.array([-cfg.u_max, 0.0, cfg.u_max], dtype=float)
    return np.array([[ux, uy] for ux in vals for uy in vals], dtype=float)


def score_terms(z_next: np.ndarray, u: np.ndarray, visited: np.ndarray, cfg: Config) -> dict[str, float]:
    i_dyn = float(dynamics_info(z_next, cfg))
    i_obs = float(observation_info(z_next, cfg))
    nov = novelty_term(z_next, visited, cfg)
    ctrl = cfg.control_cost * float(np.dot(u, u))
    acq = cfg.w_dyn * i_dyn + cfg.w_obs * i_obs + cfg.w_nov * nov - ctrl
    return {"I_dyn": i_dyn, "I_obs": i_obs, "novelty": nov, "control_penalty": ctrl, "A": acq}


def myopic_action(z: np.ndarray, visited: np.ndarray, cfg: Config) -> tuple[np.ndarray, dict[str, float]]:
    best_u = None
    best_terms = None
    best_score = -1e18
    for u in action_set(cfg):
        z1 = step_state(z, u, cfg)
        terms = score_terms(z1, u, visited, cfg)
        if terms["A"] > best_score:
            best_score = terms["A"]
            best_u = u
            best_terms = terms
    return best_u, best_terms


def planning_action(z: np.ndarray, visited: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    actions = action_set(cfg)
    best_u0 = actions[0]
    best_return = -1e18
    for _ in range(cfg.planning_candidates):
        idx = rng.integers(low=0, high=len(actions), size=cfg.planning_horizon)
        seq = actions[idx]
        z_roll = z.copy()
        visited_roll = visited.copy()
        ret = 0.0
        discount = 1.0
        for u in seq:
            z_roll = step_state(z_roll, u, cfg)
            terms = score_terms(z_roll, u, visited_roll, cfg)
            ret += discount * terms["A"]
            discount *= 0.96
            visited_roll = np.vstack([visited_roll, z_roll])
        if ret > best_return:
            best_return = ret
            best_u0 = seq[0].copy()
    return best_u0, float(best_return)


def rollout(cfg: Config, mode: str) -> dict[str, np.ndarray | float | list[float]]:
    assert mode in {"myopic", "planning"}
    rng = np.random.default_rng(cfg.random_seed)

    z = np.array(cfg.z0, dtype=float)
    visited = np.array([z.copy()])
    traj = [z.copy()]
    actions = []
    term_hist = {"I_dyn": [], "I_obs": [], "novelty": [], "control_penalty": [], "A": []}
    predicted_returns = []

    for _ in range(cfg.steps):
        if mode == "myopic":
            u, _ = myopic_action(z, visited, cfg)
            predicted_returns.append(np.nan)
        else:
            u, ret = planning_action(z, visited, cfg, rng)
            predicted_returns.append(ret)

        z1 = step_state(z, u, cfg)
        terms = score_terms(z1, u, visited, cfg)

        for k in term_hist:
            term_hist[k].append(terms[k])

        actions.append(u.copy())
        traj.append(z1.copy())
        visited = np.vstack([visited, z1])
        z = z1

    return {
        "mode": mode,
        "traj": np.asarray(traj),
        "actions": np.asarray(actions),
        "predicted_returns": np.asarray(predicted_returns),
        "I_dyn": np.asarray(term_hist["I_dyn"]),
        "I_obs": np.asarray(term_hist["I_obs"]),
        "novelty": np.asarray(term_hist["novelty"]),
        "control_penalty": np.asarray(term_hist["control_penalty"]),
        "A": np.asarray(term_hist["A"]),
        "cum_A": np.cumsum(np.asarray(term_hist["A"])),
    }


def make_maps(cfg: Config, visited_ref: np.ndarray) -> dict[str, np.ndarray]:
    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_n)
    X, Y = np.meshgrid(grid, grid)
    Z = np.stack([X, Y], axis=-1)
    I_dyn = dynamics_info(Z, cfg)
    I_obs = observation_info(Z, cfg)

    nov = np.zeros_like(I_dyn)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            nov[i, j] = novelty_term(Z[i, j], visited_ref, cfg)

    A = cfg.w_dyn * I_dyn + cfg.w_obs * I_obs + cfg.w_nov * nov
    return {"X": X, "Y": Y, "I_dyn": I_dyn, "I_obs": I_obs, "A": A}


def save_fig_a(cfg: Config, maps: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    titles = ["Dynamics info $I_{dyn}(z)$", "Observation info $I_{obs}(z)$", "Total acquisition $A(z)$ (u=0 ref)"]
    keys = ["I_dyn", "I_obs", "A"]
    cmaps = ["magma", "viridis", "cividis"]

    for ax, title, key, cmap in zip(axes, titles, keys, cmaps):
        im = ax.contourf(maps["X"], maps["Y"], maps[key], levels=28, cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.set_label(key)
        ax.set_title(title)
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_aspect("equal")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fig_b(cfg: Config, visited_ref: np.ndarray, out_path: Path) -> tuple[float, float]:
    grid = np.linspace(cfg.grid_min, cfg.grid_max, 23)
    X, Y = np.meshgrid(grid, grid)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    C = np.zeros_like(X)

    dom_counts = {"dyn": 0, "obs": 0, "nov": 0}

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = np.array([X[i, j], Y[i, j]], dtype=float)
            u, _ = myopic_action(z, visited_ref, cfg)
            z1 = step_state(z, u, cfg)
            t = score_terms(z1, u, visited_ref, cfg)
            dyn_c = cfg.w_dyn * t["I_dyn"]
            obs_c = cfg.w_obs * t["I_obs"]
            nov_c = cfg.w_nov * t["novelty"]
            dom = np.argmax([dyn_c, obs_c, nov_c])
            if dom == 0:
                dom_counts["dyn"] += 1
            elif dom == 1:
                dom_counts["obs"] += 1
            else:
                dom_counts["nov"] += 1

            U[i, j] = u[0]
            V[i, j] = u[1]
            C[i, j] = t["A"]

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4), constrained_layout=True)
    q = ax.quiver(X, Y, U, V, C, cmap="plasma", pivot="mid", scale=2.2 / cfg.u_max)
    cb = fig.colorbar(q, ax=ax)
    cb.set_label("myopic one-step acquisition")
    ax.scatter([cfg.z0[0]], [cfg.z0[1]], c="yellow", edgecolors="k", marker="s", s=54, label="reference start")
    ax.set_title("Best-action vector field (myopic)")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    tot = float(np.prod(X.shape))
    return dom_counts["dyn"] / tot, dom_counts["obs"] / tot


def save_fig_c(cfg: Config, pol: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)

    ax = axes[0]
    tr = pol["traj"]
    ax.plot(tr[:, 0], tr[:, 1], "-o", ms=2.8, lw=1.8, color="#1f77b4")
    ax.scatter(tr[0, 0], tr[0, 1], c="yellow", edgecolors="k", marker="s", s=62, label="start")
    ax.scatter(tr[-1, 0], tr[-1, 1], c="red", marker="*", s=90, label="end")
    ax.set_title("Planning rollout trajectory in latent space")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_aspect("equal")
    ax.set_xlim(cfg.grid_min, cfg.grid_max)
    ax.set_ylim(cfg.grid_min, cfg.grid_max)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    t = np.arange(cfg.steps)
    ax.plot(t, cfg.w_dyn * pol["I_dyn"], label="dyn-info term", lw=1.8)
    ax.plot(t, cfg.w_obs * pol["I_obs"], label="obs-info term", lw=1.8)
    ax.plot(t, cfg.w_nov * pol["novelty"], label="novelty term", lw=1.8)
    ax.plot(t, -pol["control_penalty"], label="-control penalty", lw=1.8)
    ax.plot(t, pol["A"], label="total A", color="k", lw=2.0, alpha=0.85)
    ax.set_title("Per-step contribution timeline (planning rollout)")
    ax.set_xlabel("step")
    ax.set_ylabel("contribution value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fig_d(cfg: Config, myopic: dict[str, np.ndarray], planning: dict[str, np.ndarray], out_path: Path) -> dict[str, float]:
    m_u0 = myopic["actions"][0]
    p_u0 = planning["actions"][0]
    first_action_angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(m_u0, p_u0) / (np.linalg.norm(m_u0) * np.linalg.norm(p_u0) + 1e-12), -1.0, 1.0))))

    N = min(15, cfg.steps)
    state_div = np.linalg.norm(myopic["traj"][1 : N + 1] - planning["traj"][1 : N + 1], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)

    ax = axes[0]
    ax.quiver([0, 0], [0, 0], [m_u0[0], p_u0[0]], [m_u0[1], p_u0[1]],
              angles="xy", scale_units="xy", scale=1, color=["#ff7f0e", "#1f77b4"])
    ax.set_xlim(-cfg.u_max * 1.3, cfg.u_max * 1.3)
    ax.set_ylim(-cfg.u_max * 1.3, cfg.u_max * 1.3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title(f"First action from same start\nangle diff={first_action_angle_deg:.1f}°")
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$u_2$")

    ax = axes[1]
    t = np.arange(cfg.steps)
    ax.plot(t, myopic["A"], label="myopic realized A", lw=1.8, color="#ff7f0e")
    ax.plot(t, planning["A"], label="planning realized A", lw=1.8, color="#1f77b4")
    ax.plot(t, planning["predicted_returns"] / max(cfg.planning_horizon, 1),
            label="planning forecast gain/H", lw=1.4, ls="--", color="#2ca02c")
    ax.set_title("Short-horizon gain diagnostic")
    ax.set_xlabel("step")
    ax.set_ylabel("gain")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(np.arange(1, N + 1), state_div, "-o", ms=3.2, lw=1.8, color="#9467bd")
    ax.set_title(f"Trajectory divergence over first {N} steps")
    ax.set_xlabel("step")
    ax.set_ylabel("||z_myopic - z_planning||")
    ax.grid(alpha=0.25)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "first_action_angle_deg": first_action_angle_deg,
        "mean_divergence_firstN": float(np.mean(state_div)),
        "max_divergence_firstN": float(np.max(state_div)),
    }


def compose_panel(fig_paths: list[Path], out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), constrained_layout=True)
    labels = ["A", "B", "C", "D"]
    for ax, p, lab in zip(axes.flatten(), fig_paths, labels):
        img = plt.imread(str(p))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(lab, loc="left", fontweight="bold")
    fig.suptitle("Acquisition qualitative paper-set: circular constant-omega with opposing information geometry", fontsize=13)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(cfg: Config, exp_dir: Path, metrics: dict) -> None:
    summary = [
        "Acquisition qualitative paper-set (circular constant-omega)",
        "=========================================================", 
        f"experiment_name: {cfg.experiment_name}",
        "",
        "Setup: dynamics information grows radially, observation information concentrates near center.",
        "Total acquisition uses weighted sum of dynamics-info, observation-info, novelty, minus control penalty.",
        "Fair comparison: myopic and planning start from the same initial latent state.",
        "",
        "Key qualitative findings:",
        f"- Opposing geometry verified: corr(radius, I_dyn)={metrics['corr_radius_I_dyn']:.4f}, corr(radius, I_obs)={metrics['corr_radius_I_obs']:.4f}",
        f"- Vector field dominance split: dyn-dominant={metrics['vectorfield_dyn_dominant_frac']:.3f}, obs-dominant={metrics['vectorfield_obs_dominant_frac']:.3f}",
        f"- First-step policy mismatch: angle={metrics['diagnostic']['first_action_angle_deg']:.2f} deg",
        f"- Early trajectory divergence (N={metrics['diagnostic']['N_divergence']}): mean={metrics['diagnostic']['mean_divergence_firstN']:.4f}, max={metrics['diagnostic']['max_divergence_firstN']:.4f}",
        f"- Cumulative acquisition: myopic={metrics['myopic']['cum_A_final']:.4f}, planning={metrics['planning']['cum_A_final']:.4f}",
    ]
    (exp_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    cfg = Config()
    exp_dir = prepare_experiment_dir(cfg)

    # same initialization for compared policies
    myopic = rollout(cfg, mode="myopic")
    planning = rollout(cfg, mode="planning")

    visited_ref = np.array([np.array(cfg.z0, dtype=float)])
    maps = make_maps(cfg, visited_ref)

    figA = exp_dir / "figA_acquisition_decomposition_maps.png"
    figB = exp_dir / "figB_best_action_vector_field.png"
    figC = exp_dir / "figC_trajectory_timeline_contributions.png"
    figD = exp_dir / "figD_myopic_vs_planning_diagnostic.png"
    panel_png = exp_dir / "panel_acquisition_qualitative_set.png"
    panel_pdf = exp_dir / "panel_acquisition_qualitative_set.pdf"

    save_fig_a(cfg, maps, figA)
    dyn_dom_frac, obs_dom_frac = save_fig_b(cfg, visited_ref, figB)
    save_fig_c(cfg, planning, figC)
    diag = save_fig_d(cfg, myopic, planning, figD)

    compose_panel([figA, figB, figC, figD], panel_png, panel_pdf)

    rr = np.linspace(0.0, 2.3, 110)
    rpts = np.stack([rr, np.zeros_like(rr)], axis=-1)
    corr_dyn = float(np.corrcoef(rr, dynamics_info(rpts, cfg))[0, 1])
    corr_obs = float(np.corrcoef(rr, observation_info(rpts, cfg))[0, 1])

    metrics = {
        "experiment_name": cfg.experiment_name,
        "figure_generation_mode": "individual_figures_first_then_compose",
        "definitions": {
            "I_dyn": "I_dyn(z)=dt^2||z||^2/sigma_d^2",
            "I_obs": "I_obs(z)=exp(-||z||^2/sigma_c^2)/sigma_o^2",
            "A": "A=w_dyn*I_dyn + w_obs*I_obs + w_nov*novelty - control_cost*||u||^2",
            "novelty": "1-exp(-min_j ||z-z_j||^2/(2*sigma_n^2)) where z_j are visited states"
        },
        "corr_radius_I_dyn": corr_dyn,
        "corr_radius_I_obs": corr_obs,
        "vectorfield_dyn_dominant_frac": float(dyn_dom_frac),
        "vectorfield_obs_dominant_frac": float(obs_dom_frac),
        "same_initialization": bool(np.allclose(myopic["traj"][0], planning["traj"][0])),
        "myopic": {
            "first_action": myopic["actions"][0].tolist(),
            "cum_A_final": float(myopic["cum_A"][-1]),
            "mean_A": float(np.mean(myopic["A"])),
        },
        "planning": {
            "first_action": planning["actions"][0].tolist(),
            "cum_A_final": float(planning["cum_A"][-1]),
            "mean_A": float(np.mean(planning["A"])),
            "planning_horizon": cfg.planning_horizon,
            "planning_candidates": cfg.planning_candidates,
        },
        "diagnostic": {
            **diag,
            "N_divergence": int(min(15, cfg.steps)),
        },
        "outputs": {
            "figA": str(figA),
            "figB": str(figB),
            "figC": str(figC),
            "figD": str(figD),
            "panel_png": str(panel_png),
            "panel_pdf": str(panel_pdf),
            "summary": str(exp_dir / "summary.txt"),
            "metrics": str(exp_dir / "metrics.json"),
        },
    }

    write_outputs(cfg, exp_dir, metrics)

    print(f"[ok] experiment_dir: {exp_dir}")
    for p in [figA, figB, figC, figD, panel_png, panel_pdf, exp_dir / "summary.txt", exp_dir / "metrics.json"]:
        print(f"[ok] wrote {p}")


if __name__ == "__main__":
    main()
