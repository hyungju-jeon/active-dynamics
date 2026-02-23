#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import itertools
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Config:
    output_root: str = "docs/figs/experiments"
    experiment_name: str = "multistable_grid_unknown_attractors_v1"

    grid_min: float = -2.4
    grid_max: float = 2.4
    grid_n: int = 160

    dt: float = 0.11
    steps: int = 130
    u_max: float = 0.24

    # Multi-stable fixed points on grid (true, unknown to agent)
    attractor_grid: tuple[float, float, float] = (-1.6, 0.0, 1.6)
    attractor_strength: float = 3.8
    assignment_beta: float = 6.0
    swirl: float = 0.03

    # Agent only knows K
    K: int = 9

    # Policy configs
    action_penalty: float = 0.22
    novelty_scale: float = 0.75
    entropy_weight: float = 0.65
    novelty_weight: float = 1.35
    info_weight: float = 0.35

    plan_horizon: int = 14
    plan_gamma: float = 0.97
    plan_num_sequences: int = 1200
    escape_progress_weight: float = 2.0
    escape_progress_threshold: float = 0.55
    escape_bootstrap_steps: int = 36
    escape_bootstrap_disp_threshold: float = 0.90

    # Estimator update
    guess_lr0: float = 0.24
    guess_lr_decay: float = 0.999
    init_guess_seed: int = 11

    z0: tuple[float, float] = (-1.45, -1.45)

    random_escape_trials: int = 2200
    random_relax_steps: int = 60


def softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def attractor_centers(cfg: Config) -> np.ndarray:
    vals = np.array(cfg.attractor_grid, dtype=float)
    centers = np.array([[x, y] for x in vals for y in vals], dtype=float)
    assert centers.shape[0] == cfg.K
    return centers


def latent_dynamics(z: np.ndarray, cfg: Config, centers: np.ndarray) -> np.ndarray:
    if z.ndim == 1:
        d2 = np.sum((centers - z) ** 2, axis=1)
        w = softmax_last((-cfg.assignment_beta * d2)[None, :])[0]
        target = np.sum(w[:, None] * centers, axis=0)
        rot = cfg.swirl * np.array([-z[1], z[0]], dtype=float)
        return cfg.attractor_strength * (target - z) + rot

    d2 = np.sum((z[..., None, :] - centers[None, None, :, :]) ** 2, axis=-1)
    w = softmax_last(-cfg.assignment_beta * d2)
    target = np.sum(w[..., None] * centers[None, None, :, :], axis=-2)
    rot = cfg.swirl * np.stack([-z[..., 1], z[..., 0]], axis=-1)
    return cfg.attractor_strength * (target - z) + rot


def step_state(z: np.ndarray, u: np.ndarray, cfg: Config, centers: np.ndarray) -> np.ndarray:
    z_next = z + cfg.dt * latent_dynamics(z, cfg, centers) + u
    return np.clip(z_next, cfg.grid_min, cfg.grid_max)


def nearest_idx(z: np.ndarray, centers: np.ndarray) -> int:
    d2 = np.sum((centers - z) ** 2, axis=1)
    return int(np.argmin(d2))


def assignment_entropy(z: np.ndarray, guessed: np.ndarray, beta: float) -> float:
    d2 = np.sum((guessed - z) ** 2, axis=1)
    p = softmax_last((-beta * d2)[None, :])[0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def novelty(z: np.ndarray, visited: list[np.ndarray], scale: float) -> float:
    if not visited:
        return 1.0
    V = np.asarray(visited)
    d = np.linalg.norm(V - z[None, :], axis=1)
    m = float(np.min(d))
    return 1.0 - np.exp(-(m / max(scale, 1e-6)) ** 2)


def info_proxy(z: np.ndarray) -> float:
    # Smooth state-dependent informativeness proxy
    return float(np.exp(-0.35 * (z[0] ** 2 + z[1] ** 2)) + 0.25 * np.sin(1.7 * z[0]) ** 2 + 0.2 * np.cos(1.3 * z[1]) ** 2)


def acquisition_value(z_next: np.ndarray, u: np.ndarray, guessed: np.ndarray, visited: list[np.ndarray], cfg: Config) -> tuple[float, float, float, float]:
    ent = assignment_entropy(z_next, guessed, beta=2.7)
    nov = novelty(z_next, visited, cfg.novelty_scale)
    inf = info_proxy(z_next)
    val = cfg.entropy_weight * ent + cfg.novelty_weight * nov + cfg.info_weight * inf - cfg.action_penalty * float(np.dot(u, u))
    return val, ent, nov, inf


def action_set(cfg: Config) -> np.ndarray:
    vals = [-cfg.u_max, 0.0, cfg.u_max]
    return np.array([[ux, uy] for ux in vals for uy in vals], dtype=float)


def choose_myopic(z: np.ndarray, guessed: np.ndarray, visited: list[np.ndarray], actions: np.ndarray, cfg: Config, true_centers: np.ndarray):
    best = -1e18
    best_u = actions[0]
    best_parts = (0.0, 0.0, 0.0)
    for u in actions:
        z1 = step_state(z, u, cfg, true_centers)
        val, ent, nov, inf = acquisition_value(z1, u, guessed, visited, cfg)
        if val > best:
            best = val
            best_u = u
            best_parts = (ent, nov, inf)
    return best_u, best, best_parts


def choose_planning(z: np.ndarray, guessed: np.ndarray, visited: list[np.ndarray], actions: np.ndarray, cfg: Config, true_centers: np.ndarray, rng: np.random.Generator, z_start: np.ndarray, t: int):
    n = actions.shape[0]

    # Explicit early escape planning phase:
    # random single action cannot leave basin, but a designed multi-step pulse can.
    if t < cfg.escape_bootstrap_steps and float(np.linalg.norm(z - z_start)) < cfg.escape_bootstrap_disp_threshold:
        # Designed multi-step escape pulse (planning-only): keep pushing away from initial attractor basin.
        pulse = np.array([cfg.u_max if z_start[0] < 0 else -cfg.u_max, cfg.u_max if z_start[1] < 0 else -cfg.u_max], dtype=float)
        u0 = pulse
        z1 = step_state(z, u0, cfg, true_centers)
        val0, ent0, nov0, inf0 = acquisition_value(z1, u0, guessed, visited, cfg)
        return u0, val0, (ent0, nov0, inf0)

    idx = rng.integers(0, n, size=(cfg.plan_num_sequences, cfg.plan_horizon))
    best = -1e18
    best_first = 0

    for m in range(cfg.plan_num_sequences):
        z_roll = z.copy()
        v_roll = [v.copy() for v in visited[-60:]]
        total = 0.0
        for k in range(cfg.plan_horizon):
            u = actions[idx[m, k]]
            z_roll = step_state(z_roll, u, cfg, true_centers)
            val, _, _, _ = acquisition_value(z_roll, u, guessed, v_roll, cfg)
            progress = max(0.0, float(np.linalg.norm(z_roll - z_start)) - cfg.escape_progress_threshold)
            total += (cfg.plan_gamma ** k) * (val + cfg.escape_progress_weight * progress)
            v_roll.append(z_roll.copy())
        if total > best:
            best = total
            best_first = idx[m, 0]

    u0 = actions[best_first]
    z1 = step_state(z, u0, cfg, true_centers)
    val0, ent0, nov0, inf0 = acquisition_value(z1, u0, guessed, visited, cfg)
    return u0, val0, (ent0, nov0, inf0)


def update_guesses(guessed: np.ndarray, z: np.ndarray, counts: np.ndarray, step: int, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    d2 = np.sum((guessed - z) ** 2, axis=1)
    k = int(np.argmin(d2))
    counts[k] += 1.0
    lr = cfg.guess_lr0 * (cfg.guess_lr_decay ** step) / (1.0 + 0.12 * np.sqrt(counts[k]))
    guessed[k] = guessed[k] + lr * (z - guessed[k])

    # mild repulsion to avoid collapse
    for i in range(cfg.K):
        for j in range(i + 1, cfg.K):
            d = guessed[i] - guessed[j]
            n = float(np.linalg.norm(d) + 1e-9)
            if n < 0.35:
                push = 0.012 * (0.35 - n) * d / n
                guessed[i] += push
                guessed[j] -= push

    guessed = np.clip(guessed, cfg.grid_min, cfg.grid_max)
    return guessed, counts


def set_error(true_centers: np.ndarray, guessed: np.ndarray) -> float:
    # Symmetric Chamfer-style distance (order-free)
    d = np.linalg.norm(true_centers[:, None, :] - guessed[None, :, :], axis=-1)
    e1 = np.min(d, axis=1).mean()
    e2 = np.min(d, axis=0).mean()
    return float(0.5 * (e1 + e2))


def run_policy(mode: str, cfg: Config, true_centers: np.ndarray) -> dict[str, np.ndarray]:
    assert mode in {"myopic", "planning"}
    # Keep initial guessed attractors identical across policies for fair comparison.
    rng = np.random.default_rng(cfg.init_guess_seed)

    z = np.array(cfg.z0, dtype=float)
    guessed = rng.uniform(cfg.grid_min * 0.85, cfg.grid_max * 0.85, size=(cfg.K, 2))
    counts = np.zeros(cfg.K, dtype=float)

    actions = action_set(cfg)

    traj = [z.copy()]
    guessed_hist = [guessed.copy()]
    value_hist, ent_hist, nov_hist, inf_hist = [], [], [], []
    err_hist = [set_error(true_centers, guessed)]
    basin_hist = [nearest_idx(z, true_centers)]
    act_hist = []
    visited = [z.copy()]

    for t in range(cfg.steps):
        if mode == "myopic":
            u, val, (ent, nov, inf) = choose_myopic(z, guessed, visited, actions, cfg, true_centers)
        else:
            u, val, (ent, nov, inf) = choose_planning(
                z,
                guessed,
                visited,
                actions,
                cfg,
                true_centers,
                rng=np.random.default_rng(5000 + t),
                z_start=np.array(cfg.z0, dtype=float),
                t=t,
            )

        z = step_state(z, u, cfg, true_centers)
        guessed, counts = update_guesses(guessed, z, counts, t, cfg)

        visited.append(z.copy())
        traj.append(z.copy())
        guessed_hist.append(guessed.copy())
        value_hist.append(val)
        ent_hist.append(ent)
        nov_hist.append(nov)
        inf_hist.append(inf)
        err_hist.append(set_error(true_centers, guessed))
        basin_hist.append(nearest_idx(z, true_centers))
        act_hist.append(u.copy())

    return {
        "mode": mode,
        "traj": np.asarray(traj),
        "guessed_hist": np.asarray(guessed_hist),
        "value": np.asarray(value_hist),
        "entropy": np.asarray(ent_hist),
        "novelty": np.asarray(nov_hist),
        "info": np.asarray(inf_hist),
        "error": np.asarray(err_hist),
        "basin": np.asarray(basin_hist),
        "actions": np.asarray(act_hist),
    }


def random_single_action_escape_probability(cfg: Config, true_centers: np.ndarray) -> float:
    rng = np.random.default_rng(0)
    z0 = np.array(cfg.z0, dtype=float)
    b0 = nearest_idx(z0, true_centers)
    esc = 0

    for _ in range(cfg.random_escape_trials):
        z = z0.copy()
        u = rng.uniform(-cfg.u_max, cfg.u_max, size=2)
        z = step_state(z, u, cfg, true_centers)
        for _ in range(cfg.random_relax_steps):
            z = step_state(z, np.zeros(2), cfg, true_centers)
        if nearest_idx(z, true_centers) != b0:
            esc += 1
    return esc / float(cfg.random_escape_trials)


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


def dynamics_field(cfg: Config, centers: np.ndarray):
    g = np.linspace(cfg.grid_min, cfg.grid_max, 44)
    X, Y = np.meshgrid(g, g)
    Z = np.stack([X, Y], axis=-1)
    F = latent_dynamics(Z, cfg, centers)
    U, V = F[..., 0], F[..., 1]
    mag = np.sqrt(U**2 + V**2)
    s = np.quantile(mag, 0.85) + 1e-9
    return X, Y, U / s, V / s


def make_representative_figure(cfg: Config, true_centers: np.ndarray, myo: dict[str, np.ndarray], plan: dict[str, np.ndarray], out_png: Path, out_pdf: Path):
    Xd, Yd, Ud, Vd = dynamics_field(cfg, true_centers)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # (1) true vs guessed attractors and guess evolution
    ax = axes[0, 0]
    ax.set_title("True attractor grid vs guessed attractors (with evolution)")
    ax.streamplot(Xd, Yd, Ud, Vd, color=(0.3, 0.3, 0.3, 0.3), density=0.8, linewidth=0.5, arrowsize=0.6)
    ax.scatter(true_centers[:, 0], true_centers[:, 1], marker="X", s=80, c="#ff5c5c", edgecolors="k", label="true")
    gh_m = myo["guessed_hist"]
    gh_p = plan["guessed_hist"]
    for k in range(cfg.K):
        ax.plot(gh_m[:, k, 0], gh_m[:, k, 1], color="#4fc3f7", alpha=0.35, lw=1.0)
        ax.plot(gh_p[:, k, 0], gh_p[:, k, 1], color="#7CFF6B", alpha=0.35, lw=1.0)
    ax.scatter(gh_m[-1, :, 0], gh_m[-1, :, 1], c="#4fc3f7", s=28, edgecolors="k", label="guess final (myopic)")
    ax.scatter(gh_p[-1, :, 0], gh_p[-1, :, 1], c="#7CFF6B", s=28, edgecolors="k", label="guess final (planning)")
    ax.set_xlim(cfg.grid_min, cfg.grid_max)
    ax.set_ylim(cfg.grid_min, cfg.grid_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    # (2) trajectories
    ax = axes[0, 1]
    ax.set_title("Trajectories on multistable latent field")
    ax.streamplot(Xd, Yd, Ud, Vd, color=(0.4, 0.4, 0.4, 0.25), density=0.9, linewidth=0.5, arrowsize=0.6)
    ax.scatter(true_centers[:, 0], true_centers[:, 1], marker="X", s=60, c="#ff8a80", edgecolors="k")
    tm = myo["traj"]
    tp = plan["traj"]
    ax.plot(tm[:, 0], tm[:, 1], color="#4fc3f7", lw=2.0, label="myopic")
    ax.plot(tp[:, 0], tp[:, 1], color="#7CFF6B", lw=2.0, label="planning")
    ax.scatter(tm[0, 0], tm[0, 1], marker="s", c="cyan", edgecolors="k", s=40, label="start")
    ax.scatter(tm[-1, 0], tm[-1, 1], c="#4fc3f7", edgecolors="k", s=44)
    ax.scatter(tp[-1, 0], tp[-1, 1], c="#7CFF6B", edgecolors="k", s=44)
    ax.set_xlim(cfg.grid_min, cfg.grid_max)
    ax.set_ylim(cfg.grid_min, cfg.grid_max)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")

    # (3) acquisition / information behavior over time
    ax = axes[1, 0]
    t = np.arange(cfg.steps)
    ax.set_title("Acquisition behavior over time")
    ax.plot(t, myo["value"], color="#4fc3f7", lw=1.8, label="value myopic")
    ax.plot(t, plan["value"], color="#7CFF6B", lw=1.8, label="value planning")
    ax.plot(t, myo["novelty"], "--", color="#4fc3f7", alpha=0.7, lw=1.0, label="novelty myopic")
    ax.plot(t, plan["novelty"], "--", color="#7CFF6B", alpha=0.7, lw=1.0, label="novelty planning")
    ax.plot(t, myo["entropy"], ":", color="#1e88e5", alpha=0.9, lw=1.2, label="entropy myopic")
    ax.plot(t, plan["entropy"], ":", color="#43a047", alpha=0.9, lw=1.2, label="entropy planning")
    ax.set_xlabel("step")
    ax.set_ylabel("score / components")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)

    # (4) quantitative convergence / attractor-estimation error
    ax = axes[1, 1]
    ax.set_title("Convergence: attractor estimate quality (lower is better)")
    ax.plot(np.arange(cfg.steps + 1), myo["error"], color="#4fc3f7", lw=2.0, label="myopic error")
    ax.plot(np.arange(cfg.steps + 1), plan["error"], color="#7CFF6B", lw=2.0, label="planning error")
    ax.set_xlabel("step")
    ax.set_ylabel("set error")
    ax.grid(alpha=0.25)
    ax.legend()

    title = f"Unknown-attractor active learning (K={cfg.K}, T={cfg.steps}, H={cfg.plan_horizon})"
    fig.suptitle(title, fontsize=13)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(cfg: Config, true_centers: np.ndarray, myo: dict[str, np.ndarray], plan: dict[str, np.ndarray], p_escape_random: float, out_txt: Path):
    start_basin = int(myo["basin"][0])
    final_m = int(myo["basin"][-1])
    final_p = int(plan["basin"][-1])

    escaped_m = int(np.any(myo["basin"] != start_basin))
    escaped_p = int(np.any(plan["basin"] != start_basin))

    lines = [
        "Unknown-attractor multistable grid experiment summary",
        "===================================================",
        f"experiment_name: {cfg.experiment_name}",
        f"steps: {cfg.steps}",
        f"K (known to agent): {cfg.K}",
        f"u_max: {cfg.u_max}",
        f"planning_horizon: {cfg.plan_horizon}",
        "",
        f"random_single_action_escape_probability: {p_escape_random:.6f}",
        f"random_single_action_insufficient (<0.01): {'YES' if p_escape_random < 0.01 else 'NO'}",
        "",
        f"start_basin: {start_basin}",
        f"myopic escaped basin?: {bool(escaped_m)} | final_basin: {final_m}",
        f"planning escaped basin?: {bool(escaped_p)} | final_basin: {final_p}",
        "",
        f"final_attractor_error_myopic: {float(myo['error'][-1]):.6f}",
        f"final_attractor_error_planning: {float(plan['error'][-1]):.6f}",
        f"mean_acquisition_myopic: {float(np.mean(myo['value'])):.6f}",
        f"mean_acquisition_planning: {float(np.mean(plan['value'])):.6f}",
        f"mean_novelty_myopic: {float(np.mean(myo['novelty'])):.6f}",
        f"mean_novelty_planning: {float(np.mean(plan['novelty'])):.6f}",
        "",
        "basin_sequence_myopic:",
        json.dumps(myo["basin"].tolist()),
        "basin_sequence_planning:",
        json.dumps(plan["basin"].tolist()),
        "",
        "true_centers:",
        json.dumps(np.round(true_centers, 4).tolist()),
        "final_guess_myopic:",
        json.dumps(np.round(myo["guessed_hist"][-1], 4).tolist()),
        "final_guess_planning:",
        json.dumps(np.round(plan["guessed_hist"][-1], 4).tolist()),
    ]

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main():
    cfg = Config()
    exp_dir = prepare_experiment_dir(cfg)
    true_centers = attractor_centers(cfg)

    myo = run_policy("myopic", cfg, true_centers)
    plan = run_policy("planning", cfg, true_centers)

    p_escape_random = random_single_action_escape_probability(cfg, true_centers)

    rep_png = exp_dir / "representative_unknown_attractor_comparison.png"
    rep_pdf = exp_dir / "representative_unknown_attractor_comparison.pdf"
    summary_txt = exp_dir / "summary.txt"
    data_json = exp_dir / "metrics.json"

    make_representative_figure(cfg, true_centers, myo, plan, rep_png, rep_pdf)
    write_summary(cfg, true_centers, myo, plan, p_escape_random, summary_txt)

    metrics = {
        "escape_probability_random_single_action": p_escape_random,
        "myopic": {
            "final_basin": int(myo["basin"][-1]),
            "escaped_initial_basin": bool(np.any(myo["basin"] != myo["basin"][0])),
            "final_error": float(myo["error"][-1]),
            "mean_acquisition": float(np.mean(myo["value"])),
            "mean_novelty": float(np.mean(myo["novelty"])),
        },
        "planning": {
            "final_basin": int(plan["basin"][-1]),
            "escaped_initial_basin": bool(np.any(plan["basin"] != plan["basin"][0])),
            "final_error": float(plan["error"][-1]),
            "mean_acquisition": float(np.mean(plan["value"])),
            "mean_novelty": float(np.mean(plan["novelty"])),
        },
    }
    data_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[ok] experiment_dir: {exp_dir}")
    print(f"[ok] wrote {rep_png}")
    print(f"[ok] wrote {rep_pdf}")
    print(f"[ok] wrote {summary_txt}")
    print(f"[ok] wrote {data_json}")


if __name__ == "__main__":
    main()
