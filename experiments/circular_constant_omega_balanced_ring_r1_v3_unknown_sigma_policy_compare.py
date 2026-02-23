import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


EXPERIMENT_NAME = "circular_constant_omega_balanced_ring_r1_v3_unknown_sigma_policy_compare"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figs" / "experiments" / EXPERIMENT_NAME
ARCHIVE_DIR = ROOT / "docs" / "figs" / "experiments" / "archive"


PARAMS = {
    "dt": 0.05,
    "T": 520,
    "omega": 1.15,
    "r_min": 1e-3,
    "r_max": 3.0,
    "u_max": 0.075,
    "k_gain": 0.21,
    "smoothness": 0.26,
    "control_noise_std": 0.0035,
    "seed": 17,
    # dynamics information term
    "a_d": 0.75,
    "r_d": 1.00,
    "C_d": 1.8,
    # observation model / unknown parameter
    "g": 1.0,
    "sigma_c_true": 1.35,
    "obs_noise_std": 0.045,
    "sigma_grid_min": 0.45,
    "sigma_grid_max": 2.20,
    "sigma_grid_n": 160,
    # combined policy weights
    "w_dyn_combined": 1.0,
    "w_obs_combined": 0.85,
}

POLICIES = {
    "combined": {
        "w_dyn": PARAMS["w_dyn_combined"],
        "w_obs": PARAMS["w_obs_combined"],
        "label": r"combined (I_comb=w_d I_dyn+w_o I_obs)",
        "color": "tab:blue",
    },
    "obs_only": {
        "w_dyn": 0.0,
        "w_obs": 1.0,
        "label": r"obs-info-only (I_obs)",
        "color": "tab:green",
    },
    "dyn_only": {
        "w_dyn": 1.0,
        "w_obs": 0.0,
        "label": r"dyn-info-only (I_dyn)",
        "color": "tab:red",
    },
}


def maybe_archive_previous(out_dir: Path, archive_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = archive_dir / f"{ts}-{out_dir.name}"
        shutil.move(str(out_dir), str(dst))
    out_dir.mkdir(parents=True, exist_ok=True)


def I_dyn(r):
    p = PARAMS
    return p["C_d"] - p["a_d"] * (r - p["r_d"]) ** 2


def I_obs(r, sigma_c):
    eps = 1e-8
    s = np.maximum(sigma_c, eps)
    return (r**2 / s**4) * np.exp(-(r**2) / (s**2))


def I_comb(r, sigma_c, w_dyn, w_obs):
    return w_dyn * I_dyn(r) + w_obs * I_obs(r, sigma_c)


def dI_dr_numeric(r, sigma_c, w_dyn, w_obs):
    h = 1e-3
    r1 = np.clip(r + h, PARAMS["r_min"], PARAMS["r_max"])
    r0 = np.clip(r - h, PARAMS["r_min"], PARAMS["r_max"])
    return (I_comb(r1, sigma_c, w_dyn, w_obs) - I_comb(r0, sigma_c, w_dyn, w_obs)) / np.maximum(r1 - r0, 1e-8)


def observation_mean(r, sigma_c):
    p = PARAMS
    return p["g"] * np.exp(-(r**2) / (2.0 * sigma_c**2))


def radial_argmax(sigma_c, r_grid, w_dyn, w_obs):
    vals = I_comb(r_grid, sigma_c, w_dyn, w_obs)
    i = int(np.argmax(vals))
    return float(r_grid[i]), float(vals[i])


def generate_shared_noise():
    p = PARAMS
    rng = np.random.default_rng(p["seed"])
    return {
        "obs_noise": rng.normal(0.0, p["obs_noise_std"], size=p["T"]),
        "ctrl_noise": rng.normal(0.0, p["control_noise_std"], size=p["T"]),
    }


def simulate_policy(policy_name: str, shared_noise):
    p = PARAMS
    pol = POLICIES[policy_name]
    T = p["T"]
    dt = p["dt"]

    sigma_grid = np.linspace(p["sigma_grid_min"], p["sigma_grid_max"], p["sigma_grid_n"])
    posterior = np.ones_like(sigma_grid) / sigma_grid.size

    rs = np.zeros(T)
    thetas = np.zeros(T)
    xs = np.zeros(T)
    ys = np.zeros(T)
    us = np.zeros(T)
    y_obs = np.zeros(T)
    sigma_hat = np.zeros(T)
    sigma_std = np.zeros(T)
    r_star_hat = np.zeros(T)
    i_dyn = np.zeros(T)
    i_obs = np.zeros(T)
    i_comb = np.zeros(T)

    # fair initialization across policies
    rs[0] = 2.15
    thetas[0] = 0.22
    prev_u = 0.0

    r_grid = np.linspace(p["r_min"], p["r_max"], 2400)

    for t in range(T):
        y_t = observation_mean(rs[t], p["sigma_c_true"]) + shared_noise["obs_noise"][t]
        y_obs[t] = y_t

        mu_grid = observation_mean(rs[t], sigma_grid)
        ll = -0.5 * ((y_t - mu_grid) / p["obs_noise_std"]) ** 2
        ll -= np.max(ll)
        post_unnorm = posterior * np.exp(ll)
        posterior = post_unnorm / np.sum(post_unnorm)

        sigma_hat[t] = float(np.sum(sigma_grid * posterior))
        sigma_std[t] = float(np.sqrt(np.sum(((sigma_grid - sigma_hat[t]) ** 2) * posterior)))

        r_star_hat[t], _ = radial_argmax(sigma_hat[t], r_grid, pol["w_dyn"], pol["w_obs"])

        i_dyn[t] = float(I_dyn(rs[t]))
        i_obs[t] = float(I_obs(rs[t], sigma_hat[t]))
        i_comb[t] = float(I_comb(rs[t], sigma_hat[t], pol["w_dyn"], pol["w_obs"]))

        if t < T - 1:
            g = dI_dr_numeric(rs[t], sigma_hat[t], pol["w_dyn"], pol["w_obs"])
            u_raw = p["k_gain"] * g - p["smoothness"] * prev_u
            u = float(np.clip(u_raw, -p["u_max"], p["u_max"]))
            u += float(shared_noise["ctrl_noise"][t])

            r_next = float(np.clip(rs[t] + u, p["r_min"], p["r_max"]))
            th_next = float((thetas[t] + p["omega"] * dt) % (2 * math.pi))
            rs[t + 1] = r_next
            thetas[t + 1] = th_next
            us[t] = u
            prev_u = u

    us[-1] = us[-2]
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    return {
        "policy": policy_name,
        "r": rs,
        "theta": thetas,
        "x": xs,
        "y": ys,
        "u": us,
        "y_obs": y_obs,
        "sigma_hat": sigma_hat,
        "sigma_std": sigma_std,
        "r_star_hat": r_star_hat,
        "i_dyn": i_dyn,
        "i_obs": i_obs,
        "i_comb": i_comb,
        "init": {
            "r0": float(rs[0]),
            "theta0": float(thetas[0]),
            "posterior": "uniform over sigma_grid",
        },
    }


def make_policy_trajectory_fig(out_dir: Path, sims):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), constrained_layout=True)
    ang = np.linspace(0, 2 * np.pi, 600)

    for ax, (name, sim) in zip(axes, sims.items()):
        pol = POLICIES[name]
        ax.plot(np.cos(ang), np.sin(ang), "--", color="black", lw=1.2, alpha=0.6, label="ring r=1")
        ax.plot(sim["x"], sim["y"], color=pol["color"], lw=1.8, label=pol["label"])
        ax.scatter(sim["x"][0], sim["y"][0], c="lime", s=35, edgecolor="black", zorder=4)
        ax.scatter(sim["x"][-1], sim["y"][-1], c="red", s=35, edgecolor="black", zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, loc="upper right")

    fig.savefig(out_dir / "fig01_policy_trajectories.png", dpi=190)
    plt.close(fig)


def make_radius_timeseries_fig(out_dir: Path, sims):
    p = PARAMS
    t = np.arange(p["T"]) * p["dt"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for name, sim in sims.items():
        pol = POLICIES[name]
        ax.plot(t, sim["r"], color=pol["color"], lw=2.0, label=name)
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="ring r=1")
    ax.set_xlabel("time")
    ax.set_ylabel("radius r")
    ax.set_title("Radius evolution across policies")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_radius_timeseries.png", dpi=190)
    plt.close(fig)


def make_cumulative_terms_fig(out_dir: Path, sims):
    p = PARAMS
    dt = p["dt"]
    labels = list(sims.keys())
    cum_dyn = [float(np.sum(sims[k]["i_dyn"]) * dt) for k in labels]
    cum_obs = [float(np.sum(sims[k]["i_obs"]) * dt) for k in labels]
    cum_comb = [float(np.sum(sims[k]["i_comb"]) * dt) for k in labels]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    ax.bar(x - w, cum_dyn, width=w, label=r"int I_dyn dt", color="tab:orange")
    ax.bar(x, cum_obs, width=w, label=r"int I_obs dt", color="tab:green")
    ax.bar(x + w, cum_comb, width=w, label=r"int I_policy dt", color="tab:blue")
    ax.set_xticks(x, labels)
    ax.set_ylabel("cumulative value")
    ax.set_title("Cumulative information terms by policy")
    ax.grid(alpha=0.2, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_cumulative_terms.png", dpi=190)
    plt.close(fig)


def make_sigma_inference_fig(out_dir: Path, sims):
    p = PARAMS
    t = np.arange(p["T"]) * p["dt"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for name, sim in sims.items():
        pol = POLICIES[name]
        ax.plot(t, sim["sigma_hat"], color=pol["color"], lw=2.0, label=name)
    ax.axhline(p["sigma_c_true"], color="crimson", ls="--", lw=1.5, label="true sigma_c")
    ax.set_xlabel("time")
    ax.set_ylabel("sigma_hat")
    ax.set_title("Unknown-parameter inference (same init belief across policies)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_sigma_inference.png", dpi=190)
    plt.close(fig)


def compose_final_panel(out_dir: Path):
    files = [
        "fig01_policy_trajectories.png",
        "fig02_radius_timeseries.png",
        "fig03_cumulative_terms.png",
        "fig04_sigma_inference.png",
    ]
    imgs = [plt.imread(out_dir / f) for f in files]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    axes = axes.ravel()
    titles = [
        "(A) policy trajectories",
        "(B) radius timeseries",
        "(C) cumulative information terms",
        "(D) sigma inference",
    ]
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(EXPERIMENT_NAME, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "final_panel.png", dpi=200)
    fig.savefig(out_dir / "final_panel.pdf")
    plt.close(fig)


def make_policy_animation(out_dir: Path, sims):
    p = PARAMS
    frame_step = 4
    frames = list(range(0, p["T"], frame_step))
    ang = np.linspace(0, 2 * np.pi, 500)

    x = np.linspace(-2.4, 2.4, 170)
    y = np.linspace(-2.4, 2.4, 170)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), constrained_layout=True)
    artists = {}

    for ax, (name, sim) in zip(axes, sims.items()):
        pol = POLICIES[name]
        sigma0 = float(sim["sigma_hat"][0])
        Z = I_comb(R, sigma0, pol["w_dyn"], pol["w_obs"])
        im = ax.imshow(
            Z,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap="magma",
            alpha=0.92,
            vmin=float(np.min(Z)),
            vmax=float(np.max(Z)),
        )
        ax.plot(np.cos(ang), np.sin(ang), "--", color="cyan", lw=1.2, alpha=0.85)
        line, = ax.plot([], [], color=pol["color"], lw=2.0)
        point = ax.scatter([], [], c="white", s=36, edgecolor="black", zorder=5)
        title = ax.set_title(f"{name} | objective: {pol['label']}")
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        artists[name] = {"im": im, "line": line, "point": point, "title": title}

    def update(frame_idx):
        t = frames[frame_idx]
        for name, sim in sims.items():
            pol = POLICIES[name]
            a = artists[name]
            Zt = I_comb(R, float(sim["sigma_hat"][t]), pol["w_dyn"], pol["w_obs"])
            a["im"].set_data(Zt)
            a["line"].set_data(sim["x"][: t + 1], sim["y"][: t + 1])
            a["point"].set_offsets(np.array([[sim["x"][t], sim["y"][t]]]))
            a["title"].set_text(f"{name} | t={t*p['dt']:.2f}s | objective: {pol['label']}")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=55, blit=False)

    gif_path = out_dir / "trajectory_policy_comparison.gif"
    mp4_path = out_dir / "trajectory_policy_comparison.mp4"

    anim.save(gif_path, writer=PillowWriter(fps=15))

    try:
        anim.save(mp4_path, writer=FFMpegWriter(fps=15, bitrate=1600))
    except Exception:
        pass

    plt.close(fig)
    return gif_path, mp4_path


def compute_metrics(sims, gif_path, mp4_path):
    p = PARAMS
    dt = p["dt"]

    def policy_metrics(sim):
        return {
            "final_radius": float(sim["r"][-1]),
            "mean_radius": float(np.mean(sim["r"])),
            "mean_radius_last_100": float(np.mean(sim["r"][-100:])),
            "cumulative_I_dyn": float(np.sum(sim["i_dyn"]) * dt),
            "cumulative_I_obs": float(np.sum(sim["i_obs"]) * dt),
            "cumulative_I_comb_policy": float(np.sum(sim["i_comb"]) * dt),
            "sigma_hat_initial": float(sim["sigma_hat"][0]),
            "sigma_hat_final": float(sim["sigma_hat"][-1]),
        }

    pm = {k: policy_metrics(v) for k, v in sims.items()}

    d_obs_dyn = np.sqrt((sims["obs_only"]["x"] - sims["dyn_only"]["x"]) ** 2 + (sims["obs_only"]["y"] - sims["dyn_only"]["y"]) ** 2)
    d_obs_comb = np.sqrt((sims["obs_only"]["x"] - sims["combined"]["x"]) ** 2 + (sims["obs_only"]["y"] - sims["combined"]["y"]) ** 2)
    d_dyn_comb = np.sqrt((sims["dyn_only"]["x"] - sims["combined"]["x"]) ** 2 + (sims["dyn_only"]["y"] - sims["combined"]["y"]) ** 2)

    return {
        "experiment_name": EXPERIMENT_NAME,
        "definitions": {
            "I_dyn": "I_dyn(r)=C_d-a_d(r-r_d)^2",
            "I_obs": "I_obs(r;sigma_c)=(r^2/sigma_c^4)*exp(-r^2/sigma_c^2)",
            "I_comb": "I_comb(r;sigma_hat;policy)=w_dyn*I_dyn(r)+w_obs*I_obs(r;sigma_hat)",
        },
        "fair_initialization": {
            "same_initial_state": True,
            "same_initial_belief": True,
            "initial_state": sims["combined"]["init"],
            "shared_noise_sequences": True,
        },
        "unknown_parameter": {
            "name": "sigma_c",
            "true": float(p["sigma_c_true"]),
            "inference": "online Bayesian grid posterior update",
        },
        "policy_metrics": pm,
        "trajectory_divergence": {
            "mean_distance_obs_vs_dyn": float(np.mean(d_obs_dyn)),
            "final_distance_obs_vs_dyn": float(d_obs_dyn[-1]),
            "mean_distance_obs_vs_combined": float(np.mean(d_obs_comb)),
            "mean_distance_dyn_vs_combined": float(np.mean(d_dyn_comb)),
        },
        "animations": {
            "gif": str(gif_path.name),
            "mp4": str(mp4_path.name),
            "mp4_exists": bool(mp4_path.exists()),
        },
        "outputs": {
            "fig01": "fig01_policy_trajectories.png",
            "fig02": "fig02_radius_timeseries.png",
            "fig03": "fig03_cumulative_terms.png",
            "fig04": "fig04_sigma_inference.png",
            "final_panel_png": "final_panel.png",
            "final_panel_pdf": "final_panel.pdf",
        },
        "params": PARAMS,
    }


def write_summary(out_dir: Path, metrics):
    pm = metrics["policy_metrics"]
    td = metrics["trajectory_divergence"]

    text = f"""Experiment: {EXPERIMENT_NAME}

Definitions
- I_dyn: {metrics['definitions']['I_dyn']}
- I_obs: {metrics['definitions']['I_obs']}
- I_comb: {metrics['definitions']['I_comb']}

Policies compared
- obs-only: optimize only I_obs (w_dyn=0, w_obs=1)
- dyn-only: optimize only I_dyn (w_dyn=1, w_obs=0)
- combined: optimize w_dyn*I_dyn + w_obs*I_obs

Unknown parameter mechanism retained
- sigma_c inferred online from noisy observations with Bayesian grid posterior.

Fair initialization
- Same initial state and initial posterior belief for all policies.
- Shared observation/control noise sequences used for fair comparison.

Quantitative comparison
- obs-only final radius: {pm['obs_only']['final_radius']:.4f}
- dyn-only final radius: {pm['dyn_only']['final_radius']:.4f}
- combined final radius: {pm['combined']['final_radius']:.4f}
- obs-only mean radius: {pm['obs_only']['mean_radius']:.4f}
- dyn-only mean radius: {pm['dyn_only']['mean_radius']:.4f}
- combined mean radius: {pm['combined']['mean_radius']:.4f}
- cumulative I_obs (obs-only): {pm['obs_only']['cumulative_I_obs']:.4f}
- cumulative I_dyn (dyn-only): {pm['dyn_only']['cumulative_I_dyn']:.4f}
- cumulative I_comb_policy (combined): {pm['combined']['cumulative_I_comb_policy']:.4f}
- trajectory divergence mean distance (obs vs dyn): {td['mean_distance_obs_vs_dyn']:.4f}
- trajectory divergence final distance (obs vs dyn): {td['final_distance_obs_vs_dyn']:.4f}

Animation outputs
- GIF: {metrics['animations']['gif']}
- MP4: {metrics['animations']['mp4']} (exists={metrics['animations']['mp4_exists']})

Representative figure
- final_panel.png
"""
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")


def main():
    maybe_archive_previous(OUT_DIR, ARCHIVE_DIR)

    shared_noise = generate_shared_noise()
    sims = {name: simulate_policy(name, shared_noise) for name in ["obs_only", "dyn_only", "combined"]}

    # individual figures first
    make_policy_trajectory_fig(OUT_DIR, sims)
    make_radius_timeseries_fig(OUT_DIR, sims)
    make_cumulative_terms_fig(OUT_DIR, sims)
    make_sigma_inference_fig(OUT_DIR, sims)

    # animation deliverables
    gif_path, mp4_path = make_policy_animation(OUT_DIR, sims)

    # final panel composition
    compose_final_panel(OUT_DIR)

    metrics = compute_metrics(sims, gif_path, mp4_path)
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_summary(OUT_DIR, metrics)
    print(f"Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
