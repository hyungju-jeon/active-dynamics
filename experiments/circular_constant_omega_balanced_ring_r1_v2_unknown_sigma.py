import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_NAME = "circular_constant_omega_balanced_ring_r1_v2_unknown_sigma"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figs" / "experiments" / EXPERIMENT_NAME
ARCHIVE_DIR = ROOT / "docs" / "figs" / "experiments" / "archive"
WRITING_DIR = ROOT / "docs" / "active-dynamics-writing"


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
    # dynamics information term (concave radial preference near ring r=1)
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
    # weighting between dynamics and observation information
    "w_dyn": 1.0,
    "w_obs": 0.85,
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
    # radial observation-state information proxy; shape changes with sigma_c
    # peak radius is near r ~= sigma_c for this form
    eps = 1e-8
    s = np.maximum(sigma_c, eps)
    return (r**2 / s**4) * np.exp(-(r**2) / (s**2))


def I_comb(r, sigma_c):
    p = PARAMS
    return p["w_dyn"] * I_dyn(r) + p["w_obs"] * I_obs(r, sigma_c)


def dI_comb_dr_numeric(r, sigma_c):
    h = 1e-3
    r1 = np.clip(r + h, PARAMS["r_min"], PARAMS["r_max"])
    r0 = np.clip(r - h, PARAMS["r_min"], PARAMS["r_max"])
    return (I_comb(r1, sigma_c) - I_comb(r0, sigma_c)) / np.maximum(r1 - r0, 1e-8)


def observation_mean(r, sigma_c):
    p = PARAMS
    return p["g"] * np.exp(-(r**2) / (2.0 * sigma_c**2))


def radial_argmax(sigma_c, r_grid):
    vals = I_comb(r_grid, sigma_c)
    i = int(np.argmax(vals))
    return float(r_grid[i]), float(vals[i])


def simulate_and_infer():
    p = PARAMS
    rng = np.random.default_rng(p["seed"])

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

    rs[0] = 2.15
    thetas[0] = 0.22
    prev_u = 0.0

    r_grid = np.linspace(p["r_min"], p["r_max"], 2400)

    for t in range(T):
        # observe with true (unknown) sigma_c
        y_t = observation_mean(rs[t], p["sigma_c_true"]) + rng.normal(0.0, p["obs_noise_std"])
        y_obs[t] = y_t

        # posterior update on sigma grid using Gaussian likelihood p(y_t|r_t,sigma)
        mu_grid = observation_mean(rs[t], sigma_grid)
        ll = -0.5 * ((y_t - mu_grid) / p["obs_noise_std"]) ** 2
        ll -= np.max(ll)
        post_unnorm = posterior * np.exp(ll)
        posterior = post_unnorm / np.sum(post_unnorm)

        sigma_hat[t] = float(np.sum(sigma_grid * posterior))
        sigma_std[t] = float(np.sqrt(np.sum(((sigma_grid - sigma_hat[t]) ** 2) * posterior)))

        r_star_hat[t], _ = radial_argmax(sigma_hat[t], r_grid)

        if t < T - 1:
            g = dI_comb_dr_numeric(rs[t], sigma_hat[t])
            u_raw = p["k_gain"] * g - p["smoothness"] * prev_u
            u = float(np.clip(u_raw, -p["u_max"], p["u_max"]))
            u += float(rng.normal(0.0, p["control_noise_std"]))

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
        "r": rs,
        "theta": thetas,
        "x": xs,
        "y": ys,
        "u": us,
        "y_obs": y_obs,
        "sigma_hat": sigma_hat,
        "sigma_std": sigma_std,
        "r_star_hat": r_star_hat,
    }


def make_vector_field_fig(out_dir: Path, sim):
    p = PARAMS
    sigma_end = float(sim["sigma_hat"][-1])

    x = np.linspace(-2.4, 2.4, 33)
    y = np.linspace(-2.4, 2.4, 33)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    g_r = dI_comb_dr_numeric(np.clip(R, p["r_min"], p["r_max"]), sigma_end)
    U_rad = p["k_gain"] * g_r

    # polar-to-cartesian field: radial push + angular swirl
    denom = np.maximum(R, 1e-6)
    Ux = U_rad * (X / denom) - p["omega"] * Y
    Uy = U_rad * (Y / denom) + p["omega"] * X

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    speed = np.sqrt(Ux**2 + Uy**2)
    q = ax.quiver(X, Y, Ux, Uy, speed, cmap="viridis", pivot="mid", alpha=0.82)
    cb = fig.colorbar(q, ax=ax, fraction=0.048, pad=0.03)
    cb.set_label("field speed")

    # context rings and trajectory
    ang = np.linspace(0, 2 * np.pi, 600)
    ax.plot(np.cos(ang), np.sin(ang), "--", color="crimson", lw=1.8, label="target ring r=1")
    r_star_end = float(sim["r_star_hat"][-1])
    ax.plot(r_star_end * np.cos(ang), r_star_end * np.sin(ang), ":", color="white", lw=2.0,
            label=fr"argmax(I_comb) at end: r={r_star_end:.2f}")

    ax.plot(sim["x"], sim["y"], color="black", lw=1.6, alpha=0.85, label="trajectory")
    ax.scatter(sim["x"][0], sim["y"][0], s=44, c="lime", edgecolor="black", zorder=4, label="start")
    ax.scatter(sim["x"][-1], sim["y"][-1], s=46, c="red", edgecolor="black", zorder=4, label="end")

    ax.set_aspect("equal")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Latent dynamics vector field (radial info-control + constant omega)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_vector_field_trajectory.png", dpi=190)
    plt.close(fig)


def make_sigma_estimate_fig(out_dir: Path, sim):
    p = PARAMS
    t = np.arange(p["T"]) * p["dt"]

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.plot(t, sim["sigma_hat"], lw=2.2, color="tab:blue", label=r"$\hat{\sigma}_c(t)$")
    ax.fill_between(
        t,
        sim["sigma_hat"] - 2.0 * sim["sigma_std"],
        sim["sigma_hat"] + 2.0 * sim["sigma_std"],
        color="tab:blue",
        alpha=0.20,
        label=r"$\pm 2\,\mathrm{std}$ (posterior)",
    )
    ax.axhline(p["sigma_c_true"], color="crimson", ls="--", lw=1.7, label=fr"true $\sigma_c={p['sigma_c_true']:.2f}$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\sigma_c$")
    ax.set_title("Unknown parameter inference over rollout")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_sigma_estimate_evolution.png", dpi=190)
    plt.close(fig)


def make_combined_map_snapshot(out_dir: Path, sim, idx: int, name: str):
    x = np.linspace(-2.4, 2.4, 330)
    y = np.linspace(-2.4, 2.4, 330)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    sigma_t = float(sim["sigma_hat"][idx])
    Z = I_comb(R, sigma_t)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.contourf(X, Y, Z, levels=55, cmap="magma")
    ang = np.linspace(0, 2 * np.pi, 600)
    ax.plot(np.cos(ang), np.sin(ang), "--", color="cyan", lw=1.5, label="r=1")

    r_star_t = float(sim["r_star_hat"][idx])
    ax.plot(r_star_t * np.cos(ang), r_star_t * np.sin(ang), ":", color="white", lw=2.0,
            label=fr"argmax radius={r_star_t:.2f}")

    ax.scatter(sim["x"][idx], sim["y"][idx], s=48, c="lime", edgecolor="black", zorder=4)
    ax.set_aspect("equal")
    ax.set_title(fr"I_comb(x,y) snapshot ({name}), sigma_hat={sigma_t:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
    fig.tight_layout()
    out = out_dir / f"fig03_combined_map_snapshot_{name}.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return out


def make_combined_snapshot_triptych(out_dir: Path, paths, labels):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)
    for ax, pth, label in zip(axes, paths, labels):
        img = plt.imread(pth)
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    fig.savefig(out_dir / "fig03_combined_map_snapshots_triptych.png", dpi=180)
    plt.close(fig)


def make_radial_argmax_fig(out_dir: Path, sim):
    t = np.arange(PARAMS["T"]) * PARAMS["dt"]
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.plot(t, sim["r_star_hat"], lw=2.1, color="tab:purple", label=r"$r^*_{comb}(t)$")
    ax.axhline(1.0, ls="--", lw=1.6, color="crimson", label="target ring r=1")
    ax.set_xlabel("time")
    ax.set_ylabel("argmax radius")
    ax.set_title("Evolution of combined-map argmax radius")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_argmax_radius_evolution.png", dpi=190)
    plt.close(fig)


def compose_final_panel(out_dir: Path):
    # user-preferred workflow: compose from already-saved individual figures
    files = [
        "fig01_vector_field_trajectory.png",
        "fig02_sigma_estimate_evolution.png",
        "fig03_combined_map_snapshots_triptych.png",
        "fig04_argmax_radius_evolution.png",
    ]
    imgs = [plt.imread(out_dir / f) for f in files]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2))
    axes = axes.ravel()
    titles = [
        "(A) Vector field + trajectory",
        "(B) Unknown parameter estimate",
        "(C) Combined information map snapshots",
        "(D) Radial argmax evolution",
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


def write_model_writeup(out_dir: Path, sim, metrics):
    p = PARAMS
    text = f"""# {EXPERIMENT_NAME} — concise model write-up

## 1) Latent dynamics
State is z_t=(r_t, theta_t) with Cartesian embedding x_t=r_t cos(theta_t), y_t=r_t sin(theta_t).

r_(t+1) = clip(r_t + u_t + eps_t, r_min, r_max)
theta_(t+1) = (theta_t + omega*dt) mod 2pi

Control uses an information gradient:
u_t = clip(k * d/dr I_comb(r_t, sigma_hat_t) - beta * u_(t-1), [-u_max, u_max]).

## 2) Observation model and unknown parameter
Observation:
y_t = g * exp(-r_t^2/(2*sigma_c^2)) + eta_t,  eta_t ~ N(0, sigma_y^2).
Unknown parameter is sigma_c (sensor radial width).

## 3) Information terms
I_dyn(r) = C_d - a_d*(r-r_d)^2
I_obs(r; sigma_c) = (r^2/sigma_c^4)*exp(-r^2/sigma_c^2)
I_comb(r; sigma_c) = w_dyn*I_dyn(r) + w_obs*I_obs(r; sigma_c)

## 4) Parameter inference/update
A posterior on sigma_c is maintained on a fixed grid:
p_t(sigma) proportional to p_(t-1)(sigma) * exp(-(y_t - mu(r_t,sigma))^2/(2*sigma_y^2)),
with mu(r,sigma)=g*exp(-r^2/(2*sigma^2)).
The online estimate is posterior mean sigma_hat_t = E[sigma | y_1:t].

## 5) Why/when the combined map changes
I_obs explicitly depends on sigma_c, so changes in sigma_hat_t reshape I_obs and therefore I_comb.
Hence the combined map is time-varying during rollout as inference progresses.

## 6) Quantitative checks (this run)
- true sigma_c: {p['sigma_c_true']:.4f}
- final sigma_hat: {metrics['unknown_parameter']['sigma_hat_final']:.4f}
- absolute estimation error: {metrics['unknown_parameter']['abs_error_final']:.4f}
- map-change L2 (t0->tmid): {metrics['map_change']['l2_t0_tmid']:.6f}
- map-change L2 (tmid->tend): {metrics['map_change']['l2_tmid_tend']:.6f}
- map-change L2 (t0->tend): {metrics['map_change']['l2_t0_tend']:.6f}
- argmax-radius shift (t0->tend): {metrics['map_change']['argmax_radius_shift_t0_tend']:.6f}
"""
    out_file = out_dir / "model_writeup_unknown_sigma.md"
    out_file.write_text(text, encoding="utf-8")

    WRITING_DIR.mkdir(parents=True, exist_ok=True)
    (WRITING_DIR / f"{EXPERIMENT_NAME}_writeup.md").write_text(text, encoding="utf-8")


def map_l2_change(sig_a, sig_b):
    x = np.linspace(-2.4, 2.4, 250)
    y = np.linspace(-2.4, 2.4, 250)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    A = I_comb(R, sig_a)
    B = I_comb(R, sig_b)
    return float(np.sqrt(np.mean((A - B) ** 2)))


def main():
    maybe_archive_previous(OUT_DIR, ARCHIVE_DIR)

    sim = simulate_and_infer()
    p = PARAMS
    t = np.arange(p["T"]) * p["dt"]

    t0 = 0
    tm = p["T"] // 2
    te = p["T"] - 1

    # 1) individual figures first
    make_vector_field_fig(OUT_DIR, sim)
    make_sigma_estimate_fig(OUT_DIR, sim)
    s0 = make_combined_map_snapshot(OUT_DIR, sim, t0, "t0")
    sm = make_combined_map_snapshot(OUT_DIR, sim, tm, "tmid")
    se = make_combined_map_snapshot(OUT_DIR, sim, te, "tend")
    make_combined_snapshot_triptych(OUT_DIR, [s0, sm, se], ["t0", "t_mid", "t_end"])
    make_radial_argmax_fig(OUT_DIR, sim)

    # 2) compose final panel from saved figures
    compose_final_panel(OUT_DIR)

    # metrics / evidence of changing combined map
    l2_0_m = map_l2_change(float(sim["sigma_hat"][t0]), float(sim["sigma_hat"][tm]))
    l2_m_e = map_l2_change(float(sim["sigma_hat"][tm]), float(sim["sigma_hat"][te]))
    l2_0_e = map_l2_change(float(sim["sigma_hat"][t0]), float(sim["sigma_hat"][te]))

    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "equations": {
            "latent": "r_{t+1}=clip(r_t + u_t + eps_t, r_min, r_max), theta_{t+1}=theta_t + omega*dt",
            "observation": "y_t = g*exp(-r_t^2/(2*sigma_c^2)) + eta_t",
            "I_dyn": "I_dyn(r)=C_d-a_d(r-r_d)^2",
            "I_obs": "I_obs(r;sigma_c)=(r^2/sigma_c^4)*exp(-r^2/sigma_c^2)",
            "I_comb": "I_comb = w_dyn*I_dyn + w_obs*I_obs",
            "posterior_update": "p_t(sigma) propto p_{t-1}(sigma)*exp(-(y_t-mu(r_t,sigma))^2/(2*sigma_y^2))",
        },
        "unknown_parameter": {
            "name": "sigma_c",
            "true": float(p["sigma_c_true"]),
            "sigma_hat_initial": float(sim["sigma_hat"][0]),
            "sigma_hat_mid": float(sim["sigma_hat"][tm]),
            "sigma_hat_final": float(sim["sigma_hat"][te]),
            "abs_error_final": float(abs(sim["sigma_hat"][te] - p["sigma_c_true"])),
        },
        "map_change": {
            "l2_t0_tmid": l2_0_m,
            "l2_tmid_tend": l2_m_e,
            "l2_t0_tend": l2_0_e,
            "argmax_radius_t0": float(sim["r_star_hat"][t0]),
            "argmax_radius_tmid": float(sim["r_star_hat"][tm]),
            "argmax_radius_tend": float(sim["r_star_hat"][te]),
            "argmax_radius_shift_t0_tend": float(sim["r_star_hat"][te] - sim["r_star_hat"][t0]),
        },
        "simulation": {
            "initial_radius": float(sim["r"][0]),
            "final_radius": float(sim["r"][-1]),
            "mean_radius_last_100": float(np.mean(sim["r"][-100:])),
            "std_radius_last_100": float(np.std(sim["r"][-100:])),
            "duration": float(t[-1]),
        },
        "params": PARAMS,
        "outputs": {
            "vector_field": "fig01_vector_field_trajectory.png",
            "sigma_evolution": "fig02_sigma_estimate_evolution.png",
            "combined_maps_t0": "fig03_combined_map_snapshot_t0.png",
            "combined_maps_tmid": "fig03_combined_map_snapshot_tmid.png",
            "combined_maps_tend": "fig03_combined_map_snapshot_tend.png",
            "combined_maps_triptych": "fig03_combined_map_snapshots_triptych.png",
            "argmax_radius": "fig04_argmax_radius_evolution.png",
            "final_panel_png": "final_panel.png",
            "final_panel_pdf": "final_panel.pdf",
            "writeup": "model_writeup_unknown_sigma.md",
        },
    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = f"""Experiment: {EXPERIMENT_NAME}

Unknown parameter and update
- Unknown parameter: sigma_c in observation model y=g*exp(-r^2/(2*sigma_c^2))+noise
- Update method: online grid-posterior Bayes update over sigma_c, using Gaussian likelihood.

Key equations
- I_dyn(r) = C_d - a_d (r-r_d)^2
- I_obs(r; sigma_c) = (r^2/sigma_c^4) exp(-r^2/sigma_c^2)
- I_comb(r; sigma_c) = w_dyn I_dyn + w_obs I_obs

Evidence combined map changes as sigma_hat updates
- sigma_hat(t0)={sim['sigma_hat'][t0]:.4f}, sigma_hat(tmid)={sim['sigma_hat'][tm]:.4f}, sigma_hat(tend)={sim['sigma_hat'][te]:.4f}
- L2(I_comb_t0, I_comb_tmid)={l2_0_m:.6f}
- L2(I_comb_tmid, I_comb_tend)={l2_m_e:.6f}
- L2(I_comb_t0, I_comb_tend)={l2_0_e:.6f}
- argmax radius: r*(t0)={sim['r_star_hat'][t0]:.4f}, r*(tmid)={sim['r_star_hat'][tm]:.4f}, r*(tend)={sim['r_star_hat'][te]:.4f}

Primary outputs
- fig01_vector_field_trajectory.png
- fig02_sigma_estimate_evolution.png
- fig03_combined_map_snapshot_t0.png / _tmid.png / _tend.png
- fig03_combined_map_snapshots_triptych.png
- fig04_argmax_radius_evolution.png
- final_panel.png and final_panel.pdf
- model_writeup_unknown_sigma.md
"""
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    write_model_writeup(OUT_DIR, sim, metrics)

    print(f"Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
