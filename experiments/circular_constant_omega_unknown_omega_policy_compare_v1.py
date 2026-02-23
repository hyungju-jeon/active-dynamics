import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


EXPERIMENT_NAME = "circular_constant_omega_unknown_omega_policy_compare_v1"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figs" / "experiments" / EXPERIMENT_NAME
ARCHIVE_DIR = ROOT / "docs" / "figs" / "experiments" / "archive"


PARAMS = {
    "dt": 0.06,
    "T": 320,
    "omega_true": 1.28,
    "omega_grid_min": 0.55,
    "omega_grid_max": 2.10,
    "omega_grid_n": 320,
    "r_min": 0.03,
    "r_max": 2.8,
    "u_max": 0.065,
    "k_gain": 0.26,
    "smoothness": 0.22,
    "control_noise_std": 0.0022,
    "theta_dyn_noise_std": 0.010,
    "radial_process_noise_std": 0.0015,
    "obs_pos_noise_std": 0.010,
    "seed": 24,
    # acquisition shape
    "obs_center_sigma": 0.90,
    "dyn_peak_width": 0.62,
    "dyn_amp_base": 0.35,
    "dyn_amp_unc_scale": 0.60,
    # plotting grid
    "grid_lim": 2.85,
    "grid_n": 240,
    "r_mid_target": 1.22,
}

POLICIES = {
    "dyn_only": {
        "w_dyn": 1.0,
        "w_obs": 0.0,
        "r_reg": 0.0,
        "label": "dyn-only",
        "color": "tab:red",
    },
    "obs_only": {
        "w_dyn": 0.0,
        "w_obs": 1.0,
        "r_reg": 0.0,
        "label": "obs-only",
        "color": "tab:green",
    },
    "combined": {
        "w_dyn": 0.60,
        "w_obs": 0.95,
        "r_reg": 2.20,
        "label": "combined",
        "color": "tab:blue",
    },
}


def maybe_archive_previous(out_dir: Path, archive_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = archive_dir / f"{ts}-{out_dir.name}"
        shutil.move(str(out_dir), str(dst))
    out_dir.mkdir(parents=True, exist_ok=True)


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def dyn_peak_radius(omega_var):
    # Most informative radius for dynamics depends on omega uncertainty.
    # Early (high uncertainty): peak farther out; later: contracts inward slightly.
    return float(np.clip(2.10 + 1.60 * np.sqrt(max(omega_var, 0.0)), 2.00, 2.70))


def dyn_info(r, omega_var):
    p = PARAMS
    rp = dyn_peak_radius(omega_var)
    amp = p["dyn_amp_base"] + p["dyn_amp_unc_scale"] * omega_var
    return amp * np.exp(-0.5 * ((r - rp) / p["dyn_peak_width"]) ** 2)


def obs_info(r):
    p = PARAMS
    return np.exp(-(r**2) / (p["obs_center_sigma"] ** 2))


def policy_objective(r, omega_var, w_dyn, w_obs):
    return w_dyn * dyn_info(r, omega_var) + w_obs * obs_info(r)


def d_obj_dr_numeric(r, omega_var, w_dyn, w_obs):
    h = 1e-3
    r1 = np.clip(r + h, PARAMS["r_min"], PARAMS["r_max"])
    r0 = np.clip(r - h, PARAMS["r_min"], PARAMS["r_max"])
    f1 = policy_objective(r1, omega_var, w_dyn, w_obs)
    f0 = policy_objective(r0, omega_var, w_dyn, w_obs)
    return (f1 - f0) / np.maximum(r1 - r0, 1e-9)


def generate_shared_noise():
    p = PARAMS
    rng = np.random.default_rng(p["seed"])
    T = p["T"]
    return {
        "theta_dyn": rng.normal(0.0, p["theta_dyn_noise_std"], size=T),
        "radial_proc": rng.normal(0.0, p["radial_process_noise_std"], size=T),
        "ctrl": rng.normal(0.0, p["control_noise_std"], size=T),
        "obs_xy": rng.normal(0.0, p["obs_pos_noise_std"], size=(T, 2)),
    }


def expected_best_radius(omega_var, w_dyn, w_obs):
    r_grid = np.linspace(PARAMS["r_min"], PARAMS["r_max"], 2000)
    vals = policy_objective(r_grid, omega_var, w_dyn, w_obs)
    return float(r_grid[int(np.argmax(vals))])


def simulate_policy(policy_name, shared_noise):
    p = PARAMS
    pol = POLICIES[policy_name]
    T = p["T"]

    omega_grid = np.linspace(p["omega_grid_min"], p["omega_grid_max"], p["omega_grid_n"])
    posterior = np.ones_like(omega_grid) / omega_grid.size

    r = np.zeros(T)
    th = np.zeros(T)
    x = np.zeros(T)
    y = np.zeros(T)
    r_obs = np.zeros(T)
    th_obs = np.zeros(T)
    u = np.zeros(T)

    omega_mean = np.zeros(T)
    omega_std = np.zeros(T)
    omega_var = np.zeros(T)
    r_star = np.zeros(T)
    i_dyn = np.zeros(T)
    i_obs = np.zeros(T)
    a_total = np.zeros(T)

    # fairness: same initialization across all policies
    r[0] = 1.55
    th[0] = 0.18
    prev_u = 0.0

    for t in range(T):
        x[t] = r[t] * math.cos(th[t])
        y[t] = r[t] * math.sin(th[t])

        # noisy position observation and implied angle observation
        x_o = x[t] + shared_noise["obs_xy"][t, 0]
        y_o = y[t] + shared_noise["obs_xy"][t, 1]
        r_obs[t] = float(np.hypot(x_o, y_o))
        th_obs[t] = float(np.arctan2(y_o, x_o))

        # online omega posterior update from observed angle increment
        if t > 0:
            dth_obs = wrap_to_pi(th_obs[t] - th_obs[t - 1])
            # angle observation noise grows near center; radius known from noisy obs
            meas_std = p["theta_dyn_noise_std"] + p["obs_pos_noise_std"] / max(r_obs[t], 0.08)
            pred = omega_grid * p["dt"]
            err = wrap_to_pi(dth_obs - pred)
            ll = -0.5 * (err / meas_std) ** 2
            ll -= np.max(ll)
            post_u = posterior * np.exp(ll)
            posterior = post_u / np.sum(post_u)

        omega_mean[t] = float(np.sum(omega_grid * posterior))
        omega_var[t] = float(np.sum(((omega_grid - omega_mean[t]) ** 2) * posterior))
        omega_std[t] = float(np.sqrt(max(omega_var[t], 0.0)))

        r_star[t] = expected_best_radius(omega_var[t], pol["w_dyn"], pol["w_obs"])
        i_dyn[t] = float(dyn_info(r[t], omega_var[t]))
        i_obs[t] = float(obs_info(r[t]))
        a_total[t] = float(pol["w_dyn"] * i_dyn[t] + pol["w_obs"] * i_obs[t])

        if t < T - 1:
            grad = d_obj_dr_numeric(r[t], omega_var[t], pol["w_dyn"], pol["w_obs"])
            grad_eff = grad - pol.get("r_reg", 0.0) * (r[t] - p["r_mid_target"])
            u_raw = p["k_gain"] * grad_eff - p["smoothness"] * prev_u
            u_t = float(np.clip(u_raw, -p["u_max"], p["u_max"])) + float(shared_noise["ctrl"][t])

            r_next = float(np.clip(r[t] + u_t + shared_noise["radial_proc"][t], p["r_min"], p["r_max"]))
            th_next = float((th[t] + p["omega_true"] * p["dt"] + shared_noise["theta_dyn"][t]) % (2.0 * np.pi))

            r[t + 1] = r_next
            th[t + 1] = th_next
            u[t] = u_t
            prev_u = u_t

    u[-1] = u[-2]
    posterior_final = posterior.copy()

    return {
        "policy": policy_name,
        "r": r,
        "theta": th,
        "x": x,
        "y": y,
        "u": u,
        "omega_mean": omega_mean,
        "omega_std": omega_std,
        "omega_var": omega_var,
        "omega_grid": omega_grid,
        "posterior_final": posterior_final,
        "r_star": r_star,
        "i_dyn": i_dyn,
        "i_obs": i_obs,
        "a_total": a_total,
        "init": {
            "r0": float(r[0]),
            "theta0": float(th[0]),
            "omega_belief": "uniform over omega_grid",
        },
    }


def map_from_policy(omega_var, policy_name):
    p = PARAMS
    pol = POLICIES[policy_name]
    lim = p["grid_lim"]
    n = p["grid_n"]
    gx = np.linspace(-lim, lim, n)
    gy = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(gx, gy)
    R = np.sqrt(X**2 + Y**2)
    I_d = dyn_info(R, omega_var)
    I_o = obs_info(R)
    A = pol["w_dyn"] * I_d + pol["w_obs"] * I_o
    return X, Y, I_d, I_o, A


def fig01_acquisition_maps(out_dir: Path, sims):
    omega_var0 = float(sims["combined"]["omega_var"][0])
    X, Y, I_d, I_o, A_c = map_from_policy(omega_var0, "combined")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    for ax, Z, title, cmap, cbl in [
        (axes[0], I_d, "I_dyn map (t=0, unknown ω)", "magma", "I_dyn"),
        (axes[1], I_o, "I_obs map", "viridis", "I_obs"),
        (axes[2], A_c, "A_total map (combined, t=0)", "cividis", "A_total"),
    ]:
        im = ax.contourf(X, Y, Z, levels=34, cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.86, label=cbl)
        ax.set_aspect("equal")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title(title)
    fig.savefig(out_dir / "fig01_acquisition_decomposition_maps.png", dpi=220)
    plt.close(fig)


def fig02_trajectories(out_dir: Path, sims):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    lim = PARAMS["grid_lim"]
    ang = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(ang), np.sin(ang), "--", color="gray", lw=1.0, alpha=0.6)
    ax.plot(1.6 * np.cos(ang), 1.6 * np.sin(ang), ":", color="gray", lw=1.0, alpha=0.5)

    for name in ["dyn_only", "combined", "obs_only"]:
        s = sims[name]
        pol = POLICIES[name]
        ax.plot(s["x"], s["y"], color=pol["color"], lw=2.0, label=pol["label"])
        ax.scatter(s["x"][0], s["y"][0], c="yellow", edgecolor="k", s=38, zorder=4)
        ax.scatter(s["x"][-1], s["y"][-1], c=pol["color"], edgecolor="k", s=36, zorder=4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.22)
    ax.set_title("Policy trajectories on latent plane")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_policy_trajectories.png", dpi=220)
    plt.close(fig)


def fig03_omega_evolution(out_dir: Path, sims):
    p = PARAMS
    t = np.arange(p["T"]) * p["dt"]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), constrained_layout=True)

    ax = axes[0]
    for name in ["dyn_only", "combined", "obs_only"]:
        s = sims[name]
        pol = POLICIES[name]
        ax.plot(t, s["omega_mean"], color=pol["color"], lw=2.0, label=pol["label"])
        ax.fill_between(t, s["omega_mean"] - 2.0 * s["omega_std"], s["omega_mean"] + 2.0 * s["omega_std"], color=pol["color"], alpha=0.12)
    ax.axhline(p["omega_true"], color="k", ls="--", lw=1.4, label="true ω")
    ax.set_ylabel("omega estimate")
    ax.set_title("Online ω inference (agents start with unknown ω)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    for name in ["dyn_only", "combined", "obs_only"]:
        s = sims[name]
        pol = POLICIES[name]
        ax.plot(t, s["omega_std"], color=pol["color"], lw=2.0, label=pol["label"])
    ax.set_xlabel("time")
    ax.set_ylabel("posterior std(ω)")
    ax.set_title("Uncertainty contraction over time")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.savefig(out_dir / "fig03_omega_inference_evolution.png", dpi=220)
    plt.close(fig)


def fig04_radius_objective(out_dir: Path, sims):
    p = PARAMS
    t = np.arange(p["T"]) * p["dt"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), constrained_layout=True)

    ax = axes[0]
    for name in ["dyn_only", "combined", "obs_only"]:
        s = sims[name]
        pol = POLICIES[name]
        ax.plot(t, s["r"], color=pol["color"], lw=2.0, label=pol["label"])
    ax.set_xlabel("time")
    ax.set_ylabel("radius r")
    ax.set_title("Radius evolution")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name in ["dyn_only", "combined", "obs_only"]:
        s = sims[name]
        pol = POLICIES[name]
        cum = np.cumsum(s["a_total"]) * p["dt"]
        ax.plot(t, cum, color=pol["color"], lw=2.0, label=pol["label"])
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative objective")
    ax.set_title("Cumulative objective comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.savefig(out_dir / "fig04_radius_and_cumulative_objective.png", dpi=220)
    plt.close(fig)


def fig05_map_uncertainty_snapshots(out_dir: Path, sims):
    p = PARAMS
    idxs = [0, p["T"] // 2, p["T"] - 1]
    names = ["t0", "tmid", "tend"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), constrained_layout=True)
    for ax, idx, nm in zip(axes, idxs, names):
        omega_var_t = float(sims["combined"]["omega_var"][idx])
        X, Y, _, _, A = map_from_policy(omega_var_t, "combined")
        im = ax.contourf(X, Y, A, levels=34, cmap="cividis")
        plt.colorbar(im, ax=ax, shrink=0.84)
        ax.set_aspect("equal")
        ax.set_title(f"A_total combined ({nm})")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        txt = f"Var(ω)={omega_var_t:.4f}"
        ax.text(0.03, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    fig.savefig(out_dir / "fig05_combined_map_uncertainty_snapshots.png", dpi=220)
    plt.close(fig)


def compose_final_panel(out_dir: Path):
    files = [
        "fig01_acquisition_decomposition_maps.png",
        "fig02_policy_trajectories.png",
        "fig03_omega_inference_evolution.png",
        "fig04_radius_and_cumulative_objective.png",
        "fig05_combined_map_uncertainty_snapshots.png",
    ]
    imgs = [plt.imread(out_dir / f) for f in files]

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 14.0))
    axes = axes.ravel()
    titles = [
        "(A) acquisition decomposition maps",
        "(B) policy trajectories",
        "(C) omega estimate/posterior evolution",
        "(D) radius + cumulative objective",
        "(E) A_total snapshots vs omega uncertainty",
    ]
    for k in range(5):
        axes[k].imshow(imgs[k])
        axes[k].set_title(titles[k])
        axes[k].axis("off")
    axes[5].axis("off")
    fig.suptitle(EXPERIMENT_NAME, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "final_panel.png", dpi=220)
    fig.savefig(out_dir / "final_panel.pdf")
    plt.close(fig)


def make_animation(out_dir: Path, sims):
    p = PARAMS
    frame_step = 3
    frames = list(range(0, p["T"], frame_step))
    lim = p["grid_lim"]

    gx = np.linspace(-lim, lim, 160)
    gy = np.linspace(-lim, lim, 160)
    X, Y = np.meshgrid(gx, gy)
    R = np.sqrt(X**2 + Y**2)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    A0 = policy_objective(R, float(sims["combined"]["omega_var"][0]), POLICIES["combined"]["w_dyn"], POLICIES["combined"]["w_obs"])
    im = ax.imshow(A0, extent=[-lim, lim, -lim, lim], origin="lower", cmap="cividis", alpha=0.84)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="combined A_total")

    lines = {}
    points = {}
    for name in ["dyn_only", "combined", "obs_only"]:
        pol = POLICIES[name]
        line, = ax.plot([], [], color=pol["color"], lw=2.0, label=pol["label"])
        point = ax.scatter([], [], c=pol["color"], s=36, edgecolor="k", zorder=4)
        lines[name] = line
        points[name] = point

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Trajectories with evolving combined acquisition map")
    ax.legend(fontsize=8, loc="upper right")

    def update(fi):
        t = frames[fi]
        A = policy_objective(R, float(sims["combined"]["omega_var"][t]), POLICIES["combined"]["w_dyn"], POLICIES["combined"]["w_obs"])
        im.set_data(A)
        for name in ["dyn_only", "combined", "obs_only"]:
            s = sims[name]
            lines[name].set_data(s["x"][: t + 1], s["y"][: t + 1])
            points[name].set_offsets(np.array([[s["x"][t], s["y"][t]]]))
        ax.set_title(f"Trajectories + evolving A_total | t={t*p['dt']:.2f}, Var(ω)={sims['combined']['omega_var'][t]:.4f}")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=55, blit=False)

    gif_path = out_dir / "trajectory_all_policies_unknown_omega.gif"
    mp4_path = out_dir / "trajectory_all_policies_unknown_omega.mp4"

    anim.save(gif_path, writer=PillowWriter(fps=15))
    try:
        anim.save(mp4_path, writer=FFMpegWriter(fps=15, bitrate=1800))
    except Exception:
        pass
    plt.close(fig)
    return gif_path, mp4_path


def compute_metrics(sims, gif_path: Path, mp4_path: Path):
    p = PARAMS
    dt = p["dt"]

    pm = {}
    for name, s in sims.items():
        pm[name] = {
            "final_radius": float(s["r"][-1]),
            "mean_radius": float(np.mean(s["r"])),
            "mean_radius_last_80": float(np.mean(s["r"][-80:])),
            "cum_objective": float(np.sum(s["a_total"]) * dt),
            "omega_mean_final": float(s["omega_mean"][-1]),
            "omega_std_final": float(s["omega_std"][-1]),
            "omega_abs_error_final": float(abs(s["omega_mean"][-1] - p["omega_true"])),
        }

    order_final = [pm["obs_only"]["final_radius"], pm["combined"]["final_radius"], pm["dyn_only"]["final_radius"]]
    order_mean = [pm["obs_only"]["mean_radius"], pm["combined"]["mean_radius"], pm["dyn_only"]["mean_radius"]]

    satisfied = {
        "final_radius_order_obs_lt_comb_lt_dyn": bool(order_final[0] < order_final[1] < order_final[2]),
        "mean_radius_order_obs_lt_comb_lt_dyn": bool(order_mean[0] < order_mean[1] < order_mean[2]),
        "dyn_outward_trend": bool(pm["dyn_only"]["final_radius"] > sims["dyn_only"]["r"][0] + 0.5),
        "obs_center_seeking": bool(pm["obs_only"]["final_radius"] < 0.45),
        "combined_middle": bool(0.65 < pm["combined"]["final_radius"] < 1.9),
    }

    unknown = {
        "parameter": "omega",
        "true_omega": float(p["omega_true"]),
        "initial_belief": "uniform over omega grid",
        "inference_method": "online Bayesian grid posterior using observed angle increments",
        "final_abs_error_combined": pm["combined"]["omega_abs_error_final"],
        "final_posterior_std_combined": pm["combined"]["omega_std_final"],
        "posterior_std_reduction_combined": float(sims["combined"]["omega_std"][0] - sims["combined"]["omega_std"][-1]),
    }

    return {
        "experiment_name": EXPERIMENT_NAME,
        "equations": {
            "latent_dynamics": "r_{t+1}=clip(r_t+u_t+eps_r), theta_{t+1}=theta_t+omega*dt+eps_theta",
            "observation": "(x^o_t,y^o_t)=(r_t cos theta_t, r_t sin theta_t)+noise",
            "posterior_update": "p_t(omega) propto p_{t-1}(omega) * exp(-0.5 * wrap(dtheta_obs-omega*dt)^2 / sigma_theta(r)^2)",
            "I_dyn": "I_dyn(r,Var[omega])=(a0+a1*Var[omega])*exp(-0.5*((r-r_dyn*(Var[omega]))/s_dyn)^2)",
            "I_obs": "I_obs(r)=exp(-r^2/s_obs^2)",
            "A_total_policy": "A=w_dyn*I_dyn + w_obs*I_obs",
            "combined_control_regularizer": "for combined: grad_eff=drA - lambda_reg*(r-r_mid)",
        },
        "fairness": {
            "same_initial_state": True,
            "same_initial_omega_belief": True,
            "shared_noise_sequences": True,
            "initial_state": sims["combined"]["init"],
        },
        "policy_metrics": pm,
        "qualitative_targets": satisfied,
        "unknown_omega": unknown,
        "outputs": {
            "acquisition_maps": "fig01_acquisition_decomposition_maps.png",
            "trajectories": "fig02_policy_trajectories.png",
            "omega_evolution": "fig03_omega_inference_evolution.png",
            "radius_cumulative": "fig04_radius_and_cumulative_objective.png",
            "uncertainty_snapshots": "fig05_combined_map_uncertainty_snapshots.png",
            "final_panel_png": "final_panel.png",
            "final_panel_pdf": "final_panel.pdf",
            "animation_gif": gif_path.name,
            "animation_mp4": mp4_path.name,
            "animation_mp4_exists": bool(mp4_path.exists()),
        },
        "params": PARAMS,
    }


def write_summary(out_dir: Path, metrics):
    pm = metrics["policy_metrics"]
    q = metrics["qualitative_targets"]
    uk = metrics["unknown_omega"]

    txt = f"""Experiment: {EXPERIMENT_NAME}

This run explicitly assumes agents do NOT know omega at start.
All policies start from the same initial state and same uniform omega posterior,
and use shared noise sequences for fair comparison.

Model equations
- latent dynamics: {metrics['equations']['latent_dynamics']}
- posterior update: {metrics['equations']['posterior_update']}
- I_dyn: {metrics['equations']['I_dyn']}
- I_obs: {metrics['equations']['I_obs']}
- A_total: {metrics['equations']['A_total_policy']}

Key quantitative checks
- final radius ordering (obs < combined < dyn): {q['final_radius_order_obs_lt_comb_lt_dyn']}
- mean radius ordering (obs < combined < dyn): {q['mean_radius_order_obs_lt_comb_lt_dyn']}
- dyn outward trend: {q['dyn_outward_trend']}
- obs center-seeking: {q['obs_center_seeking']}
- combined middle: {q['combined_middle']}

Final radii
- obs-only: {pm['obs_only']['final_radius']:.4f}
- combined: {pm['combined']['final_radius']:.4f}
- dyn-only: {pm['dyn_only']['final_radius']:.4f}

Mean radii
- obs-only: {pm['obs_only']['mean_radius']:.4f}
- combined: {pm['combined']['mean_radius']:.4f}
- dyn-only: {pm['dyn_only']['mean_radius']:.4f}

Unknown-omega inference quality (combined)
- true omega: {uk['true_omega']:.4f}
- final |omega_hat - omega_true|: {uk['final_abs_error_combined']:.6f}
- final posterior std(omega): {uk['final_posterior_std_combined']:.6f}
- posterior std reduction: {uk['posterior_std_reduction_combined']:.6f}

Deliverables
- acquisition decomposition map: fig01_acquisition_decomposition_maps.png
- trajectories: fig02_policy_trajectories.png
- omega evolution: fig03_omega_inference_evolution.png
- radius/objective comparison: fig04_radius_and_cumulative_objective.png
- map uncertainty snapshots: fig05_combined_map_uncertainty_snapshots.png
- representative panel: final_panel.png
- animation: {metrics['outputs']['animation_gif']} / {metrics['outputs']['animation_mp4']}
"""
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")


def main():
    maybe_archive_previous(OUT_DIR, ARCHIVE_DIR)

    shared_noise = generate_shared_noise()
    sims = {name: simulate_policy(name, shared_noise) for name in ["dyn_only", "obs_only", "combined"]}

    # individual figures first
    fig01_acquisition_maps(OUT_DIR, sims)
    fig02_trajectories(OUT_DIR, sims)
    fig03_omega_evolution(OUT_DIR, sims)
    fig04_radius_objective(OUT_DIR, sims)
    fig05_map_uncertainty_snapshots(OUT_DIR, sims)

    gif_path, mp4_path = make_animation(OUT_DIR, sims)

    # compose final panel from saved figures
    compose_final_panel(OUT_DIR)

    metrics = compute_metrics(sims, gif_path, mp4_path)
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_summary(OUT_DIR, metrics)
    print(f"Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
