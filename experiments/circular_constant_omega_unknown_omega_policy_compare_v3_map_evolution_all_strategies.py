import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


EXPERIMENT_NAME = "circular_constant_omega_unknown_omega_policy_compare_v3_map_evolution_all_strategies"
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
    # observation-limited omega inference
    "omega_like_sigma_floor": 0.010,
    "omega_like_sigma_base": 0.002,
    "omega_like_info_power": 1.15,
    "omega_like_eps": 1e-4,
    # plotting grid
    "grid_lim": 2.85,
    "grid_n": 200,
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
    return float(np.clip(2.10 + 1.60 * np.sqrt(max(omega_var, 0.0)), 2.00, 2.70))


def dyn_info(r, omega_var):
    p = PARAMS
    rp = dyn_peak_radius(omega_var)
    amp = p["dyn_amp_base"] + p["dyn_amp_unc_scale"] * omega_var
    return amp * np.exp(-0.5 * ((r - rp) / p["dyn_peak_width"]) ** 2)


def obs_info(r):
    p = PARAMS
    return np.exp(-(r**2) / (p["obs_center_sigma"] ** 2))


def sigma_theta_meas_from_obsinfo(r):
    p = PARAMS
    i_obs = np.maximum(obs_info(r), p["omega_like_eps"])
    return p["omega_like_sigma_floor"] + p["omega_like_sigma_base"] / (i_obs ** p["omega_like_info_power"])


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
    sigma_theta_meas = np.zeros(T)

    r[0] = 1.55
    th[0] = 0.18
    prev_u = 0.0

    for t in range(T):
        x[t] = r[t] * math.cos(th[t])
        y[t] = r[t] * math.sin(th[t])

        x_o = x[t] + shared_noise["obs_xy"][t, 0]
        y_o = y[t] + shared_noise["obs_xy"][t, 1]
        r_obs[t] = float(np.hypot(x_o, y_o))
        th_obs[t] = float(np.arctan2(y_o, x_o))

        sigma_theta_meas[t] = float(sigma_theta_meas_from_obsinfo(r_obs[t]))

        if t > 0:
            dth_obs = wrap_to_pi(th_obs[t] - th_obs[t - 1])
            pred = omega_grid * p["dt"]
            err = wrap_to_pi(dth_obs - pred)
            ll = -0.5 * (err / sigma_theta_meas[t]) ** 2
            ll -= np.max(ll)
            ll = np.clip(ll, -745.0, 0.0)
            post_u = posterior * np.exp(ll)
            z = float(np.sum(post_u))
            if np.isfinite(z) and z > 1e-300:
                posterior = post_u / z

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
        "posterior_final": posterior.copy(),
        "r_star": r_star,
        "i_dyn": i_dyn,
        "i_obs": i_obs,
        "a_total": a_total,
        "sigma_theta_meas": sigma_theta_meas,
        "init": {
            "r0": float(r[0]),
            "theta0": float(th[0]),
            "omega_belief": "uniform over omega_grid",
        },
    }


def make_grid():
    lim = PARAMS["grid_lim"]
    n = PARAMS["grid_n"]
    gx = np.linspace(-lim, lim, n)
    gy = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(gx, gy)
    R = np.sqrt(X**2 + Y**2)
    return X, Y, R


def map_from_policy(omega_var, policy_name, R):
    pol = POLICIES[policy_name]
    I_d = dyn_info(R, omega_var)
    I_o = obs_info(R)
    A = pol["w_dyn"] * I_d + pol["w_obs"] * I_o
    return I_d, I_o, A


def create_triptych_for_strategy(out_dir: Path, sims, strategy: str, X, Y, R):
    T = PARAMS["T"]
    idxs = [0, T // 2, T - 1]
    row_names = ["t0", "tmid", "tend"]
    sim = sims[strategy]

    maps = []
    for idx in idxs:
        maps.append(map_from_policy(float(sim["omega_var"][idx]), strategy, R))

    vmins = [min(float(np.min(m[k])) for m in maps) for k in range(3)]
    vmaxs = [max(float(np.max(m[k])) for m in maps) for k in range(3)]

    fig, axes = plt.subplots(3, 3, figsize=(13.6, 12.0), constrained_layout=True)
    col_titles = ["I_dyn", "I_obs", f"A_total ({strategy})"]
    cmaps = ["magma", "viridis", "cividis"]

    for r_i, (idx, rn) in enumerate(zip(idxs, row_names)):
        I_d, I_o, A = maps[r_i]
        stack = [I_d, I_o, A]
        for c_i in range(3):
            ax = axes[r_i, c_i]
            im = ax.contourf(X, Y, stack[c_i], levels=32, cmap=cmaps[c_i], vmin=vmins[c_i], vmax=vmaxs[c_i])
            if c_i == 2:
                tt = np.arange(idx + 1)
                ax.plot(sim["x"][tt], sim["y"][tt], color=POLICIES[strategy]["color"], lw=1.7, alpha=0.95)
                ax.scatter(sim["x"][idx], sim["y"][idx], s=32, c="yellow", edgecolor="k", zorder=5)
            ax.set_aspect("equal")
            ax.set_xlabel("z1")
            ax.set_ylabel("z2")
            if r_i == 0:
                ax.set_title(col_titles[c_i])
            if c_i == 0:
                ax.text(
                    0.03,
                    0.96,
                    f"{rn} | t={idx * PARAMS['dt']:.2f}\nVar(ω)={sim['omega_var'][idx]:.4f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                )
            plt.colorbar(im, ax=ax, shrink=0.82)

    fig.suptitle(f"Map evolution triptych: {strategy}", fontsize=13)
    out = out_dir / f"maps_triptych_{strategy}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def create_map_evolution_animation(out_dir: Path, sims, strategy: str, X, Y, R):
    sim = sims[strategy]
    pol = POLICIES[strategy]
    frame_step = 3
    frames = list(range(0, PARAMS["T"], frame_step))

    t0 = frames[0]
    I_d0, I_o0, A0 = map_from_policy(float(sim["omega_var"][t0]), strategy, R)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    cmaps = ["magma", "viridis", "cividis"]
    labels = ["I_dyn", "I_obs", f"A_total ({strategy})"]

    ims = []
    for ax, Z0, cmap, lab in zip(axes, [I_d0, I_o0, A0], cmaps, labels):
        im = ax.imshow(Z0, extent=[X.min(), X.max(), Y.min(), Y.max()], origin="lower", cmap=cmap, alpha=0.92)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=lab)
        ax.set_aspect("equal")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title(lab)
        ims.append(im)

    line, = axes[2].plot([], [], color=pol["color"], lw=1.9)
    point = axes[2].scatter([], [], c="yellow", edgecolor="k", s=42, zorder=4)

    def update(fi):
        t = frames[fi]
        I_d, I_o, A = map_from_policy(float(sim["omega_var"][t]), strategy, R)
        ims[0].set_data(I_d)
        ims[1].set_data(I_o)
        ims[2].set_data(A)

        line.set_data(sim["x"][: t + 1], sim["y"][: t + 1])
        point.set_offsets(np.array([[sim["x"][t], sim["y"][t]]]))

        fig.suptitle(
            f"{strategy} map evolution | t={t * PARAMS['dt']:.2f} | Var(ω)={sim['omega_var'][t]:.4f}",
            fontsize=12,
        )
        return ims + [line, point]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    gif_path = out_dir / f"maps_evolution_{strategy}.gif"
    mp4_path = out_dir / f"maps_evolution_{strategy}.mp4"

    anim.save(gif_path, writer=PillowWriter(fps=15))
    try:
        anim.save(mp4_path, writer=FFMpegWriter(fps=15, bitrate=1800))
    except Exception:
        pass
    plt.close(fig)
    return gif_path, mp4_path


def compose_representative_panel(out_dir: Path):
    files = [
        "maps_triptych_dyn_only.png",
        "maps_triptych_obs_only.png",
        "maps_triptych_combined.png",
    ]
    imgs = [plt.imread(out_dir / f) for f in files]

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 16.0), constrained_layout=True)
    titles = [
        "(A) dyn_only: [I_dyn, I_obs, A_total] at t0/tmid/tend",
        "(B) obs_only: [I_dyn, I_obs, A_total] at t0/tmid/tend",
        "(C) combined: [I_dyn, I_obs, A_total] at t0/tmid/tend",
    ]
    for i in range(3):
        axes[i].imshow(imgs[i])
        axes[i].set_title(titles[i], fontsize=11)
        axes[i].axis("off")

    fig.suptitle(f"Representative panel — {EXPERIMENT_NAME}", fontsize=14)
    fig.savefig(out_dir / "representative_panel_map_evolution.png", dpi=220)
    fig.savefig(out_dir / "representative_panel_map_evolution.pdf")
    plt.close(fig)


def l2_changes(arr0, arrm, arre):
    return {
        "t0_to_tmid": float(np.linalg.norm(arrm - arr0)),
        "tmid_to_tend": float(np.linalg.norm(arre - arrm)),
        "t0_to_tend": float(np.linalg.norm(arre - arr0)),
    }


def compute_metrics(sims, X, Y, R, triptychs, animations):
    T = PARAMS["T"]
    idx0, idxm, idxe = 0, T // 2, T - 1

    map_change = {}
    for strategy in ["dyn_only", "obs_only", "combined"]:
        sim = sims[strategy]
        d0, o0, a0 = map_from_policy(float(sim["omega_var"][idx0]), strategy, R)
        dm, om, am = map_from_policy(float(sim["omega_var"][idxm]), strategy, R)
        de, oe, ae = map_from_policy(float(sim["omega_var"][idxe]), strategy, R)

        map_change[strategy] = {
            "I_dyn_L2": l2_changes(d0, dm, de),
            "I_obs_L2": l2_changes(o0, om, oe),
            "A_total_L2": l2_changes(a0, am, ae),
        }

    outputs = {
        "triptychs": {k: str(v.relative_to(OUT_DIR)) for k, v in triptychs.items()},
        "animations": {
            k: {
                "gif": str(v["gif"].relative_to(OUT_DIR)),
                "mp4": str(v["mp4"].relative_to(OUT_DIR)),
                "mp4_exists": bool(v["mp4"].exists()),
            }
            for k, v in animations.items()
        },
        "panel_png": "representative_panel_map_evolution.png",
        "panel_pdf": "representative_panel_map_evolution.pdf",
    }

    return {
        "experiment_name": EXPERIMENT_NAME,
        "definitions": {
            "I_dyn": "I_dyn(r,Var[omega])=(a0+a1*Var[omega])*exp(-0.5*((r-r_dyn*(Var[omega]))/s_dyn)^2)",
            "I_obs": "I_obs(r)=exp(-r^2/s_obs^2)",
            "A_total": "A_total^policy = w_dyn*I_dyn + w_obs*I_obs",
        },
        "fairness": {
            "same_initial_state": True,
            "same_initial_omega_posterior": True,
            "shared_noise_sequences": True,
            "initial_state": sims["combined"]["init"],
        },
        "timepoints": {
            "indices": {"t0": idx0, "tmid": idxm, "tend": idxe},
            "times": {
                "t0": float(idx0 * PARAMS["dt"]),
                "tmid": float(idxm * PARAMS["dt"]),
                "tend": float(idxe * PARAMS["dt"]),
            },
        },
        "map_change_metrics": map_change,
        "outputs": outputs,
        "params": PARAMS,
    }


def write_summary(out_dir: Path, metrics):
    mc = metrics["map_change_metrics"]
    out = metrics["outputs"]

    lines = []
    lines.append(f"Experiment: {metrics['experiment_name']}")
    lines.append("")
    lines.append("Definitions")
    lines.append(f"- I_dyn: {metrics['definitions']['I_dyn']}")
    lines.append(f"- I_obs: {metrics['definitions']['I_obs']}")
    lines.append(f"- A_total: {metrics['definitions']['A_total']}")
    lines.append("")
    lines.append("Map-change metrics (L2 on full grid)")
    for strategy in ["dyn_only", "obs_only", "combined"]:
        lines.append(f"- {strategy}:")
        for mk in ["I_dyn_L2", "I_obs_L2", "A_total_L2"]:
            v = mc[strategy][mk]
            lines.append(
                f"  - {mk}: t0->tmid={v['t0_to_tmid']:.6f}, tmid->tend={v['tmid_to_tend']:.6f}, t0->tend={v['t0_to_tend']:.6f}"
            )
    lines.append("")
    lines.append("Outputs")
    for strategy in ["dyn_only", "obs_only", "combined"]:
        lines.append(f"- triptych ({strategy}): {out['triptychs'][strategy]}")
        lines.append(f"- animation ({strategy}) gif: {out['animations'][strategy]['gif']}")
        lines.append(f"- animation ({strategy}) mp4: {out['animations'][strategy]['mp4']}")
    lines.append(f"- representative panel png: {out['panel_png']}")
    lines.append(f"- representative panel pdf: {out['panel_pdf']}")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    maybe_archive_previous(OUT_DIR, ARCHIVE_DIR)

    shared_noise = generate_shared_noise()
    sims = {name: simulate_policy(name, shared_noise) for name in ["dyn_only", "obs_only", "combined"]}

    X, Y, R = make_grid()

    triptychs = {}
    animations = {}
    for strategy in ["dyn_only", "obs_only", "combined"]:
        triptychs[strategy] = create_triptych_for_strategy(OUT_DIR, sims, strategy, X, Y, R)

    for strategy in ["dyn_only", "obs_only", "combined"]:
        gif_path, mp4_path = create_map_evolution_animation(OUT_DIR, sims, strategy, X, Y, R)
        animations[strategy] = {"gif": gif_path, "mp4": mp4_path}

    compose_representative_panel(OUT_DIR)

    metrics = compute_metrics(sims, X, Y, R, triptychs, animations)
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_summary(OUT_DIR, metrics)
    print(f"Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
