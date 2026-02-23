import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


EXPERIMENT_NAME = "circular_constant_omega_balanced_ring_r1_v1"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figs" / "experiments" / EXPERIMENT_NAME
ARCHIVE_DIR = ROOT / "docs" / "figs" / "experiments" / "archive"


def maybe_archive_previous(out_dir: Path, archive_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = archive_dir / f"{ts}-{out_dir.name}"
        shutil.move(str(out_dir), str(dst))
    out_dir.mkdir(parents=True, exist_ok=True)


# Information profiles (explicit, analytic-friendly)
# I_dyn(r) = C_d - a_d (r-r_d)^2
# I_obs(r) = C_o - a_o (r-r_o)^2
# A_base(r) = I_dyn(r) + I_obs(r)
PARAMS = {
    "a_d": 1.0,
    "r_d": 1.6,
    "C_d": 2.0,
    "a_o": 1.0,
    "r_o": 0.4,
    "C_o": 2.0,
    "omega": 1.15,  # radius-invariant angular velocity [rad/unit-time]
    "dt": 0.05,
    "T": 440,
    "k_gain": 0.20,
    "smoothness": 0.28,
    "control_max": 0.06,
    "noise_std": 0.004,
    "seed": 7,
}


def I_dyn(r):
    p = PARAMS
    return p["C_d"] - p["a_d"] * (r - p["r_d"]) ** 2


def I_obs(r):
    p = PARAMS
    return p["C_o"] - p["a_o"] * (r - p["r_o"]) ** 2


def A_base(r):
    return I_dyn(r) + I_obs(r)


def dA_dr(r):
    p = PARAMS
    return -2.0 * p["a_d"] * (r - p["r_d"]) - 2.0 * p["a_o"] * (r - p["r_o"])


def analytic_r_star():
    p = PARAMS
    return (p["a_d"] * p["r_d"] + p["a_o"] * p["r_o"]) / (p["a_d"] + p["a_o"])


def simulate():
    p = PARAMS
    rng = np.random.default_rng(p["seed"])

    T = p["T"]
    dt = p["dt"]
    omega = p["omega"]

    rs = np.zeros(T)
    thetas = np.zeros(T)
    xs = np.zeros(T)
    ys = np.zeros(T)
    us = np.zeros(T)
    As = np.zeros(T)

    rs[0] = 2.2
    thetas[0] = 0.3
    prev_u = 0.0

    for t in range(T - 1):
        g = dA_dr(rs[t])
        u_raw = p["k_gain"] * g - p["smoothness"] * prev_u
        u = float(np.clip(u_raw, -p["control_max"], p["control_max"]))
        u += float(rng.normal(0.0, p["noise_std"]))

        r_next = np.clip(rs[t] + u, 0.0, 3.0)
        th_next = (thetas[t] + omega * dt) % (2 * math.pi)

        rs[t + 1] = r_next
        thetas[t + 1] = th_next
        us[t] = u
        As[t] = A_base(rs[t])
        prev_u = u

    us[-1] = us[-2]
    As[-1] = A_base(rs[-1])

    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    return {
        "r": rs,
        "theta": thetas,
        "x": xs,
        "y": ys,
        "u": us,
        "A": As,
    }


def make_radial_profile_fig(out_dir: Path):
    r = np.linspace(0, 3.0, 600)
    r_star_a = analytic_r_star()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(r, I_dyn(r), lw=2.2, label="I_dyn(r)")
    ax.plot(r, I_obs(r), lw=2.2, label="I_obs(r)")
    ax.plot(r, A_base(r), lw=2.8, label="A_base(r)=I_dyn+I_obs", color="black")
    ax.axvline(1.0, color="crimson", linestyle="--", lw=1.8, label="target r*=1")
    ax.axvline(r_star_a, color="gray", linestyle=":", lw=1.6, label=fr"analytic argmax={r_star_a:.3f}")
    ax.set_xlabel("radius r")
    ax.set_ylabel("information / acquisition")
    ax.set_title("Radial information profiles and summed acquisition")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_profiles.png", dpi=180)
    plt.close(fig)


def make_spatial_maps_fig(out_dir: Path):
    x = np.linspace(-2.5, 2.5, 401)
    y = np.linspace(-2.5, 2.5, 401)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)

    maps = [I_dyn(R), I_obs(R), A_base(R)]
    titles = ["I_dyn(x,y)", "I_obs(x,y)", "A_base(x,y)"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), constrained_layout=True)
    for ax, Z, title in zip(axes, maps, titles):
        im = ax.contourf(X, Y, Z, levels=45, cmap="viridis")
        ax.contour(X, Y, R, levels=[1.0], colors=["white"], linestyles=["--"], linewidths=[2.0])
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)

    fig.savefig(out_dir / "spatial_maps.png", dpi=180)
    plt.close(fig)


def make_timeseries_fig(out_dir: Path, sim):
    t = np.arange(len(sim["r"])) * PARAMS["dt"]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.3), sharex=True)
    axes[0].plot(t, sim["r"], lw=2.2)
    axes[0].axhline(1.0, color="crimson", linestyle="--", lw=1.6)
    axes[0].set_ylabel("r(t)")
    axes[0].set_title("Trajectory evolution under information-driven radial control")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, sim["A"], lw=2.1, color="black")
    axes[1].set_ylabel("A_base(r(t))")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, sim["u"], lw=1.8, color="tab:green")
    axes[2].axhline(0, color="gray", lw=1)
    axes[2].set_ylabel("u(t)")
    axes[2].set_xlabel("time")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_timeseries.png", dpi=180)
    plt.close(fig)


def make_representative_trajectory_fig(out_dir: Path, sim):
    xg = np.linspace(-2.5, 2.5, 401)
    yg = np.linspace(-2.5, 2.5, 401)
    X, Y = np.meshgrid(xg, yg)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = A_base(R)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    im = ax.contourf(X, Y, Z, levels=50, cmap="magma")
    ax.contour(X, Y, R, levels=[1.0], colors=["cyan"], linestyles=["--"], linewidths=[2])
    ax.plot(sim["x"], sim["y"], color="white", lw=2.0, alpha=0.95)
    ax.scatter(sim["x"][0], sim["y"][0], s=60, c="lime", edgecolor="black", zorder=5, label="start")
    ax.scatter(sim["x"][-1], sim["y"][-1], s=60, c="red", edgecolor="black", zorder=5, label="end")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory on acquisition landscape")
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="A_base")
    fig.tight_layout()
    fig.savefig(out_dir / "representative_trajectory_landscape.png", dpi=180)
    plt.close(fig)


def make_animation(out_dir: Path, sim):
    xg = np.linspace(-2.5, 2.5, 350)
    yg = np.linspace(-2.5, 2.5, 350)
    X, Y = np.meshgrid(xg, yg)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = A_base(R)

    t = np.arange(len(sim["r"])) * PARAMS["dt"]
    stride = 2
    idxs = np.arange(0, len(t), stride)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    ax0, ax1 = axes

    ax0.contourf(X, Y, Z, levels=48, cmap="magma")
    ax0.contour(X, Y, R, levels=[1.0], colors=["cyan"], linestyles=["--"], linewidths=[2])
    ax0.set_aspect("equal")
    ax0.set_xlim(-2.5, 2.5)
    ax0.set_ylim(-2.5, 2.5)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_title("Latent trajectory over A_base landscape")

    line_traj, = ax0.plot([], [], lw=2.2, color="white")
    point, = ax0.plot([], [], "o", color="lime", markersize=6)

    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(0.0, 2.5)
    ax1.axhline(1.0, color="crimson", linestyle="--", lw=1.6, label="target r=1")
    ax1.set_xlabel("time")
    ax1.set_ylabel("radius r(t)")
    ax1.set_title("Radius convergence")
    ax1.grid(alpha=0.25)
    line_r, = ax1.plot([], [], lw=2.3, color="tab:blue", label="r(t)")
    ax1.legend(loc="upper right", fontsize=9)

    def init():
        line_traj.set_data([], [])
        point.set_data([], [])
        line_r.set_data([], [])
        return line_traj, point, line_r

    def update(frame_i):
        i = idxs[frame_i]
        line_traj.set_data(sim["x"][: i + 1], sim["y"][: i + 1])
        point.set_data([sim["x"][i]], [sim["y"][i]])
        line_r.set_data(t[: i + 1], sim["r"][: i + 1])
        fig.suptitle(f"t = {t[i]:.2f}")
        return line_traj, point, line_r

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(idxs),
        init_func=init,
        interval=40,
        blit=False,
    )

    mp4_path = out_dir / "trajectory_evolution.mp4"
    gif_path = out_dir / "trajectory_evolution.gif"

    try:
        writer = animation.FFMpegWriter(fps=25, bitrate=2200)
        ani.save(str(mp4_path), writer=writer, dpi=150)
    except Exception:
        # Fallback to pillow-only workflow if ffmpeg is unavailable
        temp_gif = out_dir / "_temp.gif"
        ani.save(str(temp_gif), writer="pillow", fps=20, dpi=120)
        shutil.copyfile(temp_gif, gif_path)
        if temp_gif.exists():
            temp_gif.unlink()
    else:
        ani.save(str(gif_path), writer="pillow", fps=20, dpi=120)

    plt.close(fig)


def main():
    maybe_archive_previous(OUT_DIR, ARCHIVE_DIR)

    sim = simulate()

    # Figures first
    make_radial_profile_fig(OUT_DIR)
    make_spatial_maps_fig(OUT_DIR)
    make_timeseries_fig(OUT_DIR, sim)
    make_representative_trajectory_fig(OUT_DIR, sim)

    # Then animation
    make_animation(OUT_DIR, sim)

    # Numeric verification of radial argmax
    r_grid = np.linspace(0, 3.0, 20001)
    A_grid = A_base(r_grid)
    idx_max = int(np.argmax(A_grid))
    r_star_num = float(r_grid[idx_max])
    r_star_an = float(analytic_r_star())

    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "analytic": {
            "r_star_formula": "(a_d*r_d + a_o*r_o)/(a_d + a_o)",
            "r_star": r_star_an,
            "condition_for_r_star_eq_1": "a_d*(r_d-1) + a_o*(r_o-1) = 0",
        },
        "numerical": {
            "r_star_estimated": r_star_num,
            "abs_error_to_target_1": abs(r_star_num - 1.0),
            "abs_error_num_vs_analytic": abs(r_star_num - r_star_an),
            "max_A_value": float(A_grid[idx_max]),
        },
        "simulation": {
            "initial_radius": float(sim["r"][0]),
            "final_radius": float(sim["r"][-1]),
            "mean_radius_last_100": float(np.mean(sim["r"][-100:])),
            "std_radius_last_100": float(np.std(sim["r"][-100:])),
        },
        "params": PARAMS,
        "outputs": {
            "radial_profiles": "radial_profiles.png",
            "spatial_maps": "spatial_maps.png",
            "trajectory_timeseries": "trajectory_timeseries.png",
            "representative_figure": "representative_trajectory_landscape.png",
            "animation_gif": "trajectory_evolution.gif",
            "animation_mp4": "trajectory_evolution.mp4",
        },
    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({"params": PARAMS}, f, indent=2)

    summary = f"""Experiment: {EXPERIMENT_NAME}

Model / setup
- Latent dynamics in polar form:
  r_(t+1) = clip(r_t + u_t + eps_t, 0, 3),    eps_t ~ N(0, sigma^2)
  theta_(t+1) = theta_t + omega*dt
  x_t = r_t cos(theta_t), y_t = r_t sin(theta_t)
- omega = {PARAMS['omega']} rad/unit-time is constant and radius-invariant.

Information terms
- I_dyn(r) = C_d - a_d (r-r_d)^2
- I_obs(r) = C_o - a_o (r-r_o)^2
- A_base(r) = I_dyn(r) + I_obs(r)

Chosen parameters
- a_d={PARAMS['a_d']}, r_d={PARAMS['r_d']}, C_d={PARAMS['C_d']}
- a_o={PARAMS['a_o']}, r_o={PARAMS['r_o']}, C_o={PARAMS['C_o']}

Analytic condition for global radial maximizer at r*=1
- Since A_base is a concave quadratic in r (second derivative = -2(a_d+a_o)<0),
  the unique global maximizer is:
    r* = (a_d r_d + a_o r_o)/(a_d + a_o)
- For r*=1, condition is:
    a_d(r_d-1) + a_o(r_o-1) = 0
- With chosen parameters, this holds exactly:
    1.0*(1.6-1) + 1.0*(0.4-1) = 0.

Acquisition used for control
- Radial control uses acquisition gradient and smoothness:
    u_t = clip(k * dA_base/dr(r_t) - beta*u_(t-1), +/-u_max) + eps_t
  where k={PARAMS['k_gain']}, beta={PARAMS['smoothness']}, u_max={PARAMS['control_max']}.
- This drives r_t toward the maximizing ring r=1 while dynamics rotate with constant omega.

Numerical verification
- Estimated argmax radius on dense grid: r*_num = {r_star_num:.6f}
- |r*_num - 1| = {abs(r_star_num-1.0):.6e}
- |r*_num - r*_analytic| = {abs(r_star_num-r_star_an):.6e}

Trajectory behavior
- Initial radius: {sim['r'][0]:.4f}
- Final radius: {sim['r'][-1]:.4f}
- Mean radius over last 100 steps: {np.mean(sim['r'][-100:]):.4f} +- {np.std(sim['r'][-100:]):.4f}

Primary outputs
- radial_profiles.png
- spatial_maps.png
- trajectory_timeseries.png
- representative_trajectory_landscape.png
- trajectory_evolution.gif
- trajectory_evolution.mp4
"""

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
