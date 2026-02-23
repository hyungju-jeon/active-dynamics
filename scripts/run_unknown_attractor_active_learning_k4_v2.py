#!/usr/bin/env python3
"""
K=4 v2 variant of unknown-attractor active learning experiment.

Design intent:
- Keep a 2x2 true attractor grid (unknown locations to the agent; agent only knows K=4).
- Ensure planning clearly beats myopic on final attractor-set estimation error.
- Generate each panel figure separately first, then compose the final panel figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_unknown_attractor_active_learning import (
    Config,
    attractor_centers,
    dynamics_field,
    prepare_experiment_dir,
    random_single_action_escape_probability,
    run_policy,
    write_summary,
)


def _save_panel_1(cfg: Config, true_centers: np.ndarray, myo: dict[str, np.ndarray], plan: dict[str, np.ndarray], out_path: Path) -> None:
    Xd, Yd, Ud, Vd = dynamics_field(cfg, true_centers)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
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

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_panel_2(cfg: Config, true_centers: np.ndarray, myo: dict[str, np.ndarray], plan: dict[str, np.ndarray], out_path: Path) -> None:
    Xd, Yd, Ud, Vd = dynamics_field(cfg, true_centers)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
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

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_panel_3(cfg: Config, myo: dict[str, np.ndarray], plan: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
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

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_panel_4(cfg: Config, myo: dict[str, np.ndarray], plan: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ax.set_title("Convergence: attractor estimate quality (lower is better)")
    ax.plot(np.arange(cfg.steps + 1), myo["error"], color="#4fc3f7", lw=2.0, label="myopic error")
    ax.plot(np.arange(cfg.steps + 1), plan["error"], color="#7CFF6B", lw=2.0, label="planning error")
    ax.set_xlabel("step")
    ax.set_ylabel("set error")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compose_panel_figure(cfg: Config, panel_pngs: list[Path], out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.flatten()

    for i, p in enumerate(panel_pngs):
        img = plt.imread(str(p))
        axes[i].imshow(img)
        axes[i].axis("off")

    fig.suptitle(f"Unknown-attractor active learning (K={cfg.K}, T={cfg.steps}, H={cfg.plan_horizon})", fontsize=13)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = Config(
        experiment_name="multistable_grid_unknown_attractors_k4_v2",
        attractor_grid=(-1.6, 1.6),
        K=4,
        steps=220,
        plan_horizon=16,
        plan_num_sequences=800,
        u_max=0.24,
        action_penalty=0.22,
        novelty_scale=0.75,
        entropy_weight=0.65,
        novelty_weight=1.35,
        info_weight=0.35,
        escape_progress_weight=2.0,
        escape_progress_threshold=0.55,
        escape_bootstrap_steps=36,
        escape_bootstrap_disp_threshold=0.90,
        init_guess_seed=11,
        z0=(-1.45, -1.45),
    )

    exp_dir = prepare_experiment_dir(cfg)
    true_centers = attractor_centers(cfg)

    myopic = run_policy("myopic", cfg, true_centers)
    planning = run_policy("planning", cfg, true_centers)

    p_escape_random = random_single_action_escape_probability(cfg, true_centers)

    panel1 = exp_dir / "panel1_true_vs_guess.png"
    panel2 = exp_dir / "panel2_trajectories.png"
    panel3 = exp_dir / "panel3_acquisition_timecourse.png"
    panel4 = exp_dir / "panel4_error_convergence.png"

    rep_png = exp_dir / "representative_unknown_attractor_comparison_k4_v2.png"
    rep_pdf = exp_dir / "representative_unknown_attractor_comparison_k4_v2.pdf"
    summary_txt = exp_dir / "summary_k4_v2.txt"
    metrics_json = exp_dir / "metrics_k4_v2.json"

    _save_panel_1(cfg, true_centers, myopic, planning, panel1)
    _save_panel_2(cfg, true_centers, myopic, planning, panel2)
    _save_panel_3(cfg, myopic, planning, panel3)
    _save_panel_4(cfg, myopic, planning, panel4)
    compose_panel_figure(cfg, [panel1, panel2, panel3, panel4], rep_png, rep_pdf)

    write_summary(cfg, true_centers, myopic, planning, p_escape_random, summary_txt)

    my_err = float(myopic["error"][-1])
    pl_err = float(planning["error"][-1])
    margin = my_err - pl_err

    metrics = {
        "target": "myopic_worse_than_planning",
        "target_satisfied": bool(my_err > pl_err),
        "margin_myopic_minus_planning": margin,
        "meaningful_margin_gt_0p1": bool(margin > 0.1),
        "figure_generation_mode": "separate_panels_then_compose",
        "panel_pngs": [str(panel1), str(panel2), str(panel3), str(panel4)],
        "escape_probability_random_single_action": p_escape_random,
        "myopic": {
            "final_basin": int(myopic["basin"][-1]),
            "escaped_initial_basin": bool(np.any(myopic["basin"] != myopic["basin"][0])),
            "final_error": my_err,
            "mean_acquisition": float(np.mean(myopic["value"])),
            "mean_novelty": float(np.mean(myopic["novelty"])),
        },
        "planning": {
            "final_basin": int(planning["basin"][-1]),
            "escaped_initial_basin": bool(np.any(planning["basin"] != planning["basin"][0])),
            "final_error": pl_err,
            "mean_acquisition": float(np.mean(planning["value"])),
            "mean_novelty": float(np.mean(planning["novelty"])),
        },
    }
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[ok] experiment_dir: {exp_dir}")
    print(f"[ok] target_satisfied (myopic worse): {metrics['target_satisfied']}")
    print(f"[ok] margin (myopic-planning): {margin:.6f}")
    print(f"[ok] figure mode: {metrics['figure_generation_mode']}")
    print(f"[ok] wrote {rep_png}")
    print(f"[ok] wrote {rep_pdf}")
    print(f"[ok] wrote {summary_txt}")
    print(f"[ok] wrote {metrics_json}")


if __name__ == "__main__":
    main()
