#!/usr/bin/env python3
"""
K=4 variant of unknown-attractor active learning experiment.

Goal:
- 4 true attractors (2x2 grid), hidden from agent.
- Agent knows only K.
- Myopic should perform worse than planning (final attractor-set error).

Outputs are saved under an experiment-specific subfolder and archived on config changes
via prepare_experiment_dir() from the base experiment module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_unknown_attractor_active_learning import (
    Config,
    attractor_centers,
    make_representative_figure,
    prepare_experiment_dir,
    random_single_action_escape_probability,
    run_policy,
    write_summary,
)


def main() -> None:
    cfg = Config(
        experiment_name="multistable_grid_unknown_attractors_k4_v1",
        attractor_grid=(-1.6, 1.6),
        K=4,
        steps=220,
        plan_horizon=16,
        plan_num_sequences=800,
        u_max=0.24,
        # Keep same objective structure as base experiment
        action_penalty=0.22,
        novelty_scale=0.75,
        entropy_weight=0.65,
        novelty_weight=1.35,
        info_weight=0.35,
        # Escape shaping for planning objective
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

    rep_png = exp_dir / "representative_unknown_attractor_comparison_k4.png"
    rep_pdf = exp_dir / "representative_unknown_attractor_comparison_k4.pdf"
    summary_txt = exp_dir / "summary_k4.txt"
    metrics_json = exp_dir / "metrics_k4.json"

    make_representative_figure(cfg, true_centers, myopic, planning, rep_png, rep_pdf)
    write_summary(cfg, true_centers, myopic, planning, p_escape_random, summary_txt)

    my_err = float(myopic["error"][-1])
    pl_err = float(planning["error"][-1])

    metrics = {
        "target": "myopic_worse_than_planning",
        "target_satisfied": bool(my_err > pl_err),
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
    print(f"[ok] wrote {rep_png}")
    print(f"[ok] wrote {rep_pdf}")
    print(f"[ok] wrote {summary_txt}")
    print(f"[ok] wrote {metrics_json}")


if __name__ == "__main__":
    main()
