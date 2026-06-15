from __future__ import annotations

BASE_DIR = "results/tbme/exp02_hard"
DEFAULT_SEED_COUNT = 500
DEFAULT_EXP_IDS = (
    "exp02_hard_duffing",
    "exp02_hard_asymmetric_basin",
    "exp02_hard_damped_pendulum",
)

EXPERIMENT_SUITES = {
    "exp02_hard_duffing": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing_hard",
        "model_ids": [
            "active_planning_adaptive_u20_r20_h40",
            "active_planning_u20_r20_h40",
            "active_myopic",
            "prbs",
            "random",
            "flex",
            "flex_true_state",
            "ensemble",
            "rhc",
        ],
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp02_hard_asymmetric_basin": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_hard",
        "model_ids": [
            "active_planning_adaptive_u20_r20_h40",
            "active_planning_u20_r20_h40",
            "active_myopic",
            "prbs",
            "random",
            "flex",
            "flex_true_state",
            "ensemble",
            "rhc",
        ],
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp02_hard_damped_pendulum": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_damped_pendulum_hard",
        "model_ids": [
            "active_planning_adaptive_u20_r20_h40",
            "active_planning_u20_r20_h40",
            "active_myopic",
            "prbs",
            "random",
            "flex",
            "flex_true_state",
            "ensemble",
            "rhc",
        ],
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
}

from experiments.tbme.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
