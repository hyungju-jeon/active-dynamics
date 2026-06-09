from __future__ import annotations

BASE_DIR = "results/tbme/exp01_base"
DEFAULT_SEED_COUNT = 500
DEFAULT_EXP_IDS = ("exp01_duffing", "exp01_damped_pendulum", "exp01_asymmetric_basin")

EXPERIMENT_SUITES = {
    "exp01_duffing": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing",
        "model_ids": [
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
    "exp01_damped_pendulum": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_damped_pendulum",
        "model_ids": [
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
    "exp01_asymmetric_basin": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin",
        "model_ids": [
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
