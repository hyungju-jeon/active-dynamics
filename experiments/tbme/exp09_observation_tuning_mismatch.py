from __future__ import annotations

BASE_DIR = "results/tbme/exp09_observation_tuning_mismatch"
DEFAULT_SEED_COUNT = 500
DEFAULT_EXP_IDS = (
    "exp09_duffing_observation_tuning_mismatch_mild",
    "exp09_duffing_observation_tuning_mismatch_strong",
    "exp09_asymmetric_basin_observation_tuning_mismatch_mild",
    "exp09_asymmetric_basin_observation_tuning_mismatch_strong",
)

MODEL_IDS = [
    "active_planning_adaptive_u20_r20_h40",
    "active_planning_u5_r5_h40",
    "active_planning_u20_r20_h40",
    "active_myopic",
    "prbs",
    "random",
    "ensemble",
]

EXPERIMENT_SUITES = {
    "exp09_duffing_observation_tuning_mismatch_mild": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing_observation_tuning_mismatch_mild",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp09_duffing_observation_tuning_mismatch_strong": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing_observation_tuning_mismatch_strong",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp09_asymmetric_basin_observation_tuning_mismatch_mild": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_observation_tuning_mismatch_mild",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp09_asymmetric_basin_observation_tuning_mismatch_strong": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_observation_tuning_mismatch_strong",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
}

from experiments.tbme.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
