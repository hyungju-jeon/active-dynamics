from __future__ import annotations

BASE_DIR = "results/tbme/exp08_parameter_mismatch_stress"
DEFAULT_SEED_COUNT = 500
DEFAULT_EXP_IDS = (
    "exp08_duffing_parameter_mismatch_mild",
    "exp08_duffing_parameter_mismatch_strong",
    "exp08_asymmetric_basin_parameter_mismatch_mild",
    "exp08_asymmetric_basin_parameter_mismatch_strong",
)

MODEL_IDS = [
    "active_planning_u5_r5_h40",
    "active_planning_u20_r20_h40",
    "active_myopic",
    "prbs",
    "random",
    "ensemble",
]

EXPERIMENT_SUITES = {
    "exp08_duffing_parameter_mismatch_mild": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing_parameter_mismatch_mild",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp08_duffing_parameter_mismatch_strong": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing_parameter_mismatch_strong",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp08_asymmetric_basin_parameter_mismatch_mild": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_parameter_mismatch_mild",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp08_asymmetric_basin_parameter_mismatch_strong": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_parameter_mismatch_strong",
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
