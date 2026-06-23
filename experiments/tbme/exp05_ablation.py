from __future__ import annotations

BASE_DIR = "results/tbme/exp05_ablation"
DEFAULT_SEED_COUNT = 500
DEFAULT_EXP_IDS = (
    "exp05_asymmetric_basin_objective_ablation",
    "exp05_hard_asymmetric_basin_objective_ablation",
)
MODEL_IDS = [
    "active_planning_u20_r20_h40",
    "adaptive_async_realtime",
    "active_fully_observable_u20_r20_h40",
    "active_e_optimality_u20_r20_h40",
    "active_state_information_u20_r20_h40",
    "active_dynamics_u20_r20_h40",
    "active_sampling_variance_u20_r20_h40",
    "ensemble",
    "prbs",
]

EXPERIMENT_SUITES = {
    "exp05_asymmetric_basin_objective_ablation": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp05_hard_asymmetric_basin_objective_ablation": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_hard",
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
