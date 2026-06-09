from __future__ import annotations

BASE_DIR = "results/tbme/exp06_bottleneck"
DEFAULT_SEED_COUNT = 500
DEFAULT_EXP_IDS = (
    "exp06_asymmetric_basin_bottleneck_weak_observation",
    "exp06_asymmetric_basin_bottleneck_tight_action",
    "exp06_asymmetric_basin_bottleneck_combined",
)

EXPERIMENT_SUITES = {
    "exp06_asymmetric_basin_bottleneck_weak_observation": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_weak_observation",
        "model_ids": ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "prbs", "random"],
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp06_asymmetric_basin_bottleneck_tight_action": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_tight_action",
        "model_ids": ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "prbs", "random"],
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp06_asymmetric_basin_bottleneck_combined": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_bottleneck",
        "model_ids": ["active_planning_u20_r20_h40", "active_myopic", "ensemble", "prbs", "random"],
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
}

from experiments.tbme.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
