from __future__ import annotations

BASE_DIR = "results/tbme/exp03_schedule"
DEFAULT_SEED_COUNT = 500
DEFAULT_EXP_IDS = (
    "exp03_schedule_duffing",
    "exp03_schedule_damped_pendulum",
    "exp03_schedule_asymmetric_basin",
)

EXPERIMENT_SUITES = {
    "exp03_schedule_duffing": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing",
        "model_ids": [
            "active_planning_u1_r1_h40",
            "active_planning_u5_r5_h40",
            "active_planning_u10_r10_h40",
            "active_planning_u20_r20_h40",
            "active_planning_u5_r10_h40",
            "active_planning_u5_r20_h40",
            "active_planning_u10_r20_h40",
            "active_planning_adaptive_u20_r20_h40",
            "active_planning_adaptive_async_realtime_u20_r20_h40",
        ],
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp03_schedule_damped_pendulum": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_damped_pendulum",
        "model_ids": [
            "active_planning_u1_r1_h40",
            "active_planning_u5_r5_h40",
            "active_planning_u10_r10_h40",
            "active_planning_u20_r20_h40",
            "active_planning_u5_r10_h40",
            "active_planning_u5_r20_h40",
            "active_planning_u10_r20_h40",
            "active_planning_adaptive_u20_r20_h40",
            "active_planning_adaptive_async_realtime_u20_r20_h40",
        ],
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "exp03_schedule_asymmetric_basin": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin",
        "model_ids": [
            "active_planning_u1_r1_h40",
            "active_planning_u5_r5_h40",
            "active_planning_u10_r10_h40",
            "active_planning_u20_r20_h40",
            "active_planning_u5_r10_h40",
            "active_planning_u5_r20_h40",
            "active_planning_u10_r20_h40",
            "active_planning_adaptive_u20_r20_h40",
            "active_planning_adaptive_async_realtime_u20_r20_h40",
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
