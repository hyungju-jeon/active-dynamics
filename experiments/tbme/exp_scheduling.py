from __future__ import annotations

DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = (
    "duffing",
    "damped_pendulum",
    "gated_duffing",
)
MODEL_IDS = [
    "active_planning_u1_r1_h40",
    "active_planning_u5_r5_h40",
    "active_planning_u10_r10_h40",
    "active_planning_u20_r20_h40",
    "active_planning_u5_r10_h40",
    "active_planning_u5_r20_h40",
    "active_planning_u10_r20_h40",
    "adaptive",
    "adaptive_async_realtime",
]

EXPERIMENT_SUITES = {
    "duffing": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "damped_pendulum": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_damped_pendulum",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "gated_duffing": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin",
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
