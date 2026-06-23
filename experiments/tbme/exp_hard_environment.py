from __future__ import annotations


DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = (
    "duffing_hard",
    "gated_duffing_hard",
    "damped_pendulum_hard",
)
MODEL_IDS = [
    "adaptive",
    "adaptive_async_realtime",
    "adaptive_async_anytime",
    "active_planning_u20_r20_h40",
    "active_myopic",
    "prbs",
    "random",
    "flex",
    "flex_true_state",
    "ensemble",
    "rhc",
]

EXPERIMENT_SUITES = {
    "duffing_hard": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_duffing_hard",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "gated_duffing_hard": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_hard",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "damped_pendulum_hard": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_damped_pendulum_hard",
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
