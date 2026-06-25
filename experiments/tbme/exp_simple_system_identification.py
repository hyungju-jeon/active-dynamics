from __future__ import annotations

DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = ("duffing", "damped_pendulum", "gated_duffing")
MODEL_IDS = [
    "adaptive",
    "adaptive_async_anytime",
    "adaptive_async_realtime",
    "active_planning",
    "active_myopic",
    "prbs",
    "random",
    "flex",
    "flex_true_state",
    "rhc",
    "off_policy",
]
SHARED_EXP_ARGS = {
    "experiment_kind": "parameter",
    "total_steps": 2000,
    "model_ids": MODEL_IDS,
    "trajectory_eval_horizon": 200,
    "trajectory_eval_samples": 100,
}

EXPERIMENT_SUITES = {
    "duffing": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_duffing",
    },
    "damped_pendulum": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_damped_pendulum",
    },
    "gated_duffing": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing",
    },
}

from experiments.tbme.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
