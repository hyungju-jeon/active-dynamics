from __future__ import annotations

DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = (
    "gated_duffing",
    "gated_duffing_asymmetric",
    "gated_duffing_challenging",
)

MODEL_IDS = [
    "prbs",
    "adaptive",
    "active_planning",
    "active_fully_observable",
    "active_e_optimality",
    "active_state_information",
    "active_dynamics",
    "active_observation_variance",
    "active_state_variance",
]
SHARED_EXP_ARGS = {
    "experiment_kind": "parameter",
    "total_steps": 2000,
    "model_ids": MODEL_IDS,
    "trajectory_eval_horizon": 200,
    "trajectory_eval_samples": 100,
}

EXPERIMENT_SUITES = {
    "gated_duffing": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing",
    },
    "gated_duffing_asymmetric": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_asymmetric",
    },
    "gated_duffing_challenging": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_challenging",
    },
}

from experiments.tbme.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
