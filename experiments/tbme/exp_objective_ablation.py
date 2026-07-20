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
    "confounded_gate": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_confounded_gate",
        "model_ids": [*MODEL_IDS, "random"],
        "total_steps": 500,
    },
    "rank_imbalanced_gate": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_rank_imbalanced_gate",
        "model_ids": ["active_planning", "active_e_optimality", "prbs", "random"],
        "total_steps": 500,
    },
    "compound_tri_gate": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_compound_tri_gate",
        "model_ids": [
            "compound_active_planning",
            "compound_active_fully_observable",
            "compound_active_e_optimality",
            "compound_active_state_information",
            "compound_active_dynamics",
            "compound_active_observation_variance",
            "compound_active_state_variance",
            "prbs",
            "random",
        ],
        "total_steps": 2000,
    },
    "compound_tri_gate_poisson": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_compound_tri_gate_poisson",
        # Reuse the exact policies and schedules from the Gaussian benchmark;
        # only the observation likelihood and loading population change.
        "model_ids": [
            "compound_active_planning",
            "compound_active_fully_observable",
            "compound_active_e_optimality",
            "compound_active_state_information",
            "compound_active_dynamics",
            "compound_active_observation_variance",
            "compound_active_state_variance",
            "prbs",
            "random",
        ],
        "total_steps": 2000,
    },
    "simple_tri_gate_poisson": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_simple_tri_gate_poisson",
        "model_ids": [
            "compound_active_planning",
            "compound_active_fully_observable",
            "compound_active_e_optimality",
            "compound_active_state_information",
            "compound_active_dynamics",
            "compound_active_observation_variance",
            "compound_active_state_variance",
            "prbs",
            "random",
        ],
        "total_steps": 2000,
    },
}

from experiments.tbme.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
