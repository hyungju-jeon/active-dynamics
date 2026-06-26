from __future__ import annotations

DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = (
    "gated_duffing_asymmetric",
    "gated_duffing_challenging",
    "gated_duffing_observation_bottleneck_mild",
    "gated_duffing_observation_bottleneck_strong",
    "gated_duffing_action_bottleneck_mild",
    "gated_duffing_action_bottleneck_strong",
    "gated_duffing_state_noise_mild",
    "gated_duffing_state_noise_strong",
)
MODEL_IDS = [
    "adaptive",
    "adaptive_async_anytime",
    "adaptive_async_realtime",
    "active_planning",
    "active_myopic",
    "prbs",
    "random",
    "flex",
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
    "gated_duffing_asymmetric": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_asymmetric",
    },
    "gated_duffing_challenging": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_challenging",
    },
    "gated_duffing_observation_bottleneck_mild": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_observation_bottleneck_mild",
    },
    "gated_duffing_observation_bottleneck_strong": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_observation_bottleneck_strong",
    },
    "gated_duffing_action_bottleneck_mild": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_action_bottleneck_mild",
    },
    "gated_duffing_action_bottleneck_strong": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_action_bottleneck_strong",
    },
    "gated_duffing_state_noise_mild": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_state_noise_mild",
    },
    "gated_duffing_state_noise_strong": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_state_noise_strong",
    },
}

from experiments.tbme.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
