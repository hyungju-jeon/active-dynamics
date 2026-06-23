from __future__ import annotations

DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = (
    "gated_duffing_observation_bottleneck_mild",
    "gated_duffing_observation_bottleneck_strong",
    "gated_duffing_action_bottleneck_mild",
    "gated_duffing_action_bottleneck_strong",
)
MODEL_IDS = [
    "adaptive",
    "adaptive_async_realtime",
    "active_planning_u20_r20_h40",
    "active_myopic",
    "ensemble",
    "prbs",
    "random",
]

EXPERIMENT_SUITES = {
    "gated_duffing_observation_bottleneck_mild": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_observation_bottleneck_mild",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "gated_duffing_observation_bottleneck_strong": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_observation_bottleneck_strong",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "gated_duffing_action_bottleneck_mild": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_action_bottleneck_mild",
        "model_ids": MODEL_IDS,
        "trajectory_eval_horizon": 200,
        "trajectory_eval_samples": 100,
    },
    "gated_duffing_action_bottleneck_strong": {
        "experiment_kind": "parameter",
        "total_steps": 2000,
        "env_preset_id": "tbme_asymmetric_basin_action_bottleneck_strong",
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
