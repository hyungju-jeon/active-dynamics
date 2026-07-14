from __future__ import annotations

DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = (
    "duffing",
    "damped_pendulum",
    "gated_duffing",
    "gated_duffing_asymmetric",
    "gated_duffing_challenging",
    "gated_duffing_observation_bottleneck_mild",
    "gated_duffing_observation_bottleneck_strong",
)
MODEL_IDS = [
    "flex",
    "flex_filter",
    "flex_true",
    "flex_rollback",
]
SHARED_EXP_ARGS = {
    "experiment_kind": "parameter",
    "total_steps": 2000,
    "model_ids": MODEL_IDS,
    "trajectory_eval_horizon": 200,
    "trajectory_eval_samples": 100,
}

EXPERIMENT_SUITES = {
    exp_id: {
        **SHARED_EXP_ARGS,
        "env_preset_id": f"tbme_{exp_id}",
    }
    for exp_id in DEFAULT_EXP_IDS
}


from experiments.tbme.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
