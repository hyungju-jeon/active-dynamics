from __future__ import annotations

DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = (
    "gated_duffing",
    "gated_duffing_challenging",
)
MODEL_IDS = [
    "active_planning_u1_r1_h40",
    "active_planning_u5_r5_h40",
    "active_planning_u5_r10_h40",
    "active_planning_u10_r10_h40",
    "active_planning_u5_r20_h40",
    "active_planning_u10_r20_h40",
    "active_planning",
    "adaptive",
    "adaptive_async_anytime",
]
SHARED_EXP_ARGS = {
    "experiment_kind": "parameter",
    "total_steps": 2000,
    "model_ids": MODEL_IDS,
    "trajectory_eval_horizon": 200,
    "trajectory_eval_samples": 100,
}

EXPERIMENT_SUITES = {
    "gated_duffing_challenging": {
        **SHARED_EXP_ARGS,
        "env_preset_id": "tbme_gated_duffing_challenging",
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
