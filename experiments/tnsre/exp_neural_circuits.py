from __future__ import annotations

"""Wong-Wang decision-circuit suite.

Wilson-Cowan is one of the three main benchmark systems and lives in the
shared suites. Wong-Wang shares their observation model, interaction budget,
and policy set, and its preset carries the basin-switch evaluation used to
score how cheaply an identified model can reverse a committed decision.
"""

from experiments.tnsre.exp_simple_system_identification import MODEL_IDS

DEFAULT_SEED_COUNT = 100
DEFAULT_EXP_IDS = ("wong_wang",)
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

from experiments.tnsre.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
