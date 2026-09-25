from __future__ import annotations

"""Wong-Wang identification over decision sessions of the Wang (2002) spiking network.

The learner's drift is Wong-Wang with the input inside the transfer function
(``wong_wang_inside_gain``). Each run is ``spiking_sessions`` (20) decision sessions: the circuit starts in the
quiet state, the agent drives it until a decision (``|z1 - z2| > 2``), and 200 ms
later, or after 2 s without a decision, the circuit resets. Agents are given the
session rule: their filters jump to the known reset state and the iCEM planners
simulate resets inside their rollouts. The run stops after the last session;
``total_steps`` is only an upper bound (20 sessions x 400 bins).
"""

DEFAULT_SEED_COUNT = 20
DEFAULT_EXP_IDS = ("wong_wang_snn_sessions_m2",)
# Matched comparison set of the manuscript (PALDI, Myopic, FLEX, RHC-US, PRBS, Random).
MODEL_IDS = ["adaptive", "active_myopic", "flex_rollback", "rhc", "prbs", "random"]
SHARED_EXP_ARGS = {
    "experiment_kind": "parameter",
    "total_steps": 8000,
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
