"""Online PALDI/random comparisons for the Wong-Wang response and input ablations.

The transfer family uses the published response shape, with five learned effective
coefficients. Its nominal parameters are not SNN ground truth. Use held-out SNN
trajectories for predictive evaluation; synthetic reference-parameter error is
not a valid accuracy measure here. The full six-policy manuscript comparison
is in ``wong_wang_full`` and ``wong_wang_campaign``.
"""

from experiments.tnsre.run_tbme_experiments import run_experiment_entrypoint

DEFAULT_SEED_COUNT = 3
PRESETS = {
    "ww_logistic_uniform": "tbme_wong_wang_snn_sessions_m2_evidence",
    "ww_logistic_partial": "tbme_wong_wang_logistic_partial",
    "ww_logistic_synaptic": "tbme_wong_wang_logistic_synaptic",
    "ww_logistic_both": "tbme_wong_wang_logistic_both",
    "ww_transfer_uniform": "tbme_wong_wang_transfer",
    "ww_transfer_partial": "tbme_wong_wang_transfer_partial",
    "ww_transfer_synaptic": "tbme_wong_wang_transfer_synaptic",
    "ww_transfer_both": "tbme_wong_wang_transfer_both",
}
DEFAULT_EXP_IDS = tuple(PRESETS)
EXPERIMENT_SUITES = {
    name: {"experiment_kind": "parameter", "total_steps": 8000,
           "model_ids": ["adaptive", "random"], "env_preset_id": preset,
           "trajectory_eval_interval": 0}
    for name, preset in PRESETS.items()
}


def main(argv=None):
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
