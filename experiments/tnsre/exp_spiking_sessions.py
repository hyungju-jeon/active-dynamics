from __future__ import annotations

"""Wong-Wang identification over decision sessions of the Wang (2002) spiking network.

The manuscript ``wong_wang_snn_native_uniform`` suite (2026-10-08) uses native transfer,
uniform current, 20+20 recordings, five evidence-led trials with every identification bound
scaled by 0.1 (PALDI 2 pA, below the 6-pA evidence), then fifteen fixed 2-s trials at the full
bound (20 pA), and 200-ms within-trial PALDI planning. ``wong_wang_snn_native_uniform_strong``
is the 2026-10-07 protocol (ten evidence-led trials at 20 pA, then ten fixed trials). Historical
suites below remain selectable.

In the historical decision-ended protocol, each run has 20 sessions: the circuit starts in the
quiet state, the agent drives it until a decision (``|z1 - z2| > 2``), and 200 ms
later, or after 2 s without a decision, the circuit resets. Agents are given the
session rule: they learn of a reset when it happens and their filters then jump to
the known reset state. The iCEM planners do not simulate resets: a model rollout stays
in the current session for the whole horizon. The run stops after the last session;
``total_steps`` is only an upper bound (20 sessions x 400 bins).

``wong_wang_snn_sessions_m2_evidence`` adds sensory evidence to every session (6 pA to a
random pool for the first 1 s, as in the task sessions); the agents know it, so their
filters, learners, and planners use the action plus the evidence as the circuit's input.
``wong_wang_snn_sessions_m2_evidence_u2`` doubles the agents' input bound (|u| <= 2, 40 pA)
with the same evidence; the readout is calibrated as before (amplitude 1).
``wong_wang_snn_sessions_m2_fixed`` keeps every session for the full 400 bins: a decision
does not end it, so the data include the held choice (fixed-length trials).
``wong_wang_snn_sessions_m2_evmix_*`` run 10 decision-ended sessions without evidence, then 10 with
the task's evidence, which the agents know; ``wong_wang_snn_sessions_m2_fixed_evmix_*`` do the same
with fixed 2-s sessions. ``wong_wang_snn_sessions_m2_evfirst_*`` reverse the order: 10 decision-ended
sessions with evidence, then 10 fixed 2-s sessions without it.
The ``_obs`` suites compare PALDI (also with its reactive cadence) and RHC-US on fixed trials
with the default, two low-SNR (4 or 2 neurons per pool), and a biased (40 + 2 neurons) recording.
"""

DEFAULT_SEED_COUNT = 20
HISTORICAL_EXP_IDS = (
    "wong_wang_snn_sessions_m2",
    "wong_wang_snn_sessions_m2_evidence",
    "wong_wang_snn_sessions_m2_evidence_u2",
    "wong_wang_snn_sessions_m2_fixed",
    # Control-strategy matrix (2026-10-05): six policies, decision or fixed trials, low-SNR or biased recording.
    "wong_wang_snn_sessions_m2_lowsnr4",
    "wong_wang_snn_sessions_m2_biased",
    "wong_wang_snn_sessions_m2_fixed_lowsnr4",
    "wong_wang_snn_sessions_m2_fixed_biased",
    # Second matrix (2026-10-05): 10 + 10, 20 + 20, 20 + 5 neurons; decision, fixed, and mixed-evidence trials.
    "wong_wang_snn_sessions_m2_n10_10",
    "wong_wang_snn_sessions_m2_n20_20",
    "wong_wang_snn_sessions_m2_n20_5",
    "wong_wang_snn_sessions_m2_fixed_n10_10",
    "wong_wang_snn_sessions_m2_fixed_n20_20",
    "wong_wang_snn_sessions_m2_fixed_n20_5",
    "wong_wang_snn_sessions_m2_evmix_n10_10",
    "wong_wang_snn_sessions_m2_evmix_n20_20",
    "wong_wang_snn_sessions_m2_evmix_n20_5",
    "wong_wang_snn_sessions_m2_fixed_evmix_n10_10",
    "wong_wang_snn_sessions_m2_fixed_evmix_n20_20",
    "wong_wang_snn_sessions_m2_fixed_evmix_n20_5",
    "wong_wang_snn_sessions_m2_evfirst_n10_10",
    "wong_wang_snn_sessions_m2_evfirst_n20_20",
    "wong_wang_snn_sessions_m2_evfirst_n20_5",
)
# Matched comparison set of the manuscript (PALDI, Myopic, FLEX, RHC-US, PRBS, Random).
MODEL_IDS = ["adaptive", "active_myopic", "flex_rollback", "rhc", "prbs", "random"]
# PALDI with a 2-bin planning step and a 3-bin minimum cadence, with a 4-s or a 200-ms horizon (2026-10-06),
# run in every suite on request.
PALDI_VARIANTS = ["adaptive_c2", "adaptive_c2h20"]
SHARED_EXP_ARGS = {
    "experiment_kind": "parameter",
    "total_steps": 8000,
    "model_ids": [*MODEL_IDS, *PALDI_VARIANTS],
    "trajectory_eval_horizon": 200,
    "trajectory_eval_samples": 100,
}

EXPERIMENT_SUITES = {
    exp_id: {
        **SHARED_EXP_ARGS,
        "env_preset_id": f"tbme_{exp_id}",
    }
    for exp_id in HISTORICAL_EXP_IDS
}

# Manuscript protocol (2026-10-08): 5 weak evidence-led trials (2 pA), then 15 exploration trials
# (10 pA). The 2026-10-07 protocol (10 evidence-led trials at 20 pA) stays selectable for its archived runs.
EXPERIMENT_SUITES['wong_wang_snn_native_uniform'] = {
    **SHARED_EXP_ARGS,
    'env_preset_id': 'tbme_ww_weak_evidence5_explore10_transfer_uniform',
    'model_ids': ['adaptive_c2h20', *MODEL_IDS[1:]],
    'trajectory_eval_interval': 0,
}
EXPERIMENT_SUITES['wong_wang_snn_native_uniform_strong'] = {
    **EXPERIMENT_SUITES['wong_wang_snn_native_uniform'],
    'env_preset_id': 'tbme_ww_observation_transfer_uniform',
}
DEFAULT_EXP_IDS = ('wong_wang_snn_native_uniform',)

OBS_MODEL_IDS = ["adaptive_reactive", "adaptive", "rhc"]
for _suite, _preset, _models in (
    ("wong_wang_snn_sessions_m2_fixed_obs_default", "tbme_wong_wang_snn_sessions_m2_fixed", ["adaptive_reactive"]),
    ("wong_wang_snn_sessions_m2_fixed_obs_lowsnr4", "tbme_wong_wang_snn_sessions_m2_fixed_lowsnr4", OBS_MODEL_IDS),
    ("wong_wang_snn_sessions_m2_fixed_obs_lowsnr2", "tbme_wong_wang_snn_sessions_m2_fixed_lowsnr2", OBS_MODEL_IDS),
    ("wong_wang_snn_sessions_m2_fixed_obs_biased", "tbme_wong_wang_snn_sessions_m2_fixed_biased", OBS_MODEL_IDS),
):
    EXPERIMENT_SUITES[_suite] = {**SHARED_EXP_ARGS, "model_ids": _models, "env_preset_id": _preset}

from experiments.tnsre.run_tbme_experiments import run_experiment_entrypoint


def main(argv: list[str] | None = None) -> int:
    return run_experiment_entrypoint(globals(), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
