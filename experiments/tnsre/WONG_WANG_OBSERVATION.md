# Native transfer and partial stimulation under the updated manuscript protocol

Selected for the manuscript on 2026-10-07: **native transfer + uniform current**,
with the 200-ms PALDI horizon and within-trial planning below. The code is merged
into `neuro`. Its data and figures are archived in
`results/archive/experiments/20261007_snn_native_uniform/`; since 2026-10-08 the
manuscript uses five weak evidence trials (`results/tnsre/20261008_snn_native_uniform_weak5_x10_ball/`,
see `README.md`). The other arms, earlier protocols,
and original failed/recovered attempts are in `results/archive/experiments/`.
The additional manuscript checkpoints and budgets use the same frozen models
through `native_manuscript.py`. The design below records the original comparison.

This matched test asks whether the native response and partial stimulation retain
useful prediction and PALDI control performance after the manuscript update. No
result is assumed. Run in the `wong-wang-realistic-input` worktree on GCP only.

The former manuscript source is now `results/archive/experiments/20261006_snn_evidence_first_n20_20` in the
main checkout. Its executed protocol is from `snn-within-trial` (f0e2140;
planner correction 3c77303; short-horizon PALDI a2bec6f). The relevant session and
planner implementation is ported from a9aa9d8. This is a scoped port, not a merge.

## Matched design

- Four arms: sigmoid/uniform, native/uniform, sigmoid/partial, native/partial.
- Each arm: PALDI (`adaptive_c2h20`), Myopic, FLEX, RHC-US, PRBS, Random; 20
  training seeds (0–19); 20 trials. All 480 models are trained again.
- Record the first 20 neurons in each pool: 40 observed neurons, 5-ms bins.
  Readouts are calibrated per seed and shared by all methods in an arm.
- Trials 1–10: 6-pA known current evidence to one pool for the first second;
  reset 200 ms after the decision or at the 2-s limit. Trials 11–20: no evidence;
  fixed 2-s trials. Planners remain in the current trial and do not predict resets.
- PALDI: 20 planning steps × 10 ms; minimum update/replan cadence 15 ms.
  Other methods retain the manuscript settings, including their original action
  constraint conventions. The task controller uses the same L2 bound for all.
- Partial stimulation retains the original fixed half-population mask, unequal
  weights, and total-current normalization. It does not match squared-current
  dose and does not expose individual weights to the learner.
- Current evidence only in this comparison; no Poisson arm is silently added.
- Prediction: saved R² at every identification trial, 12 held-out trajectories.
- Primary control comparison: final trial 20, reversal budgets 10 and 12, peak
  norm 1, task seeds 91000–91007, as in the manuscript. Every arm includes the
  prior, separately fitted reduced model, no input, constant input, and maximum
  pulse. Reference fitting uses the independent fitting trajectories of our
  realistic-input campaign and selects only by fitting loss.

The original data, budgets, and figures are preserved. This new campaign does
not estimate isolated effects of observation count versus the changed trial
schedule: all four arms use the complete updated setup.

## Checks and analysis

The driver verifies 120 completed models per arm, 20 trial checkpoints, 40×2
readout shapes, identical per-seed readouts across methods, and the exact switch
in evidence and trial length. All expected control rows must exist without
duplicates; valid starts and numerical failures are retained. Failed valid
controllers count as unsuccessful, following the previous campaign convention.

Report all baseline means, median/IQR prediction, paired family and targeting
contrasts, and paired PALDI-baseline differences. Do not select successful seeds
or infer superiority from a higher mean alone. Intervals are conditional on the
fixed task panel and fixed stimulation mask. Comparison with the earlier
manuscript is descriptive until the fresh sigmoid/uniform arm reproduces it.

One arm per GCP VM, eight subprocess workers per 16-vCPU VM. Each process is
single-threaded. The first complete seed-0 PALDI run is a preflight and is reused
in the full cohort. Job commands, attempts, errors, and outputs are retained.

```sh
python -m experiments.tnsre.wong_wang_observation \
  --root results/ww_observation_20261006/transfer_partial \
  --arm transfer_partial --workers 8 --nshards 32
```
