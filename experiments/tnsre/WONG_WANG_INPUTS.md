# Wong–Wang transfer and input pilot

This worktree tests two independent changes: removing the learner's logistic
response and making inputs to the spiking network less idealized. The original
manuscript experiment remains available with its existing defaults.

The completed 2026-10-04 pilot ran all 24 fits without failure. The transfer
learner preserved held-out prediction accuracy across the four input conditions;
differences from logistic were small and did not establish a general ranking.
Two short online PALDI checks also completed. See the
[results report](../../results/tnsre/20261004_wong_wang_pilot/REPORT.md) for
scores, input costs, checks, and limits.

## Learner

`wong_wang_transfer` uses the response from the Appendix of
[Wong & Wang (2006)](https://www.gatsby.ucl.ac.uk/tea/tea_archive/attached_files/JNS06.pdf):

\[
H(I)=\frac{aI-b}{1-\exp[-d(aI-b)]}
\]

The implementation evaluates \(F(q)=q/(1-e^{-q})\), with \(q=d(aI-b)\).
At zero, \(F(0)=1\), \(F'(0)=1/2\), and \(F''(0)=1/6\). A local series
and separate positive/negative branches avoid division by zero and overflow.
This is an unbounded firing response, not a logistic saturation.

Both learner families use five effective coefficients and the same two NMDA
states, \(z=4(s-1/2)\), normalized time \(\tilde t=t/(100\,\mathrm{ms})\),
and Euler step \(\Delta\tilde t=0.05\):

\[
\frac{dz_i}{d\tilde t}
=4[-s_i+(1-s_i)\gamma F(q_i)],\qquad
q_i=6[w_+c_i-w_-c_j+h+gu_i]
\]

Here \(c=z/4\), \(h=h_{raw}/6\), and \(\gamma,g\) are softplus transforms
of their raw coordinates. The existing logistic learner instead uses
\(\sigma(q_i)\) in place of \(F(q_i)\). Its fitted parameters cannot be
copied to the transfer learner.

The native default maps the published Appendix values exactly:
\(a=270\), \(b=108\), \(d=0.154\), \(\gamma_N=0.641\),
\(J_{self}=0.2609\), \(J_{cross}=0.0497\), and \(I_0=0.3255\), with
currents in nA. The identities are
\(6w_+=daJ_{self}\), \(6w_-=daJ_{cross}\),
\(h_{raw}=d[a(I_0+(J_{self}-J_{cross})/2)-b]\),
\(\gamma=0.1\gamma_N/d\), and \(6g=da(0.020)\).

These are initialization values, not SNN ground truth. Fitting changes all five
effective coefficients. The published Appendix omits recurrent AMPA; the SNN
retains it. Both learners therefore remain approximate two-state models.

## Independent input changes

| Arm | Controlled current | Sensory evidence |
| --- | --- | --- |
| `uniform_current` | Every selective neuron, equal strength | Uniform current |
| `partial_current` | Fixed 50% subset, unequal strength | Uniform current |
| `synaptic_evidence` | Every selective neuron, equal strength | Independent Poisson streams through AMPA |
| `both` | Fixed 50% subset, unequal strength | Independent Poisson streams through AMPA |

Control always acts directly on membrane voltage. A seeded subset of 120 of
240 neurons per pool receives lognormal weights (target CV 0.25). Weights sum
to 240, so a unit action still delivers 4,800 pA summed over each pool.
Individual peaks and summed squared current increase. The pilot saves the
exact weights, peak current, and squared-current ratio; it does not claim
equal energy across targeting conditions.

Sensory evidence remains separate from control at the simulator interface:
`network.step(action, evidence=evidence)`. Current evidence is uniform even
when control targets a subset. For synaptic evidence, each selective neuron
receives its own Poisson stream, which increments external AMPA gating with
a 0.5-ms synaptic delay and 2-ms decay. Background input remains 2,400 Hz.
At -55 mV, 2.1-nS AMPA conductance gives 0.231 pA per Hz in steady mean.
Thus the nominal 6-pA evidence becomes 25.97 Hz per neuron. This is a match at
one voltage, not an exact equivalence of mean current at every state, variance,
or transient response.

Both learners still receive nominal `action + evidence`, with one effective
input gain. Separate current and synaptic pathways can therefore introduce
additional model mismatch. Learning separate pathway gains would be a further
ablation, not part of this comparison.

## Reproduce the bounded pilot

Run from this worktree with the shared project Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv/bin/python -m experiments.tnsre.wong_wang_input_pilot \
  --out results/tnsre/NEW_OUTPUT_DIRECTORY
```

The default pilot uses the full 1,600-excitatory/400-inhibitory network, 80
recorded neurons, 0.1-ms integration, and 5-ms control bins. It crosses two
learner families with four input arms and three replicate blocks: 24 fits.
Each arm/replicate has six training and four held-out trajectories, each 1.5 s.
Controls change every 200 ms. Evidence drives a seeded pool for the first 1 s.

Fits use true recorded NMDA proxies. Each uses four initializations, float64,
300 Adam iterations at learning rate 0.025, gradient norm bound 10, and 500-ms
rollouts with starts every 250 ms. Initialization centers are family-specific:
the old logistic reference and the published transfer mapping, with the same
seeded perturbations. Training MSE selects the initialization and iterate;
held-out data never enter fitting or selection. Equal optimization budgets do
not establish that either nonconvex fit is globally optimal.

The reported R2 pools errors over two coordinates and all held-out prediction
windows, using a separate target mean per coordinate. Each prediction starts
from the recorded state; there are no subsequent spike corrections, clipping,
or excluded trajectories. Overlapping windows are not independent replicates.
The plot shows all three replicate differences and their median, with no
confidence interval or significance claim.

The same control programs, evidence schedules, and seed labels are used across
input arms. Adding Poisson groups changes consumption of Brian2's random
stream: matching seed labels across evidence modes does not make background
spike realizations identical. Learner comparisons within an arm use exactly
the same saved trajectories. Mask seeds change between replicate blocks.

Each output directory contains a manifest, source snapshots and hashes,
working-tree patch, train/test data, all fit histories and initializations,
predictions, input diagnostics, scores, and a vector/raster comparison figure.
Existing output directories are refused to preserve previous attempts.

This pilot tests response-family fit and input feasibility. It does not measure
online identification speed, spike-filter accuracy, or closed-loop control
success. Readout calibration from privileged latent states remains an
assumption of the online experiment.

## Online entry point

`exp_wong_wang_inputs.py` exposes eight suites, with PALDI and random policies.
The default is a larger three-seed, 20-session study; it is separate from the
bounded fit pilot:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv/bin/python -m experiments.tnsre.exp_wong_wang_inputs \
  --exp-ids ww_transfer_uniform,ww_transfer_partial,ww_transfer_synaptic,ww_transfer_both \
  --seeds 0,1,2 --mode run --base-dir results/tnsre/NEW_ONLINE_DIRECTORY
```

Use held-out SNN trajectories for evaluation. Distance to the nominal
`true_params` field is not a biological parameter-recovery score; that field is
required by the existing runner and is only a reference here. Synthetic
reference-rollout evaluation is disabled for these suites. This pilot entry point
includes PALDI and random; the full campaign below also enables FLEX through its
transfer-family model adapter.

## Checks

Focused tests cover the published physical equation, threshold value and first
two derivatives, finite-difference Jacobians, pool symmetry, continuous-time
gate boundaries, input-dependent forward steps, independent preset switches,
target masks, total-current normalization, actual neuron-current assignment,
sensory spike generation, known evidence, and exact repeatability after reset.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv/bin/python -m pytest -q tests/test_input_dependent_dynamics.py \
  tests/test_spiking_decision.py tests/test_decision_sessions.py
```

## Full manuscript comparison on Lambda

`wong_wang_campaign.py` runs the full six-policy comparison on a frozen source
snapshot. Identification uses 20 seeds and 20 decision trials without sensory
evidence. Four training arms cross logistic/original response with uniform/partial
stimulation. Each trained model is then tested with current and Poisson evidence,
so sensory input changes are tested independently of training.

Prediction follows the manuscript: 12 held-out trajectories, 100-bin rollouts,
stride 20, and pooled target variance. Change-of-mind control uses checkpoints
0, 1, 2, 5, and 20; eight task seeds; budgets 10 and 12; and warm-start iCEM.
This gives 480 training runs and 66,880 task evaluations, including prior,
privileged-fit, and model-free references. The family-specific generic priors from
the pilot remain; the response comparison includes this prior choice.

The historical reference-fit script was removed. New privileged references use
48 independent training trajectories, 16 starts, and 300 Adam iterations with
learning rate 0.05. The reconstructed task-like stimulus schedule is explicit in
`wong_wang_full.data`; these references are not exact reproductions of the old fit.
Selection uses training MSE only, without control-test outcomes.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.tnsre.wong_wang_campaign --root <results-dir> --workers 32
python -m experiments.tnsre.wong_wang_compare --root <results-dir>
```

The campaign records each command, attempt log, completion state, and runtime.
The comparison checks exact job coverage, all 20 training trials, finite learned
parameters, unique task keys, paired validity, and input budgets. Tables retain
negative prediction scores. Paired bootstrap intervals use training seeds as
replicates and are conditional on the fixed task panel. Input contrasts use only
task seeds valid in both conditions; within-condition success tables use all
valid task seeds. Source hashes, the uncommitted patch, pinned `uv.lock`, and exact
external dependency files travel with the remote snapshot.

The full run exposed three singular-matrix failures in the non-sigmoid model's
state filter. Two were for PALDI training seed 6 after trial 1, task seed 91000, at
budgets 10 and 12. An isolated replay reproduced both failures at bin 172 with
zero stimulation energy spent. The estimated second state had reached 19.71,
outside the physical latent range [-2, 2]. This is a failed controller estimate;
the recovery does not regularize its matrix or change its model or planner.
Recovery found a third failure for PRBS training seed 3 after trial 1, task seed
91000, budget 10, at bin 127. All three cases use uniform current input and failed
before reaching the control goal.

`wong_wang_recover.py` waits for the frozen campaign to stop, then completes the
interrupted shards. Original shard CSVs and failure logs remain unchanged; new
rows use `worker_recovery_*.csv`. Numerical exceptions retain their SNN traces
and explicit flags. A valid task that crashes before reaching the goal is
unsuccessful and remains in the denominator. A goal reached in the recorded
prefix remains a success. Other exceptions still stop the recovery. The separate
`recovery.json` records completion without rewriting the original failed jobs.

The completed campaign is in `results/tnsre/20261004_wong_wang_full/`. Its
`comparison/REPORT.md` gives the findings, paired intervals, historical baseline
differences, and limits. PALDI retains useful prediction and control with the
original response; Myopic performs worse. Partial stimulation improves PALDI
success while increasing squared-current dose. Poisson evidence preserves useful
control. The fresh sigmoid baseline differs from the saved manuscript baseline,
so all response and input contrasts use the fresh baseline from this campaign.

The six-method ranking does not include model-free references. Maximum-amplitude
control reaches 100% success in the partial-stimulation conditions and 87.5% with
uniform stimulation and Poisson evidence, exceeding PALDI in those conditions.
The complete reference results are in `comparison/reference_control_summary.csv`
and the corrected report. Thus the input variants support useful learned control
but do not preserve the manuscript advantage over maximum-amplitude control.
The manuscript's additional budget settings 200 and 600 pA² s were not included
in this campaign; its control evaluations cover nominal budgets 400 and 480 pA² s.

## Reduced control bounds

`wong_wang_budget.py` reuses the final trial-20 models and tests action bounds
0.25 and 0.5, compared with the original bound 1. The normalized budgets are
`[5, 10, 12, 15] * action_max**2`, so all four budgets remain below the maximum
cost allowed by the two-second control window. This is a control-only change;
identification inputs and the learned models remain those of the full campaign.

Every input condition includes all six identification methods, the initial model,
the privileged fit, and all three model-free controls. Each evaluation records
the peak action norm, peak individual-neuron current, and squared-current integral
summed over the stimulated neurons. Partial targeting still uses higher physical
dose at a given nominal action. Both bounds and all budgets are retained regardless
of outcomes; no budget point is selected to favor a method.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.tnsre.wong_wang_budget --stage campaign \
  --parent <full-campaign-results> --root <new-budget-results> --workers 32 --nshards 32
```

The fixed protocol and remote status are in
`results/tnsre/20261004_wong_wang_bounds/PROTOCOL.md`. Exact expected coverage is
83,072 task evaluations. A full result requires campaign status `completed`,
the coverage and constraint checks in `comparison/verification.json`, and an
empty copyback mismatch list. Each worker starts in a fresh process and builds
one network; the preflight checks use the same process isolation.
