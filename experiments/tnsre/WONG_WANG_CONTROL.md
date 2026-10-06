# Frozen-model control study

This study carries out three requested tests: diagnose planner choices, separate
the peak-input constraint from the energy budget, and evaluate activity tracking.
The objective is to establish whether PALDI gives better held-out control than
each baseline. A desired ranking is a hypothesis, not a selection criterion.

## Fixed inputs and comparison

Reuse the 20-session models from `20261004_wong_wang_full`, including all six
identification methods, initial models, privileged fitted references, and their
calibrated readouts. Export exact parameters and input-file hashes with
`wong_wang_control_study --stage export`. Keep both response families and all
four input conditions available. The first smoke tests use native transfer with
uniform/current and partial/Poisson inputs. They do not establish a result for
the other six arms.

All learned models use the same iCEM algorithm, spike-count EKF, observations,
input bound, energy budget, and task. The SNN latent proxy is used only to score
outcomes and to announce the shared initial decision event. True-state planner
predictions are explicitly privileged diagnostics, never PALDI results.

Include no control, constant-amplitude, maximum-amplitude, tuned pulse, relay,
and PI controls. Relay and PI use static Poisson-readout inversion from smoothed
spike counts, with each training seed's calibrated readout. Pulse controls have
one outcome per task seed; do not treat repeated training seeds as additional
independent pulse observations. Report the privileged fitted controller even
when it exceeds PALDI.

## Tests

1. At the first uncontrolled decision, compare the selected iCEM plan with the
   front-loaded maximum pulse and the constant pulse. Score all three under the
   learned model from its EKF estimate and from the true onset state. Replay all
   three complete open-loop plans in the SNN from identical initializations.
   These replays match the prediction horizon; closed-loop outcomes are a
   separate test. Preserve state error, predicted soft/hard success, actual
   success, energy, and complete action/state traces.
2. Hold peak amplitude at 1 and evaluate budgets 6, 8, 10, 12, 14, 16. Separately
   evaluate bounds 0.5, 0.75, 1 with budget 10. The 0.5 bound permits at most
   energy 5 in the two-second window, so the nominal budget is nonbinding there.
   Report achieved energy and the entire grid, including zeros. Never interpret
   a joint change of bound and budget as an isolated amplitude effect.
3. Control for two seconds after the first decision and track the NMDA-proxy
   population gap. Three preset references are zero, +1, and +1 switching to -1
   at one second (in coordinates where the target opposes the original choice).
   Score every bin after the first 0.5 seconds. Primary endpoint is RMSE;
   secondary endpoints are time within ±0.35 of target, and success defined as
   at least 90% within that band. Do not terminate on first entry. Use budgets
   6, 10, 14 at bound 1 for calibration. These are gating-proxy tracking tasks,
   not claims that firing rates have the same numeric targets.

The tracking planner minimizes expected squared gap error. The reversal planner
keeps its original soft first-passage objective. Their common iCEM sampling code
is factored into `optimize_icem`; verify that this refactor preserves the original
planner's controls and scores on deterministic replay fixtures.

## Calibration and confirmation

- Diagnostics may reuse 91000–91007 to explain the previously reported outcomes.
- Smoke and calibration seeds: 120000–120015. Initial smoke uses four of these.
- Development comparison: 121000–121007.
- Untouched confirmation: 122000–122063, extended with a new disjoint block only
  if a precision calculation made before that extension requires it.
- Existing training seeds 0–19 stay paired across identification methods. This
  confirms control on new network-noise seeds, conditional on the frozen learned
  models and fixed stimulation masks; it does not supply new identification
  trials or independent network architectures.

Choose pulse timing/amplitude and feedback gains on calibration data only, using
the task's declared endpoint. Choose tracking feasibility using the privileged
fit and simple controllers, before comparing PALDI with active baselines. Keep
all tested configurations in the report. Freeze the confirmation plan before
opening confirmation results. Any subsequent controller change needs another
fresh confirmation block, applied identically to all learned models.

Use paired resampling across training seeds and task seeds; compare PALDI with
every baseline, and report intervals with multiplicity control for the declared
primary comparisons. A rank from means alone does not establish an advantage.
For overturn, success is primary and energy is reported for all valid sessions.
Do not compare energy only among successes without also showing success rates.

## Feedback controller experiment after confirmation

The completed `confirmation_v1` did not establish an advantage over every
baseline. Its results and frozen plan remain the original confirmation. Saved
traces motivate a separate controller experiment: the original planner scores
fixed future inputs, while the PI and relay baselines correct inputs every
5 ms. Retrospective scoring of their realized input sequences cannot separate
model error from the benefit of future feedback. It is not a causal test.

An opt-in plan field selects the shared controller in `wong_wang_feedback.py`:

```json
"learned_controller": {"name": "feedback_tracking", "gain": 2.0}
```

In target-swapped coordinates, let `d = (-1, 1) / sqrt(2)`. The policy is
`u_t = project(v_t + gain * (r_t - (z_1 - z_0)) * d)`. iCEM optimizes the
feedforward sequence `v_t`, with a correction in every simulated 5-ms bin.
Each simulated trajectory has its own remaining energy, and every action is
projected onto both the peak limit and that remaining budget. The reported
planner `feedforward_energy` is not the physical energy; trial reports continue
to compute physical energy from the actions actually sent to the SNN.

Execution uses the latest spike-count EKF estimate at every input bin. It has
no access to the true SNN state. The forecast assumes feedback on its simulated
state and omits future observation errors; this approximation must be assessed
in the SNN. Feedforward replanning remains every 200 ms, with 50-ms input blocks.
All six learned methods, the initial model, and the privileged fit receive the
same controller variant. Pulse, relay, and PI settings and scoring stay fixed.

The first matched GCP smoke comparison uses gains absent (original controller)
and 2, training indices 0 and 1, all eight input/model arms, all 14 controllers,
and calibration task seeds 120000--120003. Gain 2 is a declared prototype,
not a fit to confirmation scores. These reused seeds provide integration and
development evidence only. Freeze another fresh task-seed block before any new
confirmation, and retain the unsuccessful original confirmation in the report.
Do not interpret a smoke ranking as an all-baseline advantage.

Plans without the new field keep their original job keys and controller path.
Mathematical checks cover causal feedback under known additive noise, exact
linear dynamics, per-trajectory bounds, pool exchange, correction between
replans, and the zero-gain limit.

## Failures and provenance

Numerical controller failures count as unsuccessful valid tasks. Tracking
failures receive RMSE 5, the largest possible error for a bounded physical gap
in [-4,4] and references in [-1,1]. Save their traces and exceptions. Report
invalid uncontrolled trials and check that validity is identical across methods.

Each phase has an immutable plan, expected row keys, source snapshot, input
hashes, per-attempt logs, and append-only result shards. Verify complete unique
coverage, finite scores, amplitude and energy constraints, and copyback before
claiming completion. Preserve failed attempts. All remaining simulations run
on the previously approved four GCP CPU VMs. The stopped Lambda development
run's completed trials are preserved and checked before resuming its unfinished
shards with the same source commit, models, plan, and seeds on GCP.

Run one network/input condition per Python process, as the campaign workers do.
A replay check that created two different conditions sequentially in one process
did not reproduce the second condition's initial decision. Separate-process
checks reproduced all four saved PALDI/maximum-pulse cases. The failed check is
retained under the study's provenance directory; it is not a campaign outcome.

The tuning grid has 18 pulses (three amplitudes, two delays, three durations),
nine relay settings (three deadbands and three count-smoothing times), and ten
PI settings (eight gain pairs at 50 ms smoothing, plus two smoothing ablations).
Tune each on all 16 calibration task seeds and two matched readouts where needed.
The calibrated readouts are identical across response families; the selection
code checks that identity before sharing model-free settings between families.

The diagnostic report also reads each saved action/state trace and writes
`diagnostic_actions.csv` and `diagnostic_actions_summary.csv`. It resolves input
energy into orthogonal common and differential components, and reports energy
delivered in the first 40 and 100 bins plus the bin when half the energy has been
delivered. A bin is 5 ms in this experiment. These fractions describe prescribed
complete open-loop plans, including stimulation after a threshold crossing.
Common input is not necessarily ineffective in a nonlinear network. Differences
in timing and direction suggest mechanisms; only matched ablations can establish
their causal effects. Trace hashes, peak-bound checks, reconstructed peak gaps,
and the orthogonal energy identity are recorded in
`diagnostic_action_verification.json`.

## Reproducible comparison figures

After a phase has complete verified rows and a comparison report, run:

```sh
.venv/bin/python -m experiments.tnsre.figures.wong_wang_control \
  --models results/tnsre/20261005_wong_wang_control/inputs/models.json \
  --phase-dir results/tnsre/20261005_wong_wang_control/results/calibration_v1
```

The same command accepts diagnostic, smoke, development, and confirmation phase
directories. It checks the summary against raw outcomes and writes PDF, SVG,
color and grayscale previews, plotted values, captions, and input hashes under
`comparison/figures`. The canvas uses the existing 516-pt manuscript width.

Constraint curves show pointwise crossed-bootstrap intervals of mean success;
they do not test paired differences. Paired comparison panels use the report's
simultaneous intervals. Single-model-seed smoke panels have descriptive points
without confidence intervals. Diagnostic bars show complete open-loop plan
replays on the declared reused seeds, not closed-loop MPC success rates.
