# TBME Experiments

This directory contains the TBME experiment definitions, TBME-specific catalogs, and TBME figure entrypoints. The experiment execution path is intentionally thin:

1. A suite module declares one manuscript experiment family through `EXPERIMENT_SUITES`.
2. `experiments.tbme.run_tbme_experiments` installs the TBME catalogs and injects those suite definitions.
3. `experiments.run` performs the actual run and summary work.

There is no TBME suite YAML file in the current structure. Suite definitions live next to the experiment they describe.
Suite ids use clean environment slugs such as `duffing` and `gated_duffing`, not old `exp##_` prefixes.

## Directory Layout

- `config/experiment_env.yaml`: TBME environment presets.
- `config/experiment_model.yaml`: TBME policy, objective, and schedule presets.
- `exp_simple_system_identification.py`: base-environment policy comparisons.
- `exp_observation_action_bottleneck.py`: gated-Duffing bottleneck experiments.
- `exp_model_mismatch.py`: nominal model-mismatch experiments.
- `exp_parameter_mismatch_stress.py`: mild and strong parameter-mismatch stress tests.
- `exp_observation_tuning_mismatch.py`: observation-tuning mismatch experiments.
- `exp_objective_ablation.py`: objective ablations.
- `exp_scheduling.py`: planning schedule sweeps.
- `exp_hard_environment.py`: legacy hard-environment suite.
- `exp_observation_mismatch_stress.py`: legacy observation-loading mismatch stress suite.
- `run_tbme_experiments.py`: helper that connects each explicit TBME suite module to `experiments.run`.
- `generate_figures.py`: single entrypoint for TBME figure and asset generation.
- `generate_behavior_video.py`: TBME behavior frame/video renderer.
- `tbme_io.py`: shared TBME run trace and metadata-loading helpers.
- `tbme_figures.py`: shared TBME figure configuration and thin figure entrypoint delegates.
- `tbme_figures_summary.py`: per-suite summary and trajectory figures.
- `tbme_figures_experiment.py`: experiment-level manuscript figures.
- `tbme_figures_assets.py`: manuscript asset assembly used by `generate_figures.py assets`.
- `tbme_figures_diagnostics.py`: catalog-level dynamics and observation diagnostics.

The shared experiment runtime remains outside this directory:

- `experiments/run.py`
- `experiments/summarize.py`
- `experiments/experiment_definitions.py`
- `actdyn/core/experiment.py`
- `actdyn/utils/plotting.py`

## Catalogs

TBME runs use both the base experiment catalogs and the TBME catalogs:

- environments: `experiments/experiment_env.yaml`, then `experiments/tbme/config/experiment_env.yaml`
- models/policies: `experiments/experiment_model.yaml`, then `experiments/tbme/config/experiment_model.yaml`
- suites: injected from the selected suite module, with file-backed suite catalogs disabled

When the same model or policy id appears in both model catalog files, the later TBME catalog entry overrides the base entry as a whole. This is not a deep merge. Current duplicated ids include `active_myopic`, `active_planning`, and `random`.

## Shared Experiment Families

The default TBME entrypoint is the shared tracks family:

```bash
./.venv/bin/python -m experiments.tbme.run_tbme_experiments all --mode all --skip-existing
```

It writes to `results/tbme/session_<n>/tracks` and deduplicates repeated
environment-method pairs in this order:

| Shared group | Source modules |
| --- | --- |
| `simple_system_identification` | `exp_simple_system_identification.py` |
| `observation_action_bottleneck` | `exp_observation_action_bottleneck.py` |
| `model_mismatch` | `exp_model_mismatch.py`, `exp_parameter_mismatch_stress.py`, `exp_observation_tuning_mismatch.py` |
| `objective_ablation` | `exp_objective_ablation.py` |
| `scheduling` | `exp_scheduling.py` |

The default seed counts are manuscript-scale defaults. For smoke tests or debugging, always pass an explicit small `--seeds` value.

## Objective-ablation dynamics design

The objective ablation should use a factorial family rather than one catch-all
oscillator.  For a planned trajectory, PALDI accumulates

\[
J_\theta=\sum_k \gamma^k S_k^\top
  (I+P_{z,k}^{-}I_{z,k})^{-1}I_{z,k}S_k,
\qquad
\mathrm{EIG}=\tfrac12\log\det(I+P_\theta J_\theta).
\]

The dynamics therefore need independent controls over parameter sensitivity
`S`, predicted latent uncertainty `P_z^-`, observation information `I_z`, and
the rank profile of `J_theta`.  A single stronger nonlinearity changes several
of these at once and does not identify why an ablation fails.

| Policy or ablation | Situation in which it should fail | Required stress mechanism |
| --- | --- | --- |
| Fully-observable parameter EIG | Parameter sensitivity and an uncertain latent nuisance produce the same observed displacement. Omitting `P_z^-` attributes both to the parameter. | A hidden nuisance direction that is confounded with the parameter at one gate but not at another. |
| E-optimality | One weak direction determines the minimum eigenvalue, while another action gives much larger joint information. | Two gates with posterior-scaled information matrices `J_main = diag(16, 4, 0.04)` and `J_balanced = diag(0.16, 0.16, 0.16)`. E-optimality prefers the second; log-determinant EIG prefers the first. |
| State information | A region is highly informative about the current state but its vector field is insensitive to the unknown parameters. | An observable autonomous mode with large `P_z^- I_z` and `F_theta` near zero. |
| Dynamics sensitivity | A large `S` lies in a sensor-null or state-confounded direction. | High raw sensitivity at the confounded gate and lower sensitivity in a directly observed, contracting region. |
| Observation variance | Predicted rates vary because of a latent nuisance, high Poisson gain, or one redundant parameter direction. | A high-rate nuisance gate whose posterior predictive variance is large but whose conditional parameter information is small. |
| State variance | Autonomous instability or process noise creates state spread without parameter information. | A bounded noisy/chaotic mode with `F_theta` near zero. |
| Random and PRBS | Informative excitation requires reaching and dwelling in a narrow region or applying a phase-specific action sequence. | Start in the confounded basin; make the informative gate reachable within one planning horizon but unlikely under zero-mean or fixed-hold inputs. |

### Implemented factorial family

The primary system is the three-state `confounded_gate` model.  Let
`g_j(r) = exp(-(r-c_j)^2/(2 w_j^2))`, with controlled selector state `r`,
observed response `s`, and weakly observed nuisance `h`:

\[
\begin{aligned}
\dot r &= -\kappa(r-c_A)+u_r,\\
\dot s &= -\lambda s
  + b_I g_I(r)\theta_1
  + b_A g_A(r)(\theta_1+h)
  + u_s,\\
\dot h &= 0.
\end{aligned}
\]

The catalog uses `c_A=-0.5`, `c_I=-0.32`, `w_A=w_I=0.04`, `b_A=20`,
`b_I=10`, and initial nuisance variance 4.  The true nuisance and its filter
prior are both centered at zero; the observation loading sees `s` well but
scales the direct `h` loading by 0.02.  Thus the stress comes from posterior
ambiguity rather than a mismatched initial mean.  At the ambiguity gate `A`, raw sensitivity,
state information, and predictive variance are all large, but `theta_1` and
`h` are not separately identifiable.  At gate `I`, the response is contracting
and `theta_1` is identifiable without the nuisance.  Initial states should lie
inside `A`; a sustained bounded action should reach `I` within one planning
horizon.  This one system targets fully-observable EIG, state information,
dynamics sensitivity, variance objectives, and passive excitation.

The separate `rank_imbalanced_gate` system tests D- versus E-optimality.  Its
main-gate sensitivities are `(4, 2, 0.2)` and its balanced-gate sensitivities
are `(0.4, 0.4, 0.4)`, giving the two diagonal targets in the table.  Keeping
this mechanism separate prevents a deliberately weak nuisance coefficient
from dominating the parameter-error metric in the confounded-gate experiment.

### CompoundTriGate: one system for all objective ablations

`compound_tri_gate` combines the two mechanisms without hiding them in a
single nonlinear coefficient.  Its state is `z=(r,s1,s2,s3,h)`, its learned
parameter is `theta=(theta1,theta2,theta3)`, and only the selector `r` is
actuated:

\[
\begin{aligned}
\dot r &= -(r+0.5)+u,\\
\dot s &= -4s
 + g_A(r)(20\theta_1+200h,0,0)^\top\\
&\quad + g_B(r)\operatorname{diag}(0.02,0.02,0.02)\theta
 + g_M(r)\operatorname{diag}(4,2,0)\theta,\\
\dot h &= 0.
\end{aligned}
\]

The Gaussian gates have width `0.04` and centers `c_A=-0.5`, `c_B=-0.32`,
and `c_M=0`.  The true parameter is `(1,1,0)`, so the third coordinate starts
at its correct prior mean but remains a posterior direction that an
acquisition function may choose to reduce.

This original construction is retained for provenance, but it is not valid
evidence of complete parameter recovery: gate `M` has zero sensitivity to
`theta3`, while both the true value and initial estimate of `theta3` are zero.
Its aggregate error and R2 can therefore look favorable without learning that
coordinate.  Use `three_gate_diagnostic` below for the corrected comparison.

The observation model is deliberately explicit and linear Gaussian.  Its
state Fisher matrix is

\[
I_z=C^\top R^{-1}C=\operatorname{diag}(1,1,1,1,0.01).
\]

This matrix is part of the dynamics stress test.  At gate `A`, `theta1` and
the weakly observed nuisance `h` enter the same response coordinate.  PALDI's
factor `(I+P_z^- I_z)^{-1}I_z` attenuates that apparent parameter sensitivity
because predicted uncertainty in `h` can explain the displacement.
Fully-observed EIG removes this attenuation.  Dynamics sensitivity and the
variance objectives see the large coefficient at `A` but not the conditional
identifiability loss; state information rewards the same high-uncertainty
region.  At gate `M`, PALDI gets large joint information about `theta1` and
`theta2`.  E-optimality instead prefers the uniformly weak gate `B`, because
`M` has zero minimum eigenvalue.  PRBS and random excitation begin at `A` and
do not dwell at `M`.

Trajectory R2 is evaluated deterministically from starts spanning the three
gates, with `h=0`.  This prevents independent process-noise realizations or a
large shared nuisance trajectory from setting the R2 ceiling; online training
still uses the configured process noise.

The current 10-seed, 2000-step engineering audit gives:

| Policy | Final parameter error (mean +/- SEM) | Final targeted trajectory R2 | Gate A / B / M occupancy |
| --- | ---: | ---: | ---: |
| PALDI | `0.482 +/- 0.048` | `0.860 +/- 0.027` | `0.03 / 0.06 / 0.91` |
| Observation variance (nearest ablation) | `0.748 +/- 0.059` | `0.446 +/- 0.084` | `0.65 / 0.17 / 0.18` |
| Fully-observed EIG | `0.865 +/- 0.112` | `0.327 +/- 0.190` | `0.57 / 0.16 / 0.27` |
| E-optimality | `1.262 +/- 0.129` | `0.101 +/- 0.377` | `0.17 / 0.79 / 0.04` |
| PRBS / random | `1.945 / 1.794` | `-2.052 / -1.510` | no main-gate dwell |

These ten seeds are an engineering check, not the manuscript acceptance run;
the 100-paired-seed requirement below still applies.

#### Paired log-linear Poisson observation variant

`compound_tri_gate_poisson` keeps the dynamics, policies, schedules, seeds,
and targeted trajectory evaluation fixed while replacing the Gaussian decoder
with 160 conditionally independent Poisson neurons,

\[
y_j\sim\operatorname{Poisson}\!\left(\Delta t\,
\exp(c_j^\top z+b)\right),\qquad \Delta t=0.01.
\]

For every latent coordinate, 16 neurons have loading `+a_i` and 16 have
loading `-a_i`, where
`a=(0.35,0.35,0.35,0.35,0.035)`.  All neurons have center rate
`25.510204 Hz`.  The paired construction makes the Fisher matrix diagonal for
every state,

\[
I_{z,ii}(z_i)=32\,\Delta t\,(25.510204)\,a_i^2\cosh(a_i z_i),
\]

and matches the Gaussian benchmark exactly at the operating point:

\[
I_z(0)=\operatorname{diag}(1,1,1,1,0.01).
\]

Thus the variant changes the likelihood family and introduces bounded
rate-dependent Fisher information without also changing the origin
information scale.  Over the radius-six state domain the largest possible
per-neuron rate is `208.34 Hz`, below the configured `210 Hz` cap.  The
realized loading matrix and bias are written to every run's metadata.

This is a controlled observation-model transfer test, not an adversarial
Poisson benchmark: axis-aligned positive/negative tuning intentionally keeps
the dynamics-driven traps auditable.  Required diagnostics are the realized
rate range, state-Fisher curves, gate occupancy, and sequential recovery
curves; a fixed-gate Fisher table alone cannot establish the posterior-driven
switching behavior.

The 10-seed, 2000-step engineering audit gives:

| Policy | Final parameter error (mean +/- SEM) | Final targeted trajectory R2 | Gate A / B / M occupancy |
| --- | ---: | ---: | ---: |
| PALDI | `0.528 +/- 0.035` | `0.850 +/- 0.021` | `0.03 / 0.05 / 0.93` |
| Observation variance (nearest ablation) | `0.967 +/- 0.035` | `0.112 +/- 0.058` | `0.64 / 0.16 / 0.19` |
| Fully-observed EIG | `1.041 +/- 0.076` | `0.002 +/- 0.160` | `0.64 / 0.14 / 0.22` |
| E-optimality | `1.381 +/- 0.061` | `0.054 +/- 0.166` | `0.19 / 0.79 / 0.02` |
| State information | `1.447 +/- 0.028` | `-0.107 +/- 0.081` | `0.85 / 0.15 / 0.00` |
| Dynamics sensitivity | `1.438 +/- 0.027` | `-0.096 +/- 0.081` | `0.78 / 0.21 / 0.01` |
| State variance | `1.035 +/- 0.045` | `0.135 +/- 0.064` | `0.61 / 0.19 / 0.20` |
| PRBS / random | `1.450 / 1.538` | `-0.227 / -0.478` | no main-gate dwell |

PALDI beats every comparison in all ten paired seeds except fully-observed EIG,
where it wins nine of ten.  Its mean targeted trajectory R2 first exceeds
`0.8` at step `1040` and reaches `0.850` at step `2000`.  The largest rate
implied by the logged selector and first-response coordinates is `207.1 Hz`,
so the separation is not caused by violating the observation-rate cap.

### SimpleTriGate: corrected wide-gate Poisson benchmark

`three_gate_diagnostic` corrects the favorable-`theta3` initialization and
removes the free trap at the passive equilibrium.  Its state is
`z=(r,s1,s2,s3,h)`, all three learned parameters have truth one and initial
posterior mean zero, and only `r` is actuated:

\[
\begin{aligned}
\dot r &= -(r+1)+u,\\
\dot s &= -4s + 20g_A(r)(\theta_1+5h,0,0)^\top\\
&\quad + g_B(r)\operatorname{diag}(1,1,1)\theta
 + g_M(r)\operatorname{diag}(5,5,0.75)\theta,\\
\dot h &= 0.
\end{aligned}
\]

The passive equilibrium is `r=-1`, outside all three gates.  Gate centers are
`c_A=-0.5`, `c_B=-0.1`, and `c_M=0.3`, with Gaussian width `0.1`.  Thus each
gate is 2.5 times wider than in `compound_tri_gate`; adjacent `+/-2 sigma`
supports meet, while center-to-center leakage is only `exp(-8)`.  Holding the
three gates requires the explicit actions `u=(0.5,0.9,1.3)` within the action
bound `1.5`.

The roles are intentionally distinct:

- `A` has the largest raw sensitivity, but `theta1` and a weakly observed,
  process-noisy nuisance `h` drive the same response.  Fully-observed EIG and
  the state/dynamics/variance ablations are attracted to it.
- `B` gives modest isotropic information and has the strongest weakest
  parameter direction, so E-optimality stays there.
- `M` gives much larger joint information about `theta1` and `theta2` and a
  nonzero `0.75 theta3` direction, so its parameter-sensitivity matrix is full
  rank.  Because `theta3*=1` and `theta3_hat(0)=0`, success cannot come from an
  initially correct unidentifiable parameter.

The observation model uses only ten Poisson neurons, one positive and one
negative log-linear neuron per state coordinate:

\[
y_j\sim\operatorname{Poisson}\!\left(0.01\exp(c_j^\top z+\log 50)\right),
\qquad
a=(0.3,0.3,0.3,0.3,0.05).
\]

For coordinate `i`, the paired state Fisher is

\[
I_{z,ii}(z_i)=2(0.01)(50)a_i^2\cosh(a_i z_i),
\]

so `I_z(0)=diag(0.09,0.09,0.09,0.09,0.0025)`.  The nuisance is 36 times less
observable than a primary coordinate at the origin.  The radius-three domain
keeps the maximum possible rate below `125 Hz`.

Held-out recovery is computed on a fixed deterministic validation distribution
spanning the rest point and all gates, not on each controller's own visited
trajectory.  The reported score averages coordinate-wise R2 over
`(s1,s2,s3)`, preventing the large `s1` amplitude or parameter-independent
selector and nuisance coordinates from dominating the result.

The finalized 100-paired-seed, 2000-step comparison gives medians and
interquartile ranges:

| Policy | Final parameter error | Response-balanced rollout R2 | Rest / A / B / M occupancy |
| --- | ---: | ---: | ---: |
| PALDI | `0.569 [0.526,0.639]` | `0.899 [0.871,0.917]` | `0.02 / 0.15 / 0.05 / 0.78` |
| State variance | `0.858 [0.769,0.943]` | `0.767 [0.699,0.815]` | `0.03 / 0.52 / 0.11 / 0.34` |
| Fully-observed EIG | `0.839 [0.759,0.906]` | `0.758 [0.716,0.802]` | `0.02 / 0.47 / 0.08 / 0.42` |
| E-optimality | `0.756 [0.702,0.843]` | `0.750 [0.682,0.786]` | `0.02 / 0.05 / 0.92 / 0.01` |
| Observation variance | `0.895 [0.815,1.000]` | `0.746 [0.652,0.810]` | `0.03 / 0.56 / 0.10 / 0.30` |
| Dynamics sensitivity | `1.596 [1.530,1.666]` | `0.269 [0.182,0.375]` | `0.05 / 0.86 / 0.10 / 0.00` |
| State information | `1.628 [1.576,1.677]` | `0.263 [0.135,0.331]` | `0.03 / 0.90 / 0.08 / 0.00` |
| PRBS | `1.732 [1.728,1.732]` | `0.087 [0.058,0.127]` | `0.97 / 0.03 / 0.00 / 0.00` |
| Random | `1.732 [1.732,1.732]` | `0.083 [0.047,0.117]` | `1.00 / 0.00 / 0.00 / 0.00` |

PALDI wins 99 of 100 paired R2 comparisons against fully-observed EIG,
E-optimality, and state variance, and all 100 against every other comparison.
The paired mean difference from fully-observed EIG is `+0.141`, with a
deterministic 20,000-resample bootstrap 95% interval `[+0.128,+0.155]`.
PALDI's final posterior medians are `theta_hat=(0.636,0.842,0.602)`, so all
three coordinates move substantially from the zero initialization.

Occupancy is interpreted only at the cluster level.  A deterministic input
that reaches and holds M produces rest/A/B/M occupancy
`0.011/0.028/0.055/0.907`, so PALDI's B occupancy is compatible with transit,
not evidence of a deliberate theta3-specific visit.  The observed clusters
are A-dominant state-information and dynamics objectives, B-dominant
E-optimality, and split A/M variance or fully-observed objectives.  All 100
PALDI true and filtered state traces are finite; the median radial-boundary
fraction is `0.005`, and the largest realized firing rate over all 900 runs is
`122.14 Hz`, below the `125 Hz` cap.

The ambiguity term is written as `20(theta1+5h)` to expose its dimensionless
nuisance ratio.  A four-setting pilot sweep used ratios `1`, `2.5`, `5`, and
`10`: ratio `1` did not separate PALDI from fully-observed EIG, ratio `2.5` was
modest, ratio `5` produced the replicated separation above, and ratio `10`
was unnecessary.  Thus `5` is the smallest tested clearly separating value,
not an unrestricted extreme chosen after the fact.

Reproduce the audit with:

```bash
./.venv/bin/python -m experiments.tbme.exp_objective_ablation \
  --mode run \
  --exp-ids three_gate_diagnostic \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99 \
  --total-steps 2000 \
  --base-dir results/tbme/session_4 \
  --skip-existing
./.venv/bin/python -m experiments.tbme.exp_objective_ablation \
  --mode summary \
  --exp-ids three_gate_diagnostic \
  --base-dir results/tbme/session_4
```

Add mechanism-off controls, not just easier noise settings:

- set the nuisance covariance to zero; PALDI and fully-observable EIG should coincide;
- directly observe `h`; the state-ambiguity advantage should disappear;
- set the autonomous-mode amplitude and process noise to zero; its attraction to state-only objectives should disappear;
- replace the two rank profiles by isotropic matrices; D- and E-optimality should coincide;
- widen the informative gate or initialize inside it; the active/passive gap should shrink.

Before interpreting an occupancy pattern as a distinct causal mechanism,
first test objective rankings on fixed canonical rollouts and then run the
mechanism-off controls above.  The finalized 100-seed comparison supports the
objective-ordering and behavioral-cluster claims; without those online
mechanism-off controls, it does not establish a unique causal failure for
every individual ablation.

An initial 20-paired-seed engineering audit at 500 online steps gave the
following results.  These are validation targets for the implementation, not
a replacement for the 100-seed manuscript acceptance run.

Reproduce that audit from the repository root with:

```bash
./.venv/bin/python -m experiments.tbme.exp_objective_ablation \
  --exp-ids confounded_gate,rank_imbalanced_gate \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --base-dir results/tbme/paldi_objective_ablation_audit \
  --mode all
```

| System | Comparison | Final parameter error | Mechanism occupancy |
| --- | --- | --- | --- |
| `confounded_gate` | PALDI vs local objective ablations | `0.349` vs `0.436--0.507` | identifiable gate: `0.146` vs `0.054--0.087` |
| `rank_imbalanced_gate` | PALDI vs E-optimality | `0.225` vs `0.749` | main gate: `0.693` vs `0.033` |
| `rank_imbalanced_gate` | PALDI vs random / PRBS | `0.225` vs `1.095 / 1.095` | main gate: `0.693` vs `0 / 0` |

For the confounded system, paired 95% intervals for PALDI minus each local
ablation were strictly negative.  For the rank-imbalanced system, the paired
PALDI-minus-E-optimality difference was `-0.524` with interval
`[-0.718, -0.331]`.

An exploratory two-state local-amplifier design was rejected during this work:
after deterministic planner seeding, fully-observable EIG and observation
variance could exploit the same informative region as PALDI.  It should not be
added to the experiment catalog as evidence for the full objective.

## Running Experiments

Run experiment modules from the repository root with `./.venv/bin/python -m`. This keeps the package import path explicit and uses the project environment. If your shell already activates the project environment, plain `python -m` is equivalent.

```bash
./.venv/bin/python -m experiments.tbme.exp_simple_system_identification --mode run --seeds 0 --skip-existing
```

Run all shared tracks, including summaries:

```bash
./.venv/bin/python -m experiments.tbme.run_tbme_experiments all \
  --mode all \
  --seeds 0,10,20 \
  --skip-existing
```

Run a small one-policy smoke test:

```bash
./.venv/bin/python -m experiments.tbme.exp_simple_system_identification \
  --mode run \
  --exp-ids duffing \
  --policy-ids random \
  --seeds 0 \
  --total-steps 1 \
  --base-dir /tmp/tbme_smoke \
  --skip-existing
```

The helper can also run a family by module name:

```bash
./.venv/bin/python -m experiments.tbme.run_tbme_experiments exp_objective_ablation \
  --mode summary \
  --exp-ids gated_duffing \
  --seeds 0
```

Do not prefer direct file execution such as `./.venv/bin/python experiments/tbme/exp_simple_system_identification.py`. The current entrypoints use package imports and are meant to be run with `./.venv/bin/python -m` from the repository root.

## Common Arguments

Each suite module accepts these TBME-level defaults and forwards all other arguments to `experiments.run`:

- `--exp-ids`: comma-separated suite ids; defaults to that experiment family's suites.
- `--base-dir`: output root; defaults to `results/tbme`.
- `--seeds`: comma-separated integer seeds; defaults to the family seed range.

Common forwarded arguments include:

- `--mode {run,summary,all}`
- `--policy-ids`
- `--repeats`
- `--skip-existing`
- `--total-steps`
- `--q-theta`
- `--parameter-prior-covariance`
- `--eig-gamma`

Use `--help` on the TBME entrypoints to inspect the current parser. Prefer these entrypoints because they install the TBME catalog stack before calling the generic runner:

```bash
./.venv/bin/python -m experiments.tbme.exp_simple_system_identification --help
./.venv/bin/python -m experiments.tbme.run_tbme_experiments --help
```

## Outputs

Runs write one session under the selected `--base-dir`. A typical run contains:

- `session_metadata.json`: resolved catalogs, command line, experiments, policies, seeds, and run summary.
- `experiment_driver.log`: redirected stdout and stderr when running non-interactively.
- `tracks/<env>/<policy_id>/seed_<seed>/repeat_<repeat>/run_metadata.json`: per-run metadata.
- Trace CSV files such as `parameter_error_trace.csv`, `trajectory_r2_trace.csv`, `embedding_estimate_trace.csv`, `information_trace.csv`, and `state_action_trace.csv`.
- Summary artifacts when `--mode summary` or `--mode all` is used.
- Figure assets and diagnostics default to `assets/` and `diagnostics/` under the same `session_<n>` root.

## Figure Generation

Use `generate_figures.py` as the single figure entrypoint:

```bash
./.venv/bin/python -m experiments.tbme.generate_figures --help
./.venv/bin/python -m experiments.tbme.generate_figures summary
./.venv/bin/python -m experiments.tbme.generate_figures experiment
./.venv/bin/python -m experiments.tbme.generate_figures assets
./.venv/bin/python -m experiments.tbme.generate_figures diagnostics
./.venv/bin/python -m experiments.tbme.generate_figures all
./.venv/bin/python -m experiments.tbme.tbme_figures_assets --help
```

Asset generation writes the existing mean/SEM R2 figures in `assets/` and the
parallel median/IQR R2 figures in `assets/median_iqr/`. Use
`--r2-summaries mean_sem` or `--r2-summaries median_iqr` to generate only one
set. Regenerate suite summaries once after upgrading so
`trajectory_r2_over_steps.csv` contains the median and quartile columns.

The figure code keeps TBME result-group definitions in `tbme_figures.py`. If result roots are renamed, update the `GROUPS` table there before relying on the figure commands.
The diagnostics command does not require completed runs. The top-level `generate_figures diagnostics` entrypoint uses the fixed core environment list in `generate_figures.py`.

## Reproducibility Checklist

Before launching a manuscript-scale run, record:

- the exact module command;
- `--exp-ids`, if not using the family default;
- `--policy-ids`, if filtering policies;
- `--seeds` and `--repeats`;
- `--base-dir`;
- whether `--total-steps` overrides the suite default;
- the active git commit written to `session_metadata.json`.

For quick validation, prefer one suite, one policy, one seed, and `--total-steps 1` before launching the full seed grid.
