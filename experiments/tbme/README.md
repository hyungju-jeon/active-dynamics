# TBME Experiment Program

This directory contains TBME-specific catalogs and thin wrappers on top of the shared experiment runners in `experiments/`.

## Layout

- `experiment_env.yaml`: TBME environment presets
- `experiment_model.yaml`: TBME-only model and policy additions
- `experiment_suite.yaml`: TBME manuscript experiment matrices
- `run_exp1.py`: convenience launcher for Experiment 1 suites
- `run_exp2.py`: convenience launcher for Experiment 2 suites
- `run_exp3.py`: convenience launcher for the Experiment 3 neural digital-twin workflow
- `exp3_digital_twin.yaml`: default config for the spike-driven Exp 3 neural digital twin
- `exp3_digital_twin.py`: generator fitting, digital-twin construction, active-ID benchmark, and summary library
- `seqvae_mcrtt.yaml`: default config for the SeqVAE-on-MC_RTT research workflow
- `seqvae_mcrtt.py`: SeqVAE training, baseline comparison, and semi-synthetic recovery library
- `run_seqvae_mcrtt.py`: convenience launcher for the SeqVAE-on-MC_RTT workflow
- `summarize_seqvae_mcrtt.py`: summary-only entrypoint for SeqVAE-on-MC_RTT sessions
- `summarize_experiments.py`: TBME wrapper around `experiments/summarize_experiments.py`
- `render_experiment_videos.py`: TBME wrapper around `experiments/render_experiment_videos.py`

The shared runner implementation lives in:

- `experiments/experiment_common.py`
- `experiments/experiment_specs.py`
- `experiments/run_experiments.py`
- `experiments/summarize_experiments.py`
- `experiments/render_experiment_videos.py`

## Manuscript Framing

The TBME results program is organized around three experiment families:

1. Basic proof of work on simple controlled dynamics
2. Convergence and robustness under model mismatch
3. Scaling to high-dimensional neural replay data

The catalogs in this directory define the TBME-specific experiment layer only. Shared environment presets, schedules, and common policy definitions are inherited from the base catalog stack exposed through `experiments/`.

The recommended manuscript defaults are:

- synthetic policy-comparison seeds: `0,10,20`
- real-data replay seeds: `0` unless replay subsampling or randomized selection is being stress-tested
- run output root: `results/tbme`
- family pipeline commands: `python -m experiments.tbme.run_exp1 --mode all ...`, `run_exp2`, or `run_exp3`

The suite YAML encodes the default step budgets:

- Experiment 1 suites: `600` steps
- Experiment 2 suites: `800` steps
- Experiment 3 benchmark: `200` active interaction steps by default

## Experiment 1: Basic Proof of Work

### Question

Can active identification outperform passive and non-adaptive excitation on simple low-dimensional dynamics when the observation model is noisy and indirect?

### Hypothesis

For simple systems with matched model families, active policies should reduce parameter error faster than passive or random baselines, while also improving predictive trajectory quality.

### Systems and Suites

Environment presets:

- `tbme_duffing_easy`
- `tbme_pendulum_damped`
- `tbme_double_integrator`

Suites:

- `tbme_exp1_duffing_policy`
- `tbme_exp1_pendulum_policy`
- `tbme_exp1_double_integrator_policy`
- `tbme_exp1_objective_duffing`
- `tbme_exp1_duffing_challenge_policy`
- `tbme_exp1_duffing_budget_ablation_short`
- `tbme_exp1_duffing_budget_ablation_medium`
- `tbme_exp1_duffing_ig_ablation`
- `tbme_exp1_duffing_schedule_ablation`
- `tbme_exp1_duffing_competitor_compare`

### Compared Methods

Policy comparison suites:

- `active_myopic`
- `active_planning`
- `baseline_prbs`
- `random`
- `off_policy`

Objective comparison suite on Duffing:

- `active_planning`
- `e_optimality`
- `state_information`
- `dynamics`
- `sampling_variance`

### Primary Metrics

Primary:

- parameter error over time
- final parameter error
- trajectory `R^2`

Secondary:

- accumulated information trace
- representative state-action trajectories
- planned trajectory snapshots and acquisition maps for active methods

### Expected Figures and Tables

Lead figure:

- one convergence panel per system showing parameter error versus step for the policy sweep

Supporting figure:

- one Duffing objective-ablation panel comparing the acquisition objective choices under the same planning policy
- one hard-Duffing challenge panel showing the short-budget policy comparison
- one hard-Duffing budget-ablation panel comparing how the planning advantage changes with interaction budget
- one hard-Duffing IG-approximation ablation panel with schedule held fixed
- one hard-Duffing schedule-tuning panel with objective held fixed
- one hard-Duffing competitor panel comparing the default method to practical alternatives

Table:

- final parameter error and final trajectory `R^2` aggregated over seeds for all three systems

### Interpretation Goal

This experiment establishes the basic method claim: the active planner should identify dynamics faster than passive/random baselines on canonical low-dimensional systems when the estimator family is correctly specified.

### Hard-Duffing Challenge

The base matched-system suites above are intentionally simple. They establish that active excitation helps, but they do not consistently separate `active_planning` from `active_myopic`. For manuscript claims about long-horizon information gathering, the more relevant track is the constrained hard-Duffing challenge:

- `tbme_duffing_planning_challenge`
- `tbme_exp1_duffing_challenge_policy`
- `tbme_exp1_duffing_budget_ablation_short`
- `tbme_exp1_duffing_budget_ablation_medium`

This challenge makes planning matter by combining:

- weaker observations (`observation_dim: 24`)
- lower firing rates (`mean_firing_rate_target: 12`)
- tighter control bounds (`action_max: 0.45`)
- broader reachable state support (`x_range: 6`)
- moderate state and observation noise

The intended interpretation is sample-efficiency under constrained reachability. If `active_planning` does not beat `active_myopic` here, then the benchmark is still too easy or the planning objective is not aligned with the true identification bottleneck.

### Deep Ablation and Parameter Tuning

The hard-Duffing challenge is also the recommended setting for deeper method analysis. The additional suites serve three distinct purposes:

- `tbme_exp1_duffing_ig_ablation`
  This holds the schedule fixed at `u1_r1_h20` and varies only the information objective / approximation:
  `ig_parameter`, `ig_full_observable`, `ig_e_optimality`, `ig_state_information`, `ig_dynamics`, `ig_sampling_variance`.

- `tbme_exp1_duffing_schedule_ablation`
  This holds the objective fixed at parameter EIG and varies:
  planning horizon, update interval, and replan frequency through
  `sched_h2_u1_r1`, `sched_h10_u1_r1`, `sched_h20_u1_r1`, `sched_h40_u1_r1`, `sched_h20_u5_r1`, `sched_h20_u5_r5`, `sched_h40_u5_r5`, plus `baseline_prbs`.

- `tbme_exp1_duffing_competitor_compare`
  This is the compact comparison against practical alternatives already implemented in the repo:
  `active_planning`, `ig_sampling_variance`, `ig_e_optimality`, `baseline_prbs`, and `random`.

These suites are intentionally all run on the same hard-Duffing preset so the conclusions are attributable to the ablation axis rather than to environment changes.

### Proposed Experiment 1 Addition: Basin-Bridge Multistable Planning Challenge

To make the long-range-planning claim sharper than the current hard-Duffing setup, add a multistable benchmark where accurate identification requires leaving the initial basin and visiting a second basin with disjoint local parameter sensitivity.

Core design:

- two separated attractors with basin-specific parameter subsets
- a central barrier / separatrix that is crossable only through coordinated multi-step control
- starts initialized in one basin so local exploration alone is insufficient
- metrics that explicitly track basin visitation, first-cross time, and per-basin parameter recovery

Recommended suite IDs once the runtime support exists:

- `tbme_exp1_two_basin_policy`
- `tbme_exp1_two_basin_budget_ablation`
- `tbme_exp1_two_basin_schedule_ablation`

Minimal runnable spec currently registered in TBME:

- environment preset: `tbme_asymmetric_basin`
- runtime system: `asymmetric_basin`
- suites: `tbme_exp1_two_basin_policy`, `tbme_exp1_two_basin_budget_ablation`, `tbme_exp1_two_basin_schedule_ablation`

This benchmark now uses a dedicated `asymmetric_basin` runtime with basin-gated parameters while still fitting the existing 2D synthetic runner.

- rollouts start in the left basin by construction
- left and right basins expose different local parameter sensitivity through `(a_L, b_L, a_R, b_R)`
- control and observation budgets are tighter than `tbme_duffing_planning_challenge`

Suggested default parameterization for the minimal benchmark:

- `system_id: asymmetric_basin`
- `embedding_dim: 4`
- true parameters `(a_L, b_L, a_R, b_R) = (-1.2, -0.8, -0.55, -1.35)`
- `action_max: 0.35`
- `state_noise: 0.25`
- `state_init_uncertainty: 4.0`
- `observation_dim: 20`
- `observation_noise_scale: 0.2`
- `firing_rate_scale: 0.35`
- `mean_firing_rate_target: 10.0`
- `max_firing_rate_target: 45.0`
- `total_steps: 300` for policy comparison, `200` for the short budget ablation

Interpretation:

- this benchmark now implements the disjoint-parameter idea at the dynamics level through basin-gated left/right parameters
- it tests whether a policy can leave the initial basin under a constrained budget and gather information from both sides of the separatrix
- it still does **not** yet log dedicated basin-crossing summary metrics; those remain the next extension

Valid smoke test for repo integration:

```bash
python -m experiments.run_experiments \
  --env-catalog experiments/tbme/experiment_env.yaml \
  --model-catalog experiments/experiment_model.yaml \
  --model-catalog experiments/tbme/experiment_model.yaml \
  --suite-catalog experiments/tbme/experiment_suite.yaml \
  --exp-id tbme_exp1_two_basin_budget_ablation \
  --policy-ids baseline_prbs \
  --seeds 0 \
  --total-steps 8 \
  --mode run \
  --base-dir results/tbme/two_basin_smoke
```

Smoke-test success criteria:

- process exits with code `0`
- session metadata is written under `results/tbme/two_basin_smoke/...`
- one run record is marked `completed`
- `run_metadata.json` reports `env_preset_id: tbme_asymmetric_basin` and `system_id: asymmetric_basin`

Interpretation goal:

- show that planning helps not merely because it explores more, but because some parameters are effectively unidentifiable from single-basin trajectories under the same control budget

See `experiments/tbme/TBME_experiment_additions.md` for the current design spec and implementation hooks.

## Experiment 2: Convergence and Robustness

### Question

How much does performance degrade when the estimator is structurally misspecified, even when the observation regime is held fixed?

### Hypothesis

Active methods should still retain an advantage under moderate model mismatch, and the benefit should be largest when the observation model is correct but the latent dynamics model is wrong.

### Mismatch Presets

Available environment presets:

- `tbme_duffing_family_mismatch`
- `tbme_duffing_parameter_mismatch`

Main suites:

- `tbme_exp2_robustness_duffing`
- `tbme_exp2_robustness_duffing_parameter`

`tbme_duffing_family_mismatch` keeps the true Duffing generator but forces the learner to use the wrong dynamics family through `estimator_system_id: damped_pendulum`.

`tbme_duffing_parameter_mismatch` keeps the Duffing family correct but fixes the learner to the wrong non-inferred cubic coefficient. Concretely, the true generator uses

```math
\dot{x}=v,\qquad \dot{v}=a v - b x - 0.1 x^3,
```

while the learner uses

```math
\dot{x}=v,\qquad \dot{v}=a v - b x - 0.2 x^3.
```

The cubic coefficient is not inferred online; only the remaining Duffing embedding parameters are learned.

### Compared Methods

- `active_planning`
- `e_optimality`
- `baseline_prbs`
- `random`
- `off_policy`

These suites are intentionally narrower than Experiment 1. The Experiment 2 question is about model mismatch, not noise robustness or exhaustive policy sweeps.

### Primary Metrics

Primary:

- parameter error over time
- final parameter error

Secondary:

- trajectory `R^2`
- information trace
- failure or saturation behavior under strong mismatch

### Expected Figures and Tables

Lead figure:

- robustness comparison panel showing convergence traces for each mismatch regime

Supporting figure:

- one focused family-mismatch panel highlighting true-system versus estimator-system mismatch

Table:

- final parameter error by suite and policy, with notes on the mismatch type

### Interpretation Goal

This experiment supports the manuscript robustness claim. It should show where active identification remains useful, where it fails gracefully, and where estimator-family mismatch causes the strongest degradation.

### Proposed Qualitative / Follow-on Addition: Time-Varying Observation Hotspots

A complementary synthetic benchmark should test time-varying dynamics tracking when observation informativeness is concentrated in moving high-SNR regions that are not a simple monotone function of distance from the origin. A standalone prototype is now available at `experiments/tbme/run_circular_timevarying_hotspots_experiment.py`.

Core design:

- latent dynamics with unknown time-varying parameters such as drifting `omega_t`
- moving off-center hotspots or anisotropic arcs of high observation gain
- policies evaluated on phase alignment to the moving informative regions, cumulative information, and final tracking / parameter error
- explicit avoidance of the trivial radial heuristic where "farther from center" always implies greater sensitivity

Recommended first implementation path:

- run the standalone prototype `python -m experiments.tbme.run_circular_timevarying_hotspots_experiment --output-dir results/tbme/timevarying_hotspots`
- promote it to a TBME suite only after the information geometry and metrics are stable

Interpretation goal:

- show that good active tracking requires synchronizing state occupancy with a moving informative set, not simply maximizing radius or staying near the center

See `experiments/tbme/TBME_experiment_additions.md` for the current design spec and implementation hooks.

## Experiment 3: Neural Digital Twin From Real Spiking Data

### Question

Can we fit a latent neural dynamics model from real spiking data, turn it into a controlled generative simulator, and then test whether active identification recovers that latent dynamics faster than non-adaptive excitation?

### Hypothesis

A spike-driven latent digital twin fit on MC_RTT should support a closed-loop identification benchmark in which active information-seeking policies reduce latent dynamics parameter error faster than random or PRBS baselines.

### Dataset and Workflow

Data source:

- `data/mcrtt/mcrtt_replay.npz`, prepared from the published DANDI `000129` MC_RTT train session

Main launcher:

- `python -m experiments.tbme.run_exp3 --mode all ...`

Core workflow:

1. Fit a spike-only `SeqVae` with `MLP` latent dynamics on MC_RTT spike sequences.
2. Freeze the fitted encoder/decoder and calibrate a low-dimensional additive control map in latent space.
3. Treat the fitted model as the generator digital twin.
4. Instantiate a learner with the same observation model and control map, but perturbed latent dynamics parameters.
5. Run active identification on the digital twin and evaluate recovery against the generator's latent dynamics.

### Compared Methods

- `active_myopic`
- `active_planning`
- `baseline_prbs`
- `random`

### Protocol

The current digital-twin pipeline:

1. loads spike and aligned behavior arrays from the prepared `.npz` archive
2. windows the spike sequence into train and evaluation segments
3. trains the generator `SeqVae` on spikes only
4. validates the fitted generator by held-out spike prediction and behavior readout from the latent state
5. builds a controlled latent digital twin with synthetic low-dimensional actions
6. runs active identification on that simulator while only retraining the learner dynamics with supervised latent-transition updates
7. reports dynamics-parameter recovery and predictive rollout fidelity against the generator

This should be described in the manuscript as a closed-loop benchmark on a data-driven neural digital twin, not as a direct intervention experiment on the original MC_RTT recording.

### Primary Metrics

Primary:

- relative latent dynamics parameter error over interaction steps
- final relative latent dynamics parameter error

Secondary:

- latent rollout MSE against the generator
- spike-rate rollout MSE and `R^2`
- generator held-out spike prediction metrics

### Expected Figures and Tables

Lead figure:

- relative parameter error versus interaction step for the policy sweep

Supporting figure:

- spike-rate rollout MSE versus interaction step
- generator fit summary for the spike-only neural latent model

Table:

- final parameter error and rollout-fidelity metrics by policy

### Interpretation Goal

This experiment supports the manuscript large-scale claim: active identification should remain effective when the environment is a controlled digital twin fit from real neural population data.

## SeqVAE Dynamics Recovery on MC_RTT

### Purpose

This workflow is a TBME research prototype for fitting a `SeqVae` with `MLP` latent dynamics on the prepared MC_RTT replay dataset, then checking how well the learned latent simulator captures and reproduces the underlying replay dynamics.

It is intentionally separate from the policy-selection Experiment 3 pipeline above. The goal here is model-fitting and recovery analysis, not active transition selection.

### Scope

Current implementation choices:

- source data: `data/mcrtt/mcrtt_replay.npz`
- default observation stream: `behavior`
- latent dynamics model: `SeqVae` with `MLP` dynamics
- latent-dimension sweep: `2`, `4`, `8`
- baselines:
  - linear behavior dynamics fit
  - behavior-only nonlinear MLP predictor
- recovery benchmark:
  - generate synthetic sequences from the fitted SeqVAE
  - refit the same model family on those synthetic sequences
  - measure aligned latent and rollout recovery error

This first version is behavior-sequence based. It does not yet fit a joint spike-plus-behavior decoder, and it does not claim recovery of biological ground-truth neural dynamics.

### Questions Answered

1. Does a nonlinear latent SeqVAE predict held-out MC_RTT replay trajectories better than simple linear and nonlinear behavior baselines?
2. Do larger latent dimensions improve forecast quality on held-out replay windows?
3. If the fitted SeqVAE is treated as a generator, can the same model family recover that simulator from synthetic sequences?

### Outputs

Each session under `results/tbme/seqvae_mcrtt/session_N/` contains:

- `linear_behavior/`: linear baseline metrics
- `mlp_behavior/`: nonlinear behavior baseline metrics and checkpoint
- `seqvae_latent_2/`, `seqvae_latent_4/`, `seqvae_latent_8/`: fitted SeqVAE runs
- `seqvae_latent_*/recovery/`: synthetic-refit recovery artifacts
- `summary/realdata_metrics.csv`: held-out comparison across all models
- `summary/recovery_metrics.csv`: semi-synthetic recovery metrics
- `summary/figures/`: manuscript-facing comparison plots

Key metrics:

- one-step MSE on held-out replay windows
- rollout MSE by horizon
- one-step and rollout `R^2`
- aligned latent recovery MSE
- generator-versus-recovered rollout MSE

### Commands

Run the full latent-dimension sweep with summary generation:

```bash
python -m experiments.tbme.run_seqvae_mcrtt \
  --mode all \
  --base-dir results/tbme/seqvae_mcrtt
```

Run a smaller smoke configuration manually:

```bash
python -m experiments.tbme.run_seqvae_mcrtt \
  --mode all \
  --latent-dims 2 \
  --sequence-length 64 \
  --sequence-stride 32 \
  --max-train-sequences 32 \
  --max-eval-sequences 8 \
  --n-epochs 5 \
  --recovery-epochs 3 \
  --synthetic-num-sequences 8 \
  --base-dir /tmp/tbme_seqvae_smoke
```

Summarize an existing session without retraining:

```bash
python -m experiments.tbme.summarize_seqvae_mcrtt \
  --base-dir results/tbme/seqvae_mcrtt
```

## Policies and Objectives

TBME-specific model additions in this directory:

- `baseline_random`
- `baseline_prbs`
- `e_optimality`
- `e_optimality_u5_r5_h40`

The TBME catalog layer merges these with the shared base catalog, so suites here can still use shared policies such as `active_myopic`, `active_planning`, `random`, and `off_policy`.

## Reproducibility Notes

The suite YAML defines which environments and model IDs belong to each manuscript experiment, but seeds are passed on the command line. For manuscript runs, keep the seed list explicit in the command used to generate final assets.

Recommended practice:

- use the same seed set across all synthetic policy comparisons
- keep summary generation in the same output root as the corresponding runs
- archive the exact command line used for each final figure or table

## Commands

Prepare the MC_RTT replay data:

```bash
python -m experiments.tbme.prepare_exp3_data --overwrite-output
```

This downloads the published MC_RTT NWB assets into `data/mcrtt/raw/` if needed and writes the replay file expected by the current Experiment 3 config at `data/mcrtt/mcrtt_replay.npz`.

Run Experiment 1:

```bash
python -m experiments.tbme.run_exp1 \
  --mode all \
  --seeds 0,10,20 \
  --base-dir results/tbme/exp1
```

Run Experiment 2:

```bash
python -m experiments.tbme.run_exp2 \
  --mode all \
  --seeds 0,10,20 \
  --base-dir results/tbme/exp2
```

Run Experiment 3:

```bash
python -m experiments.tbme.run_exp3 \
  --mode all \
  --base-dir results/tbme/exp3
```

Run only the generator fit stage:

```bash
python -m experiments.tbme.run_exp3 \
  --mode fit \
  --base-dir results/tbme/exp3
```

Run the benchmark only against an existing fitted digital twin:

```bash
python -m experiments.tbme.run_exp3 \
  --mode benchmark \
  --base-dir results/tbme/exp3
```

Run one TBME suite through the generic config-driven runner:

```bash
python -m experiments.run_experiments \
  --env-catalog experiments/experiment_env.yaml \
  --env-catalog experiments/tbme/experiment_env.yaml \
  --model-catalog experiments/experiment_model.yaml \
  --model-catalog experiments/tbme/experiment_model.yaml \
  --suite-catalog experiments/tbme/experiment_suite.yaml \
  --exp-id tbme_exp1_duffing_policy \
  --mode all \
  --seeds 0,10,20 \
  --base-dir results/tbme/manual
```

Summarize only:

```bash
python -m experiments.tbme.summarize_experiments \
  --base-dir results/tbme \
  --exp-id tbme_exp2_robustness_duffing \
  --seeds 0,10,20
```

## Replay Dataset Format

The real-data path expects a standardized `.npz` archive referenced by `dataset_path`.

Required arrays:

- `behavior`: shape `(T, D_state)`
- `spikes`: shape `(T, D_obs)`

Optional array:

- `dt`: scalar or length-1 array in seconds

Default keys for `tbme_mcrtt_spikes`:

- `state_key: behavior`
- `observation_key: spikes`

## Outputs

Synthetic runs write:

- `parameter_error_trace.csv` or `dynamics_mse_trace.csv`
- `trajectory_r2_trace.csv`
- `information_trace.csv`
- `state_action_trace.csv`
- optional `planned_trajectory_trace.npz`
- optional `acquisition_map_trace.npz`

Experiment 3 digital-twin runs write:

- `generator/checkpoint.pt`
- `generator/fit_metrics.json`
- `benchmark/<policy>/seed_<seed>/metrics_over_steps.csv`
- `benchmark/<policy>/seed_<seed>/final_metrics.json`
- `summary/benchmark_final_metrics.csv`
- `summary/figures/parameter_error_over_steps.pdf`

Summary artifacts are written under each suite's `summary/` directory. Real-data runs skip the planar-only renderer.
