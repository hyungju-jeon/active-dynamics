# TBME Experiment Program

This directory contains TBME-specific catalogs and thin wrappers on top of the shared experiment runners in `experiments/`.

## Layout

- `experiment_env.yaml`: TBME environment presets
- `experiment_model.yaml`: TBME-only model and policy additions
- `experiment_suite.yaml`: TBME manuscript experiment matrices
- `experiment_specs.py`: TBME catalog view that merges the shared base env/model catalogs with the TBME additions
- `_run_family.py`: shared helper for TBME family launchers
- `run_exp1.py`: convenience launcher for Experiment 1 suites
- `run_exp2.py`: convenience launcher for Experiment 2 suites
- `run_exp3.py`: convenience launcher for Experiment 3 suites
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
- Experiment 3 suite: `256` selected replay transitions

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

Table:

- final parameter error and final trajectory `R^2` aggregated over seeds for all three systems

### Interpretation Goal

This experiment establishes the basic method claim: the active planner should identify dynamics faster than passive/random baselines on canonical low-dimensional systems when the estimator family is correctly specified.

## Experiment 2: Convergence and Robustness

### Question

How much does performance degrade when the estimator is misspecified or the observation regime becomes less informative?

### Hypothesis

Active methods should still retain an advantage under moderate mismatch, but the gain should shrink when observation quality drops or when the estimator family is structurally wrong.

### Mismatch Presets

Available environment presets:

- `tbme_duffing_obs_mismatch`
- `tbme_duffing_noise_mismatch`
- `tbme_duffing_family_mismatch`
- `tbme_pendulum_param_mismatch`
- `tbme_double_integrator_mismatch`

Main suites:

- `tbme_exp2_robustness_duffing`
- `tbme_exp2_robustness_pendulum`
- `tbme_exp2_robustness_double_integrator`

`tbme_duffing_family_mismatch` keeps the true Duffing environment but swaps the learned model family through `estimator_system_id: damped_pendulum`.

### Compared Methods

- `active_planning`
- `e_optimality`
- `baseline_prbs`
- `random`
- `off_policy`

These suites are intentionally narrower than Experiment 1. The robustness question is about whether the active method still converges usefully, not about exhaustively sweeping all policy/objective combinations.

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

## Experiment 3: High-Dimensional Neural Replay

### Question

Can the same active-selection logic scale to high-dimensional neural observations without requiring the closed-loop synthetic setting used in Experiments 1 and 2?

### Hypothesis

Even in an offline replay setting, actively selected transitions should improve held-out dynamics prediction faster than passive or random selection baselines.

### Dataset and Suite

Environment preset:

- `tbme_mcrtt_spikes`

Suite:

- `tbme_exp3_realdata_policy`

This path is an offline replay benchmark. It fits a linear ridge dynamics model after each selected transition and evaluates the learned model on held-out replay transitions.
The current prepared replay file is derived from the published DANDI `000129` release `0.241017.1444`, using the train NWB asset `sub-Indy_desc-train_behavior+ecephys.nwb`.

### Compared Methods

- `active_myopic`
- `active_planning`
- `baseline_prbs`
- `random`
- `off_policy`

### Protocol

The current replay pipeline:

1. loads a standardized `.npz` archive from `dataset_path`
2. projects behavior to `latent_dim` and standardizes it
3. optionally truncates observation channels to the highest-variance subset
4. splits replay transitions into train and evaluation partitions
5. selects training transitions sequentially according to the policy
6. refits a linear ridge dynamics model after each selected transition
7. reports held-out prediction quality over time

This should be described in the manuscript as counterfactual replay-based active selection, not as an online intervention experiment.

### Primary Metrics

Primary:

- held-out dynamics MSE over selected transitions
- final held-out dynamics MSE

Secondary:

- held-out trajectory `R^2`
- information trace

### Expected Figures and Tables

Lead figure:

- held-out dynamics MSE versus selected transitions for the replay policy sweep

Supporting figure:

- one representative replay-session panel combining latent state evolution, uncertainty/information trace, and policy comparison summary

Table:

- final held-out MSE and final trajectory `R^2` by policy

### Interpretation Goal

This experiment supports the manuscript scaling claim: the active-selection framework should transfer from simple synthetic systems to high-dimensional neural observations while preserving an advantage on predictive quality.

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
  --seeds 0 \
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

Replay runs write:

- `dynamics_mse_trace.csv`
- `trajectory_r2_trace.csv`
- `information_trace.csv`
- `state_action_trace.csv`

Summary artifacts are written under each suite's `summary/` directory. Real-data runs skip the planar-only renderer.
