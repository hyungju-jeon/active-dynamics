# TBME Experiment Program on the COSYNE Scaffold

This directory now carries the TBME manuscript experiments on top of the catalog-driven COSYNE runner.

## Catalog Layout

- `experiment_env.yaml`: environment presets, latent dynamics presets, observation-model presets, and replay-data presets
- `experiment_model.yaml`: policy, objective, and planning schedule presets
- `experiment_suite.yaml`: experiment matrices used for manuscript runs

The intended execution flow is:

1. Choose an environment preset from `experiment_env.yaml`
2. Choose policy or objective presets from `experiment_model.yaml`
3. Combine them into a named manuscript suite in `experiment_suite.yaml`
4. Run through `python -m experiments.cosyne.run_experiments`

## Experiment Families

### Experiment 1: Basic Proof of Work

Goal: show that active identification improves convergence on simple controlled systems.

Environment presets:

- `tbme_duffing_easy`
- `tbme_pendulum_damped`
- `tbme_double_integrator`

Policy comparison suites:

- `tbme_exp1_duffing_policy`
- `tbme_exp1_pendulum_policy`
- `tbme_exp1_double_integrator_policy`

Compared policies:

- `active_myopic`
- `active_planning`
- `baseline_prbs`
- `random`
- `off_policy`

Primary outputs:

- parameter-error traces
- trajectory-R2 traces
- information traces
- session summaries and policy-comparison figures

### Experiment 2: Convergence and Robustness Under Mismatch

Goal: test convergence under observation mismatch, noise mismatch, parameter mismatch, and model-family mismatch.

Environment presets:

- `tbme_duffing_family_mismatch`: true Duffing environment with `damped_pendulum` estimator family
- `tbme_duffing_obs_mismatch`
- `tbme_duffing_noise_mismatch`
- `tbme_pendulum_param_mismatch`
- `tbme_double_integrator_mismatch`

Main robustness suites:

- `tbme_exp2_robustness_duffing`
- `tbme_exp2_robustness_pendulum`
- `tbme_exp2_robustness_double_integrator`

The family-mismatch hook is controlled by `estimator_system_id` in `experiment_env.yaml`. The true environment still comes from `system_id`; only the learned synthetic model changes.

### Experiment 3: High-Dimensional Neural Replay Data

Goal: evaluate active transition selection on high-dimensional neural observations through an offline replay protocol.

Environment preset:

- `tbme_mcrtt_spikes`

Main suite:

- `tbme_exp3_realdata_policy`

Protocol:

- the runner loads a standardized replay dataset
- behavior is projected to `latent_dim` and standardized
- spike counts are optionally truncated to the highest-variance channels
- training transitions are selected sequentially according to the policy
- a linear ridge dynamics model is refit after each selected transition
- held-out prediction MSE and trajectory R2 are reported over time

This is an offline counterfactual selection benchmark, not a closed-loop intervention claim.

## Objective and Policy Presets

The TBME sweep adds:

- `baseline_prbs`: persistent-excitation baseline
- `baseline_random`: explicit random baseline
- `e_optimality`: smallest-eigenvalue information objective

The runner reads `policy_type` directly from the model catalog. Current supported values are:

- `mpc-icem`
- `random`
- `baseline-random`
- `baseline-prbs`
- `off-policy`

## Commands

Run one suite:

```bash
python -m experiments.cosyne.run_experiments \
  --exp-id tbme_exp1_duffing_policy \
  --mode all \
  --seeds 0,10,20 \
  --base-dir results/cosyne
```

Run the real-data suite:

```bash
python -m experiments.cosyne.run_experiments \
  --exp-id tbme_exp3_realdata_policy \
  --mode all \
  --seeds 0 \
  --base-dir results/cosyne
```

Run summary only:

```bash
python -m experiments.cosyne.run_experiments \
  --exp-id tbme_exp2_robustness_duffing \
  --mode summary \
  --seeds 0,10,20 \
  --base-dir results/cosyne
```

## Replay Dataset Format

The current TBME real-data path expects a standardized `.npz` archive, referenced by `dataset_path` in `experiment_env.yaml`.

Required arrays:

- `behavior`: shape `(T, D_state)`
- `spikes`: shape `(T, D_obs)`

Optional arrays:

- `dt`: scalar or length-1 array in seconds

Default keys for `tbme_mcrtt_spikes`:

- `state_key: behavior`
- `observation_key: spikes`

The current loader lives in `realdata_spiking.py` and is intended as a scaffold for a future direct DANDI or NWB preparation step.

## Outputs

Default outputs are written under `results/cosyne/<session>/...`.

Synthetic suites write:

- `parameter_error_trace.csv` or `dynamics_mse_trace.csv`
- `trajectory_r2_trace.csv`
- `information_trace.csv`
- `state_action_trace.csv`
- optional `planned_trajectory_trace.npz`
- optional `acquisition_map_trace.npz`

Replay suites write:

- `dynamics_mse_trace.csv`
- `trajectory_r2_trace.csv`
- `information_trace.csv`
- `state_action_trace.csv`

Summary artifacts are written under each suite's `summary/` directory. Real-data runs skip the planar-only video renderer.
