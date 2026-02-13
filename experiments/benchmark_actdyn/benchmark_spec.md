# Benchmark Spec (v1)

## Purpose
Establish a reproducible baseline benchmark for latent-state/parameter inference under active data collection policies.

## Scope
- Compare lightweight policy baselines:
  - `baseline-random`
  - `baseline-prbs`
  - `baseline-ce-mpc`
  - `baseline-thompson`
  - `baseline-ucb`
- Run across placeholder environment tracks with hidden dynamics parameters.
- Produce unified per-step metrics and aggregated analysis outputs.

## Environment Tracks (v1 placeholders)
- `linear_easy`: stationary latent dynamics with moderate noise.
- `linear_shifted`: latent dynamics parameter shifts mid-episode.

These tracks intentionally avoid heavy simulator coupling and provide deterministic smoke-coverage behavior with seed control.

## Method Contract
Each method must output an action at each step and support a lightweight online update hook when reward/information feedback is available.

## Metrics
Logged per step (`metrics.csv` and `metrics.jsonl`):
- `param_abs_error`: absolute error of estimated parameter vs true parameter.
- `latent_abs_error`: one-step latent prediction error proxy.
- `posterior_var`: estimator variance proxy.
- `info_gain`: per-step reduction in posterior variance.
- `reward`: information reward with action penalty.
- `action_norm`: action magnitude.
- `policy_cost`: internal method objective/cost output.
- `runtime_ms`: per-step policy+update wall time.

## Required Outputs
- Process stage:
  - Unified step logs (`metrics.csv`, `metrics.jsonl`).
  - Episode aggregate logs (`episode_summary.csv`).
  - Run metadata (`metadata.json`).
- Analysis stage:
  - `summary_table.csv`
  - plots under `figures/`

## Reproducibility
- Config-driven seeds and environment/method lists.
- One seed = one independent run for each `(environment, method, episode)` combination.

## Out of Scope (v1)
- Full FLEX implementation.
- Heavy simulator integration beyond placeholder tracks.
- Large-scale hyperparameter search.

## TODOs for v2
- `TODO(FLEX-v2):` replace placeholder latent environments with FLEX-compatible interfaces.
- `TODO(FLEX-v2):` add richer posterior estimators (e.g., particle/ensemble variants).
- `TODO(FLEX-v2):` extend analysis to regret/calibration metrics and significance tests.
