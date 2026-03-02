# Cosyne Parameter-ID Rerun Plan

## Goal

Create a reproducible mid-size rerun protocol for the **updated model only** on
parameter-identification experiments, with consistent metadata and summary outputs.

## Scope

- Path: `experiments/cosyne`
- Seeds: `0, 10, 20, 30, 40`
- Track IDs: `active_short, active_long, RND, random`
- Track total steps: `1000`
- Smoke total steps: `1000`
- Repeats per `(model_tag, exp_id, seed)`: `1` with `model_tag=updated`
- Writing-aligned defaults: `q_theta=1e-4`, `k_theta=10`, `eig_gamma=1.0`
- Ablation axes: planning window and parameter update frequency (`k_theta`)

## Required Artifacts

- `experiments/cosyne/README.md`
- `experiments/cosyne/ciss_rerun_plan.md`
- `experiments/cosyne/run_manifest.yaml`
- `experiments/cosyne/run_ciss_tracks.py`
- `experiments/cosyne/summarize_cosyne_results.py`

## Metadata Contract (Per Run)

Each run writes `run_metadata.json` with:

- `model_tag`
- `commit`
- `seed`
- `exp_id`
- `total_steps`
- `base_dir`
- `status`
- `start_time`
- `end_time`
- `runtime_sec`
- `results_path`
- `rollout_steps`
- `state_error_mean`
- `state_error_final`
- `embedding_error_mean`
- `embedding_error_final`
- `q_theta`
- `k_theta`
- `eig_gamma`
- `writing_ref`
- `nan_detected`

## Execution Matrix

1. Smoke:
- parameter identification (`active_short`) only, seed `0`, steps `1000`

2. Track comparison:
- updated-only for `active_short, active_long, RND, random`
- seeds `0, 10, 20, 30, 40`, steps `1000`, repeats `1`

3. Ablation:
- planning window sweep: `3, 5, 10, 15` (fixed `k_theta=10`)
- update frequency sweep: `k_theta=1, 5, 10, 20` (fixed planning window `5`)
- seeds `0, 10, 20`, steps `1000`, repeats `1`

## Output Layout

- `results/cosyne/smoke/...`
- `results/cosyne/tracks/...`
- `results/cosyne/ablation/...`
- `results/cosyne/summary/metrics.csv`
- `results/cosyne/summary/metrics.md`
- `results/cosyne/summary/figures/*.png`

## Acceptance Criteria

- Preflight passes (required configs + imports).
- Expected track matrix exists for summary:
  `len(exp_ids) * len(seeds) * 1` (`updated` only).
- No NaN/inf metric flags in completed runs.
- Summary reports per-track parameter error and trajectory-R2 trends.
- Summary includes ablation plots for both requested axes.
