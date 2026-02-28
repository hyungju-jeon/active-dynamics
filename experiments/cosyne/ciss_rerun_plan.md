# Cosyne CISS Rerun Plan

## Goal

Create a reproducible mid-size rerun protocol that compares `baseline` vs `updated`
model behavior on CISS parameter-identification experiments, with consistent metadata
and summary outputs.

## Scope

- Path: `experiments/cosyne`
- Seeds: `0, 10`
- Track IDs: `active_short, active_long, RND, random`
- Track total steps: `1000`
- Smoke total steps: `1000`
- Repeats per `(model_tag, exp_id, seed)`: `1`
- Writing-aligned defaults: `q_theta=1e-4`, `k_theta=10`, `eig_gamma=1.0`

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

1. Smoke baseline:
- parameter identification (`active_short`) only, seed `0`, steps `1000`

2. Track comparison:
- baseline/updated for `active_short, active_long, RND, random`
- seeds `0, 10`, steps `1000`, repeats `1`

## Output Layout

- `results/CISS/cosyne/smoke/...`
- `results/CISS/cosyne/tracks/...`
- `results/CISS/cosyne/summary/metrics.csv`
- `results/CISS/cosyne/summary/metrics.md`
- `results/CISS/cosyne/summary/figures/*.png`

## Acceptance Criteria

- Preflight passes (required configs + imports).
- Expected track matrix exists for summary:
  `len(exp_ids) * len(seeds) * len(model_tags)`.
- No NaN/inf metric flags in completed runs.
- Summary reports baseline-vs-updated parameter error deltas per track.
