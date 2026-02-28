# Cosyne CISS Parameter Identification Workspace

This folder contains planning and orchestration artifacts for Cosyne-oriented
parameter-identification reruns.

## Contents

- `ciss_rerun_plan.md`: implementation-oriented plan and run matrix.
- `run_manifest.yaml`: locked defaults, metadata schema, and acceptance thresholds.
- `run_ciss_tracks.py`: preflight + execution helper for parameter-ID smoke/track runs.
- `summarize_cosyne_results.py`: aggregate run metadata into CSV/Markdown/figures.

## Quick Start

1. Preflight

```bash
python experiments/cosyne/run_ciss_tracks.py --mode preflight --model-tag baseline
```

2. Parameter-ID smoke (baseline)

```bash
python experiments/cosyne/run_ciss_tracks.py \
  --mode smoke \
  --model-tag baseline \
  --q-theta 1e-4 \
  --k-theta 10 \
  --eig-gamma 1.0 \
  --base-dir results/CISS/cosyne
```

3. Mid-size parameter-ID tracks (example)

```bash
python experiments/cosyne/run_ciss_tracks.py \
  --mode tracks \
  --model-tag updated \
  --seeds 0,10 \
  --exp-ids active_short,active_long,RND,random \
  --total-steps 1000 \
  --q-theta 1e-4 \
  --k-theta 10 \
  --eig-gamma 1.0 \
  --base-dir results/CISS/cosyne
```

4. Summarize

```bash
python experiments/cosyne/summarize_cosyne_results.py \
  --base-dir results/CISS/cosyne \
  --summary-dir results/CISS/cosyne/summary \
  --fail-on-missing
```
