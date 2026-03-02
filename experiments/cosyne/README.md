# Cosyne Parameter Identification Workspace

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
python experiments/cosyne/run_ciss_tracks.py --mode preflight
```

2. Parameter-ID smoke (updated-only)

```bash
python experiments/cosyne/run_ciss_tracks.py \
  --mode smoke \
  --model-tag updated \
  --q-theta 1e-4 \
  --k-theta 10 \
  --eig-gamma 1.0 \
  --base-dir results/cosyne
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
  --base-dir results/cosyne
```

4. Ablation suite (planning window + update frequency)

```bash
python experiments/cosyne/run_ciss_tracks.py \
  --mode ablation \
  --model-tag updated \
  --seeds 0,10,20 \
  --ablation-exp-id active_short \
  --ablation-total-steps 1000 \
  --ablation-planning-windows 3,5,10,15 \
  --ablation-k-thetas 1,5,10,20 \
  --ablation-fixed-k-theta 10 \
  --ablation-fixed-planning-window 5 \
  --base-dir results/cosyne
```

5. Summarize (updated-only)

```bash
python experiments/cosyne/summarize_cosyne_results.py \
  --base-dir results/cosyne \
  --summary-dir results/cosyne/summary \
  --model-tags updated \
  --fail-on-missing
```

Baseline-vs-updated comparison is intentionally excluded from this folder.
