# Cosyne Experiment Workspace

This directory includes two experiment tracks:

- CISS rerun orchestration (`run_ciss_tracks.py`, `summarize_cosyne_results.py`)
- Mixed-family meta-dynamics workflows (training, online ID, reproducible plotting)

## Mixed-Family Meta-Dynamics Layout

- `mixed_family_lib.py`: shared data model, training, checkpoint I/O, rollout eval, online ID, plotting utilities.
- `train_metadynamics.py`: pretrain/eval workflow; writes checkpoint + summary JSON/Markdown.
- `run_online_id.py`: rollout-centered online identification using a checkpoint (or trains if checkpoint omitted).
- `plot_vectorfield_reconstructions.py`: family-wise true-vs-reconstructed vector-field figure generation.
- `plot_embedding_clusters.py`: embedding-cluster figure generation.
- `mixed_family_metadynamics.py`: compatibility wrapper for old `--mode {pretrain_eval,identify,vectorfield_figures}` usage.

## Mixed-Family Quick Start

Default outputs live under `results/cosyne/`.

1. Pretrain + evaluate (creates checkpoint and metrics)

```bash
python3 experiments/cosyne/train_metadynamics.py \
  --system-bank mixed80 \
  --embedding-mode learned_system_id \
  --d-embed 2 \
  --train-samples-per-system 1500 \
  --train-epochs 80 \
  --results-subdir metadynamics_training
```

2. Run rollout-centered online identification from a pretrained checkpoint

```bash
python3 experiments/cosyne/run_online_id.py \
  --checkpoint results/cosyne/metadynamics_training/meta_dynamics_checkpoint.pt \
  --systems duffing_single_00 duffing_bistable_00 van_der_pol_00 double_limit_cycle_00 \
  --policies active_short random \
  --repeats 2 \
  --total-steps 120 \
  --results-subdir metadynamics_online_id
```

Each online-ID session now saves:
- a finalized rollout at `.../rollouts/rollout_<steps>.pkl`
- a per-run metadata record at `.../online_id_record.json`
- the per-step inferred embedding trace inside that record for downstream video rendering

3. Generate the canonical family-wise vector-field reconstruction figure

```bash
python3 experiments/cosyne/plot_vectorfield_reconstructions.py \
  --checkpoint results/cosyne/metadynamics_training/meta_dynamics_checkpoint.pt \
  --output-dir results/cosyne/metadynamics_training
```

Defaults are canonical and reproducible: explicit representative systems, grid range `[-3, 3]`, grid density `25`, and fixed `families_x_(true,reconstructed)_streamplot` layout. The official artifact filenames are `vectorfield_family_comparison_official.png` and `vectorfield_family_comparison_official.json`.

4. (Optional) Generate embedding-cluster figure only

```bash
python3 experiments/cosyne/plot_embedding_clusters.py \
  --checkpoint results/cosyne/metadynamics_training/meta_dynamics_checkpoint.pt \
  --results-subdir metadynamics_training
```

5. (Optional) Render a saved online-ID session as a two-panel true-vs-inferred vector-field video

```bash
python3 experiments/cosyne/render_online_id_trajectory_video.py \
  --summary results/cosyne/metadynamics_online_id/summary.json \
  --system duffing_single_00 \
  --policy active_short \
  --seed 1
```

## Known Duffing Parameter ID via Meta-Dynamics

For direct Duffing parameter identification, use the `known_duffing40` bank with fixed 2D embeddings.
In this bank, each system embedding is exactly the true Duffing parameter pair `(a, b)`.

1. Train the meta-dynamics model on the Duffing-only parameter bank

```bash
python3 experiments/cosyne/train_metadynamics.py \
  --system-bank known_duffing40 \
  --embedding-mode fixed \
  --d-embed 2 \
  --results-subdir known_duffing40_training
```

2. Run online parameter identification

```bash
python3 experiments/cosyne/run_online_id.py \
  --system-bank known_duffing40 \
  --embedding-mode fixed \
  --d-embed 2 \
  --checkpoint results/cosyne/known_duffing40_training/meta_dynamics_checkpoint.pt \
  --systems duffing_single_00 duffing_bistable_00 \
  --policies active_short random no_policy \
  --results-subdir known_duffing40_online_id
```

## CISS Track

Primary Duffing comparison set: `active_short` (active learning), `random`, `no_policy`.

1. Preflight

```bash
conda run -n active-dynamics python experiments/cosyne/run_ciss_tracks.py --mode preflight
```

2. Track runs (`low_action`)

```bash
conda run -n active-dynamics python experiments/cosyne/run_ciss_tracks.py \
  --mode tracks \
  --model-tag low_action \
  --exp-ids active_short,random,no_policy \
  --seeds 0,10,20 \
  --total-steps 500 \
  --repeats 1 \
  --action-max 1 \
  --dynamics-alpha 1.0 \
  --q-theta 5e-4 \
  --k-theta 10 \
  --q-theta-meas-coeff 0.0 \
  --state-noise 0.2 \
  --eig-gamma 1.0 \
  --state-init-uncertainty 25.0 \
  --save-acq-map \
  --acq-map-interval 5 \
  --acq-map-grid 61 \
  --base-dir results/cosyne
```

3. Track runs (`high_action`)

```bash
conda run -n active-dynamics python experiments/cosyne/run_ciss_tracks.py \
  --mode tracks \
  --model-tag high_action \
  --exp-ids active_short,random,no_policy \
  --seeds 0,10,20 \
  --total-steps 500 \
  --repeats 1 \
  --action-max 3 \
  --dynamics-alpha 1.0 \
  --q-theta 5e-4 \
  --k-theta 10 \
  --q-theta-meas-coeff 0.0 \
  --state-noise 0.2 \
  --eig-gamma 1.0 \
  --state-init-uncertainty 25.0 \
  --save-acq-map \
  --acq-map-interval 5 \
  --acq-map-grid 61 \
  --base-dir results/cosyne
```

4. Summarize

```bash
conda run -n active-dynamics python experiments/cosyne/summarize_cosyne_results.py \
  --base-dir results/cosyne \
  --summary-dir results/cosyne/summary \
  --model-tags low_action,high_action \
  --seeds 0,10,20 \
  --exp-ids active_short,random,no_policy \
  --fail-on-missing
```

5. Videos (parallel, default batch set: `info_maps` + `acq_action`)

```bash
conda run -n active-dynamics python experiments/cosyne/generate_session_behavior_video.py \
  --base-dir results/cosyne \
  --model-tag low_action \
  --exp-ids active_short,random,no_policy \
  --seeds 0,10,20 \
  --video-kind all \
  --grid-lim 10 \
  --jobs 4

conda run -n active-dynamics python experiments/cosyne/generate_session_behavior_video.py \
  --base-dir results/cosyne \
  --model-tag high_action \
  --exp-ids active_short,random,no_policy \
  --seeds 0,10,20 \
  --video-kind all \
  --grid-lim 10 \
  --jobs 4

# `traj_vf` is still available for explicit one-off renders:
# conda run -n active-dynamics python experiments/cosyne/generate_session_behavior_video.py \
#   --base-dir results/cosyne \
#   --model-tag low_action \
#   --exp-id active_short \
#   --seed 0 \
#   --video-kind traj_vf
```

6. Interactive information-map GUI

```bash
conda run -n active-dynamics python experiments/cosyne/info_map_gui.py

# examples:
# conda run -n active-dynamics python experiments/cosyne/info_map_gui.py --dynamics-type "van der poll"
# conda run -n active-dynamics python experiments/cosyne/info_map_gui.py --dynamics-type snowman
# conda run -n active-dynamics python experiments/cosyne/info_map_gui.py --dynamics-type "double limit cycle"
```
