# COSYNE Action-Limit Parameter-ID Workspace

This folder contains orchestration for COSYNE parameter-identification reruns
with action-limit comparison only.

## Active Scope

- Tracks-only (no smoke, no ablation, no baseline comparison)
- Model tags:
  - `low_action` with `|u_max| = 1`
- `high_action` with `|u_max| = 3`
- Seeds: `0,10,20`
- Exp IDs: `active_short,active_long,RND,random`
- Default dynamics setting: `--dynamics-alpha 1.0`

## Core Scripts

- `run_ciss_tracks.py`: preflight + track execution
- `summarize_cosyne_results.py`: metrics/figures aggregation
- `generate_session_behavior_video.py`: video generation
  - `info_maps`: exact Jacobian-based `I_z` and `I_theta` map video (fixed log scales)
  - `traj_vf`: inferred/true trajectory over true/inferred vector fields (time-varying inferred params)
  - `acq_action`: acquisition colormap with executed action overlays (for iCEM debugging)
- `info_map_gui.py`: interactive 2x3 information-map GUI with sliders and loading controls

## Repro Commands

1. Preflight

```bash
conda run -n active-dynamics python experiments/cosyne/run_ciss_tracks.py --mode preflight
```

2. Track runs (`low_action`)

```bash
conda run -n active-dynamics python experiments/cosyne/run_ciss_tracks.py \
  --mode tracks \
  --model-tag low_action \
  --exp-ids active_short,active_long,RND,random \
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
  --exp-ids active_short,active_long,RND,random \
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
  --exp-ids active_short,active_long,RND,random \
  --fail-on-missing
```

5. Videos (parallel, all kinds)

```bash
conda run -n active-dynamics python experiments/cosyne/generate_session_behavior_video.py \
  --base-dir results/cosyne \
  --model-tag low_action \
  --exp-ids active_short,active_long,RND,random \
  --seeds 0,10,20 \
  --video-kind all \
  --grid-lim 10 \
  --jobs 4

conda run -n active-dynamics python experiments/cosyne/generate_session_behavior_video.py \
  --base-dir results/cosyne \
  --model-tag high_action \
  --exp-ids active_short,active_long,RND,random \
  --seeds 0,10,20 \
  --video-kind all \
  --grid-lim 10 \
  --jobs 4
```

6. Interactive information-map GUI

```bash
conda run -n active-dynamics python experiments/cosyne/info_map_gui.py

# examples:
# conda run -n active-dynamics python experiments/cosyne/info_map_gui.py --dynamics-type "van der poll"
# conda run -n active-dynamics python experiments/cosyne/info_map_gui.py --dynamics-type snowman
# conda run -n active-dynamics python experiments/cosyne/info_map_gui.py --dynamics-type "double limit cycle"
```
