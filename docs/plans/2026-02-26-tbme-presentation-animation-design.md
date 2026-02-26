# TBME Presentation Animation Design (Modular 9-Minute Set)

**Date:** 2026-02-26  
**Status:** Approved by user  
**Audience:** Computational neuroscientists with limited statistical signal processing background

## Goal

Create three separate, stitchable animation videos that explain:

1. Core concept and motivation
2. Intuition-first method walkthrough (state/parameter updates)
3. Active learning with trajectory planning (Louis-based)

Each video is exactly 3 minutes and split into modular 30-second chunks so clips can be reordered and assembled later in presentation editing.

## Locked Requirements

1. Runtime split: `3 + 3 + 3` minutes.
2. Style: Geometry-first, 3Blue1Brown-like motion language.
3. Output mode: Two versions for every chunk:
   - `clean`: visuals only, zero text overlays.
   - `symbol`: same visuals/timing with minimal equation symbols only.
4. Chunking: hard chunk boundaries suitable for stitching.

## Deliverable Packaging

1. Total chunks: 18 (`6` per video).
2. Total files: 36 (`18 clean + 18 symbol`).
3. Naming pattern:
   - `V1_C01_clean.mp4` ... `V1_C06_clean.mp4`
   - `V1_C01_symbol.mp4` ... `V1_C06_symbol.mp4`
   - same for `V2_*` and `V3_*`.
4. Pair constraint: each `clean/symbol` pair must have identical frame count and timing.

## Global Visual Grammar

### Timing and Render Locks

1. Resolution/FPS: `1920x1080`, `30 fps`.
2. Chunk duration: exactly `30.0s` per chunk.
3. Stitch handles: first `1.0s` and last `1.0s` hold minimal motion.

### Motion Language

1. Continuous morphs over abrupt cuts.
2. Transition set: `fade`, `morph`, `camera pan`, `zoom`, `trace`.
3. One dominant camera move per chunk.
4. Minimum dwell time on new concept object: `2.0s`.

### Color Semantics (Fixed Across All Clips)

1. Blue: latent state and state uncertainty.
2. Orange: parameter belief and parameter updates.
3. Gray: observations and noise.
4. Red: control candidates and rollout trajectories.
5. Gold: information gain/objective.
6. Hatched red subtraction: Louis missing-information correction.

### Text Policy

1. `clean`: no text or symbols.
2. `symbol`: compact math tokens only (`z^-`, `P_z`, `G`, `I_z`, `I_θ`, `logdet`), no explanatory sentences.

### Layout Safety

1. Keep all meaningful geometry inside 10% safe margins.
2. Avoid axis-label dependence.
3. Optimize for projector readability.

## Storyboard (Approved)

## Video 1: Concept (3:00 total)

1. `V1_C01` (0:00-0:30): Attractor landscape and passive trajectory collapse.
2. `V1_C02` (0:30-1:00): Latent-vs-observation gap under noisy/partial observations.
3. `V1_C03` (1:00-1:30): Separate state uncertainty and parameter uncertainty objects.
4. `V1_C04` (1:30-2:00): Online inference loop (predict -> observe -> state update -> parameter update).
5. `V1_C05` (2:00-2:30): Why active perturbation improves information.
6. `V1_C06` (2:30-3:00): Passive vs active uncertainty reduction comparison.

## Video 2: Method Walkthrough (Intuition-First, 3:00 total)

1. `V2_C01` (0:00-0:30): State prediction through nonlinear dynamics and process noise.
2. `V2_C02` (0:30-1:00): Observation linearization and local measurement information.
3. `V2_C03` (1:00-1:30): State update intuition via score pull and curvature tightening.
4. `V2_C04` (1:30-2:00): Information-form state update as additive precision.
5. `V2_C05` (2:00-2:30): Parameter update intuition via sensitivity projection (`G` mapping).
6. `V2_C06` (2:30-3:00): Alternating state/parameter update cycle at one timestep.

## Video 3: Active Learning With Trajectory Planning (Louis-Based, 3:00 total)

1. `V3_C01` (0:00-0:30): Sample candidate control sequences and roll out trajectories.
2. `V3_C02` (0:30-1:00): Propagate sensitivity along each candidate trajectory.
3. `V3_C03` (1:00-1:30): Louis observed-information contribution (additive precision blocks).
4. `V3_C04` (1:30-2:00): Missing-information correction (subtractive hatched blocks).
5. `V3_C05` (2:00-2:30): Horizon objective as uncertainty-volume reduction (`logdet` view).
6. `V3_C06` (2:30-3:00): Receding-horizon loop: apply first action, observe, update, replan.

## Equation-to-Visual Mapping

1. `\hat{z}^-`: predicted state point moving on vector field.
2. `P_z^-`: pre-update blue ellipse.
3. `P_z`: post-update blue ellipse (contracted/rotated).
4. `s_z`: state-score force arrow.
5. `I_z`: local curvature sharpness or information block.
6. `\Lambda_z`: precision gain bar/block (inverse covariance intuition).
7. `G`: parameter-to-state sensitivity deformation arrows.
8. `s_θ`: parameter-score force arrow.
9. `I_θ`: parameter-space curvature block.
10. Louis correction term: hatched subtractive block tied to state uncertainty.
11. `\log\det`: uncertainty volume reduction glyph.

## Fidelity to Paper

1. Video 2 uses the main-text intuition and update flow for non-expert accessibility.
2. Video 3 uses Louis-based planning interpretation as requested.
3. Symbol overlays remain notation-compatible with TBME writing source.

## Acceptance Criteria

1. All 36 files exist with expected naming pattern.
2. Every chunk is exactly 30.0s at 30 fps.
3. `clean/symbol` versions are frame-aligned.
4. Color semantics and symbol mapping are consistent across all chunks.
5. Chunks can be stitched in arbitrary order without timing artifacts.

