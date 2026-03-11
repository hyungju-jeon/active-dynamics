# Mixed-family meta-dynamics pretraining summary

- System bank: mixed80
- Embedding mode: learned_system_id
- Systems: 80
- Final train loss: 4.160894
- Training samples/system: 4000
- Training epochs: 200
- Batch size: 1024
- Verification passed: 80/80
- Checkpoint: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_speedconstrained80_bidirectional_d4/meta_dynamics_checkpoint.pt`
- Embedding figure: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_speedconstrained80_bidirectional_d4/embedding_family_clusters.png`

## Family rollout metrics
- duffing_single: mean rollout MSE 0.0018, mean final-state MSE 0.0016
- duffing_bistable: mean rollout MSE 0.0508, mean final-state MSE 0.2437
- van_der_pol: mean rollout MSE 0.0339, mean final-state MSE 0.0739
- double_limit_cycle: mean rollout MSE 0.0701, mean final-state MSE 0.1707

## Parameter-bank verification
- double_limit_cycle: 20/20 passed, max radius 2.828, max final radius 2.018, p95 speed max 18.879, max speed 116.745, rotations ccw, cw
- duffing_bistable: 20/20 passed, max radius 3.450, max final radius 2.125, p95 speed max 15.593, max speed 27.176, rotations cw
- duffing_single: 20/20 passed, max radius 3.210, max final radius 0.280, p95 speed max 33.311, max speed 61.838, rotations cw
- van_der_pol: 20/20 passed, max radius 3.842, max final radius 2.068, p95 speed max 81.008, max speed 117.712, rotations cw

## Added vector-field comparison figure
- Figure: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_speedconstrained80_bidirectional_d4/vectorfield_family_comparison.png`
- Metadata: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_speedconstrained80_bidirectional_d4/vectorfield_family_comparison.json`
- Layout: compact 4x2 family-wise true vs reconstructed streamplots using representative systems per family.

## Rollout-centered online ID smoke/confirmation run
- Results dir: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_onlineid_20260311_rollout_smoke_from_pretrain_d4`
- Summary JSON: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_onlineid_20260311_rollout_smoke_from_pretrain_d4/summary.json`
- Baseline checkpoint: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_speedconstrained80_bidirectional_d4/meta_dynamics_checkpoint.pt`
- Embedding mode: `learned_system_id`
- Systems: 8 total (2 per family), repeats: 2, policies: `active_short` vs `random`, total steps: 120
- Overall rollout MSE: active_short 1.7234 ± 1.4811 vs random 1.8197 ± 1.4836
- Overall final-state MSE: active_short 2.4369 vs random 2.0780
- Overall final embedding error: active_short 1.9209 vs random 2.4945

### Per-family rollout MSE snapshot
- double_limit_cycle: active_short rollout MSE 2.6200 vs random 1.7389; final error 0.7580 vs 1.6564
- duffing_bistable: active_short rollout MSE 0.3198 vs random 0.9061; final error 1.0397 vs 1.1337
- duffing_single: active_short rollout MSE 1.7523 vs random 2.2699; final error 2.6739 vs 1.6328
- van_der_pol: active_short rollout MSE 2.2014 vs random 2.3638; final error 3.2122 vs 5.5550
