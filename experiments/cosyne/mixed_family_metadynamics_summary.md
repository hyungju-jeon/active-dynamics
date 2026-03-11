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
