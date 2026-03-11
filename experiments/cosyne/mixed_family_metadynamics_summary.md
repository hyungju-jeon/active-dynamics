# Mixed-family meta-dynamics pretraining summary

- System bank: mixed80
- Embedding mode: learned_system_id
- Systems: 80
- Final train loss: 2.186745
- Training samples/system: 4000
- Training epochs: 160
- Batch size: 1024
- Verification passed: 80/80
- Checkpoint: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_speedconstrained80/meta_dynamics_checkpoint.pt`
- Embedding figure: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_speedconstrained80/embedding_family_clusters.png`

## Family rollout metrics
- duffing_single: mean rollout MSE 0.0024, mean final-state MSE 0.0025
- duffing_bistable: mean rollout MSE 0.1428, mean final-state MSE 0.3154
- van_der_pol: mean rollout MSE 0.1418, mean final-state MSE 0.3253
- double_limit_cycle: mean rollout MSE 0.1089, mean final-state MSE 0.2876

## Parameter-bank verification
- double_limit_cycle: 20/20 passed, max radius 2.828, p95 speed max 19.353, max speed 116.745
- duffing_bistable: 20/20 passed, max radius 3.450, p95 speed max 15.593, max speed 27.176
- duffing_single: 20/20 passed, max radius 3.210, p95 speed max 33.311, max speed 61.838
- van_der_pol: 20/20 passed, max radius 3.842, p95 speed max 81.008, max speed 117.712
