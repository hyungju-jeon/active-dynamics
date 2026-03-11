# Mixed-family meta-dynamics pretraining summary

- System bank: mixed80
- Embedding mode: learned_system_id
- Systems: 80
- Final train loss: 48.577294
- Training samples/system: 3000
- Training epochs: 120
- Batch size: 1024
- Verification passed: 80/80
- Checkpoint: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_large80/meta_dynamics_checkpoint.pt`
- Embedding figure: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311_large80/embedding_family_clusters.png`

## Family rollout metrics
- duffing_single: mean rollout MSE 0.0402, mean final-state MSE 0.0371
- duffing_bistable: mean rollout MSE 2.9176, mean final-state MSE 4.9351
- van_der_pol: mean rollout MSE 3.4296, mean final-state MSE 5.8063
- double_limit_cycle: mean rollout MSE 1.5434, mean final-state MSE 2.9762

## Parameter-bank verification
- double_limit_cycle: 20/20 passed, max radius 2.828, max speed 334.302
- duffing_bistable: 20/20 passed, max radius 3.656, max speed 28.003
- duffing_single: 20/20 passed, max radius 3.210, max speed 61.838
- van_der_pol: 20/20 passed, max radius 5.193, max speed 216.286
