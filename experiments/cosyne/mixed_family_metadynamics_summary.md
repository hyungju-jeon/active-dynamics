# Mixed-family meta-dynamics pretraining summary

- System bank: mixed200
- Embedding mode: learned_system_id
- Systems: 200
- Final train loss: 151.900425
- Training samples/system: 2000
- Training epochs: 100
- Batch size: 1024
- Verification passed: 200/200
- Checkpoint: `/home/hyungju/Desktop/active-dynamics/results/cosyne/metadynamics_training_geomreg_d3/meta_dynamics_checkpoint.pt`
- Embedding figure: `/home/hyungju/Desktop/active-dynamics/results/cosyne/metadynamics_training_geomreg_d3/embedding_family_clusters.png`

## Family rollout metrics
- duffing_single: mean rollout MSE 0.0491, mean final-state MSE 0.0674
- duffing_bistable: mean rollout MSE 0.4227, mean final-state MSE 1.3821
- van_der_pol: mean rollout MSE 0.7246, mean final-state MSE 1.6863
- double_limit_cycle: mean rollout MSE 1.0378, mean final-state MSE 2.3195

## Parameter-bank verification
- double_limit_cycle: 50/50 passed, max radius 2.828, max final radius 1.953, p95 speed max 10.682, max speed 63.104, rotations ccw, cw
- duffing_bistable: 50/50 passed, max radius 3.515, max final radius 2.112, p95 speed max 9.609, max speed 13.828, rotations cw
- duffing_single: 50/50 passed, max radius 3.145, max final radius 0.652, p95 speed max 19.269, max speed 32.780, rotations cw
- van_der_pol: 50/50 passed, max radius 3.882, max final radius 2.320, p95 speed max 40.641, max speed 61.420, rotations cw
