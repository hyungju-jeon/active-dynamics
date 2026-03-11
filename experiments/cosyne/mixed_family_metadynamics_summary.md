# Mixed-family meta-dynamics pretraining summary

- System bank: mixed40
- Embedding mode: learned_system_id
- Systems: 40
- Final train loss: 30181.066399
- Verification passed: 40/40
- Checkpoint: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311/meta_dynamics_checkpoint.pt`
- Embedding figure: `/home/hyungju/Desktop/al-metadynamics/results/mixed_family_metadynamics_pretrain_20260311/embedding_family_clusters.png`

## Family rollout metrics
- duffing_single: mean rollout MSE 0.2293, mean final-state MSE 0.3027
- duffing_bistable: mean rollout MSE 0.9763, mean final-state MSE 1.5596
- van_der_pol: mean rollout MSE 2.4336, mean final-state MSE 4.0033
- double_limit_cycle: mean rollout MSE 1.0593, mean final-state MSE 1.5230

## Parameter-bank verification
- double_limit_cycle: 10/10 passed, max radius 2.828, max speed 317.040
- duffing_bistable: 10/10 passed, max radius 3.457, max speed 26.641
- duffing_single: 10/10 passed, max radius 2.924, max speed 52.078
- van_der_pol: 10/10 passed, max radius 4.961, max speed 191.858
