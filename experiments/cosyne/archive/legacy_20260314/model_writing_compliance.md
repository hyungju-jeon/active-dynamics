# Model vs Writing Compliance Check

This audit treats `docs/active-dynamics-writing/methods.tex` as the source of truth.

## Status

The current parameter-identification implementation is aligned at equation level for:

- latent-state update,
- parameter posterior update,
- planning Fisher-information objective.

## Implemented Alignment

1. Parameter drift prior (`Q_theta`) is applied before each block update:
   - `P_theta^- = P_theta + Q_theta I`
   - implemented in `FilteringEmbedding._apply_embedding_block_update`.
2. Asynchronous block update (`K_theta`) is implemented:
   - score/information accumulated over `k_theta` steps,
   - information-form update applied at block boundaries.
3. Sensitivity recursion follows writing form:
   - `S_t = F_theta,t + F_z,t S_{t-1}`.
4. Parameter score/information follow writing-consistent forms:
   - `s_t = S_t^T (P_z^-)^{-1} (z_hat - z_hat^-)`
   - `I_t = S_t^T (I + P_z^- I_z)^{-1} I_z S_t`.
5. Planning information increment follows writing form:
   - per-step increment uses attenuated Fisher pullback
     `(I + P_z^- I_z)^{-1} I_z`,
   - predicted-state sensitivity is propagated recursively over horizon.
6. EIG objective uses discounted `0.5 * logdet`:
   - horizon discount `gamma` is configurable,
   - stable `slogdet` + PSD fallback are used.

## State/Parameter Update Check

1. State update path in `FilteringEmbedding.update_posterior_embedding` matches EKF covariance-form update with numerically stabilized covariance propagation and update.
2. State reset covariance is configurable via `state_init_uncertainty`; Cosyne runs use a high prior uncertainty (`25.0`) to reflect weak initial state knowledge.
3. Parameter update path matches block-wise information-form update with:
   - block sensitivity reset at boundary,
   - block score/information accumulation,
   - prior drift before information fusion.

## Verification Evidence

- `tests/test_filtering_embedding_writing_alignment.py`
- `tests/test_embedding_fisher_metric_writing_alignment.py`
- `tests/test_cosyne_scripts.py`

All passed in conda environment `active-dynamics`.

## Notes

- Scope here is parameter-identification reruns in `experiments/cosyne`.
- RBF experiment tracks are intentionally excluded from this rerun protocol.
- `Fe`/`Fz` callables are treated as continuous-time Jacobians in code and converted to discrete-time recursion using `dt`.
