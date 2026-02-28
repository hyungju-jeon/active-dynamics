# Model vs Writing Compliance Check

This audit treats `docs/active-dynamics-writing/methods.tex` as the source of truth.

## Status

The current parameter-identification implementation is **aligned at equation level** for the key online update and planning equations used in the Cosyne rerun.

## Implemented Alignment

1. Parameter drift prior (`Q_theta`) is now applied before each block update:
   - `P_theta^- = P_theta + Q_theta I`
   - implemented in `FilteringEmbedding._apply_embedding_block_update`.
2. Asynchronous block update (`K_theta`) is implemented:
   - score/information are accumulated over `k_theta` steps,
   - information-form update is applied at block boundaries.
3. Sensitivity recursion now follows writing form:
   - `S_t = F_theta,t + F_z,t S_{t-1}`.
4. Parameter score/information use writing-consistent forms:
   - `s_t = S_t^T (P_z^-)^{-1} (z_hat - z_hat^-)`
   - `I_t = S_t^T (I + P_z^- I_z)^{-1} I_z S_t`.
5. EIG objective now uses discounted `0.5 * logdet`:
   - horizon discount `gamma` is configurable,
   - stable `slogdet` + PSD fallback are used.

## Notes

- Scope here is parameter-identification reruns in `experiments/cosyne`.
- RBF experiment tracks are intentionally excluded from this rerun protocol.
