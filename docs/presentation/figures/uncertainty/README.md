# Bayesian Uncertainty Reduction Figure

This directory contains a standalone script that generates a clean academic visualization of Bayesian uncertainty reduction as a two-panel 3D surface figure:

- `Prior belief`
- `After observing`

The panels use transparent Gaussian surfaces, wireframe overlays, floor contours, and a curved transition arrow labeled `After new observation`.

## Generate the Figure

From the repository root:

```bash
python docs/presentation/figures/uncertainty/generate_bayesian_uncertainty_reduction.py
```

Default outputs:

- `docs/presentation/figures/uncertainty/bayesian_uncertainty_reduction.svg`
- `docs/presentation/figures/uncertainty/bayesian_uncertainty_reduction.png`

## Optional Flags

```bash
python docs/presentation/figures/uncertainty/generate_bayesian_uncertainty_reduction.py \
  --outdir docs/presentation/figures/uncertainty \
  --basename bayesian_uncertainty_reduction \
  --dpi 300 \
  --formats svg,png
```
